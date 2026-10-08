[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$SkipUploadDocs,
    [switch]$ForceUploadDocs,
    [switch]$OpenBrowser,
    [switch]$SmokeTest,
    [int]$ApiPort = 9900
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$LogsDir = Join-Path $Root "logs"
$PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
$UploadMarker = Join-Path $Root ".start-project-docs-indexed"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-CommandExists {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-TcpPort {
    param([int]$Port)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        return $null -ne $connections
    } catch {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
            $connected = $async.AsyncWaitHandle.WaitOne(500)
            if ($connected) {
                $client.EndConnect($async)
            }
            return $connected
        } catch {
            return $false
        } finally {
            $client.Close()
        }
    }
}

function Wait-TcpPort {
    param(
        [int]$Port,
        [string]$Name,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-TcpPort $Port) {
            Write-Ok "$Name is listening on port $Port"
            return $true
        }
        Start-Sleep -Seconds 2
    }

    Write-Warn "$Name did not listen on port $Port within $TimeoutSeconds seconds"
    return $false
}

function Get-DotEnvValue {
    param([string]$Name)
    if (-not (Test-Path ".env")) {
        return $null
    }

    $line = Select-String -Path ".env" -Pattern "^\s*$Name\s*=" | Select-Object -Last 1
    if (-not $line) {
        return $null
    }

    return (($line.Line -split "=", 2)[1]).Trim().Trim('"').Trim("'")
}

function Assert-DotEnv {
    Write-Step "Checking .env"

    if (-not (Test-Path ".env")) {
        throw ".env file is missing. Please create it and set DASHSCOPE_API_KEY first."
    }

    $dashscopeKey = Get-DotEnvValue "DASHSCOPE_API_KEY"
    if ([string]::IsNullOrWhiteSpace($dashscopeKey) -or $dashscopeKey -match "your|replace|todo") {
        throw "DASHSCOPE_API_KEY is empty or looks like a placeholder. Please edit .env first."
    }

    $clsTransport = Get-DotEnvValue "MCP_CLS_TRANSPORT"
    $clsUrl = Get-DotEnvValue "MCP_CLS_URL"
    if ($clsTransport -ne "streamable-http" -or $clsUrl -ne "http://localhost:8003/mcp") {
        Write-Warn "Local demo usually expects MCP_CLS_TRANSPORT=streamable-http and MCP_CLS_URL=http://localhost:8003/mcp"
        Write-Warn "Current CLS MCP config: transport=$clsTransport, url=$clsUrl"
    }

    Write-Ok ".env is ready"
}

function Ensure-PythonEnv {
    Write-Step "Checking Python virtual environment"

    if ($SkipInstall -and (Test-Path $PythonExe)) {
        Write-Ok "SkipInstall is set and .venv already exists"
        return
    }

    if ((Test-Path $PythonExe) -and $SkipInstall) {
        return
    }

    if (Test-Path $PythonExe) {
        Write-Ok ".venv already exists"
        return
    }

    if (Test-CommandExists "uv") {
        Write-Host "uv found, running: uv sync"
        & uv sync
        if ($LASTEXITCODE -eq 0 -and (Test-Path $PythonExe)) {
            Write-Ok "Dependencies installed with uv"
            return
        }
        Write-Warn "uv sync failed or did not create .venv, falling back to python venv"
    }

    if (-not (Test-CommandExists "python")) {
        throw "python command is not available. Please install Python 3.11, 3.12, or 3.13."
    }

    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv"
    }

    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip"
    }

    & $PythonExe -m pip install -e .
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install project dependencies"
    }

    Write-Ok "Dependencies installed with pip"
}

function Ensure-Docker {
    Write-Step "Checking Docker"

    if (-not (Test-CommandExists "docker")) {
        throw "docker command is not available. Please install Docker Desktop first."
    }

    & docker info *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Docker is running"
        return
    }

    $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerDesktop) {
        Write-Warn "Docker is not running. Trying to start Docker Desktop..."
        Start-Process -FilePath $dockerDesktop -WindowStyle Minimized
        $deadline = (Get-Date).AddSeconds(120)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 5
            & docker info *> $null
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "Docker is running"
                return
            }
        }
    }

    throw "Docker is not running. Please start Docker Desktop and run this script again."
}

function Start-Milvus {
    Write-Step "Starting Milvus"

    & docker compose -f "vector-database.yml" up -d
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed. Please check Docker Desktop."
    }

    Wait-TcpPort -Port 19530 -Name "Milvus" -TimeoutSeconds 90 | Out-Null
}

function Start-PythonServiceIfNeeded {
    param(
        [string]$Name,
        [int]$Port,
        [string[]]$Arguments,
        [string]$StdoutLog,
        [string]$StderrLog,
        [int]$WaitSeconds = 30
    )

    if (Test-TcpPort $Port) {
        Write-Ok "$Name already running on port $Port"
        return
    }

    Write-Host "Starting $Name..."
    Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Arguments `
        -WorkingDirectory $Root `
        -WindowStyle Minimized `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        | Out-Null

    Wait-TcpPort -Port $Port -Name $Name -TimeoutSeconds $WaitSeconds | Out-Null
}

function Start-McpServices {
    Write-Step "Starting MCP services"

    Start-PythonServiceIfNeeded `
        -Name "CLS MCP" `
        -Port 8003 `
        -Arguments @("mcp_servers\cls_server.py") `
        -StdoutLog (Join-Path $LogsDir "cls-mcp.out.log") `
        -StderrLog (Join-Path $LogsDir "cls-mcp.err.log") `
        -WaitSeconds 30

    Start-PythonServiceIfNeeded `
        -Name "Monitor MCP" `
        -Port 8004 `
        -Arguments @("mcp_servers\monitor_server.py") `
        -StdoutLog (Join-Path $LogsDir "monitor-mcp.out.log") `
        -StderrLog (Join-Path $LogsDir "monitor-mcp.err.log") `
        -WaitSeconds 30
}

function Wait-ApiHealth {
    param([int]$TimeoutSeconds = 120)

    $url = "http://127.0.0.1:$ApiPort/health"
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        try {
            $health = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 5
            if ($health.data.status -eq "healthy" -and $health.data.milvus.status -eq "connected") {
                Write-Ok "FastAPI health check passed"
                return $true
            }
        } catch {
            Start-Sleep -Seconds 3
        }
    }

    Write-Warn "FastAPI health check did not pass within $TimeoutSeconds seconds"
    return $false
}

function Start-ApiService {
    Write-Step "Starting FastAPI"

    if (Test-TcpPort $ApiPort) {
        Write-Ok "FastAPI already running on port $ApiPort"
    } else {
        Start-Process `
            -FilePath $PythonExe `
            -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$ApiPort") `
            -WorkingDirectory $Root `
            -WindowStyle Minimized `
            -RedirectStandardOutput (Join-Path $LogsDir "superbiz-api.out.log") `
            -RedirectStandardError (Join-Path $LogsDir "superbiz-api.err.log") `
            | Out-Null
    }

    if (-not (Wait-ApiHealth -TimeoutSeconds 120)) {
        Write-Warn "Check logs\app_YYYY-MM-DD.log and logs\superbiz-api.err.log for details"
    }
}

function Index-DocsIfNeeded {
    if ($SkipUploadDocs) {
        Write-Ok "Document indexing skipped by -SkipUploadDocs"
        return
    }

    if ((Test-Path $UploadMarker) -and (-not $ForceUploadDocs)) {
        Write-Ok "aiops-docs was already indexed once. Use -ForceUploadDocs to rebuild it."
        return
    }

    Write-Step "Indexing aiops-docs into Milvus"

    $docsDir = Join-Path $Root "aiops-docs"
    if (-not (Test-Path $docsDir)) {
        Write-Warn "aiops-docs directory not found, skipping document indexing"
        return
    }

    $encodedDir = [uri]::EscapeDataString("aiops-docs")
    $url = "http://127.0.0.1:$ApiPort/api/index_directory?directory_path=$encodedDir"
    $response = Invoke-RestMethod -Uri $url -Method Post -TimeoutSec 300

    if ($response.data.success -eq $true) {
        Set-Content -Path $UploadMarker -Value ("indexed_at=" + (Get-Date).ToString("s")) -Encoding UTF8
        Write-Ok "aiops-docs indexed: total=$($response.data.total_files), success=$($response.data.success_count), failed=$($response.data.fail_count)"
    } else {
        Write-Warn "Document indexing finished with failures"
        Write-Warn ($response.data | ConvertTo-Json -Compress)
    }
}

function Run-SmokeTest {
    if (-not $SmokeTest) {
        return
    }

    Write-Step "Running smoke test for query understanding"

    $session = "startup-smoke-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    $body = @{
        Id = $session
        Question = "aiops cpu high troubleshooting"
    } | ConvertTo-Json -Compress

    $response = Invoke-WebRequest `
        -Uri "http://127.0.0.1:$ApiPort/api/chat_stream" `
        -Method Post `
        -ContentType "application/json; charset=utf-8" `
        -Body $body `
        -UseBasicParsing `
        -TimeoutSec 120

    if ($response.StatusCode -eq 200 -and $response.Content -match "query_understanding" -and $response.Content -match "aiops_diagnosis") {
        Write-Ok "Smoke test passed: query_understanding + aiops_diagnosis found"
    } else {
        Write-Warn "Smoke test did not find expected query_understanding output"
    }
}

try {
    Write-Host "OpsPilot AI one-click startup" -ForegroundColor Cyan
    Write-Host "Project root: $Root"

    Assert-DotEnv
    Ensure-PythonEnv
    Ensure-Docker
    Start-Milvus
    Start-McpServices
    Start-ApiService
    Index-DocsIfNeeded
    Run-SmokeTest

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "OpsPilot AI is ready" -ForegroundColor Green
    Write-Host "Web UI:    http://127.0.0.1:$ApiPort"
    Write-Host "API docs:  http://127.0.0.1:$ApiPort/docs"
    Write-Host "Health:    http://127.0.0.1:$ApiPort/health"
    Write-Host "Logs:      $LogsDir"
    Write-Host "Stop:      .\stop-project.bat"
    Write-Host "========================================" -ForegroundColor Green

    if ($OpenBrowser) {
        Start-Process "http://127.0.0.1:$ApiPort"
    }
} catch {
    Write-Fail $_.Exception.Message
    Write-Host ""
    Write-Host "Useful checks:"
    Write-Host "1. Make sure Docker Desktop is running."
    Write-Host "2. Make sure .env contains a valid DASHSCOPE_API_KEY."
    Write-Host "3. Check logs\app_YYYY-MM-DD.log and logs\*.err.log."
    exit 1
}
