[CmdletBinding()]
param(
    [switch]$KeepDocker,
    [int]$ApiPort = 9900
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

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

function Stop-PortOwner {
    param(
        [int]$Port,
        [string]$Name
    )

    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $connections) {
        Write-Ok "$Name is not running on port $Port"
        return
    }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        try {
            $process = Get-Process -Id $processId -ErrorAction Stop
            Stop-Process -Id $processId -Force
            Write-Ok "Stopped ${Name}: pid=$processId, process=$($process.ProcessName)"
        } catch {
            Write-Warn "Failed to stop $Name on port ${Port}: $($_.Exception.Message)"
        }
    }
}

Write-Host "Stopping OpsPilot AI services" -ForegroundColor Cyan

Write-Step "Stopping Python services"
Stop-PortOwner -Port $ApiPort -Name "FastAPI"
Stop-PortOwner -Port 8003 -Name "CLS MCP"
Stop-PortOwner -Port 8004 -Name "Monitor MCP"

if ($KeepDocker) {
    Write-Ok "Docker containers kept because -KeepDocker is set"
} else {
    Write-Step "Stopping Milvus Docker containers"
    docker compose -f "vector-database.yml" down
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Milvus Docker containers stopped"
    } else {
        Write-Warn "docker compose down failed or Docker is not running"
    }
}

Write-Host ""
Write-Host "Stop command finished." -ForegroundColor Green
