// OpsPilot AI 前端应用
class OpsPilotApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:9900/api';
        this.currentMode = 'quick'; // 'quick' 或 'stream'
        this.sessionId = this.generateSessionId();
        this.isStreaming = false;
        this.chatHistories = this.loadChatHistories();
        this.isCurrentChatFromHistory = false;
        this.authToken = localStorage.getItem('authToken') || '';
        this.currentUser = this.loadCurrentUser();
        this.currentView = 'chat';
        this.currentChatHistory = []; // 当前对话的消息历史
        this.chatHistories = this.loadChatHistories(); // 所有历史对话
        this.isCurrentChatFromHistory = false; // 标记当前对话是否是从历史记录加载的

        this.installEnterpriseUI();
        this.initializeElements();
        this.bindEvents();
        this.updateUI();
        this.initMarkdown();
        this.checkAndSetCentered();
        this.renderChatHistory();
        this.updateAuthUI();
        this.ensureLoggedIn();
    }

    // 初始化Markdown配置
    installEnterpriseUI() {
        const newChatBtn = document.getElementById('newChatBtn');
        if (newChatBtn && !document.getElementById('knowledgeBaseBtn')) {
            newChatBtn.insertAdjacentHTML('afterend', `
                <button class="knowledge-nav-btn" id="knowledgeBaseBtn">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                    </svg>
                    <span>知识库管理</span>
                </button>
            `);
        }

        const sidebar = document.querySelector('.sidebar');
        if (sidebar && !document.getElementById('userPanel')) {
            sidebar.insertAdjacentHTML('beforeend', `
                <div class="user-panel" id="userPanel">
                    <div class="user-avatar" id="userAvatar">A</div>
                    <div class="user-meta">
                        <div class="user-name" id="userName">未登录</div>
                        <div class="user-role" id="userRole">Local demo auth</div>
                    </div>
                    <button class="user-logout-btn" id="logoutBtn" title="退出登录">退出</button>
                </div>
            `);
        }

        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer && !document.getElementById('knowledgePage')) {
            chatContainer.insertAdjacentHTML('beforebegin', `
                <section class="knowledge-page" id="knowledgePage">
                    <div class="kb-header">
                        <div>
                            <h1>知识库管理</h1>
                            <p>查看上传文档、索引状态、入库记录和 Trace。</p>
                        </div>
                        <div class="kb-actions">
                            <button class="kb-secondary-btn" id="knowledgeRefreshBtn">刷新</button>
                            <button class="kb-primary-btn" id="knowledgeUploadBtn">上传文档</button>
                        </div>
                    </div>
                    <div class="kb-stats">
                        <div class="kb-stat"><span id="kbFileCount">0</span><label>文档数</label></div>
                        <div class="kb-stat"><span id="kbIndexedCount">0</span><label>已索引</label></div>
                        <div class="kb-stat"><span id="kbErrorCount">0</span><label>异常</label></div>
                        <div class="kb-stat"><span id="kbTotalSize">0 KB</span><label>总大小</label></div>
                    </div>
                    <div class="kb-layout">
                        <section class="kb-panel kb-file-panel">
                            <div class="kb-panel-title">
                                <h2>文件列表</h2>
                                <span id="kbLastLoaded">尚未加载</span>
                            </div>
                            <div class="kb-table-wrap">
                                <table class="kb-table">
                                    <thead>
                                        <tr>
                                            <th>文件名</th>
                                            <th>大小</th>
                                            <th>索引状态</th>
                                            <th>分片</th>
                                            <th>更新时间</th>
                                            <th>操作</th>
                                        </tr>
                                    </thead>
                                    <tbody id="knowledgeTableBody"></tbody>
                                </table>
                            </div>
                        </section>
                        <aside class="kb-panel kb-trace-panel">
                            <div class="kb-panel-title">
                                <h2>入库 Trace</h2>
                                <span id="kbTraceTitle">最近记录</span>
                            </div>
                            <div class="kb-trace-list" id="knowledgeTraceList"></div>
                        </aside>
                    </div>
                    <div class="kb-ops-layout">
                        <section class="kb-panel kb-term-panel">
                            <div class="kb-panel-title">
                                <div>
                                    <h2>术语映射</h2>
                                    <span>把口语化告警归一化为稳定检索词</span>
                                </div>
                                <button class="kb-secondary-btn" id="termMappingRefreshBtn">刷新</button>
                            </div>
                            <div class="term-form">
                                <input id="termSourceInput" placeholder="用户说法，如 CPU飙高">
                                <input id="termTargetInput" placeholder="标准术语，如 CPU 使用率过高">
                                <input id="termScenarioInput" placeholder="场景，如 AIOps">
                                <button class="kb-primary-btn" id="termMappingAddBtn">添加</button>
                            </div>
                            <div class="kb-table-wrap">
                                <table class="kb-table term-table">
                                    <thead>
                                        <tr>
                                            <th>用户说法</th>
                                            <th>标准术语</th>
                                            <th>场景</th>
                                            <th>优先级</th>
                                            <th>操作</th>
                                        </tr>
                                    </thead>
                                    <tbody id="termMappingTableBody"></tbody>
                                </table>
                            </div>
                        </section>
                        <section class="kb-panel kb-retrieval-panel">
                            <div class="kb-panel-title">
                                <div>
                                    <h2>RAG Trace 2.0</h2>
                                    <span>查询理解、多通道召回、RRF 融合和 Rerank 归因</span>
                                </div>
                                <button class="kb-primary-btn" id="ragTraceRunBtn">运行 Trace</button>
                            </div>
                            <div class="trace-lab-form">
                                <input id="ragTraceQuestionInput" value="CPU飙高怎么排查？">
                            </div>
                            <div class="rag-trace-result" id="ragTraceResult">
                                <div class="kb-empty">输入问题后运行 Trace，可查看检索链路。</div>
                            </div>
                        </section>
                    </div>
                    <section class="kb-panel kb-eval-panel">
                        <div class="kb-panel-title">
                            <div>
                                <h2>RAG 量化评测</h2>
                                <span>外部评测集，对比原始向量检索和增强检索链路</span>
                            </div>
                            <button class="kb-primary-btn" id="ragEvalRunBtn">运行评测</button>
                        </div>
                        <div class="rag-eval-summary" id="ragEvalSummary">
                            <div class="kb-empty">正在加载评测集元信息...</div>
                        </div>
                        <div class="rag-eval-results" id="ragEvalResults"></div>
                        <div class="rag-eval-updated" id="ragEvalUpdatedAt"></div>
                    </section>
                </section>
            `);
        }

        if (chatContainer && !document.getElementById('diagnosticWorkbench')) {
            const shell = document.createElement('div');
            shell.className = 'ops-workbench-shell';
            chatContainer.parentNode.insertBefore(shell, chatContainer);
            shell.appendChild(chatContainer);
            shell.insertAdjacentHTML('beforeend', `
                <aside class="diagnostic-workbench" id="diagnosticWorkbench">
                    <div class="workbench-header">
                        <div>
                            <h2>诊断工作台</h2>
                            <p>查看从问题理解到证据与诊断结论的执行链路</p>
                        </div>
                        <button class="workbench-export-btn" id="exportReportBtn" disabled>导出报告</button>
                    </div>
                    <section class="workbench-card workbench-objective" id="workbenchObjective">
                        <span class="workbench-empty">发起一次故障诊断后，这里会显示 Agent 的任务目标。</span>
                    </section>
                    <section class="workbench-card">
                        <div class="workbench-card-title">故障画像</div>
                        <div class="incident-grid" id="workbenchIncident">
                            <span class="workbench-empty">等待诊断结果</span>
                        </div>
                    </section>
                    <section class="workbench-card">
                        <div class="workbench-card-title">执行步骤</div>
                        <div class="workbench-steps" id="workbenchSteps"></div>
                    </section>
                    <section class="workbench-card">
                        <div class="workbench-card-title">工具与检索</div>
                        <div class="workbench-tools" id="workbenchTools"></div>
                    </section>
                    <section class="workbench-card">
                        <div class="workbench-card-title">RAG Trace</div>
                        <div class="workbench-rag" id="workbenchRagTrace"></div>
                    </section>
                    <section class="workbench-card">
                        <div class="workbench-card-title">风险操作确认</div>
                        <div class="workbench-risks" id="workbenchRisks">
                            <span class="workbench-empty">本轮暂无待确认操作</span>
                        </div>
                    </section>
                    <section class="workbench-card">
                        <div class="workbench-card-title">下一步补充</div>
                        <div class="workbench-questions" id="workbenchQuestions"></div>
                    </section>
                </aside>
            `);
        }

        if (!document.getElementById('loginModal')) {
            document.body.insertAdjacentHTML('beforeend', `
                <div class="login-modal" id="loginModal">
                    <form class="login-card" id="loginForm">
                        <div class="login-brand">OpsPilot AI</div>
                        <h2>登录运维控制台</h2>
                        <p>使用 .env 中配置的账号和密码</p>
                        <label>用户名</label>
                        <input id="loginUsername" value="admin" autocomplete="username">
                        <label>密码</label>
                        <input id="loginPassword" type="password" autocomplete="current-password">
                        <button type="submit">登录</button>
                        <div class="login-error" id="loginError"></div>
                    </form>
                </div>
            `);
        }
    }

    initMarkdown() {
        // 等待 marked 库加载完成
        const checkMarked = () => {
            if (typeof marked !== 'undefined') {
                try {
                    // 配置marked选项
                    marked.setOptions({
                        breaks: true,  // 支持GFM换行
                        gfm: true,     // 启用GitHub风格的Markdown
                        headerIds: false,
                        mangle: false
                    });

                    // 配置代码高亮
                    if (typeof hljs !== 'undefined') {
                        marked.setOptions({
                            highlight: function(code, lang) {
                                if (lang && hljs.getLanguage(lang)) {
                                    try {
                                        return hljs.highlight(code, { language: lang }).value;
                                    } catch (err) {
                                        console.error('代码高亮失败:', err);
                                    }
                                }
                                return code;
                            }
                        });
                    }
                    console.log('Markdown 渲染库初始化成功');
                } catch (e) {
                    console.error('Markdown 配置失败:', e);
                }
            } else {
                // 如果 marked 还没加载，等待一段时间后重试
                setTimeout(checkMarked, 100);
            }
        };
        checkMarked();
    }

    // 安全地渲染 Markdown
    renderMarkdown(content) {
        if (!content) return '';

        // 检查 marked 是否可用
        if (typeof marked === 'undefined') {
            console.warn('marked 库未加载，使用纯文本显示');
            return this.escapeHtml(content);
        }

        try {
            const html = marked.parse(content);
            return html;
        } catch (e) {
            console.error('Markdown 渲染失败:', e);
            return this.escapeHtml(content);
        }
    }

    // 高亮代码块
    highlightCodeBlocks(container) {
        if (typeof hljs !== 'undefined' && container) {
            try {
                container.querySelectorAll('pre code').forEach((block) => {
                    if (!block.classList.contains('hljs')) {
                        hljs.highlightElement(block);
                    }
                });
            } catch (e) {
                console.error('代码高亮失败:', e);
            }
        }
    }

    // 初始化DOM元素
    initializeElements() {
        // 侧边栏元素
        this.sidebar = document.querySelector('.sidebar');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.aiOpsSidebarBtn = document.getElementById('aiOpsSidebarBtn');

        // 输入区域元素
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.toolsBtn = document.getElementById('toolsBtn');
        this.toolsMenu = document.getElementById('toolsMenu');
        this.uploadFileItem = document.getElementById('uploadFileItem');
        this.modeSelectorBtn = document.getElementById('modeSelectorBtn');
        this.modeDropdown = document.getElementById('modeDropdown');
        this.currentModeText = document.getElementById('currentModeText');
        this.fileInput = document.getElementById('fileInput');
        this.knowledgeBaseBtn = document.getElementById('knowledgeBaseBtn');
        this.knowledgePage = document.getElementById('knowledgePage');
        this.knowledgeRefreshBtn = document.getElementById('knowledgeRefreshBtn');
        this.knowledgeUploadBtn = document.getElementById('knowledgeUploadBtn');
        this.knowledgeTableBody = document.getElementById('knowledgeTableBody');
        this.knowledgeTraceList = document.getElementById('knowledgeTraceList');
        this.kbTraceTitle = document.getElementById('kbTraceTitle');
        this.kbLastLoaded = document.getElementById('kbLastLoaded');
        this.kbFileCount = document.getElementById('kbFileCount');
        this.kbIndexedCount = document.getElementById('kbIndexedCount');
        this.kbErrorCount = document.getElementById('kbErrorCount');
        this.kbTotalSize = document.getElementById('kbTotalSize');
        this.ragEvalRunBtn = document.getElementById('ragEvalRunBtn');
        this.ragEvalSummary = document.getElementById('ragEvalSummary');
        this.ragEvalResults = document.getElementById('ragEvalResults');
        this.ragEvalUpdatedAt = document.getElementById('ragEvalUpdatedAt');
        this.termMappingRefreshBtn = document.getElementById('termMappingRefreshBtn');
        this.termMappingAddBtn = document.getElementById('termMappingAddBtn');
        this.termMappingTableBody = document.getElementById('termMappingTableBody');
        this.termSourceInput = document.getElementById('termSourceInput');
        this.termTargetInput = document.getElementById('termTargetInput');
        this.termScenarioInput = document.getElementById('termScenarioInput');
        this.ragTraceRunBtn = document.getElementById('ragTraceRunBtn');
        this.ragTraceQuestionInput = document.getElementById('ragTraceQuestionInput');
        this.ragTraceResult = document.getElementById('ragTraceResult');
        this.workbenchShell = document.querySelector('.ops-workbench-shell');
        this.diagnosticWorkbench = document.getElementById('diagnosticWorkbench');
        this.exportReportBtn = document.getElementById('exportReportBtn');
        this.workbenchObjective = document.getElementById('workbenchObjective');
        this.workbenchIncident = document.getElementById('workbenchIncident');
        this.workbenchSteps = document.getElementById('workbenchSteps');
        this.workbenchTools = document.getElementById('workbenchTools');
        this.workbenchRagTrace = document.getElementById('workbenchRagTrace');
        this.workbenchRisks = document.getElementById('workbenchRisks');
        this.workbenchQuestions = document.getElementById('workbenchQuestions');
        this.loginModal = document.getElementById('loginModal');
        this.loginForm = document.getElementById('loginForm');
        this.loginUsername = document.getElementById('loginUsername');
        this.loginPassword = document.getElementById('loginPassword');
        this.loginError = document.getElementById('loginError');
        this.userName = document.getElementById('userName');
        this.userRole = document.getElementById('userRole');
        this.userAvatar = document.getElementById('userAvatar');
        this.logoutBtn = document.getElementById('logoutBtn');

        // 聊天区域元素
        this.chatMessages = document.getElementById('chatMessages');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.chatContainer = document.querySelector('.chat-container');
        this.welcomeGreeting = document.getElementById('welcomeGreeting');
        this.chatHistoryList = document.getElementById('chatHistoryList');

        // 初始化时检查是否需要居中
        this.checkAndSetCentered();
    }

    // 绑定事件监听器
    bindEvents() {
        // 新建对话
        if (this.newChatBtn) {
            this.newChatBtn.addEventListener('click', () => this.newChat());
        }

        if (this.knowledgeBaseBtn) {
            this.knowledgeBaseBtn.addEventListener('click', () => this.showKnowledgePage());
        }

        if (this.knowledgeRefreshBtn) {
            this.knowledgeRefreshBtn.addEventListener('click', () => this.loadKnowledgeDashboard());
        }

        if (this.knowledgeUploadBtn) {
            this.knowledgeUploadBtn.addEventListener('click', () => {
                if (this.fileInput) {
                    this.fileInput.click();
                }
            });
        }

        if (this.knowledgeTableBody) {
            this.knowledgeTableBody.addEventListener('click', (e) => this.handleKnowledgeTableAction(e));
        }

        if (this.ragEvalRunBtn) {
            this.ragEvalRunBtn.addEventListener('click', () => this.runRagEval());
        }

        if (this.termMappingRefreshBtn) {
            this.termMappingRefreshBtn.addEventListener('click', () => this.loadTermMappings());
        }

        if (this.termMappingAddBtn) {
            this.termMappingAddBtn.addEventListener('click', () => this.addTermMapping());
        }

        if (this.termMappingTableBody) {
            this.termMappingTableBody.addEventListener('click', (e) => this.handleTermMappingAction(e));
        }

        if (this.ragTraceRunBtn) {
            this.ragTraceRunBtn.addEventListener('click', () => this.runRetrievalTrace());
        }

        if (this.exportReportBtn) {
            this.exportReportBtn.addEventListener('click', () => this.exportDiagnosticReport());
        }

        if (this.workbenchRisks) {
            this.workbenchRisks.addEventListener('click', (e) => this.handleRiskAction(e));
        }

        if (this.loginForm) {
            this.loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }

        if (this.logoutBtn) {
            this.logoutBtn.addEventListener('click', () => this.logout());
        }

        // AI Ops按钮
        if (this.aiOpsSidebarBtn) {
            this.aiOpsSidebarBtn.addEventListener('click', () => this.triggerAIOps());
        }

        // 模式选择下拉菜单
        if (this.modeSelectorBtn) {
            this.modeSelectorBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleModeDropdown();
            });
        }

        // 下拉菜单项点击
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const mode = item.getAttribute('data-mode');
                this.selectMode(mode);
                this.closeModeDropdown();
            });
        });

        // 点击外部关闭下拉菜单
        document.addEventListener('click', (e) => {
            if (!this.modeSelectorBtn.contains(e.target) &&
                !this.modeDropdown.contains(e.target)) {
                this.closeModeDropdown();
            }
        });

        // 发送消息
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => this.sendMessage());
        }

        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // 工具按钮和菜单
        if (this.toolsBtn) {
            this.toolsBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleToolsMenu();
            });
        }

        // 工具菜单项点击事件
        if (this.uploadFileItem) {
            this.uploadFileItem.addEventListener('click', () => {
                if (this.fileInput) {
                    this.fileInput.click();
                }
                this.closeToolsMenu();
            });
        }

        // 点击外部关闭工具菜单
        document.addEventListener('click', (e) => {
            if (this.toolsBtn && this.toolsMenu &&
                !this.toolsBtn.contains(e.target) &&
                !this.toolsMenu.contains(e.target)) {
                this.closeToolsMenu();
            }
        });

        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
    }

    // 切换工具菜单显示/隐藏
    toggleToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) {
                wrapper.classList.toggle('active');
            }
        }
    }

    // 关闭工具菜单
    closeToolsMenu() {
        if (this.toolsMenu && this.toolsBtn) {
            const wrapper = this.toolsBtn.closest('.tools-btn-wrapper');
            if (wrapper) {
                wrapper.classList.remove('active');
            }
        }
    }

    // 新建对话
    newChat() {
        this.showChatPage();
        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成后再新建对话', 'warning');
            return;
        }

        // 如果当前有对话内容，且不是从历史记录加载的，才保存为新的历史对话
        // 如果是从历史记录加载的，只需要更新该历史记录
        if (this.currentChatHistory.length > 0) {
            if (this.isCurrentChatFromHistory) {
                // 当前对话是从历史记录加载的，更新该历史记录
                this.updateCurrentChatHistory();
            } else {
                // 当前对话是新对话，保存为新的历史对话
                this.saveCurrentChat();
            }
        }

        // 停止所有进行中的操作
        this.isStreaming = false;

        // 清空输入框
        if (this.messageInput) {
            this.messageInput.value = '';
        }

        // 清空当前对话历史
        this.currentChatHistory = [];

        // 重置标记
        this.isCurrentChatFromHistory = false;

        // 清空聊天记录
        if (this.chatMessages) {
            this.chatMessages.innerHTML = '';
        }

        // 生成新的会话ID
        this.sessionId = this.generateSessionId();

        // 重置模式为快速
        this.currentMode = 'quick';
        this.updateUI();

        // 重新设置居中样式（确保对话框居中显示）
        this.checkAndSetCentered();

        // 确保容器有过渡动画
        if (this.chatContainer) {
            this.chatContainer.style.transition = 'all 0.5s ease';
        }

        // 更新历史对话列表
        this.renderChatHistory();
    }

    // 保存当前对话到历史记录（新建）
    saveCurrentChat() {
        if (this.currentChatHistory.length === 0) {
            return;
        }

        // 检查是否已存在相同ID的历史记录
        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex !== -1) {
            // 如果已存在，更新而不是新建
            this.updateCurrentChatHistory();
            return;
        }

        // 获取对话标题（使用第一条用户消息的前30个字符）
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        const title = firstUserMessage ?
            (firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '')) :
            '新对话';

        const chatHistory = {
            id: this.sessionId,
            title: title,
            messages: [...this.currentChatHistory],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        // 添加到历史记录列表的开头
        this.chatHistories.unshift(chatHistory);

        // 限制历史记录数量（最多保存50条）
        if (this.chatHistories.length > 50) {
            this.chatHistories = this.chatHistories.slice(0, 50);
        }

        // 保存到localStorage
        this.saveChatHistories();
    }

    // 更新当前对话的历史记录
    updateCurrentChatHistory() {
        if (this.currentChatHistory.length === 0) {
            return;
        }

        const existingIndex = this.chatHistories.findIndex(h => h.id === this.sessionId);
        if (existingIndex === -1) {
            // 如果不存在，调用保存方法
            this.saveCurrentChat();
            return;
        }

        // 更新现有的历史记录
        const history = this.chatHistories[existingIndex];
        history.messages = [...this.currentChatHistory];
        history.updatedAt = new Date().toISOString();

        // 如果标题需要更新（第一条消息改变了）
        const firstUserMessage = this.currentChatHistory.find(msg => msg.type === 'user');
        if (firstUserMessage) {
            const newTitle = firstUserMessage.content.substring(0, 30) + (firstUserMessage.content.length > 30 ? '...' : '');
            if (history.title !== newTitle) {
                history.title = newTitle;
            }
        }

        // 保存到localStorage
        this.saveChatHistories();
    }

    // 加载历史对话列表
    loadChatHistories() {
        try {
            const stored = localStorage.getItem('chatHistories');
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.error('加载历史对话失败:', e);
            return [];
        }
    }

    // 保存历史对话列表到localStorage
    saveChatHistories() {
        try {
            localStorage.setItem('chatHistories', JSON.stringify(this.chatHistories));
        } catch (e) {
            console.error('保存历史对话失败:', e);
        }
    }

    // 渲染历史对话列表
    renderChatHistory() {
        if (!this.chatHistoryList) {
            return;
        }

        this.chatHistoryList.innerHTML = '';

        if (this.chatHistories.length === 0) {
            return;
        }

        this.chatHistories.forEach((history, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.dataset.historyId = history.id;

            historyItem.innerHTML = `
                <div class="history-item-content">
                    <span class="history-item-title">${this.escapeHtml(history.title)}</span>
                </div>
                <button class="history-item-delete" data-history-id="${history.id}" title="删除">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            `;

            // 点击历史项加载对话
            historyItem.addEventListener('click', (e) => {
                if (!e.target.closest('.history-item-delete')) {
                    this.loadChatHistory(history.id);
                }
            });

            // 删除历史对话
            const deleteBtn = historyItem.querySelector('.history-item-delete');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteChatHistory(history.id);
            });

            this.chatHistoryList.appendChild(historyItem);
        });
    }

    // 加载历史对话
    async loadChatHistory(historyId) {
        this.showChatPage();
        const history = this.chatHistories.find(h => h.id === historyId);
        if (!history) {
            return;
        }

        // 如果当前有对话内容，且不是同一个对话，先保存
        if (this.currentChatHistory.length > 0 && this.sessionId !== historyId) {
            if (this.isCurrentChatFromHistory) {
                // 如果当前对话也是从历史记录加载的，更新它
                this.updateCurrentChatHistory();
            } else {
                // 如果当前对话是新对话，保存为新历史
                this.saveCurrentChat();
            }
        }

        try {
            // 从后端获取会话历史
            const response = await fetch(`/api/chat/session/${historyId}`);
            if (response.ok) {
                const data = await response.json();
                const backendHistory = data.history || [];

                // 更新会话ID
                this.sessionId = history.id;
                this.isCurrentChatFromHistory = true;

                // 清空并重新渲染消息
                if (this.chatMessages) {
                    this.chatMessages.innerHTML = '';

                    // 如果后端有历史记录，使用后端的
                    if (backendHistory.length > 0) {
                        this.currentChatHistory = [];
                        backendHistory.forEach(msg => {
                            // 后端返回格式: {role: "user|assistant", content: "...", timestamp: "..."}
                            const messageType = msg.role === 'user' ? 'user' : 'bot';
                            this.addMessage(messageType, msg.content, false, false);
                        });
                    } else {
                        // 否则使用localStorage的历史记录
                        this.currentChatHistory = [...history.messages];
                        history.messages.forEach(msg => {
                            this.addMessage(msg.type, msg.content, false, false);
                        });
                    }
                }
            } else {
                // 如果后端请求失败，使用localStorage的历史记录
                console.warn('从后端加载历史失败，使用本地缓存');
                this.sessionId = history.id;
                this.currentChatHistory = [...history.messages];
                this.isCurrentChatFromHistory = true;

                if (this.chatMessages) {
                    this.chatMessages.innerHTML = '';
                    history.messages.forEach(msg => {
                        this.addMessage(msg.type, msg.content, false, false);
                    });
                }
            }
        } catch (error) {
            console.error('加载会话历史失败:', error);
            // 出错时使用localStorage的历史记录
            this.sessionId = history.id;
            this.currentChatHistory = [...history.messages];
            this.isCurrentChatFromHistory = true;

            if (this.chatMessages) {
                this.chatMessages.innerHTML = '';
                history.messages.forEach(msg => {
                    this.addMessage(msg.type, msg.content, false, false);
                });
            }
        }

        // 更新UI
        this.checkAndSetCentered();
        this.renderChatHistory();
    }

    // 删除历史对话
    async deleteChatHistory(historyId) {
        try {
            // 调用后端API清空会话
            const response = await fetch('/api/chat/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: historyId
                })
            });

            if (!response.ok) {
                throw new Error('清空会话失败');
            }

            const result = await response.json();

            if (result.status === 'success') {
                // 从本地存储中删除
                this.chatHistories = this.chatHistories.filter(h => h.id !== historyId);
                this.saveChatHistories();
                this.renderChatHistory();

                // 如果删除的是当前对话，清空当前对话
                if (this.sessionId === historyId) {
                    this.currentChatHistory = [];
                    if (this.chatMessages) {
                        this.chatMessages.innerHTML = '';
                    }
                    this.sessionId = this.generateSessionId();
                    this.checkAndSetCentered();
                }

                this.showNotification('会话已清空', 'success');
            } else {
                throw new Error(result.message || '清空会话失败');
            }
        } catch (error) {
            console.error('删除历史对话失败:', error);
            this.showNotification('删除失败: ' + error.message, 'error');
        }
    }

    // 切换模式下拉菜单
    toggleModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) {
                wrapper.classList.toggle('active');
            }
        }
    }

    // 关闭模式下拉菜单
    closeModeDropdown() {
        if (this.modeSelectorBtn && this.modeDropdown) {
            const wrapper = this.modeSelectorBtn.closest('.mode-selector-wrapper');
            if (wrapper) {
                wrapper.classList.remove('active');
            }
        }
    }

    // 选择模式
    selectMode(mode) {
        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成后再切换模式', 'warning');
            return;
        }

        this.currentMode = mode;
        this.updateUI();

        const modeNames = {
            'quick': '快速',
            'stream': '流式'
        };

        this.showNotification(`已切换到${modeNames[mode]}模式`, 'info');
    }

    // 更新UI
    updateUI() {
        // 更新模式选择器显示
        if (this.currentModeText) {
            const modeNames = {
                'quick': '快速',
                'stream': '流式'
            };
            this.currentModeText.textContent = modeNames[this.currentMode] || '快速';
        }

        // 更新下拉菜单选中状态
        const dropdownItems = document.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            const mode = item.getAttribute('data-mode');
            if (mode === this.currentMode) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // 更新发送按钮状态
        if (this.sendButton) {
            this.sendButton.disabled = this.isStreaming;
        }

        // 更新输入框状态
        if (this.messageInput) {
            this.messageInput.disabled = this.isStreaming;
            this.messageInput.placeholder = '描述告警、故障或运维问题';
        }
    }

    // 生成随机会话ID
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    // 发送消息
    async sendMessage() {
        let message = '';
        if (this.messageInput) {
            message = this.messageInput.value.trim();
        }

        if (!message) {
            this.showNotification('请输入消息内容', 'warning');
            return;
        }

        if (this.isStreaming) {
            this.showNotification('请等待当前对话完成', 'warning');
            return;
        }

        // 显示用户消息
        this.addMessage('user', message);

        // 清空输入框
        if (this.messageInput) {
            this.messageInput.value = '';
        }

        // 设置发送状态
        this.isStreaming = true;
        this.updateUI();

        try {
            if (this.currentMode === 'quick') {
                await this.sendQuickMessage(message);
            } else if (this.currentMode === 'stream') {
                await this.sendStreamMessage(message);
            }
        } catch (error) {
            console.error('发送消息失败:', error);
            this.addMessage('assistant', '抱歉，发送消息时出现错误：' + error.message);
        } finally {
            this.isStreaming = false;
            this.updateUI();

            // 如果当前对话是从历史记录加载的，更新历史记录
            if (this.isCurrentChatFromHistory && this.currentChatHistory.length > 0) {
                this.updateCurrentChatHistory();
                this.renderChatHistory(); // 更新历史对话列表显示
            }
        }
    }

    // 发送快速消息（普通对话）
    async sendQuickMessage(message) {
        // 添加等待提示消息
        const loadingMessage = this.addLoadingMessage('正在思考...');

        try {
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    Id: this.sessionId,
                    Question: message,
                    UserId: this.getCurrentUserId()
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            console.log('[sendQuickMessage] 响应数据:', JSON.stringify(data));

            // 移除等待提示消息
            if (loadingMessage && loadingMessage.parentNode) {
                loadingMessage.parentNode.removeChild(loadingMessage);
            }

            // 统一响应格式：检查 data.code 或 data.message 判断请求是否成功
            if (data.code === 200 || data.message === 'success') {
                // data.data 是 ChatResponse 对象
                const chatResponse = data.data;

                if (chatResponse && chatResponse.success) {
                    // 成功：添加实际响应消息（即使 answer 为空也显示）
                    const answer = chatResponse.answer || '（无回复内容）';
                    this.addMessage('assistant', answer, false, true, {workbench: chatResponse.workbench});
                    if (chatResponse.workbench) {
                        this.renderDiagnosticWorkbench(chatResponse.workbench);
                    }
                } else if (chatResponse && chatResponse.errorMessage) {
                    // 业务错误
                    throw new Error(chatResponse.errorMessage);
                } else {
                    // 兜底：尝试显示任何可用内容
                    const fallbackAnswer = chatResponse?.answer || chatResponse?.errorMessage || '服务返回了空内容';
                    this.addMessage('assistant', fallbackAnswer);
                }
            } else {
                // HTTP 成功但业务失败
                throw new Error(data.message || '请求失败');
            }
        } catch (error) {
            // 出错时也要移除等待提示消息
            if (loadingMessage && loadingMessage.parentNode) {
                loadingMessage.parentNode.removeChild(loadingMessage);
            }
            throw error;
        }
    }

    // 发送流式消息
    async sendStreamMessage(message) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat_stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    Id: this.sessionId,
                    Question: message,
                    UserId: this.getCurrentUserId()
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            // 创建助手消息元素
            const assistantMessageElement = this.addMessage('assistant', '', true);
            let fullResponse = '';
            let streamCompleted = false;
            const finalizeStream = () => {
                if (streamCompleted) {
                    return;
                }
                streamCompleted = true;
                fullResponse = this.removeConsecutiveDuplicateText(fullResponse);
                this.handleStreamComplete(assistantMessageElement, fullResponse);
            };

            // 处理流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();

                    if (done) {
                        // 流结束，使用统一的处理方法
                        finalizeStream();
                        break;
                    }

                    // 解码数据并添加到缓冲区
                    buffer += decoder.decode(value, { stream: true });

                    // 按行分割处理
                    const lines = buffer.split('\n');
                    // 保留最后一行（可能不完整）
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.trim() === '') continue;

                        console.log('[SSE调试] 收到行:', line);

                        // 解析SSE格式
                        if (line.startsWith('id:')) {
                            console.log('[SSE调试] 解析到ID');
                            continue;
                        } else if (line.startsWith('event:')) {
                            // 兼容 "event:message" 和 "event: message" 两种格式
                            currentEvent = line.substring(6).trim();
                            console.log('[SSE调试] 解析到事件类型:', currentEvent);
                            // 注意：后端统一使用 "message" 事件名，真正的类型在 data 的 JSON 中
                            continue;
                        } else if (line.startsWith('data:')) {
                            // 兼容 "data:xxx" 和 "data: xxx" 两种格式
                            const rawData = line.substring(5).trim();
                            console.log('[SSE调试] 解析到数据, currentEvent:', currentEvent, ', rawData:', rawData);

                            // 兼容旧格式 [DONE] 标记
                            if (rawData === '[DONE]') {
                                // 流结束标记，将内容转换为Markdown渲染
                                finalizeStream();
                                return;
                            }

                            // 处理 SSE 数据
                            try {
                                // 尝试解析为 SseMessage 格式的 JSON
                                const sseMessage = JSON.parse(rawData);
                                console.log('[SSE调试] 解析JSON成功:', sseMessage);

                                if (sseMessage && typeof sseMessage.type === 'string') {
                                    if (sseMessage.type === 'content') {
                                        const content = sseMessage.data || '';
                                        fullResponse += content;
                                        console.log('[SSE调试] 添加内容:', content);

                                        // 实时渲染 Markdown
                                        if (assistantMessageElement) {
                                            const messageContent = assistantMessageElement.querySelector('.message-content');
                                            messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                            // 高亮代码块
                                            this.highlightCodeBlocks(messageContent);
                                            this.scrollToBottom();
                                        }
                                    } else if (sseMessage.type === 'query_understanding') {
                                        console.log('[SSE调试] 查询理解结果:', sseMessage.data);
                                        if (assistantMessageElement && sseMessage.data) {
                                            assistantMessageElement.dataset.intent = sseMessage.data.intent || '';
                                            assistantMessageElement.dataset.rewrittenQuestion = sseMessage.data.rewritten_question || '';
                                        }
                                    } else if (sseMessage.type === 'agent_route') {
                                        console.log('[SSE调试] Agent 范式路由:', sseMessage.data);
                                        if (assistantMessageElement && sseMessage.data) {
                                            assistantMessageElement.dataset.agentRoute = JSON.stringify(sseMessage.data);
                                        }
                                    } else if (sseMessage.type === 'guidance') {
                                        console.log('[SSE调试] 意图引导建议:', sseMessage.data);
                                        if (assistantMessageElement && sseMessage.data) {
                                            assistantMessageElement.dataset.guidance = JSON.stringify(sseMessage.data);
                                        }
                                    } else if (sseMessage.type === 'model_route') {
                                        console.log('[SSE调试] 模型路由事件:', sseMessage.data);
                                        if (assistantMessageElement && sseMessage.data) {
                                            assistantMessageElement.dataset.modelRoute = JSON.stringify(sseMessage.data);
                                        }
                                    } else if (sseMessage.type === 'search_results') {
                                        console.log('[SSE调试] 检索预取结果:', sseMessage.data);
                                        if (assistantMessageElement && sseMessage.data) {
                                            assistantMessageElement.dataset.retrievalTrace = JSON.stringify(sseMessage.data.trace || {});
                                        }
                                    } else if (sseMessage.type === 'diagnostic_workbench') {
                                        console.log('[SSE调试] 诊断工作台:', sseMessage.data);
                                        if (sseMessage.data) {
                                            this.renderDiagnosticWorkbench(sseMessage.data);
                                            if (assistantMessageElement && !assistantMessageElement.querySelector('.message-workbench-summary')) {
                                                const wrapper = assistantMessageElement.querySelector('.message-content-wrapper');
                                                if (wrapper) {
                                                    wrapper.insertAdjacentHTML('beforeend', this.renderMessageWorkbenchSummary(sseMessage.data));
                                                }
                                            }
                                        }
                                    } else if (sseMessage.type === 'done') {
                                        console.log('[SSE调试] 收到done标记，流结束');
                                        finalizeStream();
                                        return;
                                    } else if (sseMessage.type === 'error') {
                                        console.error('[SSE调试] 收到错误:', sseMessage.data);
                                        if (assistantMessageElement) {
                                            const messageContent = assistantMessageElement.querySelector('.message-content');
                                            messageContent.innerHTML = this.renderMarkdown('错误: ' + (sseMessage.data || '未知错误'));
                                        }
                                        return;
                                    }
                                } else {
                                    // 不是标准 SseMessage 格式，尝试兼容处理
                                    console.log('[SSE调试] 非标准格式，尝试兼容处理');
                                    fullResponse += rawData;
                                    if (assistantMessageElement) {
                                        const messageContent = assistantMessageElement.querySelector('.message-content');
                                        messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                        this.highlightCodeBlocks(messageContent);
                                        this.scrollToBottom();
                                    }
                                }
                            } catch (e) {
                                // JSON 解析失败，尝试兼容旧格式
                                console.log('[SSE调试] JSON解析失败，使用兼容模式:', e.message);
                                if (rawData === '') {
                                    fullResponse += '\n';
                                } else {
                                    fullResponse += rawData;
                                }

                                if (assistantMessageElement) {
                                    const messageContent = assistantMessageElement.querySelector('.message-content');
                                    messageContent.innerHTML = this.renderMarkdown(fullResponse);
                                    this.highlightCodeBlocks(messageContent);
                                    this.scrollToBottom();
                                }
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            throw error;
        }
    }

    // 添加消息到聊天界面
    addMessage(type, content, isStreaming = false, saveToHistory = true, metadata = null) {
        // 检查是否是第一条消息，如果是则移除居中样式
        const isFirstMessage = this.chatMessages && this.chatMessages.querySelectorAll('.message').length === 0;

        // 保存消息到当前对话历史（如果不是流式消息且需要保存）
        if (!isStreaming && saveToHistory && content) {
            this.currentChatHistory.push({
                type: type,
                content: content,
                timestamp: new Date().toISOString(),
                metadata: metadata || undefined
            });
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}${isStreaming ? ' streaming' : ''}`;

        // 如果是assistant消息，添加头像图标
        if (type === 'assistant') {
            const messageAvatar = document.createElement('div');
            messageAvatar.className = 'message-avatar';
            messageAvatar.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
                </svg>
            `;
            messageDiv.appendChild(messageAvatar);
        }

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        // 如果是assistant消息且不是流式消息，使用Markdown渲染
        if (type === 'assistant' && !isStreaming) {
            messageContent.innerHTML = this.renderMarkdown(content);
            // 高亮代码块
            this.highlightCodeBlocks(messageContent);
        } else {
            // 用户消息或流式消息使用纯文本
            messageContent.textContent = content;
        }

        messageContentWrapper.appendChild(messageContent);
        if (type === 'assistant' && metadata && metadata.workbench) {
            messageContentWrapper.insertAdjacentHTML('beforeend', this.renderMessageWorkbenchSummary(metadata.workbench));
        }
        messageDiv.appendChild(messageContentWrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);

            // 如果是第一条消息，移除居中样式并添加动画
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                // 添加动画类
                this.chatContainer.style.transition = 'all 0.5s ease';
            }

            this.scrollToBottom();
        }

        return messageDiv;
    }

    // 添加带加载动画的消息
    addLoadingMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        // 添加头像图标
        const messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        messageAvatar.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
            </svg>
        `;
        messageDiv.appendChild(messageAvatar);

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content loading-message-content';

        // 创建文本和动画容器
        const textSpan = document.createElement('span');
        textSpan.textContent = content;

        // 创建旋转动画图标
        const loadingIcon = document.createElement('span');
        loadingIcon.className = 'loading-spinner-icon';
        loadingIcon.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" fill="currentColor" opacity="0.2"/>
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10c1.54 0 3-.36 4.28-1l-1.5-2.6C13.64 19.62 12.84 20 12 20c-4.41 0-8-3.59-8-8s3.59-8 8-8c.84 0 1.64.38 2.18 1l1.5-2.6C13 2.36 12.54 2 12 2z" fill="currentColor"/>
            </svg>
        `;

        messageContent.appendChild(textSpan);
        messageContent.appendChild(loadingIcon);
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);

            // 如果是第一条消息，移除居中样式
            const isFirstMessage = this.chatMessages.querySelectorAll('.message').length === 1;
            if (isFirstMessage && this.chatContainer) {
                this.chatContainer.classList.remove('centered');
                this.chatContainer.style.transition = 'all 0.5s ease';
            }

            this.scrollToBottom();
        }

        return messageDiv;
    }

    // 检查并设置居中样式
    checkAndSetCentered() {
        if (this.chatMessages && this.chatContainer) {
            const hasMessages = this.chatMessages.querySelectorAll('.message').length > 0;
            if (!hasMessages) {
                this.chatContainer.classList.add('centered');
            } else {
                this.chatContainer.classList.remove('centered');
            }
        }
    }

    // 滚动到底部
    scrollToBottom() {
        if (this.chatMessages) {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }
    }

    // 去掉模型偶发生成的“完整答案连续重复一遍”
    removeConsecutiveDuplicateText(text) {
        const original = String(text || '');
        const trimmed = original.trim();
        if (trimmed.length < 400) {
            return original;
        }

        const compact = (value) => value.replace(/\s+/g, '');
        const middle = Math.floor(trimmed.length / 2);
        const windowSize = Math.min(160, Math.floor(trimmed.length * 0.08));

        for (let split = middle - windowSize; split <= middle + windowSize; split += 1) {
            const left = trimmed.slice(0, split).trim();
            const right = trimmed.slice(split).trim();
            if (compact(left).length < 220 || compact(right).length < 220) {
                continue;
            }
            if (compact(left) === compact(right)) {
                return left;
            }
        }
        return original;
    }

    appendUniqueFinalResponse(existing, finalText) {
        const current = String(existing || '');
        const addition = String(finalText || '').trim();
        if (!addition) {
            return current;
        }

        const compact = (value) => value.replace(/\s+/g, '');
        if (compact(current).includes(compact(addition))) {
            return current;
        }
        return current ? `${current}\n\n${addition}` : addition;
    }

    // 处理流式传输完成
    handleStreamComplete(assistantMessageElement, fullResponse) {
        if (assistantMessageElement && assistantMessageElement.dataset.streamCompleted === 'true') {
            return;
        }
        if (assistantMessageElement) {
            assistantMessageElement.dataset.streamCompleted = 'true';
        }
        fullResponse = this.removeConsecutiveDuplicateText(fullResponse);
        if (assistantMessageElement) {
            assistantMessageElement.classList.remove('streaming');
            const messageContent = assistantMessageElement.querySelector('.message-content');
            if (messageContent) {
                messageContent.innerHTML = this.renderMarkdown(fullResponse);
                // 高亮代码块
                this.highlightCodeBlocks(messageContent);
            }
        }
        // 保存流式消息到历史记录
        if (fullResponse) {
            this.currentChatHistory.push({
                type: 'assistant',
                content: fullResponse,
                timestamp: new Date().toISOString()
            });
            // 如果当前对话是从历史记录加载的，更新历史记录
            if (this.isCurrentChatFromHistory) {
                this.updateCurrentChatHistory();
                this.renderChatHistory();
            }
        }
    }

    renderMessageWorkbenchSummary(workbench) {
        const incident = workbench.incident_profile || {};
        const channels = incident.retrieval_channels || [];
        const route = workbench.agent_route || {};
        return `
            <div class="message-workbench-summary">
                <span>${this.escapeHtml(route.paradigm || '诊断链路已生成')}</span>
                <b>${this.escapeHtml(incident.fault_type || '未分类')}</b>
                <small>${this.escapeHtml(route.reason || (channels.length ? channels.join(' / ') : '等待检索通道'))}</small>
            </div>
        `;
    }

    renderDiagnosticWorkbench(workbench) {
        this.lastWorkbench = workbench;
        if (this.exportReportBtn) {
            this.exportReportBtn.disabled = !workbench.report_markdown;
        }
        this.renderWorkbenchObjective(workbench);
        this.renderWorkbenchIncident(workbench.incident_profile || {});
        this.renderWorkbenchSteps(workbench.execution_steps || []);
        this.renderWorkbenchTools(workbench.tool_calls || []);
        this.renderWorkbenchRag(workbench.rag_trace || {});
        this.renderWorkbenchRisks(workbench.risk_actions || []);
        this.renderWorkbenchQuestions(workbench.next_questions || []);
    }

    renderWorkbenchObjective(workbench) {
        if (!this.workbenchObjective) return;
        const route = workbench.agent_route || {};
        const routeText = route.paradigm ? `${route.paradigm} · ${route.reason || ''}` : 'Agent Orchestrator';
        this.workbenchObjective.innerHTML = `
            <div class="objective-label">当前目标</div>
            <strong>${this.escapeHtml(workbench.objective || '等待诊断目标')}</strong>
            <small>${this.escapeHtml(routeText)} · session: ${this.escapeHtml(workbench.session_id || this.sessionId)}</small>
        `;
    }

    renderWorkbenchIncident(incident) {
        if (!this.workbenchIncident) return;
        const fields = [
            ['故障类型', incident.fault_type || '-'],
            ['服务类型', incident.service_type || '-'],
            ['严重程度', incident.severity || '-'],
            ['置信度', incident.confidence ?? '-'],
            ['时间范围', incident.time_range || '-'],
            ['地域/集群', incident.region || '-'],
            ['证据数', incident.evidence_count ?? 0],
            ['状态', incident.status || '-'],
        ];
        const missing = incident.missing_info || [];
        this.workbenchIncident.innerHTML = `
            ${fields.map(([label, value]) => `
                <div class="incident-cell">
                    <span>${this.escapeHtml(label)}</span>
                    <b>${this.escapeHtml(value)}</b>
                </div>
            `).join('')}
            <div class="incident-cell incident-wide">
                <span>缺失信息</span>
                <b>${this.escapeHtml(missing.length ? missing.join('、') : '已足够初步诊断')}</b>
            </div>
        `;
    }

    renderWorkbenchSteps(steps) {
        if (!this.workbenchSteps) return;
        if (!steps.length) {
            this.workbenchSteps.innerHTML = `<span class="workbench-empty">暂无执行步骤</span>`;
            return;
        }
        this.workbenchSteps.innerHTML = steps.map((step, index) => `
            <div class="workbench-step ${this.escapeHtml(step.status || 'unknown')}">
                <div class="step-index">${index + 1}</div>
                <div class="step-body">
                    <div>
                        <strong>${this.escapeHtml(step.title || step.id || '')}</strong>
                        <span>${this.escapeHtml(this.formatWorkbenchStatus(step.status))}</span>
                    </div>
                    <p>${this.escapeHtml(step.detail || '')}</p>
                </div>
            </div>
        `).join('');
    }

    renderWorkbenchTools(calls) {
        if (!this.workbenchTools) return;
        if (!calls.length) {
            this.workbenchTools.innerHTML = `<span class="workbench-empty">暂无工具记录</span>`;
            return;
        }
        this.workbenchTools.innerHTML = calls.map(call => `
            <div class="tool-call ${this.escapeHtml(call.status || 'unknown')}">
                <div>
                    <strong>${this.escapeHtml(call.name || '')}</strong>
                    <span>${this.escapeHtml(call.type || '')}</span>
                </div>
                <p>${this.escapeHtml(call.summary || '')}</p>
                <small>${call.latency_ms != null ? `${Number(call.latency_ms).toFixed(1)} ms · ` : ''}${call.item_count != null ? `${call.item_count} items` : ''}</small>
            </div>
        `).join('');
    }

    renderWorkbenchRag(trace) {
        if (!this.workbenchRagTrace) return;
        const channelSummary = trace.channel_summary || {};
        const raw = channelSummary.raw_by_channel || {};
        const final = channelSummary.final_by_channel || {};
        const sources = trace.sources || [];
        const channelCards = Object.keys(raw).map(name => `
            <div class="rag-channel">
                <span>${this.escapeHtml(name)}</span>
                <b>${raw[name] || 0} → ${final[name] || 0}</b>
            </div>
        `).join('');
        const sourceCards = sources.slice(0, 4).map(source => `
            <div class="rag-source">
                <strong>${this.escapeHtml(source.id || '')} ${this.escapeHtml(source.file_name || source.source || 'unknown')}</strong>
                <small>${this.escapeHtml((source.channels || []).join(', '))} · rrf=${this.formatTraceScore(source.rrf_score)} · rerank=${this.formatTraceScore(source.rerank_score || source.score)}</small>
            </div>
        `).join('');
        this.workbenchRagTrace.innerHTML = `
            <div class="rag-flags">
                <span>RRF: ${trace.rrf_enabled ? 'on' : 'off'}</span>
                <span>Rerank: ${trace.rerank_enabled ? 'on' : 'off'}</span>
                <span>Sources: ${sources.length}</span>
            </div>
            <div class="rag-channel-grid">${channelCards || '<span class="workbench-empty">暂无通道统计</span>'}</div>
            <div class="rag-source-list">${sourceCards || '<span class="workbench-empty">暂无证据来源</span>'}</div>
        `;
    }

    renderWorkbenchRisks(actions) {
        if (!this.workbenchRisks) return;
        if (!actions.length) {
            this.workbenchRisks.innerHTML = `<span class="workbench-empty">本轮暂无待确认操作</span>`;
            return;
        }
        this.workbenchRisks.innerHTML = actions.map(action => `
            <div class="risk-card ${this.escapeHtml(action.risk || 'medium')}" data-risk-card="${this.escapeHtml(action.id || '')}">
                <div class="risk-head">
                    <strong>${this.escapeHtml(action.title || '')}</strong>
                    <span>${this.escapeHtml(this.formatRiskLabel(action.risk))}</span>
                </div>
                <p>${this.escapeHtml(action.reason || '')}</p>
                <div class="risk-actions">
                    <button data-risk-action="command" data-risk-id="${this.escapeHtml(action.id || '')}">只生成命令</button>
                    <button data-risk-action="approve" data-risk-id="${this.escapeHtml(action.id || '')}">确认模拟</button>
                    <button data-risk-action="cancel" data-risk-id="${this.escapeHtml(action.id || '')}">取消</button>
                </div>
                <small class="risk-state">等待人工确认</small>
            </div>
        `).join('');
    }

    renderWorkbenchQuestions(questions) {
        if (!this.workbenchQuestions) return;
        if (!questions.length) {
            this.workbenchQuestions.innerHTML = `<span class="workbench-empty">暂无补充问题</span>`;
            return;
        }
        this.workbenchQuestions.innerHTML = questions.map(item => `
            <button class="question-chip" type="button">${this.escapeHtml(item)}</button>
        `).join('');
    }

    handleRiskAction(event) {
        const button = event.target.closest('button[data-risk-action]');
        if (!button) return;
        const card = button.closest('.risk-card');
        const state = card ? card.querySelector('.risk-state') : null;
        const action = button.dataset.riskAction;
        const textMap = {
            command: '已选择：只生成命令，不执行操作',
            approve: '已确认：当前仅做模拟审批，不会真实执行',
            cancel: '已取消：不会生成执行动作'
        };
        if (state) {
            state.textContent = textMap[action] || '已处理';
            state.classList.add('handled');
        }
        button.parentElement.querySelectorAll('button').forEach(item => {
            item.classList.toggle('active', item === button);
        });
    }

    exportDiagnosticReport() {
        const report = this.lastWorkbench && this.lastWorkbench.report_markdown;
        if (!report) {
            this.showNotification('暂无可导出的诊断报告', 'warning');
            return;
        }
        const blob = new Blob([report], {type: 'text/markdown;charset=utf-8'});
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        link.href = url;
        link.download = `opspilot-diagnostic-report-${timestamp}.md`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        this.showNotification('诊断报告已生成', 'success');
    }

    formatWorkbenchStatus(status) {
        const map = {
            success: '完成',
            warning: '需关注',
            skipped: '跳过',
            empty: '无结果',
            error: '失败',
            disabled: '未启用',
        };
        return map[status] || status || '未知';
    }

    formatRiskLabel(risk) {
        const map = {
            low: '低风险',
            medium: '中风险',
            high: '高风险',
            critical: '严重风险',
        };
        return map[risk] || risk || '风险待定';
    }

    // 显示通知
    showNotification(message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 10000;
            animation: slideIn 0.3s ease;
            max-width: 300px;
        `;

        // 根据类型设置颜色（Google Material Design配色）
        const colors = {
            info: '#1a73e8',
            success: '#34a853',
            warning: '#fbbc04',
            error: '#ea4335'
        };
        notification.style.backgroundColor = colors[type] || colors.info;

        // 添加到页面
        document.body.appendChild(notification);

        // 3秒后自动移除
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // 处理文件选择
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            // 验证文件格式
            if (!this.validateFileType(file)) {
                this.showNotification('只支持上传 TXT 或 Markdown (.md) 格式的文件', 'error');
                this.fileInput.value = '';
                return;
            }
            this.uploadFile(file);
        }
    }

    // 验证文件类型
    validateFileType(file) {
        const fileName = file.name.toLowerCase();
        const allowedExtensions = ['.txt', '.md', '.markdown'];
        return allowedExtensions.some(ext => fileName.endsWith(ext));
    }

    // 上传文件到知识库
    async uploadFile(file) {
        // 再次验证文件类型（双重保险）
        if (!this.validateFileType(file)) {
            this.showNotification('只支持上传 TXT 或 Markdown (.md) 格式的文件', 'error');
            return;
        }

        // 验证文件大小（限制为50MB）
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('文件大小不能超过50MB', 'error');
            return;
        }

        // 锁定前端并显示上传遮罩层
        this.isStreaming = true;
        this.updateUI();
        this.showUploadOverlay(true, file.name);

        try {
            // 创建 FormData
            const formData = new FormData();
            formData.append('file', file);

            // 发送上传请求
            const response = await this.apiFetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();

            if ((data.code === 200 || data.message === 'success') && data.data) {
                // 在聊天界面显示上传成功消息
                const successMessage = `${file.name} 上传到知识库成功`;
                this.addMessage('assistant', successMessage, false, true);
                if (this.currentView === 'knowledge') {
                    await this.loadKnowledgeDashboard();
                }
            } else {
                throw new Error(data.message || '上传失败');
            }
        } catch (error) {
            console.error('文件上传失败:', error);
            this.showNotification('文件上传失败: ' + error.message, 'error');
        } finally {
            // 清空文件输入
            if (this.fileInput) {
                this.fileInput.value = '';
            }
            // 解锁前端
            this.isStreaming = false;
            this.showUploadOverlay(false);
            this.updateUI();
        }
    }

    // 格式化文件大小
    loadCurrentUser() {
        try {
            const stored = localStorage.getItem('currentUser');
            return stored ? JSON.parse(stored) : null;
        } catch (e) {
            return null;
        }
    }

    getCurrentUserId() {
        return (this.currentUser && this.currentUser.username) || 'admin';
    }

    async apiFetch(path, options = {}) {
        const headers = options.headers ? {...options.headers} : {};
        if (this.authToken) {
            headers.Authorization = `Bearer ${this.authToken}`;
        }
        return fetch(`${this.apiBaseUrl}${path}`, {
            ...options,
            headers,
        });
    }

    async ensureLoggedIn() {
        if (!this.authToken) {
            this.showLoginModal(true);
            return;
        }
        try {
            const response = await this.apiFetch('/auth/me');
            if (!response.ok) throw new Error('登录已过期');
            const result = await response.json();
            this.currentUser = result.data;
            localStorage.setItem('currentUser', JSON.stringify(this.currentUser));
            this.updateAuthUI();
            this.showLoginModal(false);
        } catch (e) {
            this.logout(false);
            this.showLoginModal(true);
        }
    }

    async handleLogin(event) {
        event.preventDefault();
        if (this.loginError) this.loginError.textContent = '';
        try {
            const response = await fetch(`${this.apiBaseUrl}/auth/login`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    username: this.loginUsername ? this.loginUsername.value : '',
                    password: this.loginPassword ? this.loginPassword.value : '',
                }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '登录失败');
            this.authToken = result.data.token;
            this.currentUser = result.data.user;
            localStorage.setItem('authToken', this.authToken);
            localStorage.setItem('currentUser', JSON.stringify(this.currentUser));
            this.updateAuthUI();
            this.showLoginModal(false);
            this.showNotification('登录成功', 'success');
        } catch (e) {
            if (this.loginError) this.loginError.textContent = e.message;
        }
    }

    logout(showToast = true) {
        if (this.authToken) {
            this.apiFetch('/auth/logout', {method: 'POST'}).catch(() => {});
        }
        this.authToken = '';
        this.currentUser = null;
        localStorage.removeItem('authToken');
        localStorage.removeItem('currentUser');
        this.updateAuthUI();
        this.showLoginModal(true);
        if (showToast) this.showNotification('已退出登录', 'info');
    }

    showLoginModal(show) {
        if (this.loginModal) {
            this.loginModal.classList.toggle('active', !!show);
        }
    }

    updateAuthUI() {
        const user = this.currentUser;
        const displayName = user ? (user.display_name || user.username || 'Admin') : '未登录';
        if (this.userName) this.userName.textContent = displayName;
        if (this.userRole) this.userRole.textContent = user ? `${user.role || 'user'} · 本地登录` : 'Local demo auth';
        if (this.userAvatar) this.userAvatar.textContent = displayName.slice(0, 1).toUpperCase();
    }

    showChatPage() {
        this.currentView = 'chat';
        if (this.knowledgePage) this.knowledgePage.classList.remove('active');
        if (this.chatContainer) this.chatContainer.style.display = '';
        if (this.workbenchShell) this.workbenchShell.style.display = '';
        if (this.knowledgeBaseBtn) this.knowledgeBaseBtn.classList.remove('active');
    }

    async showKnowledgePage() {
        this.currentView = 'knowledge';
        if (this.chatContainer) this.chatContainer.style.display = 'none';
        if (this.workbenchShell) this.workbenchShell.style.display = 'none';
        if (this.knowledgePage) this.knowledgePage.classList.add('active');
        if (this.knowledgeBaseBtn) this.knowledgeBaseBtn.classList.add('active');
        await this.loadKnowledgeDashboard();
    }

    async loadKnowledgeDashboard() {
        if (!this.knowledgeTableBody) return;
        this.knowledgeTableBody.innerHTML = `<tr><td colspan="6" class="kb-empty">加载中...</td></tr>`;
        try {
            const response = await this.apiFetch('/knowledge/files');
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '加载知识库失败');
            const files = result.data.files || [];
            this.renderKnowledgeStats(files);
            this.renderKnowledgeFiles(files);
            await this.loadKnowledgeTraces();
            await this.loadTermMappings();
            await this.loadRagEvalMetadata();
            if (this.kbLastLoaded) this.kbLastLoaded.textContent = `刷新于 ${new Date().toLocaleTimeString()}`;
        } catch (e) {
            this.knowledgeTableBody.innerHTML = `<tr><td colspan="6" class="kb-empty error">${this.escapeHtml(e.message)}</td></tr>`;
        }
    }

    renderKnowledgeStats(files) {
        const indexed = files.filter(file => file.index_status === 'SUCCESS').length;
        const errors = files.filter(file => file.index_status === 'ERROR').length;
        const totalSize = files.reduce((sum, file) => sum + Number(file.size || 0), 0);
        if (this.kbFileCount) this.kbFileCount.textContent = files.length;
        if (this.kbIndexedCount) this.kbIndexedCount.textContent = indexed;
        if (this.kbErrorCount) this.kbErrorCount.textContent = errors;
        if (this.kbTotalSize) this.kbTotalSize.textContent = this.formatFileSize(totalSize);
    }

    renderKnowledgeFiles(files) {
        if (!this.knowledgeTableBody) return;
        if (!files.length) {
            this.knowledgeTableBody.innerHTML = `<tr><td colspan="6" class="kb-empty">暂无上传文档</td></tr>`;
            return;
        }
        this.knowledgeTableBody.innerHTML = files.map(file => {
            const status = this.renderIndexStatus(file.index_status);
            const updatedAt = file.updated_at ? new Date(file.updated_at).toLocaleString() : '-';
            return `
                <tr>
                    <td><div class="kb-file-name">${this.escapeHtml(file.filename)}</div><div class="kb-file-sub">${this.escapeHtml(file.extension || '')}</div></td>
                    <td>${this.formatFileSize(file.size || 0)}</td>
                    <td>${status}</td>
                    <td>${file.chunks || 0}</td>
                    <td>${updatedAt}</td>
                    <td>
                        <div class="kb-row-actions">
                            <button data-action="trace" data-filename="${this.escapeHtml(file.filename)}">Trace</button>
                            <button data-action="download" data-filename="${this.escapeHtml(file.filename)}">下载</button>
                            <button data-action="reindex" data-filename="${this.escapeHtml(file.filename)}">重建</button>
                            <button class="danger" data-action="delete" data-filename="${this.escapeHtml(file.filename)}">删除</button>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }

    renderIndexStatus(status) {
        const normalized = status || 'NOT_TRACKED';
        const textMap = {SUCCESS: '已索引', ERROR: '异常', RUNNING: '索引中', NOT_TRACKED: '未追踪'};
        return `<span class="kb-status ${normalized.toLowerCase()}">${textMap[normalized] || normalized}</span>`;
    }

    async loadKnowledgeTraces(filename = '') {
        if (!this.knowledgeTraceList) return;
        this.knowledgeTraceList.innerHTML = `<div class="kb-empty">加载 Trace...</div>`;
        const query = filename ? `?filename=${encodeURIComponent(filename)}&limit=20` : '?limit=20';
        try {
            const response = await this.apiFetch(`/knowledge/traces${query}`);
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '加载 Trace 失败');
            const traces = result.data.traces || [];
            if (this.kbTraceTitle) this.kbTraceTitle.textContent = filename || '最近记录';
            this.renderKnowledgeTraces(traces);
        } catch (e) {
            this.knowledgeTraceList.innerHTML = `<div class="kb-empty error">${this.escapeHtml(e.message)}</div>`;
        }
    }

    renderKnowledgeTraces(traces) {
        if (!this.knowledgeTraceList) return;
        if (!traces.length) {
            this.knowledgeTraceList.innerHTML = `<div class="kb-empty">暂无 Trace 记录</div>`;
            return;
        }
        this.knowledgeTraceList.innerHTML = traces.map(trace => {
            const steps = (trace.steps || []).map(step => `
                <div class="kb-trace-step ${String(step.status || '').toLowerCase()}">
                    <span>${this.escapeHtml(step.name || '')}</span>
                    <b>${this.escapeHtml(step.status || '')}</b>
                    <small>${step.duration_ms || 0} ms</small>
                </div>
            `).join('');
            return `
                <article class="kb-trace-item">
                    <div class="kb-trace-head">
                        <strong>${this.escapeHtml(trace.filename || 'unknown')}</strong>
                        <span class="kb-status ${String(trace.status || '').toLowerCase()}">${this.escapeHtml(trace.status || '')}</span>
                    </div>
                    <div class="kb-trace-time">${this.escapeHtml(trace.end_ts || trace.ts || '')}</div>
                    ${trace.error ? `<div class="kb-trace-error">${this.escapeHtml(trace.error)}</div>` : ''}
                    <div class="kb-trace-steps">${steps}</div>
                </article>
            `;
        }).join('');
    }

    async handleKnowledgeTableAction(event) {
        const button = event.target.closest('button[data-action]');
        if (!button) return;
        const action = button.dataset.action;
        const filename = button.dataset.filename;
        if (!filename) return;
        if (action === 'trace') {
            await this.loadKnowledgeTraces(filename);
            return;
        }
        if (action === 'download') {
            await this.downloadKnowledgeFile(filename);
            return;
        }
        if (action === 'delete' && !confirm(`确定删除 ${filename} 以及对应向量索引吗？`)) return;
        button.disabled = true;
        try {
            const endpoint = `/knowledge/files/${encodeURIComponent(filename)}${action === 'reindex' ? '/reindex' : ''}`;
            const response = await this.apiFetch(endpoint, {method: action === 'delete' ? 'DELETE' : 'POST'});
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '操作失败');
            this.showNotification(action === 'delete' ? '文档已删除' : '重新索引完成', 'success');
            await this.loadKnowledgeDashboard();
        } catch (e) {
            this.showNotification(e.message, 'error');
        } finally {
            button.disabled = false;
        }
    }

    async downloadKnowledgeFile(filename) {
        try {
            const response = await this.apiFetch(`/knowledge/files/${encodeURIComponent(filename)}/download`);
            if (!response.ok) {
                let detail = '下载失败';
                try {
                    const result = await response.json();
                    detail = result.detail || detail;
                } catch (_) {
                    detail = await response.text();
                }
                throw new Error(detail);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
            this.showNotification('文档下载已开始', 'success');
        } catch (e) {
            this.showNotification(e.message || '下载失败', 'error');
        }
    }

    async loadTermMappings() {
        if (!this.termMappingTableBody) return;
        this.termMappingTableBody.innerHTML = `<tr><td colspan="5" class="kb-empty">加载术语映射...</td></tr>`;
        try {
            const response = await this.apiFetch('/chat/term-mappings');
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '加载术语映射失败');
            this.renderTermMappings(result.data?.mappings || []);
        } catch (e) {
            this.termMappingTableBody.innerHTML = `<tr><td colspan="5" class="kb-empty error">${this.escapeHtml(e.message)}</td></tr>`;
        }
    }

    renderTermMappings(mappings) {
        if (!this.termMappingTableBody) return;
        if (!mappings.length) {
            this.termMappingTableBody.innerHTML = `<tr><td colspan="5" class="kb-empty">暂无术语映射</td></tr>`;
            return;
        }
        this.termMappingTableBody.innerHTML = mappings.map(item => `
            <tr>
                <td><div class="kb-file-name">${this.escapeHtml(item.source_term || '')}</div><div class="kb-file-sub">${this.escapeHtml(item.description || '')}</div></td>
                <td>${this.escapeHtml(item.target_term || '')}</td>
                <td>${this.escapeHtml(item.scenario || '')}</td>
                <td>${Number(item.priority || 0)}</td>
                <td>
                    <div class="kb-row-actions">
                        <button class="danger" data-action="delete-term" data-id="${this.escapeHtml(item.id || '')}">删除</button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    async addTermMapping() {
        const source = this.termSourceInput ? this.termSourceInput.value.trim() : '';
        const target = this.termTargetInput ? this.termTargetInput.value.trim() : '';
        const scenario = this.termScenarioInput ? this.termScenarioInput.value.trim() : 'AIOps';
        if (!source || !target) {
            this.showNotification('请填写用户说法和标准术语', 'warning');
            return;
        }
        try {
            const response = await this.apiFetch('/chat/term-mappings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    sourceTerm: source,
                    targetTerm: target,
                    scenario: scenario || 'AIOps',
                    enabled: true,
                    priority: 80,
                    matchType: 'contains',
                    description: '页面新增规则'
                })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '保存术语映射失败');
            if (this.termSourceInput) this.termSourceInput.value = '';
            if (this.termTargetInput) this.termTargetInput.value = '';
            this.showNotification('术语映射已保存', 'success');
            await this.loadTermMappings();
        } catch (e) {
            this.showNotification(e.message || '保存术语映射失败', 'error');
        }
    }

    async handleTermMappingAction(event) {
        const button = event.target.closest('button[data-action="delete-term"]');
        if (!button) return;
        const id = button.dataset.id;
        if (!id || !confirm(`确定删除映射 ${id} 吗？`)) return;
        button.disabled = true;
        try {
            const response = await this.apiFetch(`/chat/term-mappings/${encodeURIComponent(id)}`, {method: 'DELETE'});
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '删除失败');
            this.showNotification('术语映射已删除', 'success');
            await this.loadTermMappings();
        } catch (e) {
            this.showNotification(e.message || '删除失败', 'error');
        } finally {
            button.disabled = false;
        }
    }

    async runRetrievalTrace() {
        if (!this.ragTraceResult || !this.ragTraceQuestionInput) return;
        const question = this.ragTraceQuestionInput.value.trim();
        if (!question) {
            this.showNotification('请输入要分析的问题', 'warning');
            return;
        }
        if (this.ragTraceRunBtn) {
            this.ragTraceRunBtn.disabled = true;
            this.ragTraceRunBtn.textContent = '运行中...';
        }
        this.ragTraceResult.innerHTML = `<div class="kb-empty">正在执行查询理解和多通道检索...</div>`;
        try {
            const params = new URLSearchParams({
                question,
                sessionId: `trace-${Date.now()}`,
                userId: this.getCurrentUserId()
            });
            const response = await this.apiFetch(`/chat/retrieval/inspect?${params.toString()}`);
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || 'Trace 运行失败');
            this.renderRetrievalTrace(result.data || {});
        } catch (e) {
            this.ragTraceResult.innerHTML = `<div class="kb-empty error">${this.escapeHtml(e.message || 'Trace 运行失败')}</div>`;
        } finally {
            if (this.ragTraceRunBtn) {
                this.ragTraceRunBtn.disabled = false;
                this.ragTraceRunBtn.textContent = '运行 Trace';
            }
        }
    }

    renderRetrievalTrace(payload) {
        if (!this.ragTraceResult) return;
        const understanding = payload.understanding || {};
        const retrieval = payload.retrieval || {};
        const trace = retrieval.trace || {};
        const mappings = understanding.applied_mappings || [];
        const channels = trace.channels || [];
        const nodes = trace.pipeline_nodes || [];
        const sources = retrieval.sources || [];
        const subQuestions = understanding.sub_questions || [];
        const intentNodes = understanding.intent_nodes || [];
        const retrievalScope = understanding.retrieval_scope || trace.retrieval_scope || {};
        const missingSlots = understanding.missing_slots || [];
        const targetSources = retrievalScope.target_sources || [];
        this.ragTraceResult.innerHTML = `
            <div class="trace-understanding">
                <div>
                    <span>原始问题</span>
                    <strong>${this.escapeHtml(understanding.original_question || '')}</strong>
                </div>
                <div>
                    <span>归一化问题</span>
                    <strong>${this.escapeHtml(understanding.normalized_question || understanding.original_question || '')}</strong>
                </div>
                <div>
                    <span>识别意图</span>
                    <strong>${this.escapeHtml(understanding.intent || '')} · ${Number(understanding.confidence || 0).toFixed(2)}</strong>
                </div>
                <div>
                    <span>术语映射</span>
                    <strong>${this.escapeHtml(mappings.length ? mappings.map(m => `${m.source_term}→${m.target_term}`).join('；') : '未命中')}</strong>
                </div>
                <div>
                    <span>重写策略</span>
                    <strong>${this.escapeHtml(understanding.rewrite_strategy || 'rule_rewrite')}</strong>
                </div>
                <div>
                    <span>拆分子问题</span>
                    <strong>${this.escapeHtml(subQuestions.length ? subQuestions.join('；') : '未拆分')}</strong>
                </div>
                <div>
                    <span>检索范围</span>
                    <strong>${this.escapeHtml(retrievalScope.mode || 'global')} · ${this.escapeHtml(targetSources.length ? targetSources.join(', ') : 'all')}</strong>
                </div>
                <div>
                    <span>缺失槽位</span>
                    <strong>${this.escapeHtml(missingSlots.length ? missingSlots.join('；') : '无')}</strong>
                </div>
                <div>
                    <span>重排策略</span>
                    <strong>${this.escapeHtml(trace.rerank_strategy || (trace.semantic_rerank_enabled ? 'hybrid_lexical_embedding' : 'hybrid_lexical_rule'))}</strong>
                </div>
            </div>
            <div class="trace-section">
                <h3>Intent Tree</h3>
                <div class="trace-node-list">
                    ${intentNodes.map(node => `
                        <article class="trace-node-card">
                            <div class="trace-node-head">
                                <strong>${this.escapeHtml(node.path || node.name || node.id || '')}</strong>
                                <span>${this.formatTraceScore(node.score)}</span>
                            </div>
                            <p>${this.escapeHtml(node.reason || node.route || '')}</p>
                        </article>
                    `).join('') || '<div class="kb-empty">未命中意图树节点</div>'}
                </div>
            </div>
            <div class="trace-section">
                <h3>Pipeline Nodes</h3>
                <div class="trace-node-list">
                    ${nodes.map(node => this.renderTraceNode(node)).join('')}
                </div>
            </div>
            <div class="trace-section">
                <h3>Channel Attribution</h3>
                <div class="trace-channel-grid">
                    ${channels.map(channel => `
                        <div class="trace-channel-card">
                            <span>${this.escapeHtml(channel.channel || '')}</span>
                            <strong>${channel.item_count || 0} 条</strong>
                            <small>${this.escapeHtml(channel.status || '')} · ${Number(channel.latency_ms || 0).toFixed(1)} ms</small>
                        </div>
                    `).join('') || '<div class="kb-empty">暂无通道结果</div>'}
                </div>
            </div>
            <div class="trace-section">
                <h3>Final Sources</h3>
                <div class="trace-source-list">
                    ${sources.map(source => `
                        <article class="trace-source-card">
                            <div>
                                <b>${this.escapeHtml(source.id || '')}</b>
                                <strong>${this.escapeHtml(source.file_name || source.source || 'unknown')}</strong>
                            </div>
                            <p>${this.escapeHtml(source.excerpt || '')}</p>
                            <small>channels=${this.escapeHtml((source.channels || []).join(','))} · rrf=${this.formatTraceScore(source.rrf_score)} · rerank=${this.formatTraceScore(source.rerank_score || source.score)}</small>
                        </article>
                    `).join('') || '<div class="kb-empty">暂无最终证据</div>'}
                </div>
            </div>
        `;
    }

    renderTraceNode(node) {
        const details = node.details ? JSON.stringify(node.details, null, 2) : '';
        return `
            <article class="trace-node-card">
                <div class="trace-node-head">
                    <strong>${this.escapeHtml(node.name || '')}</strong>
                    <span>${this.escapeHtml(node.type || '')}</span>
                </div>
                <p>${this.escapeHtml(node.summary || '')}</p>
                ${details ? `<pre>${this.escapeHtml(details.slice(0, 1200))}</pre>` : ''}
            </article>
        `;
    }

    formatTraceScore(value) {
        if (value === null || value === undefined || value === '') return '-';
        return Number(value || 0).toFixed(4);
    }

    async loadRagEvalMetadata() {
        if (!this.ragEvalSummary || !this.ragEvalResults) return;
        try {
            const response = await this.apiFetch('/chat/eval/retrieval/meta');
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '加载评测集信息失败');
            this.renderRagEvalMetadata(result.data || {});
        } catch (e) {
            this.ragEvalSummary.innerHTML = `<div class="kb-empty error">${this.escapeHtml(e.message || '加载评测集信息失败')}</div>`;
        }
    }

    renderRagEvalMetadata(meta) {
        const categories = meta.categories || {};
        const difficulties = meta.difficulties || {};
        const categoryText = Object.entries(categories).slice(0, 6).map(([name, count]) => `${name} ${count}`).join(' · ');
        const difficultyText = Object.entries(difficulties).map(([name, count]) => `${name} ${count}`).join(' · ');
        this.ragEvalSummary.innerHTML = `
            <div class="rag-eval-card rag-eval-card-wide">
                <span>评测集规模</span>
                <strong>${meta.sample_count || 0}</strong>
                <small>${this.escapeHtml(meta.version || '')} · 外部 JSON 数据集</small>
            </div>
            <div class="rag-eval-card rag-eval-card-wide">
                <span>场景覆盖</span>
                <strong>${Object.keys(categories).length || 0}</strong>
                <small>${this.escapeHtml(categoryText || '暂无分类')}</small>
            </div>
            <div class="rag-eval-card">
                <span>难度分布</span>
                <strong>${Object.keys(difficulties).length || 0}</strong>
                <small>${this.escapeHtml(difficultyText || '暂无难度')}</small>
            </div>
            <div class="rag-eval-card">
                <span>对比方式</span>
                <strong>Baseline</strong>
                <small>点击运行后展示 Enhanced 提升</small>
            </div>
        `;
        const preview = meta.preview || [];
        this.ragEvalResults.innerHTML = preview.length ? preview.map(item => `
            <article class="rag-eval-row">
                <div class="rag-eval-row-head">
                    <strong>${this.escapeHtml(item.id || '')}</strong>
                    <span>${this.escapeHtml(item.category || '')} · ${this.escapeHtml(item.difficulty || '')}</span>
                </div>
                <div class="rag-eval-query">
                    <span>样例问题</span>
                    <p>${this.escapeHtml(item.question || '')}</p>
                </div>
                <div class="rag-eval-query">
                    <span>期望来源</span>
                    <p>${this.escapeHtml((item.expected_sources || []).join('，'))}</p>
                </div>
            </article>
        `).join('') : `<div class="kb-empty">暂无评测样例预览</div>`;
        if (this.ragEvalUpdatedAt) {
            this.ragEvalUpdatedAt.textContent = `评测集信息加载于 ${new Date().toLocaleString()}；完整评测会调用向量检索，耗时会更长。`;
        }
    }

    async runRagEval() {
        if (!this.ragEvalSummary || !this.ragEvalResults) return;
        if (this.ragEvalRunBtn) {
            this.ragEvalRunBtn.disabled = true;
            this.ragEvalRunBtn.textContent = '评测中...';
        }
        this.ragEvalSummary.innerHTML = `<div class="kb-empty">正在运行种子评测集，请稍等...</div>`;
        this.ragEvalResults.innerHTML = '';
        try {
            const response = await this.apiFetch('/chat/eval/retrieval', {method: 'POST'});
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail || '评测失败');
            this.renderRagEvalReport(result.data || {});
            this.showNotification('RAG 评测完成', 'success');
        } catch (e) {
            this.ragEvalSummary.innerHTML = `<div class="kb-empty error">${this.escapeHtml(e.message || '评测失败')}</div>`;
            this.showNotification(e.message || '评测失败', 'error');
        } finally {
            if (this.ragEvalRunBtn) {
                this.ragEvalRunBtn.disabled = false;
                this.ragEvalRunBtn.textContent = '运行评测';
            }
        }
    }

    renderRagEvalReport(report) {
        if (!this.ragEvalSummary || !this.ragEvalResults) return;
        const baseline = report.baseline || {};
        const enhanced = report.enhanced || {};
        const improvement = report.improvement || {};
        const metricCards = [
            ['Hit@1', 'hit_at_1', true],
            ['Hit@3', 'hit_at_3', true],
            ['Hit@5', 'hit_at_5', true],
            ['MRR', 'mrr', false],
            ['nDCG@5', 'ndcg_at_5', false],
        ];

        this.ragEvalSummary.innerHTML = `
            <div class="rag-eval-card rag-eval-card-wide">
                <span>样例数</span>
                <strong>${report.sample_count || 0}</strong>
                <small>外部评测集 ${this.escapeHtml(report.version || '')} · Baseline 对比 Enhanced</small>
            </div>
            ${metricCards.map(([label, key, percent]) => `
                <div class="rag-eval-card">
                    <span>${label}</span>
                    <strong>${percent ? this.formatPercent(enhanced[key]) : this.formatDecimal(enhanced[key])}</strong>
                    <small>Baseline ${percent ? this.formatPercent(baseline[key]) : this.formatDecimal(baseline[key])} · ${this.formatDelta(improvement[key], percent)}</small>
                </div>
            `).join('')}
            <div class="rag-eval-card">
                <span>意图识别</span>
                <strong>${this.formatPercent(report.intent_accuracy)}</strong>
                <small>问题理解模块命中率</small>
            </div>
            <div class="rag-eval-comparison">
                <div class="rag-eval-comparison-title">
                    <strong>Baseline vs Enhanced</strong>
                    <span>Baseline = 原始问题直接向量检索；Enhanced = 问题理解、Query 扩展、混合检索、重排后的链路</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>Baseline</th>
                            <th>Enhanced</th>
                            <th>Delta</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${metricCards.map(([label, key, percent]) => `
                            <tr>
                                <td>${label}</td>
                                <td>${percent ? this.formatPercent(baseline[key]) : this.formatDecimal(baseline[key])}</td>
                                <td>${percent ? this.formatPercent(enhanced[key]) : this.formatDecimal(enhanced[key])}</td>
                                <td class="${Number(improvement[key] || 0) >= 0 ? 'positive' : 'negative'}">${this.formatDelta(improvement[key], percent)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;

        const rows = (report.results || []).map(item => {
            const baselineHit = item.baseline?.hit_rank ? `#${item.baseline.hit_rank}` : '未命中';
            const enhancedHit = item.enhanced?.hit_rank ? `#${item.enhanced.hit_rank}` : '未命中';
            return `
                <article class="rag-eval-row">
                    <div class="rag-eval-row-head">
                        <div>
                            <strong>${this.escapeHtml(item.question || '')}</strong>
                            <span>${this.escapeHtml(item.category || '')} · 期望 ${this.escapeHtml((item.expected_sources || []).join(', '))}</span>
                        </div>
                        <div class="rag-eval-ranks">
                            <span>Baseline ${baselineHit}</span>
                            <b>Enhanced ${enhancedHit}</b>
                        </div>
                    </div>
                    <div class="rag-eval-query">
                        <span>重写问题</span>
                        <p>${this.escapeHtml(item.rewritten_question || '')}</p>
                        <small>${this.escapeHtml(item.rewrite_strategy || '')}</small>
                    </div>
                    <div class="rag-eval-source-grid">
                        <div>
                            <label>Baseline Top Sources</label>
                            ${this.renderEvalSources(item.baseline?.sources || [])}
                        </div>
                        <div>
                            <label>Enhanced Top Sources</label>
                            ${this.renderEvalSources(item.enhanced?.sources || [])}
                        </div>
                    </div>
                </article>
            `;
        }).join('');
        this.ragEvalResults.innerHTML = rows || `<div class="kb-empty">暂无评测明细</div>`;
        if (this.ragEvalUpdatedAt) {
            this.ragEvalUpdatedAt.textContent = report.ts ? `最近评测：${report.ts}` : '';
        }
    }

    renderEvalSources(sources) {
        if (!sources.length) return `<div class="rag-eval-source-empty">暂无结果</div>`;
        return sources.slice(0, 5).map(source => `
            <div class="rag-eval-source">
                <span>#${source.rank || '-'}</span>
                <strong>${this.escapeHtml(source.file_name || source.source || 'unknown')}</strong>
                <small>${this.escapeHtml((source.channels || [source.kind || '']).join(','))} · rrf ${this.formatTraceScore(source.rrf_score)}</small>
            </div>
        `).join('');
    }

    formatPercent(value) {
        return `${(Number(value || 0) * 100).toFixed(1)}%`;
    }

    formatDecimal(value) {
        return Number(value || 0).toFixed(3);
    }

    formatDelta(value, asPercent = true) {
        const numeric = Number(value || 0);
        const sign = numeric >= 0 ? '+' : '';
        return asPercent ? `${sign}${(numeric * 100).toFixed(1)}%` : `${sign}${numeric.toFixed(3)}`;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    // 发送智能运维请求（SSE 流式模式）
    async sendAIOpsRequest(loadingMessageElement) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/aiops`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            let fullResponse = '';

            // 处理 SSE 流式响应
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let currentEvent = 'message'; // 默认事件类型为 message

            try {
                while (true) {
                    const { done, value } = await reader.read();

                    if (done) {
                        // 流结束，更新最终内容
                        if (fullResponse) {
                            console.log('AI Ops 流结束，更新最终内容，长度:', fullResponse.length);
                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                        }
                        break;
                    }

                    // 解码数据并添加到缓冲区
                    buffer += decoder.decode(value, { stream: true });

                    // 按行分割处理
                    const lines = buffer.split('\n');
                    // 保留最后一行（可能不完整）
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.trim() === '') continue;

                        console.log('[AI Ops SSE] 收到行:', line);

                        // 解析 SSE 格式
                        if (line.startsWith('id:')) {
                            continue;
                        } else if (line.startsWith('event:')) {
                            currentEvent = line.substring(6).trim();
                            console.log('[AI Ops SSE] 事件类型:', currentEvent);
                            continue;
                        } else if (line.startsWith('data:')) {
                            const rawData = line.substring(5).trim();
                            console.log('[AI Ops SSE] 数据:', rawData, ', currentEvent:', currentEvent);

                            // 解析可能包含多个JSON对象的数据
                            const processJsonMessages = (data) => {
                                const jsonPattern = /\{"type"\s*:\s*"[^"]+"\s*,\s*"data"\s*:\s*(?:"[^"]*"|null)\}/g;
                                const matches = data.match(jsonPattern);

                                if (matches && matches.length > 0) {
                                    console.log('[AI Ops SSE] 匹配到', matches.length, '个JSON对象');
                                    for (const jsonStr of matches) {
                                        try {
                                            const sseMessage = JSON.parse(jsonStr);
                                            if (sseMessage.type === 'content') {
                                                fullResponse += sseMessage.data || '';
                                            } else if (sseMessage.type === 'plan') {
                                                // 处理计划创建事件
                                                const planText = `\n\n## 📋 执行计划\n${sseMessage.message}\n\n`;
                                                fullResponse += planText;
                                            } else if (sseMessage.type === 'step_complete') {
                                                // 处理步骤完成事件
                                                const stepText = `\n✅ ${sseMessage.message}\n`;
                                                fullResponse += stepText;
                                            } else if (sseMessage.type === 'status') {
                                                // 处理状态更新事件
                                                const statusText = `\n⏳ ${sseMessage.message}\n`;
                                                fullResponse += statusText;
                                            } else if (sseMessage.type === 'report') {
                                                // 处理最终报告事件 - 流式输出
                                                console.log('AI Ops 最终报告生成');
                                                const reportText = `\n\n## 🎯 诊断报告\n\n${sseMessage.report || ''}\n`;
                                                fullResponse += reportText;
                                            } else if (sseMessage.type === 'complete') {
                                                // 处理完成事件
                                                console.log('AI Ops 诊断完成');
                                                fullResponse = this.appendUniqueFinalResponse(fullResponse, sseMessage.response);
                                                fullResponse = this.removeConsecutiveDuplicateText(fullResponse);
                                                this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                                return true;
                                            } else if (sseMessage.type === 'done') {
                                                console.log('AI Ops 流完成，最终内容长度:', fullResponse.length);
                                                this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                                return true;
                                            } else if (sseMessage.type === 'error') {
                                                throw new Error(sseMessage.data || sseMessage.message || '智能运维分析失败');
                                            }
                                        } catch (e) {
                                            if (e.message.includes('智能运维')) throw e;
                                            console.log('[AI Ops SSE] 单个JSON解析失败:', jsonStr);
                                        }
                                    }
                                    if (loadingMessageElement) {
                                        this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                    }
                                    return false;
                                }
                                return null;
                            };

                            const result = processJsonMessages(rawData);
                            if (result === true) {
                                return; // 流结束
                            } else if (result === null) {
                                // 没有匹配到多个JSON，尝试单个JSON解析
                                try {
                                    const sseMessage = JSON.parse(rawData);
                                    if (sseMessage && sseMessage.type) {
                                        if (sseMessage.type === 'content') {
                                            fullResponse += sseMessage.data || '';
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'plan') {
                                            // 处理计划创建事件
                                            const planText = `\n\n## 📋 执行计划\n${sseMessage.message}\n\n`;
                                            fullResponse += planText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'step_complete') {
                                            // 处理步骤完成事件
                                            const stepText = `\n✅ ${sseMessage.message}\n`;
                                            fullResponse += stepText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'status') {
                                            // 处理状态更新事件
                                            const statusText = `\n⏳ ${sseMessage.message}\n`;
                                            fullResponse += statusText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'report') {
                                            // 处理最终报告事件 - 这是关键！
                                            console.log('AI Ops 最终报告生成，流式输出中...');
                                            const reportText = `\n\n## 🎯 诊断报告\n\n${sseMessage.report || ''}\n`;
                                            fullResponse += reportText;
                                            if (loadingMessageElement) {
                                                this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                            }
                                        } else if (sseMessage.type === 'complete') {
                                            // 处理完成事件
                                            console.log('AI Ops 诊断完成，最终内容长度:', fullResponse.length);
                                            fullResponse = this.appendUniqueFinalResponse(fullResponse, sseMessage.response);
                                            fullResponse = this.removeConsecutiveDuplicateText(fullResponse);
                                            // 使用最终的完整内容更新消息
                                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                            return;
                                        } else if (sseMessage.type === 'done') {
                                            console.log('AI Ops 流完成，最终内容长度:', fullResponse.length);
                                            this.updateAIOpsMessage(loadingMessageElement, fullResponse, []);
                                            return;
                                        } else if (sseMessage.type === 'error') {
                                            throw new Error(sseMessage.data || sseMessage.message || '智能运维分析失败');
                                        }
                                    } else {
                                        fullResponse += rawData;
                                        if (loadingMessageElement) {
                                            this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                        }
                                    }
                                } catch (e) {
                                    if (e.message.includes('智能运维')) throw e;
                                    // 非 JSON 格式，直接追加原始数据
                                    fullResponse += rawData;
                                    if (loadingMessageElement) {
                                        this.updateAIOpsStreamContent(loadingMessageElement, fullResponse);
                                    }
                                }
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            throw error;
        }
    }

    // 更新智能运维流式内容（实时显示）
    updateAIOpsStreamContent(messageElement, content) {
        if (!messageElement) return;

        // 添加 aiops-message 类
        messageElement.classList.add('aiops-message');

        const messageContentWrapper = messageElement.querySelector('.message-content-wrapper');
        if (messageContentWrapper) {
            let messageContent = messageContentWrapper.querySelector('.message-content');
            if (!messageContent) {
                messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContentWrapper.appendChild(messageContent);
            }
            // 流式显示时使用纯文本
            messageContent.textContent = content;
            this.scrollToBottom();
        }
    }

    // 更新智能运维消息（带折叠详情）
    updateAIOpsMessage(messageElement, response, details) {
        console.log('updateAIOpsMessage 被调用');
        console.log('messageElement:', messageElement);
        console.log('response:', response);
        console.log('response length:', response ? response.length : 0);
        console.log('details:', details);

        if (!messageElement) {
            // 如果没有传入消息元素，则创建新消息
            console.log('messageElement 为空，创建新消息');
            return this.addAIOpsMessage(response, details);
        }
        if (messageElement.dataset.aiopsCompleted === 'true') {
            return messageElement;
        }
        messageElement.dataset.aiopsCompleted = 'true';
        response = this.removeConsecutiveDuplicateText(response);

        // 添加aiops-message类
        messageElement.classList.add('aiops-message');

        // 获取消息内容包装器
        const messageContentWrapper = messageElement.querySelector('.message-content-wrapper');
        if (!messageContentWrapper) {
            console.error('未找到 message-content-wrapper');
            return;
        }

        // 清空现有内容（保留消息内容容器）
        const messageContent = messageContentWrapper.querySelector('.message-content');
        if (!messageContent) {
            console.error('未找到 message-content');
            return;
        }

        // 移除加载动画相关的类和内容
        messageContent.classList.remove('loading-message-content');
        messageContent.textContent = '';

        // 移除加载图标（如果存在）
        const loadingIcon = messageContent.querySelector('.loading-spinner-icon');
        if (loadingIcon) {
            loadingIcon.remove();
        }

        // 详情部分（可折叠）- 先显示
        if (details && details.length > 0) {
            // 检查是否已存在详情容器
            let detailsContainer = messageElement.querySelector('.aiops-details');
            if (!detailsContainer) {
                detailsContainer = document.createElement('div');
                detailsContainer.className = 'aiops-details';
                messageContentWrapper.insertBefore(detailsContainer, messageContent);
            } else {
                // 清空现有详情
                detailsContainer.innerHTML = '';
            }

            const detailsToggle = document.createElement('div');
            detailsToggle.className = 'details-toggle';
            detailsToggle.innerHTML = `
                <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>查看详细步骤 (${details.length}条)</span>
            `;

            const detailsContent = document.createElement('div');
            detailsContent.className = 'details-content';

            details.forEach((detail, index) => {
                const detailItem = document.createElement('div');
                detailItem.className = 'detail-item';
                detailItem.innerHTML = `<strong>步骤 ${index + 1}:</strong> ${this.escapeHtml(detail)}`;
                detailsContent.appendChild(detailItem);
            });

            // 点击切换折叠状态
            detailsToggle.addEventListener('click', () => {
                detailsContent.classList.toggle('expanded');
                detailsToggle.classList.toggle('expanded');
            });

            detailsContainer.appendChild(detailsToggle);
            detailsContainer.appendChild(detailsContent);
        }

        // 更新主要响应内容（使用Markdown渲染）
        console.log('开始渲染 Markdown');
        const renderedHtml = this.renderMarkdown(response);
        console.log('Markdown 渲染完成，HTML 长度:', renderedHtml ? renderedHtml.length : 0);
        messageContent.innerHTML = renderedHtml;
        console.log('innerHTML 已设置');
        // 高亮代码块
        this.highlightCodeBlocks(messageContent);
        console.log('代码块高亮完成');

        // 保存到历史记录
        this.currentChatHistory.push({
            type: 'assistant',
            content: response,
            timestamp: new Date().toISOString()
        });

        this.scrollToBottom();
        return messageElement;
    }

    // 添加智能运维消息（带折叠详情）- 保留用于兼容性
    addAIOpsMessage(response, details) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant aiops-message';

        // 添加头像图标
        const messageAvatar = document.createElement('div');
        messageAvatar.className = 'message-avatar';
        messageAvatar.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="white"/>
            </svg>
        `;
        messageDiv.appendChild(messageAvatar);

        // 创建消息内容包装器
        const messageContentWrapper = document.createElement('div');
        messageContentWrapper.className = 'message-content-wrapper';

        // 详情部分（可折叠）- 先显示
        if (details && details.length > 0) {
            const detailsContainer = document.createElement('div');
            detailsContainer.className = 'aiops-details';

            const detailsToggle = document.createElement('div');
            detailsToggle.className = 'details-toggle';
            detailsToggle.innerHTML = `
                <svg class="toggle-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 18L15 12L9 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>查看详细步骤 (${details.length}条)</span>
            `;

            const detailsContent = document.createElement('div');
            detailsContent.className = 'details-content';

            details.forEach((detail, index) => {
                const detailItem = document.createElement('div');
                detailItem.className = 'detail-item';
                detailItem.innerHTML = `<strong>步骤 ${index + 1}:</strong> ${this.escapeHtml(detail)}`;
                detailsContent.appendChild(detailItem);
            });

            // 点击切换折叠状态
            detailsToggle.addEventListener('click', () => {
                detailsContent.classList.toggle('expanded');
                detailsToggle.classList.toggle('expanded');
            });

            detailsContainer.appendChild(detailsToggle);
            detailsContainer.appendChild(detailsContent);
            messageContentWrapper.appendChild(detailsContainer);
        }

        // 主要响应内容 - 后显示（使用Markdown渲染）
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = this.renderMarkdown(response);
        // 高亮代码块
        this.highlightCodeBlocks(messageContent);
        messageContentWrapper.appendChild(messageContent);
        messageDiv.appendChild(messageContentWrapper);

        if (this.chatMessages) {
            this.chatMessages.appendChild(messageDiv);
            this.scrollToBottom();
        }

        return messageDiv;
    }

    // HTML转义
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 触发智能运维（点击智能运维按钮时直接调用）
    async triggerAIOps() {
        if (this.isStreaming) {
            this.showNotification('请等待当前操作完成', 'warning');
            return;
        }

        // 新建对话
        this.newChat();

        // 添加"分析中..."的消息（带旋转动画）
        const loadingMessage = this.addLoadingMessage('分析中...');
        this.currentAIOpsMessage = loadingMessage; // 保存消息引用用于后续更新

        // 设置发送状态
        this.isStreaming = true;
        this.updateUI();

        try {
            await this.sendAIOpsRequest(loadingMessage);
        } catch (error) {
            console.error('智能运维分析失败:', error);
            // 更新消息为错误信息
            if (loadingMessage) {
                const messageContent = loadingMessage.querySelector('.message-content');
                if (messageContent) {
                    messageContent.textContent = '抱歉，智能运维分析时出现错误：' + error.message;
                }
            }
        } finally {
            this.isStreaming = false;
            this.currentAIOpsMessage = null;
            this.updateUI();
        }
    }

    // 显示/隐藏加载遮罩层
    showLoadingOverlay(show) {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                // 更新文字为智能运维
                const loadingText = this.loadingOverlay.querySelector('.loading-text');
                const loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = '智能运维分析中，请稍候...';
                if (loadingSubtext) loadingSubtext.textContent = '后端正在处理，请耐心等待';
                // 防止页面滚动
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                // 恢复页面滚动
                document.body.style.overflow = '';
            }
        }
    }

    // 显示/隐藏上传遮罩层
    showUploadOverlay(show, fileName = '') {
        if (this.loadingOverlay) {
            if (show) {
                this.loadingOverlay.style.display = 'flex';
                // 更新文字为上传中
                const loadingText = this.loadingOverlay.querySelector('.loading-text');
                const loadingSubtext = this.loadingOverlay.querySelector('.loading-subtext');
                if (loadingText) loadingText.textContent = '正在上传文件...';
                if (loadingSubtext) loadingSubtext.textContent = fileName ? `上传: ${fileName}` : '请稍候';
                // 防止页面滚动
                document.body.style.overflow = 'hidden';
            } else {
                this.loadingOverlay.style.display = 'none';
                // 恢复页面滚动
                document.body.style.overflow = '';
            }
        }
    }
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new OpsPilotApp();
});
