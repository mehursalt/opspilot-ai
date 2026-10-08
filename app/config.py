"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "OpsPilot AI"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考
    enable_query_understanding: bool = True  # 是否启用意图识别、问题改写和增强会话记忆
    enable_query_term_mapping: bool = True  # 是否启用领域术语映射/口语归一化
    query_term_mapping_path: str = "data/rag/query_term_mappings.json"  # 术语映射规则持久化文件
    intent_tree_path: str = "data/rag/intent_tree.json"  # 可配置意图树文件
    intent_tree_min_score: float = 0.35  # 意图树节点最低命中分
    intent_tree_ambiguity_ratio: float = 0.82  # top2/top1 高于该比例时认为可能存在歧义
    intent_tree_max_candidates: int = 6  # 每次最多保留的意图树候选数
    enable_llm_query_understanding: bool = True  # 是否启用 LLM 结构化查询理解，关闭/失败时使用规则兜底
    llm_query_understanding_model: str = "qwen-plus"  # 查询理解使用的小模型
    llm_query_understanding_timeout_seconds: float = 15.0  # 查询理解 LLM 超时时间
    enable_llm_intent_tree_scoring: bool = True  # 是否使用 LLM 对意图树候选节点打分
    llm_intent_tree_model: str = "qwen-plus"  # 意图树打分使用的小模型
    llm_intent_tree_timeout_seconds: float = 12.0  # 意图树 LLM 打分超时时间
    query_rewrite_max_queries: int = 6  # 每次问题理解最多生成的检索查询数
    intent_guidance_min_confidence: float = 0.6  # 低于该置信度时先返回澄清/引导建议
    retrieval_supplement_ratio: float = 0.25  # 定向检索时留给补充召回的比例
    enable_rag_prefetch: bool = True  # 是否在生成前执行检索引擎预取上下文
    rag_prefetch_top_k_per_query: int = 2  # 每条候选 query 预取的文档数
    rag_prefetch_max_context_docs: int = 5  # 注入 prompt 的最大文档片段数
    enable_keyword_retrieval: bool = True  # 是否启用本地关键词检索，补足错误码/告警名/服务名精确匹配
    keyword_search_dirs: str = "aiops-docs,uploads"  # 关键词检索扫描目录，逗号分隔
    keyword_search_top_k: int = 5  # 关键词检索候选数
    enable_rerank: bool = True  # 是否启用检索结果重排
    enable_rrf_fusion: bool = True  # 是否在多路召回后使用 RRF 做粗排融合
    rrf_k: int = 60  # Reciprocal Rank Fusion 的平滑参数
    enable_semantic_rerank: bool = True  # 是否在重排中融合 query-doc 向量语义相似度
    rerank_candidate_limit: int = 20  # 进入重排的最大候选数
    rerank_top_n: int = 5  # 重排后注入 Prompt 的最大证据片段数
    semantic_rerank_content_chars: int = 900  # 语义重排时每个候选最多参与 embedding 的字符数
    enable_llm_retrieval_eval: bool = False  # 评测时是否也启用 LLM Query Rewrite/意图树打分，默认关闭以控制成本
    enable_output_trace: bool = True  # 是否记录 RAG 对话 Trace 和消息持久化日志
    enable_source_citation: bool = True  # 回答时是否要求引用检索来源
    enhanced_memory_recent_turns: int = 6  # 增强记忆保留的最近对话轮数
    enhanced_memory_summary_max_chars: int = 1200  # 增强记忆摘要最大字符数
    enhanced_memory_relevant_turns: int = 3  # 每次问题理解召回的相关历史轮数
    enhanced_memory_topic_stack_size: int = 8  # 记忆中保留的最近主题数量
    enhanced_memory_entity_limit: int = 12  # 每类实体最多保留数量
    enable_long_term_user_memory: bool = True  # 是否启用跨会话长期用户记忆
    long_term_memory_path: str = "data/memory/user_memory.json"  # 长期用户记忆持久化文件
    long_term_memory_recent_sessions: int = 12  # 长期记忆中保留的最近会话摘要数量

    # 模型路由配置
    enable_model_routing: bool = True  # 是否启用候选模型失败切换
    llm_candidate_models: str = "qwen-max,qwen-plus,qwen-turbo"  # 候选模型，逗号分隔
    llm_failure_threshold: int = 2  # 连续失败多少次后临时熔断
    llm_cooldown_seconds: int = 30  # 熔断冷却时间
    llm_first_token_timeout_seconds: float = 60.0  # 流式首包超时

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置（transport: stdio | sse | streamable-http）
    # 腾讯云托管 MCP 的 URL 通常含 /sse/，需使用 sse；本地 FastMCP 使用 streamable-http
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # Prometheus
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    # Set credentials in .env before starting the public release.
    auth_enabled: bool = True
    auth_username: str = "admin"
    auth_password: str = ""
    auth_display_name: str = "Ops Admin"
    auth_secret: str = ""
    auth_token_ttl_hours: int = 12

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
