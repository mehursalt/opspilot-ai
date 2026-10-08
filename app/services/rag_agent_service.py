"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

import asyncio
from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS
from app.agent.mcp_client import (
    get_mcp_client_with_retry,
    load_mcp_tools_safe,
    format_exception_chain,
    suggest_mcp_transport,
)
from app.services.query_understanding_service import (
    QueryUnderstandingResult,
    query_understanding_service,
)
from app.services.intent_guidance_service import intent_guidance_service
from app.services.diagnostic_workbench_service import diagnostic_workbench_service
from app.services.model_router_service import model_router_service
from app.services.output_service import output_service
from app.services.retrieval_engine_service import (
    RetrievalEngineResult,
    retrieval_engine_service,
)

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.active_model_name = self.model_name
        self.model = model_router_service.create_chat_model(self.active_model_name, streaming)

        # 定义基础工具（与 AIOps Planner/Executor 使用同一套默认本地工具）
        self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS)

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 创建内存检查点（用于会话管理）
        self.checkpointer = MemorySaver()

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False
        self._tools_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self, model_name: str | None = None, force: bool = False):
        """异步初始化 Agent（包括 MCP 工具）"""
        target_model = model_name or model_router_service.available_models(self.model_name)[0]
        if self._agent_initialized and self.active_model_name == target_model and not force:
            return

        if not self._tools_initialized:
            for name, server in config.mcp_servers.items():
                hint = suggest_mcp_transport(
                    str(server.get("url", "")),
                    str(server.get("transport", "")),
                )
                if hint:
                    logger.warning(f"MCP 配置 [{name}]: {hint}")

            mcp_client = await get_mcp_client_with_retry()
            mcp_tools, mcp_err = await load_mcp_tools_safe(mcp_client)
            if mcp_err:
                logger.warning(
                    f"MCP 工具加载失败，将仅使用本地工具继续运行:\n{mcp_err}"
                )
                self.mcp_tools = []
            else:
                self.mcp_tools = mcp_tools
                logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")
            self._tools_initialized = True

        all_tools = self.tools + self.mcp_tools
        self.active_model_name = target_model
        self.model = model_router_service.create_chat_model(target_model, self.streaming)

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    def _build_runtime_system_prompt(
        self,
        understanding: QueryUnderstandingResult,
        retrieval_result: RetrievalEngineResult | None = None,
    ) -> str:
        """构建带意图识别和增强记忆的运行时系统提示词。"""
        from textwrap import dedent

        context_parts = [
            self.system_prompt,
            dedent(f"""
                ## 当前用户问题理解
                - 识别意图: {understanding.intent}
                - 置信度: {understanding.confidence:.2f}
                - 路由建议: {understanding.route_hint}
                - 改写后问题: {understanding.rewritten_question}
                - 候选检索查询: {"；".join(understanding.search_queries)}

                使用要求:
                1. 如果改写后问题与原始问题冲突，以原始问题为准。
                2. 如果识别意图为 knowledge_query，优先考虑知识库检索工具。
                3. 如果识别意图为 aiops_diagnosis，优先考虑告警、监控、日志和知识库工具。
                4. 如果识别意图为 time_query，优先考虑时间工具。
                5. 回答时结合会话记忆，但不要编造用户没有说过的信息。
                6. 如果下方已经提供检索上下文，优先基于检索上下文作答；上下文不足时再调用工具或说明不确定。
                7. 长期记忆只用于理解用户偏好或明确相关的历史主题，不要把“项目评测、Baseline、面试包装”等旧话题硬套进运维排障答案。
                8. 不要重复输出同一套完整步骤；如果需要补充，只补充新增差异点。
                9. 不要编造具体日期、时间范围、地域、日志主题或工具执行结果；未真实调用工具时，只能写成“建议查询/可执行”。
            """).strip(),
        ]

        if understanding.memory_context:
            context_parts.append(
                dedent(f"""
                    ## 增强会话记忆
                    {understanding.memory_context}
                """).strip()
            )

        if retrieval_result and retrieval_result.context:
            context_parts.append(
                dedent(f"""
                    ## 检索引擎预取上下文
                    {retrieval_result.context}
                """).strip()
            )

        if config.enable_source_citation and retrieval_result and retrieval_result.sources:
            source_lines = []
            for source in retrieval_result.sources:
                source_lines.append(
                    f"- [{source.get('id')}] {source.get('source', 'unknown')} "
                    f"({source.get('kind', 'unknown')})"
                )
            context_parts.append(
                dedent(f"""
                    ## Source Citation Requirement
                    When the answer uses retrieved context, cite the evidence with [S1], [S2] style markers.
                    Do not cite a source if the conclusion is not supported by that source.
                    Available sources:
                    {chr(10).join(source_lines)}
                """).strip()
            )

        return "\n\n".join(context_parts)

    def _build_user_message_content(
        self,
        question: str,
        understanding: QueryUnderstandingResult,
    ) -> str:
        """构造包含原始问题、改写问题和检索提示的用户消息。"""
        from textwrap import dedent

        return dedent(f"""
            原始问题:
            {question}

            改写后问题:
            {understanding.rewritten_question}

            候选检索查询:
            {chr(10).join(f"- {query}" for query in understanding.search_queries)}

            请优先围绕原始问题作答；当需要查知识库或工具时，可以参考改写后问题和候选检索查询。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
        user_id: str | None = None,
        include_workbench: bool = False,
    ) -> str | dict[str, Any]:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()
            understanding = await query_understanding_service.analyze_async(
                question,
                session_id,
                user_id=user_id,
            )
            guidance = intent_guidance_service.evaluate(understanding)

            output_service.persist_message(
                session_id=session_id,
                role="user",
                content=question,
                metadata={
                    "intent": understanding.intent,
                    "rewritten_question": understanding.rewritten_question,
                },
            )
            output_service.record_trace(
                session_id,
                "query_understanding",
                {
                    "understanding": understanding.to_dict(),
                    "guidance": guidance.to_dict(),
                },
            )

            logger.info(
                f"[会话 {session_id}] RAG Agent 收到查询（非流式）: "
                f"{question}, intent={understanding.intent}, rewritten={understanding.rewritten_question}"
            )

            if guidance.should_guide:
                query_understanding_service.update_memory(
                    session_id=session_id,
                    question=question,
                    answer=guidance.message,
                    understanding=understanding,
                    user_id=user_id,
                )
                output_service.persist_message(
                    session_id=session_id,
                    role="assistant",
                    content=guidance.message,
                    metadata={"guided": True, "reasons": guidance.reasons},
                )
                output_service.record_trace(
                    session_id,
                    "intent_guidance",
                    {"message": guidance.message, "reasons": guidance.reasons},
                )
                if include_workbench:
                    return {
                        "answer": guidance.message,
                        "workbench": diagnostic_workbench_service.build(
                            question=question,
                            answer=guidance.message,
                            session_id=session_id,
                            understanding=understanding,
                            guidance=guidance,
                            model=self.active_model_name,
                            mode="quick",
                        ),
                    }
                return guidance.message

            retrieval_result = await retrieval_engine_service.retrieve(understanding)
            output_service.record_trace(
                session_id,
                "retrieval",
                retrieval_result.to_dict(),
            )

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self._build_runtime_system_prompt(understanding, retrieval_result)),
                HumanMessage(content=self._build_user_message_content(question, understanding))
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = None
            used_model = self.active_model_name
            last_error: Exception | None = None
            for model_name in model_router_service.available_models(self.model_name):
                await self._initialize_agent(
                    model_name=model_name,
                    force=model_name != self.active_model_name,
                )
                output_service.record_trace(
                    session_id,
                    "model_route_attempt",
                    {"model": model_name, "streaming": False},
                )
                try:
                    result = await self.agent.ainvoke(
                        input=agent_input,
                        config=config_dict,
                    )
                    used_model = model_name
                    model_router_service.mark_success(model_name)
                    output_service.record_trace(
                        session_id,
                        "model_route_success",
                        {"model": model_name, "streaming": False},
                    )
                    break
                except Exception as e:
                    last_error = e
                    model_router_service.mark_failure(model_name, e)
                    output_service.record_trace(
                        session_id,
                        "model_route_failure",
                        {
                            "model": model_name,
                            "streaming": False,
                            "error": str(e),
                        },
                    )
                    logger.warning(
                        "[session {}] model {} failed, trying next candidate: {}",
                        session_id,
                        model_name,
                        e,
                    )

            if result is None:
                raise RuntimeError("All candidate LLM models failed") from last_error

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
                answer_text = answer if isinstance(answer, str) else str(answer)
                query_understanding_service.update_memory(
                    session_id=session_id,
                    question=question,
                    answer=answer_text,
                    understanding=understanding,
                    user_id=user_id,
                )
                output_service.persist_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer_text,
                    metadata={
                        "title": output_service.generate_title(question),
                        "retrieval": retrieval_result.trace,
                        "sources": retrieval_result.sources,
                        "model": used_model,
                    },
                )
                output_service.record_trace(
                    session_id,
                    "generation_complete",
                    {
                        "answer_chars": len(answer_text),
                        "title": output_service.generate_title(question),
                        "model": used_model,
                        "sources": retrieval_result.sources,
                    },
                )

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                if include_workbench:
                    return {
                        "answer": answer_text,
                        "workbench": diagnostic_workbench_service.build(
                            question=question,
                            answer=answer_text,
                            session_id=session_id,
                            understanding=understanding,
                            retrieval_result=retrieval_result,
                            guidance=guidance,
                            model=used_model,
                            mode="quick",
                        ),
                    }
                return answer_text

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            if include_workbench:
                return {
                    "answer": "",
                    "workbench": diagnostic_workbench_service.build(
                        question=question,
                        answer="",
                        session_id=session_id,
                        understanding=understanding,
                        retrieval_result=retrieval_result,
                        guidance=guidance,
                        model=used_model,
                        mode="quick",
                    ),
                }
            return ""

        except Exception as e:
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（非流式）: "
                f"{format_exception_chain(e)}"
            )
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
        user_id: str | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()
            understanding = await query_understanding_service.analyze_async(
                question,
                session_id,
                user_id=user_id,
            )
            guidance = intent_guidance_service.evaluate(understanding)

            logger.info(
                f"[会话 {session_id}] RAG Agent 收到查询（流式）: "
                f"{question}, intent={understanding.intent}, rewritten={understanding.rewritten_question}"
            )

            output_service.persist_message(
                session_id=session_id,
                role="user",
                content=question,
                metadata={
                    "intent": understanding.intent,
                    "rewritten_question": understanding.rewritten_question,
                },
            )
            output_service.record_trace(
                session_id,
                "query_understanding",
                {
                    "understanding": understanding.to_dict(),
                    "guidance": guidance.to_dict(),
                },
            )

            yield {
                "type": "query_understanding",
                "data": understanding.to_dict(),
            }

            if guidance.should_guide:
                yield {
                    "type": "guidance",
                    "data": guidance.to_dict(),
                }
                yield {
                    "type": "content",
                    "data": guidance.message,
                    "node": "intent_guidance",
                }
                query_understanding_service.update_memory(
                    session_id=session_id,
                    question=question,
                    answer=guidance.message,
                    understanding=understanding,
                    user_id=user_id,
                )
                output_service.persist_message(
                    session_id=session_id,
                    role="assistant",
                    content=guidance.message,
                    metadata={"guided": True, "reasons": guidance.reasons},
                )
                output_service.record_trace(
                    session_id,
                    "intent_guidance",
                    {"message": guidance.message, "reasons": guidance.reasons},
                )
                yield {
                    "type": "diagnostic_workbench",
                    "data": diagnostic_workbench_service.build(
                        question=question,
                        answer=guidance.message,
                        session_id=session_id,
                        understanding=understanding,
                        guidance=guidance,
                        model=self.active_model_name,
                        mode="stream",
                    ),
                }
                yield {"type": "complete"}
                return

            retrieval_result = await retrieval_engine_service.retrieve(understanding)
            output_service.record_trace(
                session_id,
                "retrieval",
                retrieval_result.to_dict(),
            )
            yield {
                "type": "search_results",
                "data": retrieval_result.to_dict(),
            }

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self._build_runtime_system_prompt(understanding, retrieval_result)),
                HumanMessage(content=self._build_user_message_content(question, understanding))
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            answer_chunks: list[str] = []
            used_model = self.active_model_name
            last_error: Exception | None = None

            for model_name in model_router_service.available_models(self.model_name):
                await self._initialize_agent(
                    model_name=model_name,
                    force=model_name != self.active_model_name,
                )
                output_service.record_trace(
                    session_id,
                    "model_route_attempt",
                    {"model": model_name, "streaming": True},
                )
                yield {
                    "type": "model_route",
                    "data": {"model": model_name, "status": "start"},
                }

                stream = self.agent.astream(
                    input=agent_input,
                    config=config_dict,
                    stream_mode="messages",
                )
                first_packet_seen = False

                try:
                    while True:
                        try:
                            if not first_packet_seen:
                                token, metadata = await asyncio.wait_for(
                                    stream.__anext__(),
                                    timeout=config.llm_first_token_timeout_seconds,
                                )
                                first_packet_seen = True
                                yield {
                                    "type": "model_route",
                                    "data": {
                                        "model": model_name,
                                        "status": "first_packet",
                                    },
                                }
                            else:
                                token, metadata = await stream.__anext__()
                        except StopAsyncIteration:
                            break

                        node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                        message_type = type(token).__name__

                        if message_type in ("AIMessage", "AIMessageChunk"):
                            content_blocks = getattr(token, 'content_blocks', None)

                            if content_blocks and isinstance(content_blocks, list):
                                for block in content_blocks:
                                    if isinstance(block, dict) and block.get('type') == 'text':
                                        text_content = block.get('text', '')
                                        if text_content:
                                            answer_chunks.append(text_content)
                                            yield {
                                                "type": "content",
                                                "data": text_content,
                                                "node": node_name
                                            }

                    used_model = model_name
                    model_router_service.mark_success(model_name)
                    output_service.record_trace(
                        session_id,
                        "model_route_success",
                        {"model": model_name, "streaming": True},
                    )
                    yield {
                        "type": "model_route",
                        "data": {"model": model_name, "status": "success"},
                    }
                    break
                except Exception as e:
                    last_error = e
                    model_router_service.mark_failure(model_name, e)
                    output_service.record_trace(
                        session_id,
                        "model_route_failure",
                        {
                            "model": model_name,
                            "streaming": True,
                            "error": str(e),
                        },
                    )
                    close_stream = getattr(stream, "aclose", None)
                    if close_stream:
                        await close_stream()
                    yield {
                        "type": "model_route",
                        "data": {
                            "model": model_name,
                            "status": "failure",
                            "error": str(e),
                        },
                    }
                    if answer_chunks:
                        raise
                    continue
            else:
                raise RuntimeError("All candidate LLM models failed") from last_error

            query_understanding_service.update_memory(
                session_id=session_id,
                question=question,
                answer="".join(answer_chunks),
                understanding=understanding,
                user_id=user_id,
            )
            answer_text = "".join(answer_chunks)
            output_service.persist_message(
                session_id=session_id,
                role="assistant",
                content=answer_text,
                metadata={
                    "title": output_service.generate_title(question),
                    "retrieval": retrieval_result.trace,
                    "sources": retrieval_result.sources,
                    "model": used_model,
                },
            )
            output_service.record_trace(
                session_id,
                "generation_complete",
                {
                    "answer_chars": len(answer_text),
                    "title": output_service.generate_title(question),
                    "model": used_model,
                    "sources": retrieval_result.sources,
                },
            )

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {
                "type": "diagnostic_workbench",
                "data": diagnostic_workbench_service.build(
                    question=question,
                    answer=answer_text,
                    session_id=session_id,
                    understanding=understanding,
                    retrieval_result=retrieval_result,
                    guidance=guidance,
                    model=used_model,
                    mode="stream",
                ),
            }
            yield {"type": "complete"}

        except Exception as e:
            detail = format_exception_chain(e)
            logger.error(
                f"[会话 {session_id}] RAG Agent 查询失败（流式）: {detail}"
            )
            yield {"type": "error", "data": detail}

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            # 使用 checkpointer 的 get 方法获取最新的检查点
            config = {"configurable": {"thread_id": session_id}}

            # 获取该 thread 的最新检查点
            checkpoint_tuple = self.checkpointer.get(config)

            if not checkpoint_tuple:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # checkpoint_tuple 可能是命名元组或普通元组，安全地提取 checkpoint
            # 通常第一个元素是 checkpoint 数据
            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint  # type: ignore
            else:
                # 如果是普通元组，第一个元素是 checkpoint
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)
            query_understanding_service.clear_memory(session_id)

            logger.info(f"已清除会话历史: {session_id}")
            return True

        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    def get_enhanced_memory(self, session_id: str) -> dict[str, Any]:
        """获取增强会话记忆，便于调试和前端展示。"""
        return query_understanding_service.get_memory(session_id).to_dict()

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)
