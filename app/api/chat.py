"""Chat API routes for the RAG agent."""

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.agent.mcp_client import format_exception_chain
from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse
from app.services.agent_orchestrator_service import agent_orchestrator_service
from app.services.diagnostic_workbench_service import diagnostic_workbench_service
from app.services.feedback_service import FeedbackRecord, feedback_service
from app.services.long_term_memory_service import long_term_memory_service
from app.services.model_router_service import model_router_service
from app.services.query_understanding_service import query_understanding_service
from app.services.rag_agent_service import rag_agent_service
from app.services.retrieval_engine_service import retrieval_engine_service
from app.services.term_mapping_service import term_mapping_service


router = APIRouter()


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(..., alias="sessionId")
    message_id: str = Field("", alias="messageId")
    vote: int = 0
    comment: str = ""
    question: str = ""
    answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TermMappingRequest(BaseModel):
    id: str = ""
    source_term: str = Field(..., alias="sourceTerm")
    target_term: str = Field(..., alias="targetTerm")
    scenario: str = "AIOps"
    enabled: bool = True
    priority: int = 50
    match_type: str = Field("contains", alias="matchType")
    description: str = ""

    model_config = ConfigDict(populate_by_name=True)


def _build_fallback_answer(
    question: str,
    error: Exception,
    understanding: Any | None,
    retrieval: Any | None,
) -> str:
    """Build a useful answer when LLM generation fails."""
    if not understanding:
        return f"模型生成暂时失败：{error}。请稍后重试，或检查模型 API Key / 网络连接。"

    sources = getattr(retrieval, "sources", []) if retrieval else []
    trace = getattr(retrieval, "trace", {}) if retrieval else {}
    channels = (trace.get("channel_summary", {}) or {}).get("raw_by_channel", {})
    source_lines = []
    for source in sources[:5]:
        source_lines.append(
            f"- [{source.get('id')}] {source.get('file_name') or source.get('source', 'unknown')} "
            f"(channels={','.join(source.get('channels') or [])})"
        )

    if understanding.intent in ("aiops_diagnosis", "metrics_alerts"):
        base_steps = [
            "1. 先确认故障对象：服务名、实例、地域/集群、告警时间范围。",
            "2. 查看监控趋势：判断是瞬时毛刺、持续高负载，还是周期性任务触发。",
            "3. 结合日志和进程维度定位：关注 ERROR/WARN、PID、线程、GC、慢查询等线索。",
            "4. 先做低风险止血：扩容、限流、摘除异常实例；重启/回滚需要人工确认。",
            "5. 恢复后验证：CPU/内存/错误率/响应时间回落，并观察至少一个告警窗口。",
        ]
    else:
        base_steps = [
            "1. 我已经完成问题理解和检索链路整理。",
            "2. 当前大模型生成失败，建议先根据右侧工作台查看 Trace、证据来源和下一步问题。",
            "3. 检查模型 API Key、网络连接或切换候选模型后可重新提问。",
        ]

    return "\n".join(
        [
            "模型生成暂时失败，但系统已保留本轮诊断链路，可先按工作台结果继续排查。",
            "",
            f"- 原始问题：{question}",
            f"- 识别意图：{understanding.intent}",
            f"- 重写问题：{understanding.rewritten_question}",
            f"- 失败原因：{error}",
            f"- 检索通道：{', '.join(channels.keys()) if channels else '暂无'}",
            "",
            "## 兜底排查建议",
            *base_steps,
            "",
            "## 已检索证据",
            *(source_lines or ["- 暂无可用来源"]),
        ]
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        logger.info("[session {}] chat request: {}", request.id, request.question)
        chat_result = await agent_orchestrator_service.query(
            request.question,
            session_id=request.id,
            user_id=request.user_id,
            include_workbench=True,
        )
        if isinstance(chat_result, dict):
            answer = str(chat_result.get("answer") or "")
            workbench = chat_result.get("workbench")
        else:
            answer = str(chat_result)
            understanding = await query_understanding_service.analyze_async(
                request.question,
                request.id,
                user_id=request.user_id,
            )
            workbench = diagnostic_workbench_service.build(
                question=request.question,
                answer=answer,
                session_id=request.id,
                understanding=understanding,
                model=rag_agent_service.active_model_name,
                mode="quick",
            )
        logger.info("[session {}] chat finished", request.id)

        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,
                "workbench": workbench,
                "errorMessage": None,
            },
        }

    except Exception as e:
        logger.error("Chat endpoint failed: {}", e)
        try:
            understanding = await query_understanding_service.analyze_async(
                request.question,
                request.id,
                user_id=request.user_id,
            )
            retrieval = None
            try:
                retrieval = await retrieval_engine_service.retrieve(understanding)
            except Exception as retrieval_error:
                logger.warning("Fallback retrieval failed: {}", retrieval_error)

            answer = _build_fallback_answer(request.question, e, understanding, retrieval)
            workbench = diagnostic_workbench_service.build(
                question=request.question,
                answer=answer,
                session_id=request.id,
                understanding=understanding,
                retrieval_result=retrieval,
                model=rag_agent_service.active_model_name,
                mode="quick-fallback",
            )
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "success": True,
                    "answer": answer,
                    "workbench": workbench,
                    "errorMessage": None,
                },
            }
        except Exception as fallback_error:
            logger.error("Chat fallback failed: {}", fallback_error)
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e),
            },
        }


@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint based on SSE."""
    logger.info("[session {}] stream chat request: {}", request.id, request.question)

    async def event_generator():
        try:
            async for chunk in agent_orchestrator_service.query_stream(
                request.question,
                session_id=request.id,
                user_id=request.user_id,
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                if chunk_type == "complete":
                    payload = {"type": "done", "data": chunk_data}
                elif chunk_type == "debug":
                    payload = {
                        "type": "debug",
                        "node": chunk.get("node", "unknown"),
                        "message_type": chunk.get("message_type", "unknown"),
                    }
                elif chunk_type == "error":
                    payload = {"type": "error", "data": str(chunk_data)}
                else:
                    payload = {"type": chunk_type, "data": chunk_data}

                yield {
                    "event": "message",
                    "data": json.dumps(payload, ensure_ascii=False),
                }

            logger.info("[session {}] stream chat finished", request.id)

        except Exception as e:
            detail = format_exception_chain(e)
            logger.error("Stream chat endpoint failed: {}", detail)
            yield {
                "event": "message",
                "data": json.dumps(
                    {"type": "error", "data": detail},
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """Clear chat history and enhanced query-understanding memory."""
    try:
        success = rag_agent_service.clear_session(request.session_id)
        logger.info("Clear session: {}, success={}", request.session_id, success)
        return ApiResponse(
            status="success" if success else "error",
            message="session cleared" if success else "clear session failed",
            data=None,
        )

    except Exception as e:
        logger.error("Clear session failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """Load LangGraph checkpoint chat history."""
    try:
        history = rag_agent_service.get_session_history(session_id)
        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history,
        )

    except Exception as e:
        logger.error("Get session info failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/memory/{session_id}", response_model=ApiResponse)
async def get_enhanced_memory(session_id: str) -> ApiResponse:
    """Inspect enhanced conversation memory for query understanding."""
    try:
        memory = rag_agent_service.get_enhanced_memory(session_id)
        return ApiResponse(
            status="success",
            message="enhanced memory loaded",
            data=memory,
        )

    except Exception as e:
        logger.error("Get enhanced memory failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/memory/user/{user_id}", response_model=ApiResponse)
async def get_long_term_user_memory(user_id: str) -> ApiResponse:
    """Inspect persistent long-term user memory."""
    try:
        memory = long_term_memory_service.get_user_memory(user_id)
        return ApiResponse(
            status="success",
            message="long-term user memory loaded",
            data=memory,
        )

    except Exception as e:
        logger.error("Get long-term user memory failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/memory/user/{user_id}", response_model=ApiResponse)
async def clear_long_term_user_memory(user_id: str) -> ApiResponse:
    """Clear persistent long-term user memory."""
    try:
        long_term_memory_service.clear_user_memory(user_id)
        return ApiResponse(
            status="success",
            message="long-term user memory cleared",
            data=None,
        )

    except Exception as e:
        logger.error("Clear long-term user memory failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/feedback", response_model=ApiResponse)
async def record_chat_feedback(request: FeedbackRequest) -> ApiResponse:
    """Record user feedback for later RAG quality analysis."""
    try:
        record = FeedbackRecord(
            session_id=request.session_id,
            message_id=request.message_id,
            vote=request.vote,
            comment=request.comment,
            question=request.question,
            answer=request.answer,
            metadata=request.metadata,
        )
        data = feedback_service.record_feedback(record)
        return ApiResponse(
            status="success",
            message="feedback recorded",
            data=data,
        )
    except Exception as e:
        logger.error("Record feedback failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/eval/retrieval", response_model=ApiResponse)
async def run_retrieval_eval() -> ApiResponse:
    """Run the external retrieval evaluation set."""
    try:
        data = await feedback_service.run_retrieval_eval()
        return ApiResponse(
            status="success",
            message="retrieval eval finished",
            data=data,
        )
    except Exception as e:
        logger.error("Retrieval eval failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/eval/retrieval/meta", response_model=ApiResponse)
async def get_retrieval_eval_metadata() -> ApiResponse:
    """Load retrieval eval dataset metadata without running the expensive eval."""
    try:
        return ApiResponse(
            status="success",
            message="retrieval eval metadata loaded",
            data=feedback_service.get_retrieval_eval_metadata(),
        )
    except Exception as e:
        logger.error("Retrieval eval metadata failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/model/health", response_model=ApiResponse)
async def get_model_health() -> ApiResponse:
    """Inspect candidate model health used by the lightweight router."""
    try:
        return ApiResponse(
            status="success",
            message="model health loaded",
            data=model_router_service.health_snapshot(),
        )
    except Exception as e:
        logger.error("Get model health failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/term-mappings", response_model=ApiResponse)
async def list_term_mappings() -> ApiResponse:
    """List query term mapping rules used before intent recognition."""
    try:
        return ApiResponse(
            status="success",
            message="term mappings loaded",
            data=term_mapping_service.list_mappings(),
        )
    except Exception as e:
        logger.error("List term mappings failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/term-mappings", response_model=ApiResponse)
async def upsert_term_mapping(request: TermMappingRequest) -> ApiResponse:
    """Create or update one query term mapping rule."""
    try:
        data = term_mapping_service.upsert_mapping(request.model_dump(by_alias=False))
        return ApiResponse(
            status="success",
            message="term mapping saved",
            data=data,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Save term mapping failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/term-mappings/{mapping_id}", response_model=ApiResponse)
async def delete_term_mapping(mapping_id: str) -> ApiResponse:
    """Delete one query term mapping rule."""
    try:
        deleted = term_mapping_service.delete_mapping(mapping_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="term mapping not found")
        return ApiResponse(
            status="success",
            message="term mapping deleted",
            data={"id": mapping_id},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete term mapping failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/retrieval/inspect", response_model=ApiResponse)
async def inspect_retrieval(
    question: str = Query(..., min_length=1, max_length=500),
    session_id: str = Query("inspect-session", alias="sessionId"),
    user_id: str = Query("admin", alias="userId"),
) -> ApiResponse:
    """Run query understanding and retrieval only, returning an explainable trace."""
    try:
        understanding = await query_understanding_service.analyze_async(
            question,
            session_id,
            user_id=user_id,
        )
        retrieval = await retrieval_engine_service.retrieve(understanding)
        return ApiResponse(
            status="success",
            message="retrieval trace inspected",
            data={
                "understanding": understanding.to_dict(),
                "retrieval": retrieval.to_dict(),
            },
        )
    except Exception as e:
        logger.error("Inspect retrieval failed: {}", e)
        raise HTTPException(status_code=500, detail=str(e))
