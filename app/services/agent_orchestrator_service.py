"""Main-chat agent orchestrator.

The user-facing chat entry should not expose two disconnected agents. This
service routes one question to the execution paradigm that fits the task:

- AIOps incident diagnosis -> Plan-and-Execution.
- Knowledge/project/general questions -> ReAct + RAG agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from loguru import logger

from app.agent.aiops.executor import executor
from app.agent.aiops.planner import planner
from app.agent.aiops.replanner import replanner
from app.agent.aiops.state import PlanExecuteState
from app.services.diagnostic_workbench_service import diagnostic_workbench_service
from app.services.intent_guidance_service import GuidanceDecision, intent_guidance_service
from app.services.output_service import output_service
from app.services.query_understanding_service import (
    QueryUnderstandingResult,
    query_understanding_service,
)
from app.services.rag_agent_service import rag_agent_service


@dataclass
class AgentRouteDecision:
    """Route selected for a single user question."""

    mode: str
    paradigm: str
    reason: str
    confidence: float
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "paradigm": self.paradigm,
            "reason": self.reason,
            "confidence": self.confidence,
            "signals": self.signals,
        }


class AgentOrchestratorService:
    """Route main chat requests to ReAct or Plan-and-Execution."""

    FAULT_SIGNALS = (
        "排查",
        "故障",
        "告警",
        "异常",
        "过高",
        "飙高",
        "不可用",
        "宕机",
        "超时",
        "响应慢",
        "变慢",
        "cpu",
        "内存",
        "oom",
        "gc",
        "磁盘",
        "日志",
        "监控",
        "线上",
        "定位",
        "怎么处理",
        "怎么办",
    )
    KNOWLEDGE_SIGNALS = (
        "什么是",
        "解释",
        "讲解",
        "原理",
        "实现",
        "代码",
        "面试",
        "文档",
        "项目",
        "设计",
        "范式",
        "区别",
        "对比",
        "rag",
        "mcp",
        "ragent",
    )

    async def query(
        self,
        question: str,
        *,
        session_id: str,
        user_id: str | None = None,
        include_workbench: bool = False,
    ) -> str | dict[str, Any]:
        understanding = await query_understanding_service.analyze_async(
            question,
            session_id,
            user_id=user_id,
        )
        guidance = intent_guidance_service.evaluate(understanding)
        route = self.route(understanding)

        output_service.record_trace(
            session_id,
            "agent_orchestrator_route",
            {
                "route": route.to_dict(),
                "understanding": understanding.to_dict(),
                "guidance": guidance.to_dict(),
            },
        )

        if guidance.should_guide:
            return self._guidance_result(
                question,
                session_id,
                user_id,
                understanding,
                guidance,
                route,
                include_workbench,
            )

        if route.mode == "plan_execute":
            return await self._run_plan_execute(
                question,
                session_id=session_id,
                user_id=user_id,
                understanding=understanding,
                route=route,
                include_workbench=include_workbench,
            )

        result = await rag_agent_service.query(
            question,
            session_id=session_id,
            user_id=user_id,
            include_workbench=include_workbench,
        )
        if isinstance(result, dict):
            self._attach_route(result.get("workbench"), route)
        return result

    async def query_stream(
        self,
        question: str,
        *,
        session_id: str,
        user_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        understanding = await query_understanding_service.analyze_async(
            question,
            session_id,
            user_id=user_id,
        )
        guidance = intent_guidance_service.evaluate(understanding)
        route = self.route(understanding)

        yield {"type": "agent_route", "data": route.to_dict()}
        yield {"type": "query_understanding", "data": understanding.to_dict()}

        if guidance.should_guide:
            yield {"type": "guidance", "data": guidance.to_dict()}
            yield {"type": "content", "data": guidance.message}
            yield {"type": "complete", "data": {"answer": guidance.message, "route": route.to_dict()}}
            return

        if route.mode != "plan_execute":
            async for chunk in rag_agent_service.query_stream(
                question,
                session_id=session_id,
                user_id=user_id,
            ):
                yield chunk
            return

        yield {
            "type": "content",
            "data": "已识别为 AIOps 故障诊断任务，切换到 Plan-and-Execution 范式。\n\n",
        }
        result = await self._run_plan_execute(
            question,
            session_id=session_id,
            user_id=user_id,
            understanding=understanding,
            route=route,
            include_workbench=True,
        )
        answer = str(result.get("answer") if isinstance(result, dict) else result)
        workbench = result.get("workbench") if isinstance(result, dict) else None
        yield {"type": "content", "data": answer}
        if workbench:
            yield {"type": "diagnostic_workbench", "data": workbench}
        yield {"type": "complete", "data": {"answer": answer, "route": route.to_dict()}}

    def route(self, understanding: QueryUnderstandingResult) -> AgentRouteDecision:
        original_text = " ".join(
            [
                understanding.original_question,
                understanding.normalized_question,
            ]
        ).lower()
        text = " ".join(
            [
                understanding.original_question,
                understanding.normalized_question,
                understanding.rewritten_question,
                " ".join(understanding.sub_questions),
                " ".join(str(node.get("path", "")) for node in understanding.intent_nodes[:3]),
            ]
        ).lower()
        original_knowledge_hits = [
            item for item in self.KNOWLEDGE_SIGNALS if item.lower() in original_text
        ]
        original_fault_hits = [
            item for item in self.FAULT_SIGNALS if item.lower() in original_text
        ]
        knowledge_hits = [item for item in self.KNOWLEDGE_SIGNALS if item.lower() in text]
        fault_hits = [item for item in self.FAULT_SIGNALS if item.lower() in text]

        if original_knowledge_hits and not self._has_incident_signal(original_text):
            return AgentRouteDecision(
                mode="react_rag",
                paradigm="ReAct + RAG",
                reason="原始问题是概念/项目讲解，不进入多步骤排障",
                confidence=understanding.confidence,
                signals=original_knowledge_hits[:8],
            )

        if (
            understanding.intent == "aiops_diagnosis"
            and (original_fault_hits or fault_hits)
            and not self._is_knowledge_about_aiops(original_text)
        ):
            return AgentRouteDecision(
                mode="plan_execute",
                paradigm="Plan-and-Execution",
                reason="AIOps 故障诊断需要规划、工具执行和 replan",
                confidence=max(understanding.confidence, 0.75),
                signals=(original_fault_hits or fault_hits)[:8],
            )
        if understanding.intent == "metrics_alerts":
            return AgentRouteDecision(
                mode="plan_execute",
                paradigm="Plan-and-Execution",
                reason="监控告警排查属于多步骤诊断任务",
                confidence=max(understanding.confidence, 0.72),
                signals=fault_hits[:8] or ["metrics_alerts"],
            )
        return AgentRouteDecision(
            mode="react_rag",
            paradigm="ReAct + RAG",
            reason="知识问答/项目讲解更适合检索证据后直接推理回答",
            confidence=understanding.confidence,
            signals=(knowledge_hits or fault_hits)[:8],
        )

    async def _run_plan_execute(
        self,
        question: str,
        *,
        session_id: str,
        user_id: str | None,
        understanding: QueryUnderstandingResult,
        route: AgentRouteDecision,
        include_workbench: bool,
    ) -> str | dict[str, Any]:
        output_service.persist_message(
            session_id=session_id,
            role="user",
            content=question,
            metadata={
                "intent": understanding.intent,
                "agent_route": route.to_dict(),
                "rewritten_question": understanding.rewritten_question,
            },
        )

        state: PlanExecuteState = {
            "input": self._build_plan_input(question, understanding),
            "plan": [],
            "past_steps": [],
            "response": "",
        }
        trace_steps: list[dict[str, Any]] = []

        plan_update = await planner(state)
        state.update(plan_update)
        trace_steps.append({"node": "planner", "output": plan_update})

        max_loops = 8
        for _ in range(max_loops):
            if state.get("response"):
                break
            if state.get("plan"):
                exec_update = await executor(state)
                state["plan"] = exec_update.get("plan", state.get("plan", []))
                state["past_steps"] = [
                    *state.get("past_steps", []),
                    *exec_update.get("past_steps", []),
                ]
                trace_steps.append({"node": "executor", "output": exec_update})

            replan_update = await replanner(state)
            if "plan" in replan_update:
                state["plan"] = replan_update["plan"]
            if "response" in replan_update:
                state["response"] = replan_update["response"]
            trace_steps.append({"node": "replanner", "output": replan_update})

            if not state.get("plan") and not state.get("response"):
                final_update = await replanner(state)
                if "response" in final_update:
                    state["response"] = final_update["response"]
                trace_steps.append({"node": "final_response", "output": final_update})
                break

        answer = state.get("response") or self._fallback_plan_answer(question, state)
        query_understanding_service.update_memory(
            session_id=session_id,
            question=question,
            answer=answer,
            understanding=understanding,
            user_id=user_id,
        )
        output_service.persist_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            metadata={
                "title": output_service.generate_title(question),
                "agent_route": route.to_dict(),
                "plan_execute_trace": trace_steps,
            },
        )
        output_service.record_trace(
            session_id,
            "plan_execute_agent",
            {"route": route.to_dict(), "steps": trace_steps, "final_state": state},
        )

        if not include_workbench:
            return answer

        workbench = diagnostic_workbench_service.build(
            question=question,
            answer=answer,
            session_id=session_id,
            understanding=understanding,
            model="Plan-and-Execution",
            mode="plan_execute",
        )
        self._attach_route(workbench, route, trace_steps)
        return {"answer": answer, "workbench": workbench}

    def _guidance_result(
        self,
        question: str,
        session_id: str,
        user_id: str | None,
        understanding: QueryUnderstandingResult,
        guidance: GuidanceDecision,
        route: AgentRouteDecision,
        include_workbench: bool,
    ) -> str | dict[str, Any]:
        query_understanding_service.update_memory(
            session_id=session_id,
            question=question,
            answer=guidance.message,
            understanding=understanding,
            user_id=user_id,
        )
        if not include_workbench:
            return guidance.message
        workbench = diagnostic_workbench_service.build(
            question=question,
            answer=guidance.message,
            session_id=session_id,
            understanding=understanding,
            guidance=guidance,
            model=route.paradigm,
            mode="guidance",
        )
        self._attach_route(workbench, route)
        return {"answer": guidance.message, "workbench": workbench}

    def _build_plan_input(self, question: str, understanding: QueryUnderstandingResult) -> str:
        return "\n".join(
            [
                question,
                "",
                "## Query Understanding",
                f"- intent: {understanding.intent}",
                f"- rewritten_question: {understanding.rewritten_question}",
                f"- sub_questions: {'; '.join(understanding.sub_questions)}",
                f"- missing_slots: {'; '.join(understanding.missing_slots) or 'none'}",
                f"- search_queries: {'; '.join(understanding.search_queries)}",
            ]
        )

    def _attach_route(
        self,
        workbench: dict[str, Any] | None,
        route: AgentRouteDecision,
        trace_steps: list[dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(workbench, dict):
            return
        workbench["agent_route"] = route.to_dict()
        workbench.setdefault("summary", {})["agent_paradigm"] = route.paradigm
        workbench.setdefault("summary", {})["agent_route_reason"] = route.reason
        if trace_steps is not None:
            workbench["plan_execute_trace"] = trace_steps
            plan_steps = self._workbench_plan_steps(trace_steps)
            if plan_steps:
                workbench["execution_steps"] = [
                    {
                        "id": "agent_orchestrator",
                        "title": "Agent 范式路由",
                        "status": "success",
                        "detail": f"{route.paradigm}: {route.reason}",
                    },
                    *plan_steps,
                    *workbench.get("execution_steps", []),
                ]

    def _fallback_plan_answer(self, question: str, state: PlanExecuteState) -> str:
        steps = state.get("past_steps", [])
        if not steps:
            return f"已切换到 Plan-and-Execution，但尚未获取到可用执行结果。原始问题：{question}"
        lines = ["# AIOps 诊断执行结果", "", f"原始问题：{question}", ""]
        for idx, (step, result) in enumerate(steps, start=1):
            lines.append(f"## 步骤 {idx}: {step}")
            lines.append(str(result))
            lines.append("")
        return "\n".join(lines).strip()

    def _workbench_plan_steps(self, trace_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for idx, item in enumerate(trace_steps, start=1):
            node = str(item.get("node") or "plan_execute")
            output = item.get("output") or {}
            if not isinstance(output, dict):
                detail = str(output)
            elif node == "planner":
                detail = " -> ".join(str(step) for step in output.get("plan", [])[:5]) or "未生成计划"
            elif node == "executor":
                past_steps = output.get("past_steps", [])
                detail = str(past_steps[0][0]) if past_steps else "执行下一个计划步骤"
            elif "response" in output:
                detail = "生成最终诊断响应"
            elif output.get("plan"):
                detail = "调整剩余计划: " + " -> ".join(str(step) for step in output.get("plan", [])[:4])
            else:
                detail = "评估是否继续执行或响应"
            steps.append(
                {
                    "id": f"plan_execute_{idx}",
                    "title": {
                        "planner": "Planner 制定排障计划",
                        "executor": "Executor 执行计划步骤",
                        "replanner": "Replanner 评估与重规划",
                        "final_response": "生成最终响应",
                    }.get(node, node),
                    "status": "success",
                    "detail": detail,
                }
            )
        return steps

    def _is_knowledge_about_aiops(self, text: str) -> bool:
        if any(marker in text for marker in ("怎么排查", "怎么处理", "告警", "线上", "故障", "过高", "不可用")):
            return False
        return any(marker.lower() in text for marker in self.KNOWLEDGE_SIGNALS)

    def _has_incident_signal(self, text: str) -> bool:
        incident_markers = (
            "怎么排查",
            "怎么处理",
            "怎么办",
            "告警",
            "线上",
            "故障",
            "过高",
            "飙高",
            "不可用",
            "宕机",
            "超时",
            "响应慢",
            "定位",
        )
        return any(marker in text for marker in incident_markers)


agent_orchestrator_service = AgentOrchestratorService()
