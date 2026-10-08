"""Build a Codex-style diagnostic workbench view for 运维 answers.

This module keeps product-facing observability data outside the core Agent
flow. The Agent still answers normally, while the frontend can render the same
turn as an incident workflow: goal, steps, evidence, tool activity, risky
actions and a report.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.services.intent_guidance_service import GuidanceDecision
from app.services.query_understanding_service import QueryUnderstandingResult
from app.services.retrieval_engine_service import RetrievalEngineResult


class DiagnosticWorkbenchService:
    """Create structured metadata for the 运维 diagnostic workspace."""

    def build(
        self,
        *,
        question: str,
        answer: str,
        session_id: str,
        understanding: QueryUnderstandingResult,
        retrieval_result: RetrievalEngineResult | None = None,
        guidance: GuidanceDecision | None = None,
        model: str = "",
        mode: str = "quick",
    ) -> dict[str, Any]:
        trace = retrieval_result.trace if retrieval_result else {}
        sources = retrieval_result.sources if retrieval_result else []
        incident = self._incident_profile(question, answer, understanding, trace)
        steps = self._execution_steps(understanding, trace, guidance, model)
        tool_calls = self._tool_calls(trace, guidance, model)
        risk_actions = self._risk_actions(question, answer, incident)
        report = self._report_markdown(
            question=question,
            answer=answer,
            session_id=session_id,
            incident=incident,
            steps=steps,
            tool_calls=tool_calls,
            sources=sources,
            risk_actions=risk_actions,
            understanding=understanding,
        )

        return {
            "version": "workbench-v1",
            "session_id": session_id,
            "mode": mode,
            "objective": self._objective(question, understanding),
            "incident_profile": incident,
            "execution_steps": steps,
            "tool_calls": tool_calls,
            "rag_trace": {
                "pipeline_nodes": trace.get("pipeline_nodes", []),
                "channel_summary": trace.get("channel_summary", {}),
                "channels": trace.get("channels", []),
                "sources": sources,
                "errors": trace.get("errors", []),
                "term_mapping_applied": understanding.applied_mappings,
                "rrf_enabled": trace.get("rrf_enabled"),
                "rerank_enabled": trace.get("rerank_enabled"),
            },
            "evidence_sources": sources[:8],
            "risk_actions": risk_actions,
            "next_questions": self._next_questions(incident),
            "report_markdown": report,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _objective(self, question: str, understanding: QueryUnderstandingResult) -> str:
        if understanding.intent == "aiops_diagnosis":
            return f"诊断并处理运维故障：{understanding.rewritten_question or question}"
        if understanding.intent == "metrics_alerts":
            return f"查询并解释监控告警：{understanding.rewritten_question or question}"
        if understanding.intent == "knowledge_query":
            return f"基于知识库回答：{understanding.rewritten_question or question}"
        return understanding.rewritten_question or question

    def _incident_profile(
        self,
        question: str,
        answer: str,
        understanding: QueryUnderstandingResult,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        text = " ".join(
            [
                question,
                answer[:1200],
                understanding.normalized_question,
                understanding.rewritten_question,
            ]
        ).lower()
        fault_type = self._fault_type(text, understanding.intent)
        service_type = self._service_type(text)
        missing_info = self._missing_info(text)
        confidence = round(float(understanding.confidence or 0), 2)

        if any(word in text for word in ["不可用", "宕机", "挂了", "oom", "雪崩", "p0", "p1"]):
            severity = "high"
        elif understanding.intent in ("aiops_diagnosis", "metrics_alerts"):
            severity = "medium"
        else:
            severity = "low"

        channel_summary = trace.get("channel_summary", {}) if trace else {}
        evidence_count = int(trace.get("final_item_count") or len(trace.get("sources", [])) or 0)

        return {
            "fault_type": fault_type,
            "service_type": service_type,
            "severity": severity,
            "confidence": confidence,
            "time_range": self._time_range(text),
            "region": self._region(text),
            "evidence_count": evidence_count,
            "retrieval_channels": list((channel_summary.get("raw_by_channel") or {}).keys()),
            "missing_info": missing_info,
            "status": "needs_more_context" if missing_info else "diagnosed",
        }

    def _fault_type(self, text: str, intent: str) -> str:
        patterns = [
            ("CPU 高负载", ["cpu", "load", "高负载", "飙高", "打满", "使用率过高"]),
            ("内存异常", ["内存", "memory", "oom", "泄漏"]),
            ("磁盘空间异常", ["磁盘", "disk", "空间不足", "inode"]),
            ("服务不可用", ["不可用", "服务挂", "接口挂", "宕机", "5xx"]),
            ("响应慢 / 超时", ["响应慢", "超时", "timeout", "慢查询", "latency"]),
            ("JVM / GC 异常", ["jvm", "gc", "full gc", "jstack", "jcmd"]),
            ("数据库异常", ["mysql", "redis", "数据库", "慢 sql", "连接池"]),
        ]
        for label, keywords in patterns:
            if any(keyword in text for keyword in keywords):
                return label
        return "运维故障诊断" if intent == "aiops_diagnosis" else "通用问题"

    def _service_type(self, text: str) -> str:
        patterns = [
            ("Java 服务", ["java", "jvm", "jstack", "jcmd", "gc"]),
            ("Python 服务", ["python", "py-spy", "gunicorn", "uvicorn"]),
            ("Redis", ["redis"]),
            ("MySQL", ["mysql", "sql", "innodb"]),
            ("Kubernetes 服务", ["k8s", "kubernetes", "pod", "deployment"]),
            ("Nginx / 网关", ["nginx", "gateway", "网关"]),
        ]
        for label, keywords in patterns:
            if any(keyword in text for keyword in keywords):
                return label
        return "未指定"

    def _time_range(self, text: str) -> str:
        if "最近" in text or "分钟" in text or "小时" in text:
            match = re.search(r"(最近\s*)?\d+\s*(分钟|小时|天)", text)
            if match:
                return match.group(0).replace(" ", "")
        if "昨天" in text:
            return "昨天"
        if "今天" in text:
            return "今天"
        return "最近 30 分钟（默认）"

    def _region(self, text: str) -> str:
        region_match = re.search(r"ap-[a-z]+-[a-z]+|cn-[a-z]+-\d+", text)
        if region_match:
            return region_match.group(0)
        if "广州" in text:
            return "ap-guangzhou"
        if "上海" in text:
            return "ap-shanghai"
        if "北京" in text:
            return "ap-beijing"
        return "未指定"

    def _missing_info(self, text: str) -> list[str]:
        missing: list[str] = []
        if not any(token in text for token in ["服务", "service", "java", "redis", "mysql", "pod", "应用"]):
            missing.append("服务名/应用名")
        if not any(token in text for token in ["最近", "分钟", "小时", "今天", "昨天", "时间", "告警"]):
            missing.append("故障时间范围")
        if not any(token in text for token in ["ap-", "cn-", "广州", "上海", "北京", "地域", "集群"]):
            missing.append("地域/集群")
        return missing

    def _execution_steps(
        self,
        understanding: QueryUnderstandingResult,
        trace: dict[str, Any],
        guidance: GuidanceDecision | None,
        model: str,
    ) -> list[dict[str, Any]]:
        pipeline_nodes = trace.get("pipeline_nodes", []) if trace else []
        node_names = {node.get("name") for node in pipeline_nodes if isinstance(node, dict)}

        steps = [
            {
                "id": "receive_goal",
                "title": "接收诊断目标",
                "status": "success",
                "detail": understanding.original_question,
            },
            {
                "id": "query_understanding",
                "title": "问题理解与意图识别",
                "status": "success",
                "detail": self._confidence_detail(understanding),
            },
            {
                "id": "query_rewrite_split",
                "title": "Query 重写与子问题拆分",
                "status": "success" if understanding.sub_questions else "skipped",
                "detail": (
                    f"{understanding.rewrite_strategy}; "
                    f"sub_questions={len(understanding.sub_questions or [])}; "
                    f"missing_slots={len(understanding.missing_slots or [])}"
                ),
            },
            {
                "id": "intent_tree_routing",
                "title": "意图树路由与检索范围选择",
                "status": "success" if understanding.intent_nodes else "skipped",
                "detail": self._intent_tree_detail(understanding),
            },
            {
                "id": "term_mapping",
                "title": "术语归一化",
                "status": "success" if understanding.applied_mappings else "skipped",
                "detail": self._mapping_detail(understanding.applied_mappings),
            },
        ]

        if guidance and guidance.should_guide:
            steps.append(
                {
                    "id": "intent_guidance",
                    "title": "低置信度澄清",
                    "status": "warning",
                    "detail": "已返回澄清建议，避免盲目执行诊断。",
                }
            )
            return steps

        for name, title in [
            ("parallel_retrieval", "多通道检索"),
            ("deduplication", "候选去重"),
            ("rrf_fusion", "RRF 融合排序"),
            ("rerank", "语义重排"),
        ]:
            node = next((item for item in pipeline_nodes if item.get("name") == name), None)
            steps.append(
                {
                    "id": name,
                    "title": title,
                    "status": node.get("status", "success") if node else ("success" if name in node_names else "skipped"),
                    "detail": node.get("summary", "") if node else "",
                }
            )

        steps.append(
            {
                "id": "model_generation",
                "title": "模型生成诊断结论",
                "status": "success",
                "detail": model or "default model",
            }
        )
        steps.append(
            {
                "id": "safety_review",
                "title": "风险操作检查",
                "status": "success",
                "detail": "重启、回滚、删除、限流等操作仅生成建议，需要人工确认。",
            }
        )
        return steps

    def _tool_calls(
        self,
        trace: dict[str, Any],
        guidance: GuidanceDecision | None,
        model: str,
    ) -> list[dict[str, Any]]:
        calls = [
            {
                "name": "query_understanding",
                "type": "local_service",
                "status": "success",
                "summary": "完成意图识别、问题重写和记忆拼接",
            }
        ]
        if guidance and guidance.should_guide:
            calls.append(
                {
                    "name": "intent_guidance",
                    "type": "local_service",
                    "status": "warning",
                    "summary": "置信度不足，优先引导用户补充信息",
                }
            )
            return calls

        for channel in trace.get("channels", []) if trace else []:
            name = str(channel.get("channel", "unknown"))
            calls.append(
                {
                    "name": self._channel_tool_name(name),
                    "type": "retrieval_channel",
                    "status": channel.get("status", "unknown"),
                    "latency_ms": channel.get("latency_ms"),
                    "item_count": channel.get("item_count", 0),
                    "summary": f"{name}: {channel.get('item_count', 0)} candidates",
                }
            )
        calls.append(
            {
                "name": model or "llm_generation",
                "type": "model",
                "status": "success",
                "summary": "生成最终诊断回答",
            }
        )
        return calls

    def _risk_actions(
        self,
        question: str,
        answer: str,
        incident: dict[str, Any],
    ) -> list[dict[str, Any]]:
        text = f"{question}\n{answer}".lower()
        candidates = [
            (
                "restart_service",
                "重启服务实例",
                "high",
                ["重启", "kill -15", "kill -9", "systemctl restart", "rollout restart"],
                "可能造成短暂不可用，执行前需要确认实例、流量摘除和回滚方案。",
            ),
            (
                "rollback_release",
                "回滚发布版本",
                "high",
                ["回滚", "rollback"],
                "可能影响当前版本功能，需要确认目标版本和数据兼容性。",
            ),
            (
                "scale_out",
                "扩容服务实例",
                "medium",
                ["扩容", "scale out", "增加实例"],
                "会增加资源成本，需确认容量和限流策略。",
            ),
            (
                "traffic_limit",
                "启用限流 / 熔断 / 降级",
                "medium",
                ["限流", "熔断", "降级"],
                "可能影响部分用户体验，需要确认保护阈值和降级范围。",
            ),
            (
                "delete_or_kill",
                "强制终止或删除资源",
                "critical",
                ["删除", "drop ", "truncate", "kill -9"],
                "属于高风险动作，只能在明确授权后执行。",
            ),
        ]
        actions: list[dict[str, Any]] = []
        for action_id, title, risk, keywords, reason in candidates:
            if any(keyword in text for keyword in keywords):
                actions.append(
                    {
                        "id": action_id,
                        "title": title,
                        "risk": risk,
                        "reason": reason,
                        "approval_state": "pending",
                        "suggested_mode": "generate_command_only",
                        "incident_type": incident.get("fault_type"),
                    }
                )
        return actions[:4]

    def _next_questions(self, incident: dict[str, Any]) -> list[str]:
        prompts = []
        for item in incident.get("missing_info", [])[:3]:
            prompts.append(f"请补充{item}，我可以继续收敛根因。")
        if not prompts:
            prompts.append("是否需要我把本次诊断整理成复盘报告？")
        return prompts

    def _mapping_detail(self, mappings: list[dict[str, Any]]) -> str:
        if not mappings:
            return "未命中术语映射，直接使用清洗后的问题。"
        return "；".join(
            f"{item.get('source_term')} -> {item.get('target_term')}"
            for item in mappings
        )

    def _intent_tree_detail(self, understanding: QueryUnderstandingResult) -> str:
        nodes = understanding.intent_nodes or []
        scope = understanding.retrieval_scope or {}
        if not nodes:
            return "未命中意图树节点，使用全局检索。"
        top = nodes[0]
        targets = scope.get("target_sources") or []
        target_text = ", ".join(str(item) for item in targets[:4]) if targets else "all"
        return (
            f"{top.get('path') or top.get('name')} "
            f"score={float(top.get('score') or 0):.2f}; "
            f"scope={scope.get('mode', 'global')}; "
            f"targets={target_text}"
        )

    def _confidence_detail(self, understanding: QueryUnderstandingResult) -> str:
        sources = understanding.confidence_sources or {}
        rule = sources.get("rule", {})
        tree = sources.get("intent_tree", {})
        llm = sources.get("llm", {})
        fusion = sources.get("fusion", {})
        parts = [
            f"{understanding.intent} / final={understanding.confidence:.2f}",
            f"rule={float(rule.get('confidence') or 0):.2f}",
        ]
        if tree.get("score") is not None:
            parts.append(f"tree={float(tree.get('score') or 0):.2f}")
        if llm.get("applied"):
            parts.append(f"llm={float(llm.get('confidence') or 0):.2f}")
        else:
            parts.append("llm=not_applied")
        if fusion.get("source"):
            parts.append(f"source={fusion.get('source')}")
        return "; ".join(parts)

    def _channel_tool_name(self, channel: str) -> str:
        return {
            "intent_vector": "intent_vector_search",
            "supplement_vector": "supplement_vector_search",
            "global_vector": "global_vector_search",
            "keyword": "keyword_search",
            "tool_context": "prometheus_alert_prefetch",
        }.get(channel, channel)

    def _report_markdown(
        self,
        *,
        question: str,
        answer: str,
        session_id: str,
        incident: dict[str, Any],
        steps: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        risk_actions: list[dict[str, Any]],
        understanding: QueryUnderstandingResult,
    ) -> str:
        lines = [
            "# 运维 诊断报告",
            "",
            f"- 会话 ID: `{session_id}`",
            f"- 原始问题: {question}",
            f"- 重写问题: {understanding.rewritten_question}",
            f"- 故障类型: {incident.get('fault_type')}",
            f"- 服务类型: {incident.get('service_type')}",
            f"- 严重程度: {incident.get('severity')}",
            f"- 置信度: {incident.get('confidence')}",
            "",
            "## 执行步骤",
        ]
        lines.extend(
            f"- {step.get('title')}: {step.get('status')} - {step.get('detail', '')}"
            for step in steps
        )
        lines.extend(["", "## 工具与检索"])
        lines.extend(
            f"- {call.get('name')}: {call.get('status')} ({call.get('summary', '')})"
            for call in tool_calls
        )
        lines.extend(["", "## 证据来源"])
        if sources:
            lines.extend(
                f"- [{source.get('id')}] {source.get('source')} | channels={','.join(source.get('channels') or [])}"
                for source in sources[:8]
            )
        else:
            lines.append("- 暂无检索来源")
        lines.extend(["", "## 风险动作"])
        if risk_actions:
            lines.extend(
                f"- {action.get('title')} / risk={action.get('risk')}: {action.get('reason')}"
                for action in risk_actions
            )
        else:
            lines.append("- 本轮回答未检测到需要人工审批的高风险操作。")
        lines.extend(["", "## 诊断回答", "", answer.strip()])
        return "\n".join(lines)


diagnostic_workbench_service = DiagnosticWorkbenchService()
