# 架构速览

## 一次诊断如何完成

1. [`app/api/chat.py`](../app/api/chat.py) 接收问题，调用统一编排服务。
2. [`app/services/query_understanding_service.py`](../app/services/query_understanding_service.py) 理解问题、改写查询并判断意图；[`app/services/agent_orchestrator_service.py`](../app/services/agent_orchestrator_service.py) 选择诊断或知识问答路线。
3. 诊断路线进入 [`app/services/aiops_service.py`](../app/services/aiops_service.py) 的 LangGraph 状态图，依次规划、执行、重规划；执行节点通过 [`app/agent/mcp_client.py`](../app/agent/mcp_client.py) 调用工具。
4. 知识路线使用向量检索与本地关键词检索，结合重排结果生成回答。
5. [`app/services/diagnostic_workbench_service.py`](../app/services/diagnostic_workbench_service.py) 将阶段、证据和结论整理给前端。

## 数据和边界

- Milvus 保存向量索引；本地 `data/auth/`、`data/memory/` 保存运行状态，均不随仓库发布。
- `mcp_servers/` 中的日志和监控工具是模拟服务。接入真实系统需要替换工具实现与凭据管理。
- Web 界面、API 和示例 MCP 服务默认用于本机演示。不要将此示例配置直接作为公网运维控制台。
