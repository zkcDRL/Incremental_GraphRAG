# Incremental GraphRAG 项目阶段总结

## 1. 项目概况

本项目基于 Microsoft GraphRAG，目标是在新增少量文档时减少全图社区重算和社区报告重复生成的成本，并支持 Neo4j 存储与版本隔离。

当前主要能力：

- GraphRAG 核心表可存入 Neo4j，通过 `namespace` 隔离不同项目；
- 增量文档继续执行实体、关系和 TextUnit 抽取；
- 社区更新支持三级策略：
  - 受影响边 `<1%` 且受影响社区 `<5%`：受影响社区加一跳社区的局部 Leiden；
  - 受影响边 `1%～10%`：全图 seeded Leiden；
  - 受影响边 `>10%`：全图冷启动 Leiden；
- 支持累计漂移检测：连续局部更新次数、累计边变化率、累计成员变化率或模块度下降超限后，升级为全图 seeded Leiden；仍不合格时执行冷启动；
- 局部更新后重新生成变化社区及其祖先报告，复用未变化报告；
- 可通过 `adaptive_community_update` 开关，在自适应更新和每次全图冷启动之间切换；
- 对只支持 SSE 的 OpenAI 兼容对话接口，已增加强制流式请求并聚合为普通响应的适配。

## 2. 运行与测试环境

- 运行目录：`rag_workspace/`
- 表存储：Neo4j 5 Community
- Neo4j Browser：`http://localhost:17474`
- Bolt：`bolt://localhost:17687`
- 当前 GraphRAG namespace：`incremental-graphrag`
- 向量存储：LanceDB
- 实验数据：`GraphRAG-Bench-incremental-101/`
- 查询模式：仅测试 Local Search

模型连通性测试已通过：

- 对话模型可通过 SSE 适配返回完整结果；
- Embedding 模型可返回 1024 维向量；
- 同步、异步和显式流式调用均已验证。

代码质量与单元测试已多次执行，相关社区更新、漂移检测、SSE 适配和 benchmark 辅助测试均通过 Ruff、Pyright 与 pytest。

## 3. Benchmark 设计

`scripts/run_incremental_benchmark.py` 按三阶段执行：

1. 851 个基础 chunk 完整建图，并查询 40 道题；
2. 增加 40 个 chunk，执行第一次增量更新，并查询 41 道题；
3. 增加 38 个 chunk，执行第二次增量更新，并查询 20 道题。

采集指标包括：

- 完整建图和每次增量更新时间；
- workflow 级耗时与峰值内存；
- 查询平均延迟与 P95；
- LLM Prompt、Completion 和 Total Token；
- FB、MC、MS、TF 的规则准确率；
- Gold evidence 召回率；
- 每次更新的策略、边变化比例、成员变化率和模块度漂移；
- OE 开放题输出到文件，后续人工或助手评判，不调用当前 API 作为 Judge。

## 4. 当前实验进展

首次 Base 构建已完成以下步骤：

- 851 个 Document，内部切分为 867 个 TextUnit；
- 实体关系抽取；
- 图最终化；
- 社区聚类；
- 社区报告；
- 已进入 `generate_text_embeddings` 阶段。

已生成的 Neo4j 数据规模约为：

- Documents：851
- TextUnits：867
- Entities：1,639
- Relationships：2,526
- Communities：259
- Community Reports：247

主要耗时中，实体关系抽取约 4 小时 14 分钟，社区聚类约 10 秒，说明当前主要瓶颈是远程 LLM，而不是 Leiden 或 Neo4j。

## 5. 当前关键问题

### 5.1 社区报告结构化输出不稳定

SSE 对话通道能返回普通文本，但部分社区报告没有返回合法 JSON，出现 `JSONDecodeError`，导致少数社区报告缺失。后续需要加强结构化响应聚合、JSON 清洗和失败重试，否则会影响 Local Search 质量。

### 5.2 长时间运行后 Embedding 阶段悬挂

Embedding API 独立压力测试结果正常：

- batch size 16 成功；
- 20,000 字符输入成功；
- 25 并发、每请求 16 条，共 400 条文本全部成功，耗时不足 1 秒。

因此此前 Embedding 卡住并非固定并发上限，更可能是长时间对话模型调用后 HTTP 连接池出现大量 `CLOSE-WAIT`，或某个无超时请求永久等待。建议增加请求超时、单批重试、异常隔离、客户端重建和 Embedding 阶段断点恢复。

### 5.3 Benchmark 尚不支持可靠断点续跑

当前阶段中断后不能直接从某个 workflow 继续，SSE 内部流式请求也不会进入现有非流式缓存。后续应增加：

- `--start-stage` / `--stop-after-stage`；
- 已完成阶段自动跳过；
- 从 Neo4j 核心表继续执行 Embedding；
- SSE 聚合结果缓存；
- 失败社区报告单独重试。

## 6. 下一步建议

1. 修复社区报告的 SSE 结构化 JSON 兼容和重试；
2. 给 Embedding 请求增加超时、重试及独立并发参数；
3. 增加 benchmark 阶段和 workflow 断点恢复；
4. 完成三阶段 Local Search 实验，再比较自适应更新与全图冷启动的耗时、内存、Token、质量和漂移指标。
