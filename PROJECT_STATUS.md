# Incremental GraphRAG 项目阶段总结

更新日期：2026-09-11。本文区分代码已实现能力、已验证的运行结果与待执行工作。

1800 条 Base 已从候选版本 `20260908-103737` 恢复完成并发布。最终活动版本包含 1,800 个文档、1,829 个文本单元、12,009 个实体、24,131 条关系、2,691 个社区及 2,691 份有效报告；三类版本化向量分别为 12,009、2,691 和 1,829 条，最终结果为 `status=complete`、`errors=[]`。

`scripts/recover_community_benchmark_base.py` 已改为通用恢复入口：通过必填的 `--base` 接受任意规模 Base，支持 `building` 和 `failed` 候选；完整报告直接复用，只有缺失或无效时才基于响应检查点重建。恢复会重建 Neo4j 图投影和版本化向量，校验报告覆盖、`REPORTS_ON` 连接及向量 ID 后才发布；`--check-only` 只读验证候选，`--force-report-regeneration` 可显式要求重新生成全部报告。

最新进展：Base 已扩充为 `00_base_1800.jsonl`。已验证 1,800 个唯一 chunk，前 931 条 ID 和正文一致；恢复过程复用了完整图投影和实体向量，以低内存参数重建社区与文本单元向量并完成发布。恢复日志为 `rag_workspace/benchmark_results/incremental-164/recover_base_1800.log`。

社区 benchmark 已切换到 1800 条 Base 文件，Stage 1～4 仍使用 `_30`。报告模型已接入 SSE 校验和最多 8 次尝试：空响应、非法 JSON、字段不完整或报告内容为空均重试，成功响应保存检查点；超过次数仍失败会阻止发布。此前 Stage 1 的社区 5 由于模型响应不能解析为 JSON 而失败，该次运行也实际触发了 `global_cold`（变更边占比 18.02%）。

## 1. 项目概况

本项目基于 Microsoft GraphRAG，主要修改增量更新时的社区更新策略，并将图数据存入 Neo4j，支持关联查询与版本隔离。

主要代码位于 `packages/`：

- `graphrag`：CLI/API、索引工作流、社区更新和查询；
- `graphrag-storage`：存储抽象与 Neo4j 表提供者；
- `graphrag-vectors`：向量存储与版本化索引；
- `graphrag-llm`：对话、Embedding、SSE 适配与模型调用统计；
- 其余包提供输入、分块、缓存与公共能力。

增量流程先抽取新增文档的实体关系，再合并旧图，更新 TextUnit、社区、社区报告和向量。

### 社区更新

以下为默认阈值，实际运行可通过 `update` 配置调整：

- 受影响边比例 `<1%` 且直接受影响根社区比例 `<5%`：局部 seeded Leiden，默认将范围扩展一跳社区；
- 不满足局部条件，但受影响边比例 `<=10%`：全图 seeded Leiden；即使边比例低于 1%，社区影响范围过大也会进入此分支；
- 受影响边比例 `>10%`，或关闭 `adaptive_community_update`：全图冷启动 Leiden。

边变更包括新增、删除，以及权重、描述、证据引用的变化。社区 ID 通过同层成员的 Jaccard 重叠尽量复用。

累计漂移检测覆盖连续局部更新次数、累计边变化率、累计成员迁移率和模块度下降。达到阈值后升级到全图 seeded Leiden；seeded 结果的模块度仍不满足要求时再执行全图冷启动。

社区报告通常只重建变化社区及其祖先，并补齐新社区报告；稳定报告复用。全图冷启动开关本身不等于强制重建所有报告，基准的 `full` 方案额外设置 `force_full_community_rebuild=True`。

### Neo4j 与查询

Neo4j 将表行保存为 `GraphRAGRecord` 节点并保留完整 JSON，同时建立实体关系、文本证据、社区成员和报告等图连接，兼容原有 DataFrame 读写接口。

逻辑 `namespace` 隔离实验，增量构建使用独立版本命名空间。新版本先标记为 `building`，成功完成后事务切换为 active；失败版本不替换原 active 版本。查询绑定 active 图版本及对应向量版本。

Neo4j 增量路径现已改为数据库内部复制：每次更新先在 Neo4j 内完整克隆 active 表记录到候选版本和 previous 快照，不再把六张旧表读入 Python 后写回。previous、delta 与 building 候选在工作流期间均不构建图投影；所有表、社区报告和向量完成后，只为最终候选重建一次投影，投影成功后才发布。

实体合并只从 delta 提取实体标题键，并查询 previous 中同标题的候选旧实体；匹配项合并描述、频次和 TextUnit 引用，未匹配项直接插入。关系只按 delta 的 `(source, target)` 键读取旧记录。degree 直接从变化边端点的关联关系计算，只更新这些端点；`combined_degree` 只回写这些端点关联的关系，关联关系另一端复用实体表中已有的 degree。GraphRAG 最终实体表本身按标题去重，因此当前持久化实体键仍为 `title`；抽取阶段虽按 `(title, type)` 聚合，finalize 阶段会折叠为标题唯一实体。

局部社区策略在加载完整图前完成判定：变化边来自 delta 与按键查询到的旧关系，总关系数由 Neo4j `count()` 获取；局部 Leiden 只查询直接受影响的根社区、配置跳数内的邻接社区，以及该区域的实体和诱导关系。只有选择或漂移升级为全图 seeded/cold 时，才分页流式读取完整实体、关系和社区表。局部报告只重新生成变化社区及其现存祖先，其他报告保留在完整候选副本中。

当前基准使用 Local Search：向量检索选取种子实体，再从 Neo4j 加载关联实体、关系、文本证据和社区报告，组成回答上下文。

## 2. 当前环境与实验入口

- 运行配置：`rag_workspace/settings.yaml` 及该目录的模型配置；
- 表存储：Neo4j，Browser 为 `http://localhost:17474`，Bolt 为 `bolt://localhost:17687`；
- 向量存储：LanceDB；
- 当前数据集：`GraphRAG-Bench-community-incremental-v2/`；
- 当前基准脚本：`scripts/run_community_incremental_benchmark.py`；
- 原始 851 文档基础图：`graphrag-bench-incremental-101`；
- 新 Base 目标逻辑命名空间：`graphrag-bench-incremental-164`；
- 新 Base 工作目录：`rag_workspace/benchmark_results/incremental-164/`；
- 对比实验默认输出：`rag_workspace/benchmark_results/community-incremental-v2/`；
- 对比实验命名空间：`graphrag-community-incremental-v2-incremental` 与 `graphrag-community-incremental-v2-full`。

旧数据集 `GraphRAG-Bench-incremental-101/` 和脚本 `scripts/run_incremental_benchmark.py` 保留用于历史实验及公共辅助函数，不再作为当前社区更新对比实验的主入口。

## 3. 当前数据集与 Benchmark 设计

此前测试中 Base 较小，Stage 1 触发了全图冷启动。为调整实验规模，Base 已从 851 个 chunk 先扩充到 931 个，再扩充到 1,800 个；Stage 1～4 的增量语料与题量已减少。

已核对 `00_base_1800.jsonl` 的前 931 条与已发布 Base 的 chunk ID、正文一致，新增 869 条为追加内容。

| 阶段 | Corpus 文件 | 本阶段 chunk | 累计 chunk | 累计题量 |
|---|---|---:|---:|---:|
| Base | `00_base_1800.jsonl` | 1,800 | 1,800 | 40 |
| Stage 1 | `01_micro_local_30.jsonl` | 30 | 1,830 | 74 |
| Stage 2 | `02_near_region_30.jsonl` | 30 | 1,860 | 104 |
| Stage 3 | `03_far_existing_books_30.jsonl` | 30 | 1,890 | 134 |
| Stage 4 | `04_new_textbook_regions_30.jsonl` | 30 | 1,920 | 164 |

问题文件为 `questions/cumulative/stage_0.jsonl` 至 `stage_4.jsonl`。检索和回答只使用问题及选项，标准答案与 gold 字段用于回答生成后的离线评分。

对比两种方案：

- `incremental`：自适应社区更新与选择性报告重建；
- `full`：合并增量图后，全图冷启动 Leiden 并全量重建社区报告。

两种方案从共同 Base 复制到独立命名空间，逐阶段追加相同语料并运行累计问题集。主要目标指标为更新时间、社区报告 Token 消耗、问答准确率与同层社区 NMI。详细要求见 `GraphRAG-Bench-community-incremental-v2/evaluation/EVALUATION_METRICS.md`。

## 4. 当前实验进展

### 原基础图

已将 `namespace=graphrag-bench-incremental-101` 的六张核心表复制到 `graphrag-bench-incremental-164`，逐表校验一致。后续构建和恢复完成后再次核对，原命名空间未修改。

原始 851 文档 Base 的规模如下：

| 表 | 行数 |
|---|---:|
| Documents | 851 |
| TextUnits | 867 |
| Entities | 1,639 |
| Relationships | 2,526 |
| Communities | 259 |
| Community Reports | 247 |

原基础图的 Embedding 已独立恢复完成。`rag_workspace/benchmark_results/incremental-101/01_base/embedding_recovery.json` 记录向量维度 1024、耗时约 106.6 秒、`errors=[]`。副本复制仅包含 Neo4j 表及图连接；931 文档新 Base 已生成独立的版本化向量。

### 已发布 Base 版本

新增 80 条已完成实体关系抽取、合并和全图冷启动 Leiden。首次全量报告生成缺少 3 个社区（62、515、531），候选版本未发布；随后通过 `scripts/recover_community_benchmark_base.py` 复用成功响应并重试缺失报告，完成向量生成和校验后发布。

931 文档版本 `20260906-155524` 已被 1800 文档版本取代。当前 active 版本为 `20260908-103737`，物理命名空间为 `graphrag-bench-incremental-164/__graphrag_versions__/20260908-103737`，策略记录为 `global_cold`。最终规模如下：

| 表/向量 | 行数 |
|---|---:|
| Documents | 1,800 |
| TextUnits | 1,829 |
| Entities | 12,009 |
| Relationships | 24,131 |
| Communities | 2,691 |
| Community Reports | 2,691 |
| Entity description vectors | 12,009 |
| Community full-content vectors | 2,691 |
| TextUnit text vectors | 1,829 |

社区与报告一一覆盖，三类向量 ID 分别与对应 Neo4j 表完全一致。通用恢复脚本最终结果记录为 `status=complete`、`errors=[]`。

用户已明确确认以下 API 目的地与处理范围，构建命令已获准执行：

- 对话：`https://www.micuapi.ai/v1`，模型 `deepseek-v4-flash`；
- Embedding：本机 `http://127.0.0.1:11434`，模型 `bge-m3`。

此前完成的 Neo4j 复制未调用模型；新 Base 构建使用上述 API 处理了新增语料和全量社区报告。

运行记录位于 `rag_workspace/benchmark_results/incremental-164/`：

- `copy_base.log`：六张表复制与校验成功记录；
- `source_snapshot.json`：原表内容哈希；
- `base_extension_80.jsonl`：Base 第 852～931 条；
- `prepare_base.py`：此前完成复制时使用的准备脚本；
- `prepare_base.log`、`prepare_base_retry.log`：完整构建尝试日志；
- `recover_base.log`、`base_recovery.json`：报告恢复、向量生成、发布及最终校验记录。

完整执行入口为 `scripts/prepare_community_benchmark_base.py`，从项目根目录运行 `.venv/bin/python -m scripts.prepare_community_benchmark_base`。可使用 `--check-only` 检查输入与副本；对已完成的新 Base 重复运行时只校验并复用，不再次追加。已有候选版本应使用 `.venv/bin/python -m scripts.recover_community_benchmark_base --version <候选版本> --base <Base JSONL>` 恢复，可先附加 `--check-only` 做只读预检。

### 基准脚本适配

`scripts/run_community_incremental_benchmark.py` 已修改为 Base 使用 `00_base_1800.jsonl`，四轮增量全部使用 `_30.jsonl`；默认基础命名空间和基础运行目录均指向 `incremental-164`。

增量测试默认使用索引并发 4、普通 Embedding 每批 4 条、社区 Embedding 每批 1 条、Embedding 并发 1、单批 1024 tokens，并将约 128 条向量合并后写入 LanceDB；这些参数均可通过命令行覆盖。

Base 副本支持 `--resume`：完整的 `base_copy_checkpoint.json` 会直接跳过复制；未完成的副本会在 Neo4j 内重新执行完整 namespace 克隆，再为两个可查询的 benchmark 基础命名空间各构建一次投影。复制过程不再逐表经过 Python DataFrame。成功的阶段更新记录在 `stage_N/update_metrics.json`，问题评测从已保存且顺序一致的题目结果后继续，NMI 也按阶段复用检查点。

已完成脚本语法、数据路径、chunk 数量、累计题量和 diff 空白检查。Neo4j 原生增量路径、社区漂移、Base 恢复和 benchmark 辅助逻辑通过 55 项定向单元测试；Neo4j 存储通过 3 项真实数据库集成测试，覆盖服务器内部复制、按键读取、局部 upsert、版本发布和复制后零投影边。尚未执行完整基准、全套 Pyright 或全仓 pytest，不能据此宣称当前工作区已通过全部检查。

## 5. 已知问题与评估口径

- **旧 Base 报告数少于社区数**：259 个社区对应 247 个报告；931 文档新 Base 已经真实运行验证，为 527 个社区和 527 份报告。
- **SSE 响应仍可能不稳定**：本次确实遇到错误结束、超时和非法 JSON。Base 构建脚本增加了请求总时限、结束标记检查、结构化响应提醒、成功响应检查点和恢复入口；最终完成了全部报告，但外部接口的不稳定性仍需通过这些保护处理。
- **历史 Embedding 悬挂已有恢复结果**：原索引已成功独立恢复，但这不能证明长时间运行问题的根因已解决，不应将连接池问题写成已确认结论。
- **断点恢复粒度**：社区对比脚本可跳过已完整复制的 Base、已成功发布的阶段更新、已完成题目和 NMI；Base 内部复制若中断，会重新执行一次 Neo4j namespace 克隆。如果阶段更新尚未成功结束并写入 `update_metrics.json`，该阶段会从头重跑，不能从阶段内部某个 workflow 原地续接。
- **局部漂移的模块度**：为满足局部更新不读取完整图，局部路径不会每轮重新计算全图模块度；它累计精确的边变化率和按全图根成员数缩放的局部成员迁移率，并保留最近一次全图构建的模块度。连续次数、边变化或成员迁移达到阈值时仍会升级为全图 seeded，届时重新计算模块度。
- **Base 评估已接入主循环**：社区对比脚本会在任何增量更新之前，直接使用已发布的 1800 文档 Base 和完整版本化向量运行一次 Stage 0 的 40 道题，结果单独写入 `base/`。
- **实验公平性与计时仍需对齐**：当前两种方案分别抽取新增语料，尚未共享同一份抽取结果；`update_wall_time_s` 包含整条更新流水线，包括抽取与 Embedding，未限定为抽取完成后的图与社区更新区间；也未自动执行三次重复取中位数。
- **报告 Token 统计范围偏宽**：当前 `collect_report_metrics` 汇总该阶段 index 目录下的 completion 指标，尚未只筛选社区报告调用。
- **开放题尚待离线判分**：OE 的 `correct` 暂为 `None`，但阶段 `accuracy` 当前分母包含全部题目。未合并开放题判分前，该值不能作为最终全题准确率。

## 6. 下一步

1. 在任何 Stage 1 写入前完成 Stage 0 的 40 道基线题，并保存共同 Base 的查询结果。
2. 对齐抽取共享、计时、Token 和开放题判分口径。
3. 运行四轮 `_30.jsonl` 的社区更新对比实验，再报告加速比、Token 节省率、准确率变化和 NMI。

## 7. 目前存在的问题
   如果数据量多，community_full_content__graphrag阶段占用内存很大，目前是减小并发量暂时解决
