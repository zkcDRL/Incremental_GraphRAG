# GraphRAG-Bench 200 题 evidence 标注

本目录是基于本地 `GraphRAG-Bench/textbooks/*_structured.json` 制作的派生标注，不是数据集官方发布的 supporting-passage ground truth。

## 主文件

- `graphrag_bench_evidence_200_reviewed.jsonl`：200 道题的完整标注；每行一道题。
- `graphrag_bench_evidence_200_review_summary.csv`：便于人工查看的摘要表。
- `graphrag_bench_evidence_200_supported.qrels.tsv`：只包含 `supported` 题，可直接用于主检索实验。
- `graphrag_bench_evidence_200_review_manifest.json`：抽样、状态统计和使用规则。
- `manual_overrides.json`：人工复核决策，便于审计和修改。

## 状态含义

- `supported`：所选教材 chunk 直接支持标准答案。
- `partial`：只支持部分答案，或仍需要非平凡推导。
- `not_found`：在本地教材语料中没有找到充分支持段落，`gold_evidence` 为空。
- `data_issue`：原题存在序列化或答案冲突，应从主评测排除。

主检索评测只使用 `annotation.eligible_for_primary_retrieval_eval=true`（即 `supported`）的题。`partial` 可以单独报告；不要把 `not_found` 当成检索失败样本。

## 关键字段

- `qid`：稳定题号，如 `FB-0001`。
- `gold_evidence[].chunk_id`：证据块 ID。
- `gold_evidence[].chunk_text`：完整证据块文本。
- `gold_evidence[].evidence_snippet`：用于快速复核的关键片段。
- `gold_evidence[]` 中的教材、章节、结构化记录序号和字符/词偏移用于追溯来源。
- `annotation.support_status`：上述四种支持状态。
- `top5_candidates`：保留的自动检索候选，方便二次复核。

## 当前统计

- 总计：200（FB、MC、MS、OE、TF 各 40）
- `supported`：101
- `partial`：36
- `not_found`：61
- `data_issue`：2

`qrels.tsv` 格式为 `qid<TAB>0<TAB>chunk_id<TAB>1`。101 道 supported 题对应 121 条相关 chunk 标注。
