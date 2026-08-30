# GraphRAG 运行目录

该目录已经初始化，可以独立存放输入文档、配置、提示词和索引输出。

## 需要填写的 LLM 配置

1. 在 `.env` 中把 `GRAPHRAG_API_KEY=<API_KEY>` 替换为真实密钥。
2. 在 `settings.yaml` 中检查并修改：
   - `completion_models.default_completion_model.model_provider`
   - `completion_models.default_completion_model.model`
   - completion 模型所需的 endpoint/base URL、API 版本等字段
   - `embedding_models.default_embedding_model` 下对应字段
3. 如果 completion 与 embedding 使用不同密钥，请在 `.env` 中定义两个环境变量，并在 `settings.yaml` 中分别引用。

## 目录

- `input/`：放入待索引的文本文件。
- `prompts/`：GraphRAG 默认提示词。
- `output/`：当前已发布索引。
- `update_output/`：增量更新快照和中间表。
- `cache/`：LLM 缓存。
- `logs/`：运行日志。

## Neo4j

当前表存储配置使用：

- Browser：`http://localhost:17474`
- Bolt：`bolt://localhost:17687`
- 用户名：`neo4j`
- GraphRAG namespace：`incremental-graphrag`

在仓库根目录启动数据库：

```bash
docker-compose -f docker-compose.neo4j.yml up -d
```

第一次运行需要从 Docker Hub 下载 `neo4j:5-community`。如果下载超时，请配置 Docker 镜像加速或网络代理后重试。

## 命令

在仓库根目录 `/home/zkc/Incremental_GraphRAG` 执行：

```bash
# 测试 completion 与 embedding 配置
uv run --frozen --package graphrag python \
  /home/zkc/Incremental_GraphRAG/rag_workspace/test_models.py

# 首次全量索引
uv run --frozen --package graphrag graphrag index \
  --root /home/zkc/Incremental_GraphRAG/rag_workspace \
  --method standard

# 后续增量索引（包含局部社区更新与累计漂移检测）
uv run --frozen --package graphrag graphrag index \
  --root /home/zkc/Incremental_GraphRAG/rag_workspace \
  --method standard-update
```

更新输入文档后再运行 `standard-update`。当前增量文档检测按文档标题识别新增文档。
