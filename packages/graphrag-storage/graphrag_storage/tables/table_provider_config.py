# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""表提供者的配置模型。"""

from pydantic import BaseModel, ConfigDict, Field

from graphrag_storage.tables.table_type import TableType


class TableProviderConfig(BaseModel):
    """表提供者的默认配置节。"""

    model_config = ConfigDict(extra="allow")
    """允许额外字段，以支持自定义表提供者实现。"""

    type: str = Field(
        description="要使用的表类型。内置类型包括 'parquet'、'csv'、'cosmosdb' 和 'neo4j'。",
        default=TableType.Parquet,
    )

    container_name: str | None = Field(
        description="用于表存储的 Cosmos DB 容器名称。",
        default=None,
    )

    legacy_container: str | None = Field(
        description="可选的旧版 Cosmos DB 容器名称，用于读取时的迁移回退。",
        default=None,
    )

    batch_size: int = Field(
        description="Cosmos DB 事务批量写入的文档数，最大为 100。",
        default=50,
    )
