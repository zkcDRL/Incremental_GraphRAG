# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License


"""内置表存储实现类型。"""

from enum import StrEnum


class TableType(StrEnum):
    """表存储类型枚举。"""

    Parquet = "parquet"
    CSV = "csv"
    CosmosDB = "cosmosdb"
    Neo4j = "neo4j"
