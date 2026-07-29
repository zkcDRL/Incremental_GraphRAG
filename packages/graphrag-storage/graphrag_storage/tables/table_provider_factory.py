# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License


"""表提供者工厂实现。"""

from collections.abc import Callable

from graphrag_common.factory import Factory, ServiceScope

from graphrag_storage.storage import Storage
from graphrag_storage.tables.table_provider import TableProvider
from graphrag_storage.tables.table_provider_config import TableProviderConfig
from graphrag_storage.tables.table_type import TableType


class TableProviderFactory(Factory[TableProvider]):
    """用于创建表存储实现的工厂类。"""


table_provider_factory = TableProviderFactory()


def register_table_provider(
    table_type: str,
    table_initializer: Callable[..., TableProvider],
    scope: ServiceScope = "transient",
) -> None:
    """注册自定义表存储实现。

    Args
    ----
        - table_type: 要注册的表类型标识。
        - table_initializer: 要注册的表提供者初始化器。
    """
    table_provider_factory.register(table_type, table_initializer, scope)


def create_table_provider(
    config: TableProviderConfig, storage: Storage | None = None
) -> TableProvider:
    """根据给定配置创建表提供者实现。

    Args
    ----
        - config: 要使用的表提供者配置。
        - storage: 供 Parquet、CSV 等基于文件的表提供者使用的存储实现。

    Returns
    -------
        TableProvider
            创建出的表提供者实现。
    """
    config_model = config.model_dump()
    table_type = config.type

    if table_type not in table_provider_factory:
        match table_type:
            case TableType.Parquet:
                from graphrag_storage.tables.parquet_table_provider import (
                    ParquetTableProvider,
                )

                register_table_provider(TableType.Parquet, ParquetTableProvider)
            case TableType.CSV:
                from graphrag_storage.tables.csv_table_provider import (
                    CSVTableProvider,
                )

                register_table_provider(TableType.CSV, CSVTableProvider)
            case TableType.CosmosDB:
                from graphrag_storage.tables.cosmos_table_provider import (
                    CosmosTableProvider,
                )

                register_table_provider(TableType.CosmosDB, CosmosTableProvider)
            case _:
                msg = f"TableProviderConfig.type '{table_type}' is not registered in the TableProviderFactory. Registered types: {', '.join(table_provider_factory.keys())}."
                raise ValueError(msg)

    if storage:
        config_model["storage"] = storage

    # 对 CosmosDB 表提供者，从关联的 Storage 实例中提取连接信息，
    # 以便用户只需在 output_storage 中配置一次凭据。表专属字段
    # （container_name、batch_size、legacy_container）仍由 TableProviderConfig 提供。
    if table_type == TableType.CosmosDB and storage is not None:
        from graphrag_storage.azure_cosmos_storage import AzureCosmosStorage

        if isinstance(storage, AzureCosmosStorage):
            config_model.setdefault(
                "connection_string",
                storage._connection_string,  # noqa: SLF001
            )
            config_model.setdefault(
                "account_url",
                storage._cosmosdb_account_url,  # noqa: SLF001
            )
            config_model.setdefault(
                "database_name",
                storage._database_name,  # noqa: SLF001
            )

    return table_provider_factory.create(table_type, config_model)
