# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""表提供者的抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from graphrag_storage.tables.table import RowTransformer, Table


class TableProvider(ABC):
    """提供支持 DataFrame 与行字典的表存储接口。"""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """创建表提供者实例。

        Args
        ----
            **kwargs: 初始化参数，可包含底层 Storage 实例。
        """

    @abstractmethod
    async def read_dataframe(self, table_name: str) -> pd.DataFrame:
        """将整张表读取为 pandas DataFrame。

        Args
        ----
            table_name: 要读取的表名。

        Returns
        -------
            pd.DataFrame:
                表数据对应的 DataFrame。
        """

    @abstractmethod
    async def write_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """从 pandas DataFrame 写入整张表。

        Args
        ----
            table_name: 要写入的表名。
            df: 要写入为表的 DataFrame。
        """

    @abstractmethod
    async def has(self, table_name: str) -> bool:
        """检查提供者中是否存在指定表。

        Args
        ----
            table_name: 要检查的表名。

        Returns
        -------
            bool:
                表存在时返回 True，否则返回 False。
        """

    @abstractmethod
    def list(self) -> list[str]:
        """列出提供者中的全部表名。

        Returns
        -------
            list[str]:
                表名列表，不包含文件扩展名。
        """

    async def clear(self) -> None:
        pass

    @abstractmethod
    def open(
        self,
        table_name: str,
        transformer: RowTransformer | None = None,
        truncate: bool = True,
    ) -> Table:  # 返回 Table 实例
        """打开一张表以进行逐行流式操作。

        Args
        ----
            table_name: 要打开的表名。
            transformer: 可选的行转换函数。
            truncate: 为 True（默认）时，首次写入前清空已有表；
                为 False 时，向已有表追加数据，行为类似数据库。

        Returns
        -------
            Table:
                用于逐行流式操作的 Table 实例。
        """

    def child(self, name: str | None) -> "TableProvider":
        """创建带命名空间的子提供者。

        更新管线使用它隔离 delta 与 previous 表集合。默认实现返回自身，
        即不提供命名空间隔离；子类应按需覆写以提供正确的隔离能力。

        Args
        ----
            name: 子提供者的命名空间名称。

        Returns
        -------
            TableProvider:
                子表提供者实例。
        """
        return self
