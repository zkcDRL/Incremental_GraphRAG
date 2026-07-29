# Copyright (C) 2025 Microsoft
# Licensed under the MIT License

"""支持逐行流式访问的表抽象。"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import Any

from typing_extensions import Self

RowTransformer = Callable[[dict[str, Any]], Any]


class Table(ABC):
    """流式访问表的抽象基类。

    提供逐行迭代与写入能力，以便内存高效地处理大型数据集，
    并支持异步上下文管理器协议以自动释放资源。

    Examples
    --------
        将行读取为字典：
        >>> async with (
        ...     provider.open(
        ...         "documents"
        ...     ) as table
        ... ):
        ...     async for (
        ...         row
        ...     ) in table:
        ...         process(row)

        使用 Pydantic 模型作为转换器：
        >>> async with (
        ...     provider.open(
        ...         "entities",
        ...         Entity,
        ...     ) as table
        ... ):
        ...     async for entity in table:  # 产出 Entity 实例
        ...         print(
        ...             entity.name
        ...         )
    """

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Any]:
        """异步产出行；提供转换器时产出转换后的结果。

        Yields
        ------
            Any:
                每一行，类型为字典或转换后的对象（如 Pydantic 模型）。
        """
        ...

    @abstractmethod
    async def length(self) -> int:
        """异步返回表中的行数。

        Returns
        -------
            int:
                表中的行数。
        """

    @abstractmethod
    async def has(self, row_id: str) -> bool:
        """检查是否存在具有指定 ID 的行。

        Args
        ----
            row_id: 要查找的 ID 值。

        Returns
        -------
            bool:
                存在匹配 ID 的行时返回 True，否则返回 False。
        """

    @abstractmethod
    async def write(self, row: dict[str, Any]) -> None:
        """向表中写入单行数据。

        Args
        ----
            row: 表示单行数据的字典。
        """

    @abstractmethod
    async def close(self) -> None:
        """刷新缓冲写入并释放资源。

        退出异步上下文管理器时会自动调用此方法，也可显式调用。
        """

    async def __aenter__(self) -> Self:
        """进入异步上下文管理器。

        Returns
        -------
            Table:
                当前实例。
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """退出异步上下文管理器，并确保调用 close()。

        Args
        ----
            exc_type: 发生异常时的异常类型。
            exc_val: 发生异常时的异常值。
            exc_tb: 发生异常时的异常回溯。
        """
        await self.close()
