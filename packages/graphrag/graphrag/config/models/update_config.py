# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""增量索引快照保留配置."""

from pydantic import BaseModel, Field


class UpdateConfig(BaseModel):
    """控制增量索引生成的历史快照数量."""

    snapshot_retention_count: int = Field(
        default=7,
        ge=1,
        description="保留最近的增量索引快照轮数。",
    )