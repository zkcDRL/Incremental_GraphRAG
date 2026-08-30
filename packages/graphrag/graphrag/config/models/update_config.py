# Copyright (c) 2026 Microsoft Corporation.
# Licensed under the MIT License

"""增量索引快照保留配置."""

from pydantic import BaseModel, Field, model_validator


class UpdateConfig(BaseModel):
    """控制增量索引生成的历史快照数量."""

    snapshot_retention_count: int = Field(
        default=7,
        ge=1,
        description="保留最近的增量索引快照轮数。",
    )
    local_edge_ratio: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="允许局部社区更新的最大受影响边比例(不含边界值)。",
    )
    local_community_ratio: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="允许局部社区更新的最大直接受影响社区比例(不含边界值)。",
    )
    cold_start_edge_ratio: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="超过该受影响边比例时执行冷启动全图 Leiden。",
    )
    community_hops: int = Field(
        default=1,
        ge=0,
        le=3,
        description="局部更新时从直接受影响社区扩展的社区图跳数。",
    )
    max_consecutive_local_updates: int = Field(
        default=20,
        ge=1,
        description="触发全图 seeded Leiden 的累计局部更新次数。",
    )
    max_cumulative_edge_change_ratio: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="触发全图 seeded Leiden 的累计边变更比例。",
    )
    max_cumulative_membership_churn: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description="触发全图 seeded Leiden 的累计成员变更率。",
    )
    max_modularity_drop: float = Field(
        default=0.01,
        ge=0,
        le=1,
        description="相对基线模块度允许下降的最大比例。",
    )

    @model_validator(mode="after")
    def _validate_community_thresholds(self) -> "UpdateConfig":
        if self.cold_start_edge_ratio < self.local_edge_ratio:
            msg = "cold_start_edge_ratio must be greater than or equal to local_edge_ratio"
            raise ValueError(msg)
        return self