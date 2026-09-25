"""FastAPI 应用入口：野外观测记录匿名化发布服务。

服务为纯内存、无状态处理：不写日志正文、不落盘任何原始记录。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .anonymizer import (
    apply_levels,
    prepare_hierarchies,
    prepare_sensitive,
    solve_levels,
    unprocessable,
)
from .intersection import audit_intersections

API_PREFIX = "/api/v1"

app = FastAPI(
    title="野外观测记录匿名化 API",
    description=(
        "对自然保护站公开野外观测记录进行统一层级泛化，"
        "使发布后的每种字段值组合至少出现 k 次；"
        "可选地要求每组包含至少 l 种不同的敏感属性值。"
        "另提供两次发布的关联风险审计：对同一批记录两次公开的"
        "分组按 id 求交集，标记成员数不足 k 的交集。"
    ),
    version="1.1.0",
)


class AnonymizeRequest(BaseModel):
    """匿名化请求体。"""

    # 拒绝 bool 冒充整数；k 必须为正整数。
    k: StrictInt = Field(..., description="每种字段值组合发布后至少出现的次数")
    records: list[dict[str, Any]] = Field(..., description="待匿名化的原始记录")
    # 每个字段：原值 -> 从原值（含）到统一通配符（含）的层级路径，至多 4 级。
    fields: dict[str, dict[str, list[str]]] = Field(
        ..., description="最多 3 个待匿名字段的泛化层级映射"
    )
    # 可选的 l-多样性约束：两者必须同时提供；
    # sensitive_field 不得与待匿名字段重合，且每条记录的敏感值都必须是非空字符串。
    sensitive_field: str | None = Field(
        default=None,
        description="敏感属性字段名（非待匿名字段），需与 l 同时提供",
    )
    l: StrictInt | None = Field(
        default=None,
        description="每个发布分组至少包含的不同敏感值种数，需与 sensitive_field 同时提供",
    )

    model_config = ConfigDict(extra="forbid")


class Group(BaseModel):
    """一个发布分组：泛化后的字段值组合与记录数量。"""

    values: dict[str, str]
    count: int


class AnonymizeResponse(BaseModel):
    """匿名化结果。"""

    generalized_records: list[dict[str, Any]]
    selected_levels: dict[str, int]
    groups: list[Group]
    k: int
    total_records: int


class AuditIntersectionRequest(BaseModel):
    """两次发布关联风险审计请求体。"""

    records_a: list[dict[str, Any]] = Field(..., description="第一次公开发布的记录")
    fields_a: list[str] = Field(
        ..., description="第一次发布用于分组的公开字段名列表（可为空列表）"
    )
    records_b: list[dict[str, Any]] = Field(..., description="第二次公开发布的记录")
    fields_b: list[str] = Field(
        ..., description="第二次发布用于分组的公开字段名列表（可为空列表）"
    )
    id_field: str = Field(..., description="两批记录共同的 id 字段名")
    # 拒绝 bool 冒充整数；k 必须为正整数。
    k: StrictInt = Field(..., description="交集成员数的风险阈值")

    model_config = ConfigDict(extra="forbid")


class Intersection(BaseModel):
    """一个交集单元：两次分组的一对等价组按 id 求交的结果。"""

    ids: list[str]
    count: int
    values_a: dict[str, str]
    values_b: dict[str, str]
    at_risk: bool


class AuditIntersectionResponse(BaseModel):
    """两次发布关联风险审计结果。"""

    intersections: list[Intersection]
    k: int
    total_records: int
    risk_count: int


@app.get("/")
def root() -> dict[str, str]:
    """服务信息。"""
    return {
        "service": "wildlife-observation-anonymizer",
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
        "anonymize": f"{API_PREFIX}/anonymize",
        "audit_intersection": f"{API_PREFIX}/audit-intersection",
    }


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@app.post(f"{API_PREFIX}/anonymize", response_model=AnonymizeResponse)
def anonymize(request: AnonymizeRequest) -> AnonymizeResponse:
    """对记录执行统一层级泛化，返回可公开发布的匿名化结果。"""
    if request.k < 1:
        raise unprocessable("k 必须是正整数（>= 1）")

    # 记录总数小于 k：任何分组（至少存在一个组）都不可能达到 k 条。
    if len(request.records) < request.k:
        raise unprocessable(
            f"记录总数 {len(request.records)} 小于 k={request.k}，无法满足 k-匿名"
        )

    sensitive_values = prepare_sensitive(
        request.records, request.fields, request.sensitive_field, request.l
    )

    names, hierarchies = prepare_hierarchies(request.records, request.fields)
    combo, _ = solve_levels(
        request.records,
        names,
        hierarchies,
        request.k,
        sensitive_values=sensitive_values,
        l=request.l,
    )
    selected_levels = dict(zip(names, combo))
    generalized_records, groups = apply_levels(
        request.records, names, hierarchies, combo
    )

    return AnonymizeResponse(
        generalized_records=generalized_records,
        selected_levels=selected_levels,
        groups=[Group(**group) for group in groups],
        k=request.k,
        total_records=len(request.records),
    )


@app.post(
    f"{API_PREFIX}/audit-intersection",
    response_model=AuditIntersectionResponse,
)
def audit_intersection(request: AuditIntersectionRequest) -> AuditIntersectionResponse:
    """审计同一批观测记录两次公开后的关联风险：对两次分组按 id 求交集。"""
    if request.k < 1:
        raise unprocessable("k 必须是正整数（>= 1）")

    intersections = audit_intersections(
        request.records_a,
        request.fields_a,
        request.records_b,
        request.fields_b,
        request.id_field,
        request.k,
    )

    return AuditIntersectionResponse(
        intersections=[Intersection(**item) for item in intersections],
        k=request.k,
        total_records=len(request.records_a),
        risk_count=sum(1 for item in intersections if item["at_risk"]),
    )
