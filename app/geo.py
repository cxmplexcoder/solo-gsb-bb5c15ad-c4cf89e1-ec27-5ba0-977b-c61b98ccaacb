"""地理自适应网格匿名化：把野外观测坐标归入满足 k-匿名的递归四分格。

以全球经纬范围（经度 [-180, 180]、纬度 [-90, 90]）为根格，按经纬中点
递归四分；仅当当前格的**每个非空子格**都至少有 ``k`` 条记录时才拆分，
否则整格停止，最多拆到 ``max_zoom`` 层。落在拆分边界上的点归入最北或
最东的子格。输出为标准 z/x/y 格编号（x 自西向东、y 自北向南）。
"""

from __future__ import annotations

import math
from typing import Any

from .anonymizer import unprocessable

# 根格（z=0/x=0/y=0）覆盖全球。
ROOT_LON_MIN, ROOT_LON_MAX = -180.0, 180.0
ROOT_LAT_MIN, ROOT_LAT_MAX = -90.0, 90.0


def prepare_points(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[float], list[float]]:
    """校验记录并返回 (ids, latitudes, longitudes)，三者等长且与输入同序。

    每条记录必须含唯一非空字符串 id、有限数值 latitude（-90～90）与
    longitude（-180～180）；布尔值、NaN、无穷均视为非法。
    """
    if not records:
        raise unprocessable("records 至少需要包含 1 条记录")

    ids: list[str] = []
    seen_ids: set[str] = set()
    latitudes: list[float] = []
    longitudes: list[float] = []

    for index, record in enumerate(records):
        obs_id = record.get("id")
        if not isinstance(obs_id, str) or not obs_id.strip():
            raise unprocessable(f"第 {index + 1} 条记录的 id 必须是非空字符串")
        if obs_id in seen_ids:
            raise unprocessable(f"id {obs_id!r} 重复出现，id 必须唯一")
        seen_ids.add(obs_id)
        ids.append(obs_id)
        latitudes.append(_coordinate(record, "latitude", 90.0, obs_id))
        longitudes.append(_coordinate(record, "longitude", 180.0, obs_id))

    return ids, latitudes, longitudes


def _coordinate(
    record: dict[str, Any], field: str, bound: float, obs_id: str
) -> float:
    """取出并校验一个经纬度值：有限数值且在 [-bound, bound] 内。"""
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise unprocessable(f"记录 {obs_id!r} 的 {field} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise unprocessable(f"记录 {obs_id!r} 的 {field} 必须是有限数值")
    if not -bound <= result <= bound:
        raise unprocessable(
            f"记录 {obs_id!r} 的 {field} 必须在 {-bound:g} 到 {bound:g} 之间"
        )
    return result


def build_grid(
    latitudes: list[float],
    longitudes: list[float],
    k: int,
    max_zoom: int,
) -> tuple[list[tuple[int, int, int]], list[dict[str, Any]]]:
    """递归四分并返回 (每个点的最终格 (z, x, y), 发布格列表)。

    发布格按格内记录在输入中的首次出现顺序排列，含 z/x/y、经纬边界与
    记录数；每个发布格的记录数必然 >= k。
    """
    final: list[tuple[int, int, int] | None] = [None] * len(latitudes)
    # 每个元素为 (格内记录的最小输入序号, 发布格)，用于按首次出现排序。
    cells: list[tuple[int, dict[str, Any]]] = []

    def recurse(
        z: int,
        x: int,
        y: int,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        indices: list[int],
    ) -> None:
        if z < max_zoom:
            mid_lon = (lon_min + lon_max) / 2
            mid_lat = (lat_min + lat_max) / 2
            buckets: dict[tuple[bool, bool], list[int]] = {}
            for i in indices:
                # 边界点归入最北或最东的子格。
                east = longitudes[i] >= mid_lon
                north = latitudes[i] >= mid_lat
                buckets.setdefault((east, north), []).append(i)
            # 仅当每个非空子格都至少有 k 条时才拆分，否则整格停止。
            if all(len(bucket) >= k for bucket in buckets.values()):
                for (east, north), bucket in buckets.items():
                    recurse(
                        z + 1,
                        2 * x + (1 if east else 0),
                        2 * y + (0 if north else 1),
                        mid_lon if east else lon_min,
                        lon_max if east else mid_lon,
                        mid_lat if north else lat_min,
                        lat_max if north else mid_lat,
                        bucket,
                    )
                return
        # 整格停止：当前格成为发布格。
        for i in indices:
            final[i] = (z, x, y)
        cells.append(
            (
                min(indices),
                {
                    "z": z,
                    "x": x,
                    "y": y,
                    "min_latitude": lat_min,
                    "max_latitude": lat_max,
                    "min_longitude": lon_min,
                    "max_longitude": lon_max,
                    "count": len(indices),
                },
            )
        )

    recurse(
        0,
        0,
        0,
        ROOT_LON_MIN,
        ROOT_LON_MAX,
        ROOT_LAT_MIN,
        ROOT_LAT_MAX,
        list(range(len(latitudes))),
    )

    cells.sort(key=lambda item: item[0])

    # 递归把点集不断二分，每个点必然落入恰好一个叶子格。
    result: list[tuple[int, int, int]] = []
    for cell in final:
        assert cell is not None
        result.append(cell)
    return result, [cell for _, cell in cells]
