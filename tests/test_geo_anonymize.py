"""坐标自适应网格匿名化接口 POST /api/v1/geo-anonymize 的端到端测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ENDPOINT = "/api/v1/geo-anonymize"


# ---------------------------------------------------------------- 请求样例

# 8 条记录：4 条长江口附近、2 条松嫩平原附近（obs-005 纬度恰为 45，落在
# z=2 的拆分边界上）、2 条悉尼附近。
RECORDS = [
    {"id": "obs-001", "latitude": 31.5, "longitude": 121.0, "species": "震旦鸦雀"},
    {"id": "obs-002", "latitude": 32.0, "longitude": 121.5, "species": "震旦鸦雀"},
    {"id": "obs-003", "latitude": 30.5, "longitude": 120.5, "species": "白头鹤"},
    {"id": "obs-004", "latitude": 31.0, "longitude": 122.0, "species": "白头鹤"},
    {"id": "obs-005", "latitude": 45.0, "longitude": 126.0, "species": "丹顶鹤"},
    {"id": "obs-006", "latitude": 44.5, "longitude": 125.5, "species": "丹顶鹤"},
    {"id": "obs-007", "latitude": -33.9, "longitude": 151.2, "species": "笑翠鸟"},
    {"id": "obs-008", "latitude": -33.5, "longitude": 151.0, "species": "笑翠鸟"},
]


def payload(k: int = 2, max_zoom: int = 3, records=None) -> dict:
    return {
        "k": k,
        "max_zoom": max_zoom,
        "records": records if records is not None else RECORDS,
    }


def cell_tuples(body: dict) -> dict[tuple[int, int, int], dict]:
    return {(c["z"], c["x"], c["y"]): c for c in body["cells"]}


# ---------------------------------------------------------------- 基本功能

def test_adaptive_grid_basic():
    response = client.post(ENDPOINT, json=payload(k=2, max_zoom=3))
    assert response.status_code == 200
    body = response.json()
    assert body["k"] == 2
    assert body["max_zoom"] == 3
    assert body["total_records"] == 8

    # 北半球 6 条在 z=1 整格停止：继续拆会分出只含 obs-005 一条的子格。
    # 悉尼 2 条同格一路拆到 max_zoom=3。
    assignments = {item["id"]: item for item in body["assignments"]}
    for obs_id in ("obs-001", "obs-002", "obs-003", "obs-004", "obs-005", "obs-006"):
        cell = assignments[obs_id]
        assert (cell["z"], cell["x"], cell["y"]) == (1, 1, 0)
    for obs_id in ("obs-007", "obs-008"):
        cell = assignments[obs_id]
        assert (cell["z"], cell["x"], cell["y"]) == (3, 7, 5)

    # assignments 与输入等长、等序。
    assert [item["id"] for item in body["assignments"]] == [
        record["id"] for record in RECORDS
    ]

    # 发布格：z/x/y、经纬边界与人数；按格内记录首次出现顺序排列。
    assert [(c["z"], c["x"], c["y"]) for c in body["cells"]] == [(1, 1, 0), (3, 7, 5)]
    cells = cell_tuples(body)
    north = cells[(1, 1, 0)]
    assert north["count"] == 6
    assert north["min_longitude"] == 0.0
    assert north["max_longitude"] == 180.0
    assert north["min_latitude"] == 0.0
    assert north["max_latitude"] == 90.0
    sydney = cells[(3, 7, 5)]
    assert sydney["count"] == 2
    assert sydney["min_longitude"] == 135.0
    assert sydney["max_longitude"] == 180.0
    assert sydney["min_latitude"] == -45.0
    assert sydney["max_latitude"] == -22.5

    # 每个发布格都满足 k-匿名。
    for cell in body["cells"]:
        assert cell["count"] >= 2


def test_response_excludes_original_fields():
    # 响应只含 id 与 z/x/y，不回传原坐标或其他原字段。
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    body = response.json()
    for item in body["assignments"]:
        assert set(item) == {"id", "z", "x", "y"}
    for cell in body["cells"]:
        assert set(cell) == {
            "z", "x", "y",
            "min_latitude", "max_latitude",
            "min_longitude", "max_longitude",
            "count",
        }
    text = response.text
    assert "species" not in text
    assert "震旦鸦雀" not in text
    assert "121.5" not in text
    assert "-33.9" not in text


def test_max_zoom_zero_single_root_cell():
    response = client.post(ENDPOINT, json=payload(max_zoom=0))
    assert response.status_code == 200
    body = response.json()
    assert body["cells"] == [
        {
            "z": 0,
            "x": 0,
            "y": 0,
            "min_latitude": -90.0,
            "max_latitude": 90.0,
            "min_longitude": -180.0,
            "max_longitude": 180.0,
            "count": 8,
        }
    ]
    for item in body["assignments"]:
        assert (item["z"], item["x"], item["y"]) == (0, 0, 0)


def test_k_equals_total_forces_root_cell():
    # k 等于记录总数：拆出的子格不可能都达到 k，全部并入根格。
    response = client.post(ENDPOINT, json=payload(k=8, max_zoom=12))
    assert response.status_code == 200
    body = response.json()
    assert len(body["cells"]) == 1
    assert body["cells"][0]["z"] == 0
    assert body["cells"][0]["count"] == 8


def test_boundary_points_go_north_and_east():
    # 边界点归入最北或最东格：纬度等于中点归北，经度等于中点归东。
    records = [
        {"id": "origin", "latitude": 0, "longitude": 0},
        {"id": "far-north-east", "latitude": 90, "longitude": 180},
        {"id": "far-south-west", "latitude": -90, "longitude": -180},
        {"id": "plain-south-west", "latitude": -10, "longitude": -10},
    ]
    response = client.post(
        ENDPOINT, json={"k": 1, "max_zoom": 1, "records": records}
    )
    assert response.status_code == 200
    assignments = {item["id"]: item for item in response.json()["assignments"]}
    # 原点 (0, 0) 与全球东北角 (90, 180)：最北最东格 x=1, y=0。
    assert (assignments["origin"]["x"], assignments["origin"]["y"]) == (1, 0)
    far_ne = assignments["far-north-east"]
    assert (far_ne["x"], far_ne["y"]) == (1, 0)
    # 全球西南角 (-90, -180) 与普通西南点：x=0, y=1。
    far_sw = assignments["far-south-west"]
    assert (far_sw["x"], far_sw["y"]) == (0, 1)
    plain_sw = assignments["plain-south-west"]
    assert (plain_sw["x"], plain_sw["y"]) == (0, 1)


def test_split_stops_when_any_child_below_k():
    # 3 条在北、1 条在南，k=2：根的南子格只有 1 条，整格（根）停止。
    records = [
        {"id": "a", "latitude": 10, "longitude": 10},
        {"id": "b", "latitude": 20, "longitude": 20},
        {"id": "c", "latitude": 30, "longitude": 30},
        {"id": "d", "latitude": -10, "longitude": -10},
    ]
    response = client.post(
        ENDPOINT, json={"k": 2, "max_zoom": 5, "records": records}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["cells"]) == 1
    assert body["cells"][0]["z"] == 0
    assert body["cells"][0]["count"] == 4


def test_empty_children_do_not_block_split():
    # 空子格不参与判断：全部点同在一个子格也能继续拆到 max_zoom。
    records = [
        {"id": "a", "latitude": 10, "longitude": 10},
        {"id": "b", "latitude": 20, "longitude": 20},
    ]
    response = client.post(
        ENDPOINT, json={"k": 1, "max_zoom": 2, "records": records}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cells"] == [
        {
            "z": 2,
            "x": 2,
            "y": 1,
            "min_latitude": 0.0,
            "max_latitude": 45.0,
            "min_longitude": 0.0,
            "max_longitude": 90.0,
            "count": 2,
        }
    ]


def test_symmetric_split_into_two_deep_cells():
    # 东北、西南各 2 条，k=2：两个子格都达标，两侧各自继续拆到底。
    records = [
        {"id": "a", "latitude": 10, "longitude": 10},
        {"id": "b", "latitude": 20, "longitude": 20},
        {"id": "c", "latitude": -10, "longitude": -10},
        {"id": "d", "latitude": -20, "longitude": -20},
    ]
    response = client.post(
        ENDPOINT, json={"k": 2, "max_zoom": 2, "records": records}
    )
    assert response.status_code == 200
    body = response.json()
    assignments = {item["id"]: item for item in body["assignments"]}
    for obs_id in ("a", "b"):
        cell = assignments[obs_id]
        assert (cell["z"], cell["x"], cell["y"]) == (2, 2, 1)
    for obs_id in ("c", "d"):
        cell = assignments[obs_id]
        assert (cell["z"], cell["x"], cell["y"]) == (2, 1, 2)
    cells = cell_tuples(body)
    assert cells[(2, 2, 1)]["count"] == 2
    assert cells[(2, 1, 2)]["count"] == 2
    assert cells[(2, 1, 2)]["min_longitude"] == -90.0
    assert cells[(2, 1, 2)]["max_latitude"] == 0.0


def test_boundary_coordinates_accepted():
    # 纬度 ±90、经度 ±180 是合法边界值。
    records = [
        {"id": "a", "latitude": 90, "longitude": 180},
        {"id": "b", "latitude": -90, "longitude": -180},
    ]
    response = client.post(
        ENDPOINT, json={"k": 1, "max_zoom": 1, "records": records}
    )
    assert response.status_code == 200


# ---------------------------------------------------------------- 422 校验

def test_422_when_total_less_than_k():
    response = client.post(ENDPOINT, json=payload(k=9))
    assert response.status_code == 422
    assert "小于 k" in response.json()["detail"]


def test_422_when_k_not_positive_integer():
    for bad_k in (0, -2, True, "2", 2.5):
        response = client.post(ENDPOINT, json=payload(k=bad_k))
        assert response.status_code == 422


def test_422_when_max_zoom_invalid():
    for bad_max_zoom in (-1, 13, True, "3", 2.5):
        response = client.post(ENDPOINT, json=payload(max_zoom=bad_max_zoom))
        assert response.status_code == 422


def test_422_on_empty_records():
    response = client.post(ENDPOINT, json={"k": 1, "max_zoom": 3, "records": []})
    assert response.status_code == 422


def test_422_when_id_missing_or_not_non_empty_string():
    variants = [
        [{"latitude": 1, "longitude": 1}],
        [{"id": "", "latitude": 1, "longitude": 1}],
        [{"id": "   ", "latitude": 1, "longitude": 1}],
        [{"id": 12, "latitude": 1, "longitude": 1}],
        [{"id": None, "latitude": 1, "longitude": 1}],
    ]
    for records in variants:
        response = client.post(
            ENDPOINT, json={"k": 1, "max_zoom": 3, "records": records}
        )
        assert response.status_code == 422
        assert "非空字符串" in response.json()["detail"]


def test_422_when_id_duplicated():
    records = [
        {"id": "a", "latitude": 1, "longitude": 1},
        {"id": "a", "latitude": 2, "longitude": 2},
    ]
    response = client.post(
        ENDPOINT, json={"k": 1, "max_zoom": 3, "records": records}
    )
    assert response.status_code == 422
    assert "重复" in response.json()["detail"]


def test_422_when_coordinate_not_numeric():
    for bad_value in ("1", True, None, [1], {"v": 1}):
        for field in ("latitude", "longitude"):
            record = {"id": "a", "latitude": 1, "longitude": 1}
            record[field] = bad_value
            response = client.post(
                ENDPOINT, json={"k": 1, "max_zoom": 3, "records": [record]}
            )
            assert response.status_code == 422
            assert "有限数值" in response.json()["detail"]


def test_422_when_coordinate_missing():
    for record in ({"id": "a", "longitude": 1}, {"id": "a", "latitude": 1}):
        response = client.post(
            ENDPOINT, json={"k": 1, "max_zoom": 3, "records": [record]}
        )
        assert response.status_code == 422


def test_422_when_coordinate_not_finite():
    # JSON 扩展写法 NaN / Infinity 可被服务端解析，但必须被判定为非法坐标。
    for bad_literal in ("NaN", "Infinity", "-Infinity"):
        body = (
            '{"k": 1, "max_zoom": 3, "records": '
            '[{"id": "a", "latitude": ' + bad_literal + ', "longitude": 1}]}'
        )
        response = client.post(
            ENDPOINT,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "有限数值" in response.json()["detail"]


def test_422_when_coordinate_out_of_range():
    for bad_latitude in (-90.0001, 90.0001):
        record = {"id": "a", "latitude": bad_latitude, "longitude": 1}
        response = client.post(
            ENDPOINT, json={"k": 1, "max_zoom": 3, "records": [record]}
        )
        assert response.status_code == 422
        assert "-90 到 90" in response.json()["detail"]
    for bad_longitude in (-180.0001, 180.0001):
        record = {"id": "a", "latitude": 1, "longitude": bad_longitude}
        response = client.post(
            ENDPOINT, json={"k": 1, "max_zoom": 3, "records": [record]}
        )
        assert response.status_code == 422
        assert "-180 到 180" in response.json()["detail"]


def test_422_on_missing_or_extra_body_fields():
    assert client.post(ENDPOINT, json={"k": 1}).status_code == 422
    body = payload()
    body["extra"] = 1
    assert client.post(ENDPOINT, json=body).status_code == 422


# ---------------------------------------------------------------- 持久化

def test_no_records_written_to_disk(tmp_path, monkeypatch):
    """处理请求前后工作目录下不应出现任何数据文件（服务纯内存）。"""
    monkeypatch.chdir(tmp_path)
    response = client.post(ENDPOINT, json=payload())
    assert response.status_code == 200
    assert list(tmp_path.rglob("*")) == []
