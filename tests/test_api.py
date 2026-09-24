"""/audit 与 /health 的 HTTP 层测试。"""

from __future__ import annotations

import json

import pytest

from app.web import app


@pytest.fixture()
def client():
    app.testing = True
    return app.test_client()


def _payload(**over):
    payload = {
        "window": 2,
        "detectors": ["A", "B", "C"],
        "hits": [
            {"id": "h1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "h2", "detector": "B", "time": 1, "confidence": 7},
            {"id": "h3", "detector": "C", "time": 1, "confidence": 3},
            {"id": "h4", "detector": "A", "time": 9, "confidence": 4},
        ],
    }
    payload.update(over)
    return payload


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_audit_basic(client):
    resp = client.post("/audit", json=_payload())
    assert resp.status_code == 200
    data = resp.get_json()
    # h1,h2,h3 在窗内（0..2），三元事件可信度 15
    assert data["optimal_confidence"] == 15
    assert data["event_count"] == 1
    assert data["canonical_groups"] == [["h1", "h2", "h3"]]
    assert data["solution_count"] == "1"
    rel = {(r["a"], r["b"]): r["relation"] for r in data["pair_relations"]}
    assert rel[("h1", "h2")] == "always"
    assert rel[("h2", "h3")] == "always"
    # h1 与 h4 同探测器，不出现在可同组对中
    assert ("h1", "h4") not in rel
    assert ("h3", "h4") not in rel  # 时刻差超窗


def test_audit_window_boundary(client):
    # 窗 =1：h1(t0) 与 h2/h3(t1) 之差恰为 1，闭窗边界包含 -> 三元组仍可行
    resp = client.post("/audit", json=_payload(window=1))
    assert resp.get_json()["optimal_confidence"] == 15
    # 窗 =0：仅同时刻命中可组 -> h2+h3=10
    resp0 = client.post("/audit", json=_payload(window=0))
    assert resp0.get_json()["optimal_confidence"] == 10


def test_audit_missing_fields_no_result(client):
    resp = client.post("/audit", json={"detectors": ["A"], "hits": []})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "errors" in data
    assert "optimal_confidence" not in data
    paths = {e["path"] for e in data["errors"]}
    assert "window" in paths
    assert "detectors" in paths  # 只有 1 个探测器
    assert "hits" in paths  # 不足 4 个


def test_audit_field_errors_by_path(client):
    payload = _payload()
    payload["hits"] = [
        {"id": "x", "detector": "A", "time": 0, "confidence": 1},
        {"id": "x", "detector": "Z", "time": "q", "confidence": -2},
        {"id": "", "detector": "B", "time": 0, "confidence": 0},
        {"id": "ok", "detector": "B", "time": 0},
    ]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    paths = {e["path"] for e in resp.get_json()["errors"]}
    assert "hits[1].id" in paths          # 重复 id
    assert "hits[1].detector" in paths    # 未声明探测器
    assert "hits[1].time" in paths        # 非整数
    assert "hits[1].confidence" in paths  # 非正
    assert "hits[2].id" in paths          # 空 id
    assert "hits[2].confidence" in paths  # 零
    assert "hits[3].confidence" in paths  # 缺字段


def test_audit_bool_rejected_as_integer(client):
    payload = _payload()
    payload["window"] = True
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    assert any(e["path"] == "window" for e in resp.get_json()["errors"])


def test_audit_too_many_hits_in_window(client):
    payload = _payload()
    payload["hits"] = [
        {"id": f"h{k}", "detector": "ABC"[k % 3], "time": 0,
         "confidence": 1}
        for k in range(11)
    ]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["errors"]
    assert "optimal_confidence" not in data


def test_audit_non_json_body(client):
    resp = client.post("/audit", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    assert resp.get_json()["errors"][0]["path"] == "$"


def test_audit_solution_count_is_string(client):
    resp = client.post("/audit", json=_payload())
    data = resp.get_json()
    # 任意精度以字符串传输
    assert isinstance(data["solution_count"], str)
    int(data["solution_count"])


# ---------------------------------------------------- 恢复期（dead_times）


def test_dead_times_basic(client):
    # A@0-B@1 与 A@2-B@3：无恢复期可组两个事件；
    # d=2（边界相等冲突）使任一事件的两探测器在 t=3 前都无法再分组，
    # 故最优只能保留单个二元事件（共 4 个候选，各占一个最优解）。
    payload = {
        "window": 3,
        "detectors": ["A", "B"],
        "dead_times": [2, 2],
        "hits": [
            {"id": "a1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "b1", "detector": "B", "time": 1, "confidence": 5},
            {"id": "a2", "detector": "A", "time": 2, "confidence": 5},
            {"id": "b2", "detector": "B", "time": 3, "confidence": 5},
        ],
    }
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["optimal_confidence"] == 10
    assert data["event_count"] == 1
    assert data["solution_count"] == "4"
    rel = {(r["a"], r["b"]): r["relation"] for r in data["pair_relations"]}
    # 四个跨探测器候选事件各恰在一个最优解中
    assert rel[("a1", "b1")] == "optional"
    assert rel[("b1", "a2")] == "optional"
    assert rel[("a1", "b2")] == "optional"
    assert rel[("a2", "b2")] == "optional"
    # 规范裁决取成员标识序列最小者
    assert data["canonical_groups"] == [["a1", "b1"]]


def test_dead_times_gap_strictly_greater(client):
    payload = {
        "window": 3,
        "detectors": ["A", "B"],
        "dead_times": [2, 2],
        "hits": [
            {"id": "a1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "b1", "detector": "B", "time": 0, "confidence": 5},
            {"id": "a2", "detector": "A", "time": 3, "confidence": 5},
            {"id": "b2", "detector": "B", "time": 3, "confidence": 5},
        ],
    }
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    # 间隔 3 > 2：两个事件都可保留
    assert data["optimal_confidence"] == 20
    assert data["event_count"] == 2


def test_dead_times_omitted_keeps_old_semantics(client):
    payload = {
        "window": 3,
        "detectors": ["A", "B"],
        "hits": [
            {"id": "a1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "b1", "detector": "B", "time": 1, "confidence": 5},
            {"id": "a2", "detector": "A", "time": 2, "confidence": 5},
            {"id": "b2", "detector": "B", "time": 3, "confidence": 5},
        ],
    }
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["optimal_confidence"] == 20
    assert data["event_count"] == 2
    # 响应不含恢复期相关字段
    assert "dead_times" not in data


def test_dead_times_zero(client):
    # 显式 0：同刻同探测器命中不得分属两个事件
    payload = {
        "window": 0,
        "detectors": ["A", "B"],
        "dead_times": [0, 0],
        "hits": [
            {"id": "a1", "detector": "A", "time": 0, "confidence": 1},
            {"id": "a2", "detector": "A", "time": 0, "confidence": 1},
            {"id": "b1", "detector": "B", "time": 0, "confidence": 1},
            {"id": "b2", "detector": "B", "time": 0, "confidence": 1},
        ],
    }
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["optimal_confidence"] == 2
    assert data["event_count"] == 1


def test_dead_times_not_array(client):
    payload = _payload()
    payload["dead_times"] = 3
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert any(e["path"] == "dead_times" for e in data["errors"])
    assert "optimal_confidence" not in data


def test_dead_times_length_mismatch(client):
    payload = _payload()
    payload["dead_times"] = [1]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    paths = {e["path"] for e in resp.get_json()["errors"]}
    assert "dead_times" in paths
    # 缺字段的请求行为不变：不会出现 dead_times 错误
    ok = client.post("/audit", json=_payload())
    assert ok.status_code == 200


def test_dead_times_negative_and_non_integer(client):
    payload = _payload()
    payload["dead_times"] = [-1, "x", True]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    paths = {e["path"] for e in resp.get_json()["errors"]}
    assert "dead_times[0]" in paths
    assert "dead_times[1]" in paths
    assert "dead_times[2]" in paths  # 布尔值不得当作整数


def test_dead_times_exceeds_window(client):
    payload = _payload()
    payload["dead_times"] = [3, 1, 0]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    paths = {e["path"] for e in resp.get_json()["errors"]}
    assert "dead_times[0]" in paths
    assert "dead_times[1]" not in paths
    assert "dead_times[2]" not in paths


def test_dead_times_equal_window_allowed(client):
    payload = _payload()
    payload["dead_times"] = [2, 2, 2]
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 200


def test_dead_times_errors_reported_by_index_without_results(client):
    payload = {
        "window": 2,
        "detectors": ["A", "B", "C"],
        "dead_times": [1, -5, 9],
        "hits": [
            {"id": "h1", "detector": "A", "time": 0, "confidence": 5},
            {"id": "h2", "detector": "B", "time": 1, "confidence": 7},
            {"id": "h3", "detector": "C", "time": 1, "confidence": 3},
            {"id": "h4", "detector": "A", "time": 9, "confidence": 4},
        ],
    }
    resp = client.post("/audit", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert "optimal_confidence" not in data
    paths = {e["path"] for e in data["errors"]}
    assert paths == {"dead_times[1]", "dead_times[2]"}
