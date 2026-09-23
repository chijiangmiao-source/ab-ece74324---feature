"""确定性边界与语义测试。"""

from __future__ import annotations

import random
import time
from typing import Dict, List

from app.solver import SolveError, solve


def _ids(n):
    return [f"h{k:02d}" for k in range(n)]


def test_window_closed_boundary_inclusive():
    # t=0 与 t=W 在闭窗内可同组；t=W+1 不可
    times = [0, 5, 6]
    detectors = [0, 1, 1]
    weights = [5, 7, 9]
    got = solve(times, detectors, weights, 5, id_order=_ids(3))
    # (0,1) 可同组（差 5）；(0,2) 差 6 不可；1、2 同探测器
    assert got["best_score"] == 12
    assert (0, 1) in got["pair_count"]
    assert (0, 2) not in got["pair_count"]


def test_no_possible_events_all_noise():
    times = [0, 1, 2, 3]
    detectors = [0, 0, 0, 0]  # 全部同一探测器
    weights = [1, 1, 1, 1]
    got = solve(times, detectors, weights, 10)
    assert got["best_score"] == 0
    assert got["best_events"] == 0
    assert got["total_count"] == 1
    assert got["canonical"] == []
    assert got["pair_count"] == {}


def test_max_confidence_then_min_events():
    # 命中 0,1,2 互相兼容；两两可信度相同，三分组不可能（同一探测器限制）
    times = [0, 1, 2]
    detectors = [0, 1, 0]
    weights = [5, 4, 5]
    # 事件只能是 (0,1) 和 (1,2)；二者可信度 9，各 1 事件
    got = solve(times, detectors, weights, 10, id_order=_ids(3))
    assert got["best_score"] == 9
    assert got["best_events"] == 1
    assert got["total_count"] == 2
    # 规范裁决取标识序列更小者：[h00,h01] < [h01,h02]
    assert got["canonical"] == [(0, 1)]


def test_min_events_prefers_fewer_groups():
    # 四个命中两两兼容（探测器互异）：
    # 方案 A：一个四元事件 = 全部可信度
    # 方案 B：两个二元事件，可信度和相同（构造为等权）
    times = [0, 0, 0, 0]
    detectors = [0, 1, 2, 3]
    weights = [1, 1, 1, 1]
    got = solve(times, detectors, weights, 0, id_order=_ids(4))
    assert got["best_score"] == 4
    assert got["best_events"] == 1  # 四元事件最少
    assert (0, 1, 2, 3) in got["canonical"]


def test_pair_always_optional_never():
    # 0-1-2 链式兼容（探测器 0,1,0），等权；
    # 最优取一个二元事件（可信度 2），另一个噪声。
    times = [0, 1, 2]
    detectors = [0, 1, 0]
    weights = [1, 1, 1]
    got = solve(times, detectors, weights, 2, id_order=_ids(3))
    assert got["total_count"] == 2
    # (0,1) 与 (1,2) 各在恰好一个最优解中 -> 可选
    assert got["pair_count"][(0, 1)] == 1
    assert got["pair_count"][(1, 2)] == 1
    # 0 与 2 同探测器，从不兼容
    assert (0, 2) not in got["pair_count"]
    # 中间命中 1 在两个最优解中都被分组 -> 必然归属（成员层面）
    assert got["member_count"][1] == 2


def test_pair_always_relation():
    times = [0, 1, 1]
    detectors = [0, 1, 2]
    weights = [10, 10, 1]
    # 最优必含 (0,1)（可信度 20），命中 2 噪声
    got = solve(times, detectors, weights, 1, id_order=_ids(3))
    assert got["pair_count"][(0, 1)] == got["total_count"]
    assert got["total_count"] == 1


def test_zero_window_same_timestamp():
    times = [0, 0, 0, 1]
    detectors = [0, 1, 2, 0]
    weights = [3, 4, 5, 100]
    # W=0：前三个同刻可成三元事件（12）；命中 3 高可信但无人同时刻 -> 噪声
    got = solve(times, detectors, weights, 0, id_order=_ids(4))
    assert got["best_score"] == 12
    assert got["best_events"] == 1


def test_arbitrary_precision_count():
    # k 个相互独立的链式三元组（0,1,0 探测器），每组等权有 2 个最优，
    # 组间间隔超过符合窗 -> 总方案数 2^k，验证任意精度大整数。
    k = 26
    window = 2
    times, detectors, weights = [], [], []
    for i in range(k):
        base = 10 * i
        times += [base, base + 1, base + 2]
        detectors += [0, 1, 0]
        weights += [1, 1, 1]
    n = 3 * k
    got = solve(times, detectors, weights, window, id_order=_ids(n))
    assert got["total_count"] == 2 ** k
    assert isinstance(got["total_count"], int)
    assert got["best_score"] == 2 * k
    assert got["best_events"] == k


def test_full_scale_performance_and_density():
    rng = random.Random(2026)
    n = 80
    window = 5
    # 构造满足密度约束的时刻：逐个放置，确保任意窗 ≤10
    times = []
    t = 0
    while len(times) < n:
        window_count = sum(1 for x in times if t - window <= x <= t)
        if window_count < 10:
            times.append(t)
            if rng.random() < 0.5:
                t += 1
        else:
            t += 1
    times.sort()
    detectors = [rng.randrange(8) for _ in range(n)]
    weights = [rng.randint(1, 100) for _ in range(n)]
    start = time.perf_counter()
    got = solve(times, detectors, weights, window, id_order=_ids(n))
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"solve too slow: {elapsed:.2f}s"
    assert got["best_score"] > 0
    # 大整数计数一致性：必然成员计数之和等
    assert isinstance(got["total_count"], int) and got["total_count"] >= 1


def test_dense_window_rejected_via_solver():
    # 11 个命中落在同一闭窗 -> 拒绝
    times = [0] * 11
    detectors = [k % 8 for k in range(11)]
    weights = [1] * 11
    try:
        solve(times, detectors, weights, 0, id_order=[f"h{k}" for k in range(11)])
    except SolveError:
        return
    raise AssertionError("expected SolveError")


# ---------------------------------------------------------------- 恢复期

def test_dead_time_boundary_equal_is_conflict():
    # A@0、B@0、A@2、B@2，W=2，全高权。
    times = [0, 0, 2, 2]
    detectors = [0, 1, 0, 1]
    weights = [10, 10, 10, 10]
    ids = _ids(4)
    # D_A=2：两个 A 命中时刻差恰为 2，边界相等算冲突，无法双双入组
    got = solve(times, detectors, weights, 2, id_order=ids,
                dead_times=[2, 0])
    assert got["best_score"] == 20
    assert got["best_events"] == 1
    # 任取一个 A 与一个 B 配对，共 4 个最优单事件
    assert got["total_count"] == 4
    assert _norm(got["canonical"]) == [(0, 1)]
    assert (0, 2) not in got["pair_count"]  # 同探测器
    # 每个跨探测器配对恰出现在一个最优解中 -> 全部可选
    assert got["pair_count"][(0, 1)] == 1
    assert got["pair_count"][(2, 3)] == 1
    # D_A=1：差 2 严格大于恢复期，两个事件可行；两种交叉配对：
    # (A0,B0)+(A2,B2) 与 (A0,B2)+(B0,A2)
    got1 = solve(times, detectors, weights, 2, id_order=ids,
                 dead_times=[1, 0])
    assert got1["best_score"] == 40
    assert got1["best_events"] == 2
    assert got1["total_count"] == 2
    assert _norm(got1["canonical"]) == [(0, 1), (2, 3)]


def test_dead_time_noise_does_not_trigger():
    # A@0,B@0 成事件；A@1 低权必为噪声；A@2,C@2 成事件。
    times = [0, 0, 1, 2, 2]
    detectors = [0, 1, 0, 0, 2]
    weights = [100, 100, 1, 100, 100]
    ids = _ids(5)
    # D_A=1：A@1 是噪声不触发恢复，故 A@0 与 A@2（差 2 > 1）可分别入组。
    # 若噪声也触发恢复，A@1 会封 A@2，两个事件的 400 分将不可达。
    got = solve(times, detectors, weights, 2, id_order=ids,
                dead_times=[1, 0, 0])
    assert got["best_score"] == 400
    assert got["best_events"] == 2
    assert _norm(got["canonical"]) == [(0, 1), (3, 4)]
    # D_A=2：A@0 与 A@2 差 2（边界相等）冲突，不能两个事件；
    # 噪声 A@1 与该冲突无关，最优为单事件 (A@0,B@0,C@2)=300
    got2 = solve(times, detectors, weights, 2, id_order=ids,
                 dead_times=[2, 0, 0])
    assert got2["best_score"] == 300
    assert got2["best_events"] == 1
    assert 3 not in {j for g in got2["canonical"] for j in g}


def test_dead_time_per_detector_independent():
    # 同刻两对：A@0/B@0 与 A@1/B@1（时间差均为 1）
    times = [0, 0, 1, 1]
    detectors = [0, 1, 0, 1]
    weights = [10, 10, 10, 10]
    ids = _ids(4)
    # 仅 A 的恢复期为 1：A 的两个命中不能都入组 -> 至多一个事件
    got = solve(times, detectors, weights, 1, id_order=ids,
                dead_times=[1, 0])
    assert got["best_score"] == 20
    # A=0、B=1：B 被封而 A 不被封，结论同样至多一个事件
    got2 = solve(times, detectors, weights, 1, id_order=ids,
                 dead_times=[0, 1])
    assert got2["best_score"] == 20
    # 双方恢复期均为 0：两个事件，差 1 > 0
    got3 = solve(times, detectors, weights, 1, id_order=ids,
                 dead_times=[0, 0])
    assert got3["best_score"] == 40


def test_dead_time_spans_multiple_candidate_events():
    # A@0, B@1, B@2, A@5；W=5。无恢复时最优是两个事件
    # (A0,Bx)+(By,A5)（x≠y），把两个 A 命中分别归入前后候选事件。
    times = [0, 1, 2, 5]
    detectors = [0, 1, 1, 0]
    weights = [50, 100, 100, 50]
    ids = _ids(4)
    base = solve(times, detectors, weights, 5, id_order=ids)
    assert base["best_score"] == 300 and base["best_events"] == 2
    # D_A=5：两个 A 命中差 5（边界相等）冲突，双事件方案需要两个 A
    # 分别入组，全部非法；联合求解须在选事件时维护恢复占用，
    # 最优退化为单个二元事件 150（三元事件也容不下：两个 B 同探测器）。
    got = solve(times, detectors, weights, 5, id_order=ids,
                dead_times=[5, 0])
    assert got["best_score"] == 150
    assert got["best_events"] == 1
    assert len(got["canonical"][0]) == 2
    assert 0 not in {j for g in got["canonical"] for j in g} or \
        3 not in {j for g in got["canonical"] for j in g}
    # D_A=4：差 5 严格大于恢复期，两种交叉配对都是最优（计数共享状态语义）
    got2 = solve(times, detectors, weights, 5, id_order=ids,
                 dead_times=[4, 0])
    assert got2["best_score"] == 300
    assert got2["best_events"] == 2
    assert got2["total_count"] == 2


def test_dead_time_zero_blocks_same_timestamp_double_grouping():
    # 四个同刻命中 A,B,A,B：缺省时可拆成两个事件 (40)；
    # 显式 D=0 要求时差严格大于 0，同刻同探测器边界相等即冲突 -> 至多一事件。
    times = [0, 0, 0, 0]
    detectors = [0, 1, 0, 1]
    weights = [10] * 4
    ids = _ids(4)
    base = solve(times, detectors, weights, 0, id_order=ids)
    assert base["best_score"] == 40 and base["best_events"] == 2
    got = solve(times, detectors, weights, 0, id_order=ids,
                dead_times=[0, 0])
    assert got["best_score"] == 20
    assert got["best_events"] == 1


def test_dead_time_pair_attribution_recomputed():
    # 0-1-2 链式兼容（探测器 0,1,0），等权，D_0=W=2：
    # 两个 A 命中时刻差 2 冲突，链式两选其一不再合法，只剩单事件空间。
    times = [0, 1, 2]
    detectors = [0, 1, 0]
    weights = [1, 1, 1]
    got = solve(times, detectors, weights, 2, id_order=_ids(3),
                dead_times=[2, 0])
    assert got["best_score"] == 2
    assert got["total_count"] == 2  # (0,1) 或 (1,2)，二者择一
    assert got["pair_count"][(0, 1)] == 1
    assert got["pair_count"][(1, 2)] == 1
    # 中心命中在两个最优解中都入组
    assert got["member_count"][1] == 2


def test_dead_time_full_scale_performance():
    rng = random.Random(4052)
    n = 80
    window = 5
    times = []
    t = 0
    while len(times) < n:
        window_count = sum(1 for x in times if t - window <= x <= t)
        if window_count < 10:
            times.append(t)
            if rng.random() < 0.5:
                t += 1
        else:
            t += 1
    times.sort()
    detectors = [rng.randrange(8) for _ in range(n)]
    weights = [rng.randint(1, 100) for _ in range(n)]
    dead = [rng.randint(0, window) for _ in range(8)]
    start = time.perf_counter()
    got = solve(times, detectors, weights, window, id_order=_ids(n),
                dead_times=dead)
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"solve too slow: {elapsed:.2f}s"
    assert isinstance(got["total_count"], int) and got["total_count"] >= 1
    # 每个规范事件内探测器互异
    for g in got["canonical"]:
        assert len({detectors[j] for j in g}) == len(g)
    # 同探测器任意两个已分组命中时刻差严格大于恢复期（边界相等算冲突）
    grouped = sorted(
        j for g in got["canonical"] for j in g
    )
    by_det: Dict[int, List[int]] = {}
    for j in grouped:
        by_det.setdefault(detectors[j], []).append(times[j])
    for d, ts in by_det.items():
        for a, b in zip(ts, ts[1:]):
            assert b - a > dead[d]


def _norm(groups):
    return sorted(tuple(sorted(g)) for g in groups)
