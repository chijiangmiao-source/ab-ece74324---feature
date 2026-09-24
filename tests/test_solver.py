"""确定性边界与语义测试。"""

from __future__ import annotations

import random
import time

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


def test_dead_time_boundary_equality_is_conflict():
    # 同探测器命中时刻差恰好等于恢复期 -> 不得分属两个事件
    times = [0, 0, 2, 2]
    detectors = [0, 1, 0, 1]
    weights = [1, 1, 1, 1]
    no_dt = solve(times, detectors, weights, 2, id_order=_ids(4))
    assert no_dt["best_score"] == 4  # 两个二元事件
    with_dt = solve(
        times, detectors, weights, 2, id_order=_ids(4),
        dead_times=[2, 2],
    )
    assert with_dt["best_score"] == 2  # 至多一个事件
    assert with_dt["best_events"] == 1


def test_dead_time_strictly_greater_gap_allowed():
    # 时刻差 d+1（严格大于）允许分属两个事件
    times = [0, 0, 3, 3]
    detectors = [0, 1, 0, 1]
    weights = [1, 1, 1, 1]
    got = solve(
        times, detectors, weights, 3, id_order=_ids(4),
        dead_times=[2, 2],
    )
    assert got["best_score"] == 4
    assert got["best_events"] == 2


def test_dead_time_zero_blocks_same_timestamp():
    # 显式 d=0：同刻同探测器的两个命中不能分属两个事件
    times = [0, 0, 0, 0]
    detectors = [0, 0, 1, 1]
    weights = [1, 1, 1, 1]
    absent = solve(times, detectors, weights, 0, id_order=_ids(4))
    assert absent["best_score"] == 4  # 未提供：两个事件
    enabled = solve(
        times, detectors, weights, 0, id_order=_ids(4),
        dead_times=[0, 0],
    )
    assert enabled["best_score"] == 2  # 每探测器只一个命中可分组
    assert enabled["best_events"] == 1


def test_noise_does_not_trigger_recovery():
    # A@0(低权) 作噪声时不得阻塞 A@2：最优把 A@0 丢弃，
    # 事件 (B@1, A@2) 拿到 200；若噪声也触发恢复则至多 101。
    times = [0, 1, 2, 3]
    detectors = [0, 1, 0, 1]
    weights = [1, 100, 100, 100]
    got = solve(
        times, detectors, weights, 3, id_order=_ids(4),
        dead_times=[2, 2],
    )
    assert got["best_score"] == 200
    assert got["canonical"] == [(1, 2)]
    # A@0 从不出现在任何最优事件中
    assert 0 not in got["member_count"]


def test_dead_time_spans_multiple_candidate_events():
    # 链式候选：A 在 t=0 被分组后，恢复期 d=4 跨越 t=2、t=4 两个
    # 候选事件位置，只有 t=5 的 A 命中可再次分组。
    times = [0, 1, 2, 3, 4, 5]
    detectors = [0, 1, 1, 1, 1, 0]
    weights = [10] * 6
    got = solve(
        times, detectors, weights, 5, id_order=_ids(6),
        dead_times=[4, 0],
    )
    # (A0, 某个 B) 与 (A5, 某个 B)：A 差 5 > 4 允许；
    # 两次分组共 4 个命中 = 40
    assert got["best_score"] == 40
    # 两个 A 命中必然各自入组（member 次数 = 总方案数）
    total = got["total_count"]
    assert got["member_count"][0] == total
    assert got["member_count"][5] == total
    # 它们探测器相同，从不在同一事件中
    assert (0, 5) not in got["pair_count"]


def test_dead_time_attribution_recomputed_over_optima():
    # 恢复期改变最优解集合：任一事件占用 A、B 后，同探测器在恢复结束
    # （t 差 ≤ 3）前不得再分组，故最优只能是单个二元事件；
    # 四个候选事件各占一个最优解 -> 跨探测器对均为 optional。
    times = [0, 1, 2, 3]
    detectors = [0, 1, 0, 1]
    weights = [1, 1, 1, 1]
    got = solve(
        times, detectors, weights, 3, id_order=_ids(4),
        dead_times=[3, 3],
    )
    assert got["total_count"] == 4
    assert got["best_events"] == 1
    for pair in ((0, 1), (0, 3), (1, 2), (2, 3)):
        assert got["pair_count"][pair] == 1
    # 规范裁决取成员标识序列最小者
    assert got["canonical"] == [(0, 1)]


def test_dead_time_full_scale_performance():
    rng = random.Random(909)
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
    got = solve(
        times, detectors, weights, window, id_order=_ids(n),
        dead_times=dead,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"solve too slow: {elapsed:.2f}s"
    assert got["best_score"] > 0
    assert isinstance(got["total_count"], int) and got["total_count"] >= 1
    # 规范解本身必须满足恢复约束（事件按标识排序，逐探测器收集时刻）
    times_by_det = {d: [] for d in range(8)}
    for g in got["canonical"]:
        for j in g:
            times_by_det[detectors[j]].append(times[j])
    for d, ts in times_by_det.items():
        ts.sort()
        for a, b in zip(ts, ts[1:]):
            assert b - a > dead[d], f"detector {d}: {b}-{a} <= {dead[d]}"
