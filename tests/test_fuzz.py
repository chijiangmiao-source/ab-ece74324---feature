"""随机小规模用例：DP 结果与暴力枚举逐项一致。"""

from __future__ import annotations

import itertools
import random

import pytest

from app.solver import solve

from .brute import brute_solve


def _normalize(canonical):
    return tuple(sorted(tuple(sorted(g)) for g in canonical))


@pytest.mark.parametrize("seed", range(240))
def test_matches_brute(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 12)
    ndet = rng.randint(2, min(5, n))
    window = rng.randint(0, 6)

    # 生成时刻并保证任意长度 window 的闭窗内至多 10 个命中
    times = sorted(rng.randint(0, 12) for _ in range(n))
    for _ in range(100):
        if all(
            sum(1 for t2 in times if t1 <= t2 <= t1 + window) <= 10
            for t1 in times
        ):
            break
        times = sorted(rng.randint(0, 12) for _ in range(n))
    detectors = [rng.randrange(ndet) for _ in range(n)]
    weights = [rng.randint(1, 8) for _ in range(n)]

    # 约一半用例启用恢复期：每探测器 D ∈ [0, window]
    dead_times = None
    if seed % 2 == 1:
        dead_times = [rng.randint(0, window) for _ in range(ndet)]

    expected = brute_solve(times, detectors, weights, window,
                           dead_times=dead_times)
    got = solve(times, detectors, weights, window,
                dead_times=dead_times)

    assert got["best_score"] == expected["best_score"]
    assert got["best_events"] == expected["best_events"]
    assert got["total_count"] == expected["total_count"]
    assert _normalize(got["canonical"]) == _normalize(expected["canonical"])
    assert got["pair_count"] == expected["pair_count"]
    assert got["member_count"] == expected["member_count"]


def test_no_dead_times_matches_zero_dead_times_when_no_same_time_same_det():
    # 数据中不存在同时刻同探测器的两个命中时，D=0 与缺省结果一致
    # （严格大于 0 的额外约束只禁止同时刻双入组）。
    rng = random.Random(7)
    n, ndet, window = 10, 3, 4
    # 直接抽取唯一的 (时刻, 探测器) 组合
    combos = rng.sample(
        [(t, d) for t in range(12) for d in range(ndet)], n
    )
    combos.sort()
    times = [c[0] for c in combos]
    detectors = [c[1] for c in combos]
    weights = [rng.randint(1, 8) for _ in range(n)]
    a = solve(times, detectors, weights, window)
    b = solve(times, detectors, weights, window,
              dead_times=[0] * ndet)
    assert (a["best_score"], a["best_events"], a["total_count"],
            a["canonical"], a["pair_count"], a["member_count"]) == (
        b["best_score"], b["best_events"], b["total_count"],
        b["canonical"], b["pair_count"], b["member_count"],
    )


def test_sorted_ids_break_ties_by_time_then_id():
    # 同一时刻多命中时按标识排序，规范事件内成员标识升序
    raw = [
        ("z", 0, 0, 1),
        ("a", 0, 1, 1),
        ("m", 0, 2, 1),
    ]
    raw.sort(key=lambda r: (r[1], r[0]))
    ids = [r[0] for r in raw]
    times = [r[1] for r in raw]
    detectors = [r[2] for r in raw]
    weights = [r[3] for r in raw]
    got = solve(times, detectors, weights, 0, id_order=ids)
    assert len(got["canonical"]) == 1
    assert tuple(ids[j] for j in got["canonical"][0]) == ("a", "m", "z")


def test_window_too_dense_rejected():
    from app.solver import SolveError

    times = list(range(11))
    detectors = [k % 8 for k in range(11)]
    weights = [1] * 11
    with pytest.raises(SolveError):
        solve(times, detectors, weights, 100)
