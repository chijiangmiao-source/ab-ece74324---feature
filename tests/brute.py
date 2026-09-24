"""小规模暴力枚举参照，仅用于测试比对（会枚举完整分组方案）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from app.solver import _sweep_groups


def brute_solve(
    times: Sequence[int],
    detectors: Sequence[int],
    weights: Sequence[int],
    window: int,
    dead_times: Optional[Sequence[int]] = None,
) -> dict:
    n = len(times)
    groups = _sweep_groups(times, detectors, window)

    best_score = 0
    best_events = 0
    packings: List[Tuple[Tuple[int, ...], ...]] = []

    # 恢复期：used_times[d] 记录探测器 d 上所有已提交（含早先事件中
    # 下标更大的未来成员）的分组命中时刻；新事件成员必须与每个同探测器
    # 已提交时刻严格相差 > d（边界相等即冲突）。噪声不登记。
    def add_times(
        g: Tuple[int, ...], used_times: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        updated = {d: list(ts) for d, ts in used_times.items()}
        for j in g:
            d = detectors[j]
            updated.setdefault(d, []).append(times[j])
        return updated

    def feasible(
        g: Tuple[int, ...], used_times: Dict[int, List[int]]
    ) -> bool:
        if dead_times is None:
            return True
        for j in g:
            d = detectors[j]
            for t in used_times.get(d, ()):
                if abs(times[j] - t) <= dead_times[d]:
                    return False
        return True

    def rec(
        i: int,
        used_mask: int,
        used_times: Dict[int, List[int]],
        chosen: List[Tuple[int, ...]],
        score: int,
    ) -> None:
        nonlocal best_score, best_events, packings
        while i < n and (used_mask & (1 << i)):
            i += 1
        if i == n:
            events = len(chosen)
            if score > best_score:
                best_score, best_events = score, events
                packings = [tuple(chosen)]
            elif score == best_score:
                if events < best_events:
                    best_events = events
                    packings = [tuple(chosen)]
                elif events == best_events:
                    packings.append(tuple(chosen))
            return
        # i 作为噪声（不触发恢复）
        rec(i + 1, used_mask, used_times, chosen, score)
        # 以 i 为锚点建事件
        for g in groups[i]:
            gm = 0
            gain = 0
            for j in g:
                gm |= 1 << j
                gain += weights[j]
            if used_mask & gm:
                continue
            if not feasible(g, used_times):
                continue
            chosen.append(g)
            rec(
                i + 1,
                used_mask | gm,
                add_times(g, used_times),
                chosen,
                score + gain,
            )
            chosen.pop()

    rec(0, 0, {}, [], 0)

    count = len(packings)
    pair_count: Dict[Tuple[int, int], int] = {}
    member_count: Dict[int, int] = {}
    canonical = None
    for chosen in packings:
        seq = tuple(sorted(tuple(sorted(g)) for g in chosen))
        if canonical is None or seq < canonical:
            canonical = seq
        for g in chosen:
            for j in g:
                member_count[j] = member_count.get(j, 0) + 1
            for a in range(len(g)):
                for b in range(a + 1, len(g)):
                    key = (g[a], g[b])
                    pair_count[key] = pair_count.get(key, 0) + 1

    return {
        "best_score": best_score,
        "best_events": best_events,
        "total_count": count,
        "canonical": canonical or (),
        "pair_count": pair_count,
        "member_count": member_count,
    }
