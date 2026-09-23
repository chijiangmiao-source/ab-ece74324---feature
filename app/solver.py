"""最优命中分组求解器。

目标层级：
  1. 最大化已分组命中的可信度之和；
  2. 在满足 1 的前提下最少化事件数；
  3. 规范解按成员标识排序后的事件序列做词典序裁决。

复杂度保证
----------
约定：任意长度为 ``W`` 的闭窗内至多 10 个命中。按 (时刻, 标识) 排序后，
处理第 i 个命中时，所有“已被早先事件占用、尚未关窗”的命中都落在
``[t_i, t_i+W]`` 内，至多 10 个。因此 DP 状态是至多 10 位的位掩码
（每步至多 1024 个状态），整体为 O(n * 2^10 * 2^10) 上界内的扫描线
动态规划，不枚举任何完整分组方案。方案计数为 Python 任意精度整数。

探测器恢复期（dead time）
------------------------
``dead_times[d]`` 给出探测器 d 的非负恢复期 D_d（D_d ≤ W）。同一探测器
任意两个“归入事件”的命中时刻差必须严格大于 D_d（相等即冲突）；噪声命中
不触发恢复。启用后每个扫描状态维护两个互不相交的 ≤10 位掩码：

- ``occ``：已被早先事件占用的命中（归入事件，扫描到达时强制为噪声并
  触发自身恢复）；
- ``blk``：被某个已分组同探测器命中的恢复封锁的命中（强制为噪声，
  自身不触发恢复）。

建事件转移额外要求事件成员与 occ 中同探测器命中的时刻差严格大于各自
恢复期（预计算冲突掩码），于是恢复约束在“选择事件”时即被联合维护，
无需先求无恢复最优解再删除冲突事件。状态数上界为 3^10（每窗至多 10 个
命中，各自空闲 / 占用 / 封锁三态），仍不枚举任何完整分组或恢复历史。
未提供 ``dead_times`` 时所有恢复位置零，退化为单掩码引擎，语义与旧版
完全一致。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


class SolveError(ValueError):
    """输入数据违反约束（如符合窗内命中数超限）。"""


def _sweep_groups(
    times: Sequence[int],
    detectors: Sequence[int],
    window: int,
) -> List[List[Tuple[int, ...]]]:
    """返回每个锚点 i 可以创建的事件。

    事件以其最小下标成员为锚点；成员下标升序、探测器互不相同，
    且最大时刻与最小时刻之差不超过 window。
    """
    n = len(times)
    groups: List[List[Tuple[int, ...]]] = [[] for _ in range(n)]
    for i in range(n):
        ti = times[i]
        det_used = {detectors[i]}
        chosen: List[int] = [i]
        open_hits: List[int] = []
        for j in range(i + 1, n):
            if times[j] - ti > window:
                break
            open_hits.append(j)

        def gen(pos: int) -> None:
            if pos == len(open_hits):
                if len(chosen) >= 2:
                    groups[i].append(tuple(chosen))
                return
            # 不选 open_hits[pos]
            gen(pos + 1)
            # 选（探测器须不冲突）
            j = open_hits[pos]
            d = detectors[j]
            if d not in det_used:
                det_used.add(d)
                chosen.append(j)
                gen(pos + 1)
                chosen.pop()
                det_used.remove(d)

        gen(0)
    return groups


def _open_members(times: Sequence[int], window: int) -> List[List[int]]:
    """open(i)：时刻落在 [t_i, t_i+W] 闭窗内的下标（含 i），至多 10 个。"""
    n = len(times)
    result: List[List[int]] = []
    for i in range(n):
        members = [i]
        for j in range(i + 1, n):
            if times[j] - times[i] > window:
                break
            members.append(j)
        result.append(members)
    return result


# (最优可信度和, 最少事件数, 最优方案数)
_Cell = Tuple[int, int, int]


def _better(a: _Cell, b: Optional[_Cell]) -> bool:
    if b is None:
        return True
    if a[0] != b[0]:
        return a[0] > b[0]
    return a[1] < b[1]


def solve(
    times: Sequence[int],
    detectors: Sequence[int],
    weights: Sequence[int],
    window: int,
    id_order: Optional[Sequence[str]] = None,
    dead_times: Optional[Sequence[int]] = None,
) -> dict:
    """计算最优分组、规范解、任意精度方案数与成对归属。

    调用前输入已按 (时刻, 标识) 稳定排序；id_order 给出排序后的标识，
    用于规范解的词典序比较（若为 None 则按下标比较）。

    ``dead_times`` 为 None 时不启用恢复期（与旧语义完全一致）；否则须与
    探测器声明对齐，各值为不超过 window 的非负整数。
    """
    n = len(times)
    if n == 0:
        raise SolveError("no hits")
    # dead_times 由调用方保证与探测器声明对齐：下标即探测器编号，
    # 命中里出现的探测器编号都在其范围内。

    groups = _sweep_groups(times, detectors, window)
    opens = _open_members(times, window)
    for members in opens:
        if len(members) > 10:
            raise SolveError(
                "coincidence window contains more than 10 hits"
            )

    group_masks: List[List[int]] = []
    group_gains: List[List[int]] = []
    open_masks: List[int] = []
    for i in range(n):
        ms, gains = [], []
        om = 0
        for g in groups[i]:
            m = 0
            gain = 0
            for j in g:
                m |= 1 << j
                gain += weights[j]
            ms.append(m)
            gains.append(gain)
        group_masks.append(ms)
        group_gains.append(gains)
        for j in opens[i]:
            om |= 1 << j
        open_masks.append(om)
    # gm -> gain 映射，配合子掩码枚举使用
    gain_maps = [
        dict(zip(group_masks[i], group_gains[i])) for i in range(n)
    ]

    # ---------- 恢复期数据 ----------
    # rec_bits[i]：命中 i 归入事件后触发的恢复封锁位——同窗内、同探测器、
    # 时刻差 0 < t_j - t_i <= D_d 的未来下标 j（边界相等算冲突，故取 ≤）。
    # D_d ≤ W，故所有相关 j 都在 opens[i] 内。
    # block_maps[i][gm]：候选事件 gm 与 occ 中已占用命中发生恢复冲突的位：
    # 成员 j 与同探测器命中 k 的时刻差 |t_j-t_k| ≤ D_d 时 k 入掩码。
    n_det = len(set(detectors))
    if dead_times is None:
        rec_bits = [0] * n
        block_maps: List[Dict[int, int]] = [
            {m: 0 for m in group_masks[i]} for i in range(n)
        ]
    else:
        rec_bits = [0] * n
        # 先求每个 (i, j∈open(i)) 的同探测器恢复冲突位
        conflict_bits: List[Dict[int, int]] = [{} for _ in range(n)]
        for i in range(n):
            members = opens[i]
            cb = conflict_bits[i]
            for j in members:
                dj = detectors[j]
                dline = dead_times[dj]
                m = 0
                tj = times[j]
                for k in members:
                    if k != j and detectors[k] == dj and abs(times[k] - tj) <= dline:
                        m |= 1 << k
                cb[j] = m
            # rec_bits[i] 只含未来位
            rec_bits[i] = conflict_bits[i].get(i, 0)
        block_maps = []
        for i in range(n):
            cb = conflict_bits[i]
            bm: Dict[int, int] = {}
            for gm in group_masks[i]:
                blocked = 0
                m = gm
                while m:
                    bit = m & -m
                    blocked |= cb[bit.bit_length() - 1]
                    m ^= bit
                bm[gm] = blocked
            block_maps.append(bm)

    # 状态编码：单整数 key = occ | (blk << n)；occ、blk 互不相交。
    SH = n
    LOW = (1 << n) - 1

    def _key(occ: int, blk: int) -> int:
        return occ | (blk << SH)

    def iter_event_masks(occ: int, blk: int, i: int):
        """枚举在占用/封锁掩码下、以 i 为锚点可创建的事件 (gm, gain)。

        仅枚举“既未占用也未封锁的开放命中（锚点除外）”的子掩码，
        三态计数 3^(k-1)（k≤10）；另以恢复冲突掩码拒绝与已占用命中
        时差不超过各自恢复期的候选。
        """
        bit_i = 1 << i
        free = open_masks[i] & ~occ & ~blk & ~bit_i
        bm = block_maps[i]
        s = free
        while True:
            gm = s | bit_i
            gain = gain_maps[i].get(gm)
            if gain is not None and not (occ & bm[gm]):
                yield gm, gain
            if s == 0:
                break
            s = (s - 1) & free

    def _merge(nxt: Dict[int, _Cell], key: int, cell: _Cell) -> None:
        old = nxt.get(key)
        if old is None:
            nxt[key] = cell
        elif _better(cell, old):
            nxt[key] = cell
        elif not _better(old, cell):
            nxt[key] = (old[0], old[1], old[2] + cell[2])

    # ---------- 前向 DP ----------
    # F[i][key]：处理完前 i 个命中后，待决议占用/封锁掩码为 (occ, blk) 时的
    # (已锁定可信度和, 已锁定事件数, 方案数)。
    forward: List[Dict[int, _Cell]] = [{0: (0, 0, 1)}]
    for i in range(n):
        cur = forward[i]
        nxt: Dict[int, _Cell] = {}
        bit_i = 1 << i
        ri = rec_bits[i]
        for key, (sc, ev, cnt) in cur.items():
            occ = key & LOW
            blk = key >> SH
            if occ & bit_i:
                # i 已被早先事件占用：归入事件，强制噪声并触发恢复。
                # 与另一已占用命中冲突（理论上建事件时已拦住）则丢弃。
                if ri & occ:
                    continue
                nk = _key(occ & ~bit_i, (blk | ri) & ~bit_i)
                _merge(nxt, nk, (sc, ev, cnt))
                continue
            # 选择一：i 作为噪声（被恢复封锁时同样走此路，且不触发恢复）
            nk = _key(occ & ~bit_i, blk & ~bit_i)
            _merge(nxt, nk, (sc, ev, cnt))
            if blk & bit_i:
                continue
            # 选择二：以 i 为锚点创建事件（枚举空闲开放位的子掩码）
            for gm, gain in iter_event_masks(occ, blk, i):
                # 锚点 i 归入事件，立即触发其恢复；与 occ 的冲突已由
                # block_maps 拦住（含成员 i 自身的同探测器冲突）。
                nk = _key((occ | gm) & ~bit_i, (blk | ri) & ~bit_i)
                _merge(nxt, nk, (sc + gain, ev + 1, cnt))
        forward.append(nxt)

    final = forward[n][0]
    best_score, best_events, total_count = final

    # ---------- 后向 DP ----------
    # B[i][key]：从步骤 i、掩码 (occ, blk) 出发，后缀可达的
    # (可信度和, 事件数, 方案数) 最优值。只需前向可达的掩码（每步 ≤ 3^10）。
    backward: List[Dict[int, _Cell]] = [{} for _ in range(n + 1)]
    backward[n] = {0: (0, 0, 1)}
    for i in range(n - 1, -1, -1):
        table: Dict[int, _Cell] = {}
        bit_i = 1 << i
        ri = rec_bits[i]
        b_next = backward[i + 1]
        for key in forward[i]:
            occ = key & LOW
            blk = key >> SH
            if occ & bit_i:
                if ri & occ:
                    continue
                nk = _key(occ & ~bit_i, (blk | ri) & ~bit_i)
                suffix = b_next.get(nk)
                if suffix is not None:
                    table[key] = suffix
                continue
            # 选择一：噪声（被封锁时为强制转移）
            best: Optional[_Cell] = b_next.get(
                _key(occ & ~bit_i, blk & ~bit_i)
            )
            if not (blk & bit_i):
                # 选择二：以 i 为锚点创建事件（枚举空闲开放位的子掩码）
                for gm, gain in iter_event_masks(occ, blk, i):
                    nk = _key((occ | gm) & ~bit_i, (blk | ri) & ~bit_i)
                    suffix = b_next.get(nk)
                    if suffix is None:
                        continue
                    cell = (suffix[0] + gain, suffix[1] + 1, suffix[2])
                    if best is None or _better(cell, best):
                        best = cell
                    elif not _better(best, cell):
                        best = (best[0], best[1], best[2] + cell[2])
            if best is not None:
                table[key] = best
        backward[i] = table

    # ---------- 规范解（词典序裁决） ----------
    # 对每个可达状态 (i, occ, blk) 求后缀的“成员标识排序后的事件序列”
    # 的词典序最小值（仅限达到该状态最优 (可信度, 事件数) 的转移）。
    # 事件在锚点处加入，与后缀已排序序列做单点插入后比较。
    # 预生成每个候选事件的比较键（按掩码索引）。
    # 有标识序列时按标识（字符串元组）裁决，否则直接按下标（整数元组）。
    def _idkey(j: int):
        return id_order[j] if id_order is not None else j

    group_by_mask: List[Dict[int, Tuple[int, ...]]] = [
        {m: g for g, m in zip(groups[i], group_masks[i])}
        for i in range(n)
    ]
    gm_to_keys: List[Dict[int, Tuple[object, ...]]] = [
        {
            m: tuple(_idkey(j) for j in g)
            for g, m in zip(groups[i], group_masks[i])
        }
        for i in range(n)
    ]

    memo: Dict[Tuple[int, int], Tuple[Tuple[object, ...], ...]] = {}

    def canonical_suffix(
        i: int, key: int
    ) -> Tuple[Tuple[object, ...], ...]:
        if i == n:
            return ()
        memo_key = (i, key)
        if memo_key in memo:
            return memo[memo_key]
        occ = key & LOW
        blk = key >> SH
        target = backward[i][key]
        bit_i = 1 << i
        ri = rec_bits[i]
        best_seq: Optional[Tuple[Tuple[object, ...], ...]] = None

        def consider(
            gain: int,
            events_added: int,
            nk: int,
            inserted: Optional[Tuple[object, ...]],
        ) -> None:
            nonlocal best_seq
            suf = backward[i + 1].get(nk)
            if suf is None:
                return
            if gain + suf[0] != target[0] or events_added + suf[1] != target[1]:
                return
            suffix_seq = canonical_suffix(i + 1, nk)
            if inserted is None:
                cand = suffix_seq
            else:
                cand_list = list(suffix_seq)
                lo, hi = 0, len(cand_list)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if cand_list[mid] < inserted:
                        lo = mid + 1
                    else:
                        hi = mid
                cand_list.insert(lo, inserted)
                cand = tuple(cand_list)
            if best_seq is None or cand < best_seq:
                best_seq = cand

        if occ & bit_i:
            # 占用命中：唯一强制转移，触发恢复
            if not (ri & occ):
                nk = _key(occ & ~bit_i, (blk | ri) & ~bit_i)
                if backward[i + 1].get(nk) is not None:
                    consider(0, 0, nk, None)
        else:
            # 噪声（被封锁时为强制转移）
            consider(0, 0, _key(occ & ~bit_i, blk & ~bit_i), None)
            if not (blk & bit_i):
                for gm, gain in iter_event_masks(occ, blk, i):
                    nk = _key((occ | gm) & ~bit_i, (blk | ri) & ~bit_i)
                    consider(gain, 1, nk, gm_to_keys[i][gm])

        assert best_seq is not None
        memo[memo_key] = best_seq
        return best_seq

    canonical_seq = canonical_suffix(0, 0)
    # 将比较键还原为下标元组
    if id_order is not None:
        key_to_index = {id_order[j]: j for j in range(n)}
        canonical_groups = [
            tuple(key_to_index[x] for x in gt) for gt in canonical_seq
        ]
    else:
        canonical_groups = [tuple(gt) for gt in canonical_seq]

    # ---------- 逐事件 / 逐对在最优解中的出现次数 ----------
    # 前向方案数 × 后向方案数，前/后向共享同一 (occ, blk) 状态语义。
    member_count: Dict[int, int] = {}
    pair_count: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        f_table = forward[i]
        b_next = backward[i + 1]
        bit_i = 1 << i
        ri = rec_bits[i]
        for fkey, (fsc, fev, fcnt) in f_table.items():
            occ = fkey & LOW
            blk = fkey >> SH
            if occ & bit_i or blk & bit_i:
                continue
            for gm, gain in iter_event_masks(occ, blk, i):
                nk = _key((occ | gm) & ~bit_i, (blk | ri) & ~bit_i)
                suf = b_next.get(nk)
                if suf is None:
                    continue
                if fsc + gain + suf[0] != best_score:
                    continue
                if fev + 1 + suf[1] != best_events:
                    continue
                g = group_by_mask[i][gm]
                ways = fcnt * suf[2]
                for j in g:
                    member_count[j] = member_count.get(j, 0) + ways
                for a_idx in range(len(g)):
                    for b_idx in range(a_idx + 1, len(g)):
                        a, b = g[a_idx], g[b_idx]
                        key = (a, b)
                        pair_count[key] = pair_count.get(key, 0) + ways

    return {
        "best_score": best_score,
        "best_events": best_events,
        "total_count": total_count,
        "canonical": canonical_groups,
        "member_count": member_count,
        "pair_count": pair_count,
        "groups": groups,
    }
