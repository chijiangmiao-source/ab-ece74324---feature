"""最优命中分组求解器。

目标层级：
  1. 最大化已分组命中的可信度之和；
  2. 在满足 1 的前提下最少化事件数；
  3. 规范解按成员标识排序后的事件序列做词典序裁决。

恢复期（dead time）
------------------
调用方可为每个探测器给出非负整数恢复期 d（不超过符合窗 W），语义：

* 只有被归入事件的命中触发其探测器的恢复；噪声不触发；
* 同一探测器任意两个被分组命中的时刻差必须 **严格大于** d，差恰为 d
  视为冲突（d=0 时同刻同探测器的两个命中也不能分属两个事件）；
* 求解在选择事件时联合维护尚未结束的恢复占用，恢复限制可跨越多个
  候选事件；不存在“先求旧结果再删除冲突事件”的后处理。

未提供恢复期时完全保持原有语义（同探测器命中可任意分属不同事件）。

状态与复杂度
------------
约定：任意长度为 W 的闭窗内至多 10 个命中。按 (时刻, 标识) 排序。
第 i 步的状态由两个落在 open(i)=[t_i, t_i+W]（至多 10 个下标）上的
位掩码组成，打包为单个整数：

* ``cm``（committed）：已被早先创建的事件占用的未来命中；
* ``bm``（blocked）：因早先 *已处理* 的分组命中尚在恢复期而被阻塞的
  未来命中——分组命中 j 在自身被处理那一步把同探测器、时刻差 ≤ d 的
  后续命中投影为阻塞位（shadow[j]）；这些位随处理推进逐位释放，
  无需记录恢复历史。

阻塞只与未来命中有关：恢复期 d ≤ W，shadow[j] 的目标全部位于
open(j) 内，此后随扫描逐位释放，故两个平面始终不超过 10 位
（硬上界 3^10 态/步）。事件转移枚举空闲位子掩码，并按预算好的
``forbid[gm]`` 一次性排除与早先事件未来成员的同探测器恢复冲突。
整体不枚举任何完整分组方案或恢复历史；方案计数为 Python 任意精度
整数。前向/后向计数、规范见证与归属统计共享同一套转移与状态语义。
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
    且最大时刻与最小时刻之差不超过 window。事件形态与恢复期无关：
    同一事件内每探测器至多一个命中，恢复只约束分属不同事件的命中。
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


def _accumulate(dst: Dict[int, _Cell], nm: int, cell: _Cell) -> None:
    """把 cell 按两级目标并入 dst[nm]（同级则累加任意精度方案数）。"""
    old = dst.get(nm)
    if old is None:
        dst[nm] = cell
    elif _better(cell, old):
        dst[nm] = cell
    elif not _better(old, cell):
        dst[nm] = (old[0], old[1], old[2] + cell[2])


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

    ``dead_times`` 为 None 时保持无恢复期的原有语义；否则给出下标与
    探测器编号对齐的恢复期序列（调用层已校验为 0 ≤ d ≤ window 的
    非负整数）。
    """
    n = len(times)
    if n == 0:
        raise SolveError("no hits")

    recovery_enabled = dead_times is not None
    if recovery_enabled:
        dead = list(dead_times)
    else:
        dead = []

    groups = _sweep_groups(times, detectors, window)
    opens = _open_members(times, window)
    for members in opens:
        if len(members) > 10:
            raise SolveError(
                "coincidence window contains more than 10 hits"
            )

    bit = [1 << i for i in range(n)]
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
                m |= bit[j]
                gain += weights[j]
            ms.append(m)
            gains.append(gain)
        group_masks.append(ms)
        group_gains.append(gains)
        for j in opens[i]:
            om |= bit[j]
        open_masks.append(om)
    # gm -> gain 映射，配合子掩码枚举使用
    gain_maps = [
        dict(zip(group_masks[i], group_gains[i])) for i in range(n)
    ]

    # ---------- 恢复投影 ----------
    # shadow[i]：分组命中 i 在其被处理那一步需要阻塞的后续命中
    # （同探测器、0 < t_p - t_i ≤ d；闭区间，边界相等为冲突）。
    # forbid[i][gm]：以 i 为锚点创建事件 gm 时，读取状态 cm 中不得
    # 出现的位——早先事件已占用的未来命中 q，与 gm 某成员 p 同探测器
    # 且 |t_p - t_q| ≤ d。过去分组命中的恢复已具体化为 bm 阻塞位，
    # 空闲掩码会直接排除，无需进入 forbid。
    shadow = [0] * n
    forbid: List[Dict[int, int]] = [dict() for _ in range(n)]
    if recovery_enabled:
        for i in range(n):
            di = dead[detectors[i]]
            sh = 0
            for p in range(i + 1, n):
                if times[p] - times[i] > window:
                    break
                if detectors[p] == detectors[i] and times[p] - times[i] <= di:
                    sh |= bit[p]
            shadow[i] = sh
        for i in range(n):
            om = open_masks[i]
            fb_by_gm: Dict[int, int] = {}
            for gm in group_masks[i]:
                fb = 0
                rest = om & ~gm
                bits = rest
                while bits:
                    b = bits & -bits
                    q = b.bit_length() - 1
                    # q 是否与 gm 中某成员同探测器且时刻差 ≤ d_q/d_p
                    # （同探测器恢复期相同，用任一者的 d 即可）
                    gbits = gm
                    clash = False
                    while gbits:
                        gb = gbits & -gbits
                        p = gb.bit_length() - 1
                        if (
                            detectors[p] == detectors[q]
                            and abs(times[p] - times[q])
                            <= dead[detectors[q]]
                        ):
                            clash = True
                            break
                        gbits ^= gb
                    if clash:
                        fb |= b
                    bits ^= b
                fb_by_gm[gm] = fb
            forbid[i] = fb_by_gm

    # 打包状态：低 n 位为 cm，高 n 位为 bm
    SHIFT = n
    FMASK = (1 << n) - 1

    def pack(cm: int, bm: int) -> int:
        return cm | (bm << SHIFT)

    def iter_event_masks(cm: int, bm: int, i: int):
        """枚举状态 (cm, bm) 下以 i 为锚点可创建的事件 (gm, gain)。

        仅枚举“未占用/未阻塞的开放命中（锚点除外）”的子掩码，
        并按 forbid 排除与早先事件未来成员的跨事件恢复冲突。
        """
        bit_i = bit[i]
        free = open_masks[i] & ~(cm | bm) & ~bit_i
        s = free
        fb_table = forbid[i]
        while True:
            gm = s | bit_i
            gain = gain_maps[i].get(gm)
            if gain is not None and (cm & fb_table.get(gm, 0)) == 0:
                yield gm, gain
            if s == 0:
                break
            s = (s - 1) & free

    def step_transitions(i: int, state: int):
        """生成步骤 i 从打包状态 state 出发的全部最优性无关转移。

        返回 (kind, nxt, gain, events_added, gm)：
          kind="grouped-skip"：i 早先已被分组，处理时投影恢复；
          kind="noise-skip"：  i 被恢复阻塞，只能作噪声（不触发恢复）；
          kind="noise"：       空闲 i 作为噪声；
          kind="event"：       空闲 i 锚定新事件 gm。
        """
        cm = state & FMASK
        bm = (state >> SHIFT) & FMASK
        bit_i = bit[i]
        if cm & bit_i:
            nxt = pack(cm & ~bit_i, (bm | shadow[i]) & ~bit_i)
            yield "grouped-skip", nxt, 0, 0, 0
            return
        if bm & bit_i:
            nxt = pack(cm, bm & ~bit_i)
            yield "noise-skip", nxt, 0, 0, 0
            return
        # 空闲：噪声
        yield "noise", pack(cm, bm), 0, 0, 0
        # 以 i 为锚点创建事件：i 被分组，处理同时投影 shadow[i]
        for gm, gain in iter_event_masks(cm, bm, i):
            nxt = pack((cm | gm) & ~bit_i, (bm | shadow[i]) & ~bit_i)
            yield "event", nxt, gain, 1, gm

    # ---------- 前向 DP ----------
    # F[i][state]：处理完前 i 个命中、状态为 state 时的
    # (已锁定可信度和, 已锁定事件数, 方案数)。
    forward: List[Dict[int, _Cell]] = [{pack(0, 0): (0, 0, 1)}]
    for i in range(n):
        cur = forward[i]
        nxt: Dict[int, _Cell] = {}
        for state, (sc, ev, cnt) in cur.items():
            for kind, nstate, gain, added, _gm in step_transitions(i, state):
                _accumulate(nxt, nstate, (sc + gain, ev + added, cnt))
        forward.append(nxt)

    final = forward[n].get(pack(0, 0))
    if final is None:  # 理论上不可达（全部作噪声总是可行）
        raise SolveError("no feasible grouping")
    best_score, best_events, total_count = final

    # ---------- 后向 DP ----------
    # B[i][state]：从步骤 i、状态 state 出发，后缀可达的
    # (可信度和, 事件数, 方案数) 最优值。只需前向可达的状态。
    backward: List[Dict[int, _Cell]] = [{} for _ in range(n + 1)]
    backward[n] = {pack(0, 0): (0, 0, 1)}
    for i in range(n - 1, -1, -1):
        table: Dict[int, _Cell] = {}
        b_next = backward[i + 1]
        for state in forward[i]:
            best: Optional[_Cell] = None
            for kind, nstate, gain, added, _gm in step_transitions(i, state):
                suffix = b_next.get(nstate)
                if suffix is None:
                    continue
                cell = (suffix[0] + gain, suffix[1] + added, suffix[2])
                if best is None or _better(cell, best):
                    best = cell
                elif not _better(best, cell):
                    best = (best[0], best[1], best[2] + cell[2])
            if best is not None:
                table[state] = best
        backward[i] = table

    # ---------- 规范解（词典序裁决） ----------
    # 对每个可达状态 (i, state) 求后缀的“成员标识排序后的事件序列”
    # 的词典序最小值（仅限达到该状态最优 (可信度, 事件数) 的转移）。
    # 事件在锚点处加入，与后缀已排序序列做单点插入后比较。
    # 有标识序列时按标识（字符串元组）裁决，否则直接按下标（整数元组）。
    def _key(j: int):
        return id_order[j] if id_order is not None else j

    group_by_mask: List[Dict[int, Tuple[int, ...]]] = [
        {m: g for g, m in zip(groups[i], group_masks[i])}
        for i in range(n)
    ]
    gm_to_keys: List[Dict[int, Tuple[object, ...]]] = [
        {
            m: tuple(_key(j) for j in g)
            for g, m in zip(groups[i], group_masks[i])
        }
        for i in range(n)
    ]

    memo: Dict[Tuple[int, int], Tuple[Tuple[object, ...], ...]] = {}

    def canonical_suffix(
        i: int, state: int
    ) -> Tuple[Tuple[object, ...], ...]:
        if i == n:
            return ()
        key = (i, state)
        if key in memo:
            return memo[key]
        target = backward[i][state]
        best_seq: Optional[Tuple[Tuple[object, ...], ...]] = None

        for kind, nstate, gain, added, gm in step_transitions(i, state):
            suf = backward[i + 1].get(nstate)
            if suf is None:
                continue
            if gain + suf[0] != target[0] or added + suf[1] != target[1]:
                continue
            suffix_seq = canonical_suffix(i + 1, nstate)
            if kind != "event":
                cand = suffix_seq
            else:
                inserted = gm_to_keys[i][gm]
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

        assert best_seq is not None
        memo[key] = best_seq
        return best_seq

    initial = pack(0, 0)
    canonical_seq = canonical_suffix(0, initial)
    # 将比较键还原为下标元组
    if id_order is not None:
        key_to_index = {id_order[j]: j for j in range(n)}
        canonical_groups = [
            tuple(key_to_index[x] for x in gt) for gt in canonical_seq
        ]
    else:
        canonical_groups = [tuple(gt) for gt in canonical_seq]

    # ---------- 逐事件 / 逐对在最优解中的出现次数 ----------
    # 事件转移以 (i, 读取状态 state, gm) 唯一确定；前向方案数 ×
    # 后向方案数即采用该转移的最优完整方案数。前/后向计数与规范见证
    # 共享同一 (cm, bm) 恢复状态语义。
    member_count: Dict[int, int] = {}
    pair_count: Dict[Tuple[int, int], int] = {}
    for i in range(n):
        f_table = forward[i]
        b_next = backward[i + 1]
        for fstate, (fsc, fev, fcnt) in f_table.items():
            for kind, nstate, gain, added, gm in step_transitions(i, fstate):
                if kind != "event":
                    continue
                suf = b_next.get(nstate)
                if suf is None:
                    continue
                if fsc + gain + suf[0] != best_score:
                    continue
                if fev + added + suf[1] != best_events:
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
