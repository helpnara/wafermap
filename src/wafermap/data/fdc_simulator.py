"""FDC 시뮬레이터 — 가상 팹의 공정 이력과 설비 센서 데이터를 생성한다.

무엇을: lot/wafer 타임라인, 스텝별 설비·챔버 라우팅, FDC 파라미터 값, 그리고
        "어떤 웨이퍼가 어떤 불량 패턴을 갖는가"를 함께 만들어 낸다. 정답지
        (ground_truth)도 동시에 기록한다.

어떻게 — 핵심은 **이상 사건(excursion) 주도 생성**이다:
        웨이퍼마다 독립적으로 불량 패턴을 뽑지 않는다. 대신 "ETCH-B/ch3 챔버가
        3월 5일~9일 사이에 이상했다"는 사건을 만들고, 그 시간·그 챔버를 지나간
        웨이퍼들이 패턴을 갖게 한다.

왜 이 방식인가: 웨이퍼별 독립 추출로 만들면 불량이 시간·설비에 고르게 흩어져서
        ① SPC 관리도(M2)에 잡힐 이상 구간이 존재하지 않고
        ② 커미널리티 분석(M3)에서 특정 설비가 유의하게 나올 수가 없다.
        즉 분석 파이프라인 전체가 검증 불가능해진다. 실제 팹의 불량은 항상
        특정 설비·특정 기간에 몰려서 발생하며, 그 구조를 재현해야 의미가 있다.

참고: docs/00_design.md §2.2(합성 전략), §2.5(공정 플로우), §2.6(스키마)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wafermap.features.trace_feat import trace_features
from wafermap.config import (
    CAUSE_RULES,
    CauseRule,
    DEFAULT_DIFFICULTY,
    DEFAULT_SEED,
    PRODUCT_NAME,
    PROBE_CARDS,
    DRIFT_CAUSE_RULES,
    INTERACTION_CAUSE_RULES,
    INTERACTION_INDUCED_SHARE,
    DRIFT_INDUCED_SHARE,
    SPIKE_CAUSE_RULES,
    SPIKE_INDUCED_SHARE,
    TOUCHDOWNS_PER_WAFER,
    PROCESS_STEPS,
    STEPS_BY_ID,
    TEST_CAUSE_RULES,
    TEST_INDUCED_SHARE,
    TEST_STEP,
    TESTERS,
    TECH_NODE,
    WAFERS_PER_LOT,
    WM811K_LABEL_COUNTS,
    ParamSpec,
    ProcessStep,
    SimulationDifficulty,
)

#: 스텝 간 평균 소요 시간(시간). 실제 팹의 사이클 타임을 단순화한 값이다.
#: 웨이퍼 1장 처리 시계열의 시점 수. 처리 시간 60초를 1초 간격으로 본다.
#: 왜 촘촘해야 하나: 2~3초짜리 스파이크를 보려면 그보다 잘게 잘라야 한다.
#: 2.5초 간격이면 3초 스파이크가 한 점에 뭉개져 노이즈와 구분되지 않는다.
TRACE_POINTS = 60
TRACE_SECONDS = 60.0

STEP_INTERVAL_HOURS = 6.0
#: 마지막 스텝 이후 EDS 테스트까지의 지연
EDS_DELAY_HOURS = 12.0

DEFAULT_RECIPE = "RCP-STD-01"
WRONG_RECIPE = "RCP-ALT-07"  # Near-full 유발 시 오적용되는 레시피


@dataclass(frozen=True)
class Excursion:
    """이상 사건 1건 — '언제, 어느 챔버가, 어떤 패턴을 만들었는가'.

    Attributes:
        excursion_id: 식별자
        pattern: 유발되는 불량 패턴
        step_id: 원인 스텝 (Random처럼 원인 설비가 없으면 None)
        chamber_id: 원인 챔버 (`ETCH-B/ch3` 형식)
        t_start, t_end: 이상 구간
        attack_rate: 해당 챔버를 지난 웨이퍼 중 실제로 불량이 된 비율
        is_test_induced: 공정이 아니라 **검사 설비**(프로브 카드)가 원인인 사건인가.
            이 경우 `chamber_id`는 프로브 카드 ID이고, 시간 창은 공정 투입 시각이
            아니라 **EDS 검사 시각** 기준이다.
    """

    excursion_id: str
    pattern: str
    step_id: str | None
    chamber_id: str | None
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    attack_rate: float
    is_test_induced: bool = False


@dataclass
class SimulationResult:
    """시뮬레이션 산출물 묶음."""

    wafer_master: pd.DataFrame
    fdc_summary: pd.DataFrame
    fdc_trace: pd.DataFrame
    ground_truth: pd.DataFrame
    excursions: list[Excursion] = field(default_factory=list)
    #: 의도적으로 주입한 교락 쌍 (stepA, chamberA, stepB, chamberB).
    #: M3의 층화 검정이 "진짜 원인"과 "같이 흘렀을 뿐인 설비"를 갈라내는지 채점할 때 쓴다.
    affinities: list[tuple[str, str, str, str]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────
# 1. 타임라인과 라우팅
# ──────────────────────────────────────────────────────────────────────────


def _build_lots(
    n_wafers: int, n_days: int, start: pd.Timestamp, rng: np.random.Generator
) -> pd.DataFrame:
    """lot/wafer 타임라인을 만든다.

    무엇을: n_wafers를 25장씩 lot으로 묶고, lot 투입 시각을 기간에 균등 배치한다.
    왜 lot 단위인가: 실제 팹은 lot 단위로 설비에 배정된다(§2.5). 웨이퍼 하나하나가
        다른 챔버로 흩어지지 않기 때문에, 커미널리티 분석의 표본 단위도 사실상
        lot이 된다. 이 구조를 어기면 통계적 검정력이 비현실적으로 부풀려진다.
    """
    n_lots = int(np.ceil(n_wafers / WAFERS_PER_LOT))
    total_hours = n_days * 24.0

    # lot 투입 시각: 균등 간격 + 지터(실제 라인은 정확히 등간격으로 투입되지 않는다)
    base = np.linspace(0, total_hours, n_lots, endpoint=False)
    jitter = rng.uniform(0, total_hours / max(n_lots, 1), n_lots)
    lot_hours = np.sort(base + jitter)

    rows = []
    wafer_seq = 0
    for i, h in enumerate(lot_hours):
        lot_id = f"LOT{i + 1:05d}"
        fab_in = start + pd.Timedelta(hours=float(h))
        for slot in range(1, WAFERS_PER_LOT + 1):
            if wafer_seq >= n_wafers:
                break
            rows.append(
                {
                    "wafer_id": f"{lot_id}-W{slot:02d}",
                    "lot_id": lot_id,
                    "slot_no": slot,
                    "fab_in_time": fab_in,
                }
            )
            wafer_seq += 1

    return pd.DataFrame(rows)


def _build_routing(
    lots: np.ndarray, rng: np.random.Generator, difficulty: SimulationDifficulty
) -> tuple[pd.DataFrame, list[tuple[str, str, str, str]]]:
    """lot별로 각 스텝에서 사용할 챔버를 배정한다.

    무엇을: (lot_id × step_id) → chamber_id 배정표를 만든다.
    어떻게: 기본은 무작위 배정이되, **디스패치 친화(affinity)** 를 일부 주입한다.
            즉 특정 챔버 조합이 우연보다 자주 함께 흘러가게 만든다.
    왜 친화를 넣는가: 이것이 곧 **교락(confounding)** 이다. ETCH-B/ch3가 진짜 원인인데
            CMP-01/ch2가 늘 같이 흘렀다면, 단순 교차표로는 두 설비가 똑같이 유의하게
            나온다. M3의 CMH 층화 검정이 이 둘을 갈라낼 수 있어야 하며, 그러려면
            데이터에 교락이 실제로 존재해야 한다.

    Returns:
        (라우팅 DataFrame, 주입한 친화 쌍 목록)
    """
    n_lots = len(lots)
    routing: dict[str, np.ndarray] = {}

    for step in PROCESS_STEPS:
        chambers = np.array(step.chamber_ids)
        routing[step.step_id] = rng.choice(chambers, size=n_lots)

    # 디스패치 친화: 서로 다른 두 스텝에서 챔버 한 쌍을 골라 함께 흐르게 만든다
    affinities: list[tuple[str, str, str, str]] = []
    multi_chamber_steps = [s for s in PROCESS_STEPS if len(s.chamber_ids) >= 3]
    n_affinity = min(3, len(multi_chamber_steps) - 1)

    for _ in range(n_affinity):
        step_a, step_b = rng.choice(len(multi_chamber_steps), size=2, replace=False)
        sa, sb = multi_chamber_steps[step_a], multi_chamber_steps[step_b]
        ch_a = str(rng.choice(sa.chamber_ids))
        ch_b = str(rng.choice(sb.chamber_ids))

        # sa에서 ch_a를 쓴 lot 중 일부를 sb에서도 ch_b로 몰아준다
        hit = routing[sa.step_id] == ch_a
        move = hit & (rng.random(n_lots) < difficulty.confound_rate * 3.0)
        routing[sb.step_id] = np.where(move, ch_b, routing[sb.step_id])
        affinities.append((sa.step_id, ch_a, sb.step_id, ch_b))

    df = pd.DataFrame({"lot_id": lots})
    for step_id, chambers in routing.items():
        df[step_id] = chambers
    return df, affinities


def _build_test_routing(
    lots: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """lot별 검사 셀(테스터 + 프로브 카드)을 배정한다.

    왜 lot 단위인가: EDS도 lot 단위로 검사 셀에 걸린다. 공정 라우팅과 같은 이유로
        표본의 독립성이 lot에서 끊긴다.

    왜 테스터와 카드를 따로 뽑나 ★: 프로브 카드는 **소모품이라 테스터 사이를 옮겨
        다닌다.** 둘을 묶어 배정하면 "ATE-02가 문제"와 "PC-05가 문제"가 항상 같이
        움직여 영원히 구분되지 않는다. 독립으로 뽑아야 분석이 둘을 가려낼 수 있다.
    """
    n_lots = len(lots)
    return pd.DataFrame(
        {
            "lot_id": lots,
            "tester_id": rng.choice(np.array(TESTERS), size=n_lots),
            "probe_card_id": rng.choice(np.array(PROBE_CARDS), size=n_lots),
        }
    )


def _touchdown_counts(
    card_of_wafer: np.ndarray, pm_limit: float, rng: np.random.Generator
) -> np.ndarray:
    """프로브 카드별 누적 터치다운을 계산한다 (세정 시 0으로 리셋).

    터치다운 1회 = 동시 측정 site 묶음 1회 접촉. 웨이퍼 1장이면
    `die 수 ÷ 동시측정 수` = TOUCHDOWNS_PER_WAFER 회만큼 니들이 pad를 찍는다.

    왜 별도 함수인가 ★: 일반 카운터(`_counter_values`)는 처리 **순번**을 그대로
        사용량으로 쓴다. 그러면 웨이퍼 6,000장을 돌려도 카운터가 6,000까지밖에
        안 올라가 25만 회 세정 주기에 근처도 못 간다. 터치다운은 웨이퍼당 99회씩
        쌓이므로 증가율을 제대로 반영해야 마모 구간이 데이터에 나타난다.
    """
    counts = np.zeros(len(card_of_wafer), dtype=float)
    for card in np.unique(card_of_wafer):
        sel = card_of_wafer == card
        cumulative = (np.arange(int(sel.sum())) + 1) * TOUCHDOWNS_PER_WAFER
        counts[sel] = cumulative % pm_limit
    # 계수 오차를 조금 섞되 음수는 만들지 않는다 — 누적 횟수가 음수일 수는 없다
    noisy = counts + rng.normal(0.0, pm_limit * 0.005, len(card_of_wafer))
    return np.clip(noisy, 0.0, None)


# ──────────────────────────────────────────────────────────────────────────
# 2. 이상 사건 생성과 패턴 배정
# ──────────────────────────────────────────────────────────────────────────


def _target_counts(n_wafers: int) -> dict[str, int]:
    """WM-811K 실측 라벨 비율을 목표 장수로 환산한다.

    왜: 합성 데이터가 실데이터와 다른 클래스 비율을 가지면, 합성에서 잘 나온
        macro-F1이 실데이터에서 무너진다. 특히 Near-full은 전체의 0.086%뿐이라
        이 극단적 불균형 자체가 모델링의 핵심 난이도다(§M1).
    """
    total = sum(WM811K_LABEL_COUNTS.values())
    counts = {
        p: int(round(n_wafers * c / total))
        for p, c in WM811K_LABEL_COUNTS.items()
        if p != "none"
    }
    # 희소 클래스가 0장이 되면 학습 자체가 불가능하므로 최소 1장은 보장한다
    return {p: max(c, 1) for p, c in counts.items()}


def _infect(
    *,
    pattern: pd.Series,
    severity: pd.Series,
    mechanism_out: np.ndarray,
    label: str,
    target: int,
    unit_of_wafer: np.ndarray,
    units: tuple[str, ...],
    times: np.ndarray,
    t_min: pd.Timestamp,
    span_hours: float,
    step_id: str,
    mechanism: str,
    rng: np.random.Generator,
    difficulty: SimulationDifficulty,
    excursions: list[Excursion],
) -> int:
    """설비 하나 × 시간 창 하나를 골라 그 안의 웨이퍼를 감염시킨다.

세 가지 경로가 **완전히 같은 절차**를 쓰되 단위·시각·기전만 다르다.
      · process : 단위=챔버, 시각=공정 투입 시각, 파라미터 평균이 지속 이탈
      · test    : 단위=프로브 카드, 시각=EDS 검사 시각
      · spike   : 단위=챔버, 시각=공정 투입 시각, **순간만** 튄다(평균 거의 불변)

    왜 시각 기준이 다른가 ★: 검사 이상은 검사할 때 일어난다. 공정 이상은 공정
        투입 때 일어나고 EDS에서는 66시간 뒤에 보인다(§M3 발견 3). 즉 SPC를
        EDS 시각으로 돌리면 **검사 기인 이상만 제 시각에 보이고 공정 기인은 늦게
        보인다.** 이 차이 자체가 둘을 가르는 단서가 된다.

    Returns:
        실제로 배정한 웨이퍼 수
    """
    assigned = 0
    guard = 0
    lo, hi = difficulty.severity_range

    while assigned < target and guard < 400:
        guard += 1
        unit = str(rng.choice(np.array(units)))

        win_hours = span_hours * rng.uniform(0.03, 0.12)
        start_off = rng.uniform(0, max(span_hours - win_hours, 1e-6))
        t0 = t_min + pd.Timedelta(hours=float(start_off))
        t1 = t0 + pd.Timedelta(hours=float(win_hours))

        in_window = (times >= np.datetime64(t0)) & (times <= np.datetime64(t1))
        eligible = np.where(
            (unit_of_wafer == unit) & in_window & (pattern.to_numpy() == "none")
        )[0]
        if not len(eligible):
            continue

        attack = rng.uniform(0.45, 0.85)
        hit = eligible[rng.random(len(eligible)) < attack]
        if not len(hit):
            continue
        hit = hit[: target - assigned]

        pattern.iloc[hit] = label
        severity.iloc[hit] = rng.uniform(lo, hi, size=len(hit))
        mechanism_out[hit] = mechanism
        assigned += len(hit)

        excursions.append(
            Excursion(
                excursion_id=f"EXC-{len(excursions) + 1:04d}",
                pattern=label,
                step_id=step_id,
                chamber_id=unit,
                t_start=t0,
                t_end=t1,
                attack_rate=float(attack),
                is_test_induced=mechanism == "test",
            )
        )

    return assigned


def _assign_patterns(
    wafers: pd.DataFrame,
    routing: pd.DataFrame,
    test_routing: pd.DataFrame,
    eds_time: np.ndarray,
    rng: np.random.Generator,
    difficulty: SimulationDifficulty,
) -> tuple[pd.Series, pd.Series, np.ndarray, list[Excursion]]:
    """이상 사건을 만들고, 각 웨이퍼에 불량 패턴을 배정한다.

    어떻게:
        1. 패턴별 목표 장수를 계산한다 (WM-811K 실측 비율).
        2. 목표를 **공정 기인 / 검사 기인으로 나눈다** (`TEST_INDUCED_SHARE`).
           같은 패턴이 두 경로로 생기며, 맵만 봐서는 구분되지 않는다.
        3. 각 경로마다 이상 사건을 반복 생성해 목표를 채운다.
        4. Random 패턴은 원인 설비가 없으므로 시간·설비와 무관하게 흩뿌린다.
        5. 끝까지 배정되지 않은 웨이퍼는 'none'.

    Returns:
        (패턴 Series, 심각도 Series, 기전 배열, 이상 사건 목록)
        기전 배열의 값: "none" | "process" | "test" | "spike"
    """
    n = len(wafers)
    pattern = pd.Series(["none"] * n, index=wafers.index)
    severity = pd.Series(np.nan, index=wafers.index)
    mechanism_arr = np.full(n, "none", dtype=object)
    excursions: list[Excursion] = []

    lot_to_row = wafers["lot_id"].to_numpy()
    t = wafers["fab_in_time"].to_numpy()
    t_min, t_max = wafers["fab_in_time"].min(), wafers["fab_in_time"].max()
    span_hours = (t_max - t_min).total_seconds() / 3600.0

    eds_min = pd.Timestamp(eds_time.min())

    targets = _target_counts(n)
    # 희소 패턴부터 배정한다. 흔한 패턴이 웨이퍼를 먼저 차지해 버리면
    # Near-full(전체의 0.09%)처럼 드문 패턴이 목표 장수를 못 채운다.
    order = sorted(targets, key=lambda p: targets[p])

    route_map = routing.set_index("lot_id")
    card_of_wafer = (
        test_routing.set_index("lot_id")["probe_card_id"].reindex(lot_to_row).to_numpy()
    )

    for pat in order:
        rule = CAUSE_RULES[pat]
        target = targets[pat]

        # ── 검사 기인 몫을 먼저 떼어 낸다 ───────────────────────────────
        share = TEST_INDUCED_SHARE.get(pat, 0.0)
        n_test = int(round(target * share)) if pat in TEST_CAUSE_RULES else 0
        n_process = target - n_test

        if n_test:
            n_test -= _infect(
                pattern=pattern, severity=severity, mechanism_out=mechanism_arr,
                label=pat, target=n_test,
                unit_of_wafer=card_of_wafer, units=TEST_STEP.chamber_ids,
                times=eds_time, t_min=eds_min, span_hours=span_hours,
                step_id=TEST_STEP.step_id, mechanism="test",
                rng=rng, difficulty=difficulty, excursions=excursions,
            )
            # 검사 기인이 목표를 못 채웠으면 공정 기인이 그만큼 더 가져간다
            n_process += n_test

        if rule.step_id is None:
            # 원인 설비 없음(Random) — 시간·챔버와 무관하게 산발 배정.
            # 커미널리티 분석에서 '유의한 설비 없음'이 정답이 되는 케이스다.
            free = np.where(pattern.to_numpy() == "none")[0]
            if not len(free):
                continue
            lo, hi = difficulty.severity_range
            pick = rng.choice(free, size=min(n_process, len(free)), replace=False)
            pattern.iloc[pick] = pat
            severity.iloc[pick] = rng.uniform(lo, hi, size=len(pick))
            mechanism_arr[pick] = "process"
            excursions.append(
                Excursion(
                    excursion_id=f"EXC-{len(excursions) + 1:04d}",
                    pattern=pat, step_id=None, chamber_id=None,
                    t_start=t_min, t_end=t_max, attack_rate=float("nan"),
                )
            )
            continue

        step = STEPS_BY_ID[rule.step_id]
        unit_of_wafer = route_map[step.step_id].reindex(lot_to_row).to_numpy()

        # ── 스파이크 기인 몫 — 같은 스텝·같은 챔버지만 **이상의 모양**이 다르다 ──
        # 지속형은 평균이 통째로 밀리고, 스파이크형은 2~3초만 튄다.
        # 요약통계 분석이 후자를 놓치는지 측정하려면 둘 다 데이터에 있어야 한다.
        n_spike = (
            int(round(n_process * SPIKE_INDUCED_SHARE.get(pat, 0.0)))
            if pat in SPIKE_CAUSE_RULES
            else 0
        )
        # ── 교호작용 기인 ★ ────────────────────────────────────────────
        # 시간 창이 아니라 **설비 조합**으로 감염된다. 그 조합에서 앞뒤 공정의
        # 파라미터가 같은 방향으로 치우치기 쉽기 때문이다(디스패치·레시피 관행).
        n_inter = (
            int(round(n_process * INTERACTION_INDUCED_SHARE.get(pat, 0.0)))
            if pat in INTERACTION_CAUSE_RULES
            else 0
        )
        if n_inter:
            irule = INTERACTION_CAUSE_RULES[pat]
            chamber_a = route_map[irule.step_a].reindex(lot_to_row).to_numpy()
            chamber_b = route_map[irule.step_b].reindex(lot_to_row).to_numpy()
            on_pair = (
                (chamber_a == irule.equip_pair[0]) & (chamber_b == irule.equip_pair[1])
            )
            free = np.where(on_pair & (pattern.to_numpy() == "none"))[0]
            if len(free):
                lo, hi = difficulty.severity_range
                take = min(n_inter, len(free))
                pick = rng.choice(free, size=take, replace=False)
                pattern.iloc[pick] = pat
                severity.iloc[pick] = rng.uniform(lo, hi, size=take)
                mechanism_arr[pick] = "interaction"
                excursions.append(
                    Excursion(
                        excursion_id=f"EXC-{len(excursions) + 1:04d}",
                        pattern=pat,
                        step_id=f"{irule.step_a}×{irule.step_b}",
                        chamber_id=f"{irule.equip_pair[0]}×{irule.equip_pair[1]}",
                        t_start=t_min, t_end=t_max, attack_rate=float("nan"),
                    )
                )
                n_process -= take

        n_drift = (
            int(round(n_process * DRIFT_INDUCED_SHARE.get(pat, 0.0)))
            if pat in DRIFT_CAUSE_RULES
            else 0
        )
        for count, rules, mech in (
            (n_spike, SPIKE_CAUSE_RULES, "spike"),
            (n_drift, DRIFT_CAUSE_RULES, "drift"),
        ):
            if not count:
                continue
            sub_rule = rules[pat]
            sub_step = STEPS_BY_ID[sub_rule.step_id]
            leftover = count - _infect(
                pattern=pattern, severity=severity, mechanism_out=mechanism_arr,
                label=pat, target=count,
                unit_of_wafer=route_map[sub_step.step_id].reindex(lot_to_row).to_numpy(),
                units=sub_step.chamber_ids,
                times=t, t_min=t_min, span_hours=span_hours,
                step_id=sub_step.step_id, mechanism=mech,
                rng=rng, difficulty=difficulty, excursions=excursions,
            )
            n_process = n_process - count + leftover

        _infect(
            pattern=pattern, severity=severity, mechanism_out=mechanism_arr,
            label=pat, target=n_process,
            unit_of_wafer=unit_of_wafer,
            units=step.chamber_ids,
            times=t, t_min=t_min, span_hours=span_hours,
            step_id=step.step_id, mechanism="process",
            rng=rng, difficulty=difficulty, excursions=excursions,
        )

    return pattern, severity, mechanism_arr, excursions


# ──────────────────────────────────────────────────────────────────────────
# 3. FDC 파라미터 값 생성
# ──────────────────────────────────────────────────────────────────────────


def _equipment_bias(
    chambers: tuple[str, ...], param: ParamSpec, rng: np.random.Generator, scale: float
) -> dict[str, float]:
    """챔버별 고유 baseline 편차를 만든다.

    왜: 같은 모델의 설비라도 챔버마다 미세한 오프셋이 있다(chamber matching 이슈).
        이 편차가 없으면 "파라미터가 규격을 벗어났다 = 원인이다"가 너무 쉽게 성립한다.
        실제로는 정상 챔버도 서로 다른 값을 갖기 때문에, 분석은 절대값이 아니라
        **평소 대비 변화**를 봐야 한다. 그 난이도를 데이터에 넣는 장치다.
    """
    if param.kind == "counter":
        return {c: 0.0 for c in chambers}
    return {c: float(rng.normal(0.0, param.sigma * scale)) for c in chambers}


def _counter_values(
    n: int, param: ParamSpec, order: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """소모품 누적 사용량을 톱니파로 만든다 (PM 시 0으로 리셋).

    무엇을: edge_ring_rf_hours, pad_life 처럼 쓸수록 증가하다가 정비 때 초기화되는 값.
    어떻게: 챔버 내 처리 순서를 PM 주기로 나눈 나머지를 사용량으로 쓴다.
    왜:    이 값이 '시간에 따라 단조 증가하다 급락'하는 모양이어야, M3의 트렌드 분석과
           "PM 주기를 400h에서 300h로 단축" 같은 개선안(§M5)이 데이터로 뒷받침된다.
    """
    period = param.nominal
    usage = (order % period).astype(float)
    return usage + rng.normal(0.0, period * 0.01, n)


def _simulate_step_fdc(
    step: ProcessStep,
    wafers: pd.DataFrame,
    chamber_of_wafer: np.ndarray,
    pattern: np.ndarray,
    severity: np.ndarray,
    has_signal: np.ndarray,
    is_false_positive: np.ndarray,
    rng: np.random.Generator,
    difficulty: SimulationDifficulty,
    *,
    rules: dict[str, CauseRule] | None = None,
    equip_of_wafer: np.ndarray | None = None,
    override_values: dict[str, np.ndarray] | None = None,
    mechanism: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """스텝 1개의 FDC 요약 통계를 생성한다.

    어떻게 값을 만드나 (파라미터별):
        값 = 목표값 + 챔버 baseline 편차 + PM 주기 드리프트 + 정상 산포 N(0, σ)
             + (해당 웨이퍼가 이 스텝을 원인으로 갖는 경우) 인과 섭동

    인과 섭동을 건너뛰는 두 경우:
        - is_unexplained: 불량인데 FDC에 신호가 없다 (계측하지 못한 원인)
        - 반대로 is_false_positive: 정상인데 파라미터는 이탈해 있다
        둘 다 실제 팹에서 흔하며, 원인 분석 정확도의 현실적 상한을 만든다.
    """
    n = len(wafers)
    chambers = step.chamber_ids
    is_test_step = rules is not None
    mech = mechanism if mechanism is not None else np.full(n, "process", dtype=object)
    # 이 스텝의 지속형 규칙이 적용될 웨이퍼: 경로가 맞아야 한다
    want = "test" if is_test_step else "process"

    # 챔버별 처리 순서 — 소모품 사용량과 드리프트 계산에 쓴다
    order = np.zeros(n, dtype=np.int64)
    for c in chambers:
        sel = chamber_of_wafer == c
        order[sel] = np.arange(int(sel.sum()))

    # 이 스텝이 원인인 패턴 목록 (예: P050 → Center, Scratch)
    # 검사 스텝은 별도 규칙표(TEST_CAUSE_RULES)를 쓰므로 호출자가 넘겨준다.
    source = CAUSE_RULES if rules is None else rules
    rules_here = {
        pat: rule for pat, rule in source.items() if rule.step_id == step.step_id
    }
    # 교호작용은 **두 스텝에 걸쳐** 있으므로 어느 쪽이든 이 스텝이면 적용한다
    interaction_rules = {} if is_test_step else {
        pat: rule
        for pat, rule in INTERACTION_CAUSE_RULES.items()
        if step.step_id in (rule.step_a, rule.step_b)
    }

    out: dict[str, np.ndarray] = {}
    trace_params: dict[str, np.ndarray] = {}

    for param in step.params:
        bias = _equipment_bias(chambers, param, rng, difficulty.equip_bias_sigma)
        bias_vec = np.array([bias[c] for c in chamber_of_wafer])

        if override_values is not None and param.name in override_values:
            # 호출자가 직접 계산한 값(예: 프로브 카드 누적 터치다운)
            values = np.asarray(override_values[param.name], dtype=float)
        elif param.kind == "counter":
            values = _counter_values(n, param, order, rng)
        else:
            # PM 주기(챔버당 300장 가정)에 따른 완만한 드리프트
            drift_phase = (order % 300) / 300.0
            drift = drift_phase * param.sigma * difficulty.drift_sigma_per_pm
            values = (
                param.nominal
                + bias_vec
                + drift
                + rng.normal(0.0, param.sigma, n)
            )

        # ── 인과 섭동 주입 ──────────────────────────────────────────────
        for pat, rule in rules_here.items():
            pert = next((p for p in rule.perturbations if p.param == param.name), None)
            if pert is None:
                continue

            # 같은 패턴이라도 **다른 경로로 생긴 웨이퍼에는 섭동을 넣지 않는다.**
            # Edge-Ring 중 검사 기인 웨이퍼에 식각 파라미터까지 흔들어 놓으면,
            # 두 원인이 데이터상 구분 불가능해져 분석 과제 자체가 사라진다.
            affected = (pattern == pat) & has_signal & (mech == want)
            # 위양성: 정상인데 이 파라미터가 이탈한 웨이퍼
            spurious = (pattern == "none") & is_false_positive
            targets = affected | spurious
            if not targets.any():
                continue

            sev = np.where(np.isnan(severity), 1.0, severity)
            # 위양성은 진짜 이상보다 약하게 흔들리게 한다(그래야 구분이 가능해진다)
            sev = np.where(spurious, sev * rng.uniform(0.3, 0.6, n), sev)

            if pert.counter_ratio is not None:
                forced = pert.counter_ratio * param.nominal
                values = np.where(
                    targets,
                    forced * rng.normal(1.0, 0.03, n),
                    values,
                )
            else:
                shift = pert.shift_sigma * param.sigma * sev
                values = np.where(targets, values + shift, values)

        # ── 교호작용 섭동 ★ ────────────────────────────────────────────
        # 이 스텝의 파라미터를 **규격 안에서만** 민다. 방향은 웨이퍼마다 뒤집어
        # (두껍+약함 / 얇음+강함) 주변부 평균이 정상군과 같게 유지한다.
        # 그래야 "주효과로는 안 보인다"가 데이터에서 실제로 성립한다.
        for pat, irule in interaction_rules.items():
            if param.name not in (irule.param_a, irule.param_b):
                continue
            hit = np.flatnonzero((pattern == pat) & has_signal & (mech == "interaction"))
            if not len(hit):
                continue
            # 두 파라미터가 **같은 부호**를 써야 조합이 성립한다. 스텝별로 따로
            # 뽑으면 무작위로 어긋나 교호작용 자체가 만들어지지 않으므로,
            # 웨이퍼 id에서 결정적으로 부호를 만든다.
            ids = wafers["wafer_id"].to_numpy()[hit]
            sign = np.array([1.0 if int(w[-2:]) % 2 else -1.0 for w in ids])
            lo, hi = irule.shift_range
            magnitude = rng.uniform(lo, hi, len(hit))
            direction = irule.sign_b if param.name == irule.param_b else 1.0
            values[hit] += sign * magnitude * param.sigma * direction

        out[f"{param.name}_mean"] = values

        # 요약 통계: 웨이퍼 1장 처리 중의 시계열 변동으로부터 파생
        noise = param.trace_noise if param.trace_noise > 0 else param.sigma * 0.25
        std = np.abs(rng.normal(noise, noise * 0.25, n))
        out[f"{param.name}_std"] = std
        out[f"{param.name}_min"] = values - std * rng.uniform(1.5, 2.5, n)
        out[f"{param.name}_max"] = values + std * rng.uniform(1.5, 2.5, n)

        if param.trace_noise > 0:
            trace_params[param.name] = values

    # 레시피: Near-full은 레시피 오적용으로 모델링한다(§2.4)
    recipe = np.full(n, DEFAULT_RECIPE, dtype=object)
    if "Near-full" in rules_here:
        recipe = np.where((pattern == "Near-full") & has_signal, WRONG_RECIPE, recipe)

    df = pd.DataFrame(
        {
            "wafer_id": wafers["wafer_id"].to_numpy(),
            "step_id": step.step_id,
            "step_name": step.name_en,
            "equip_id": (
                equip_of_wafer
                if equip_of_wafer is not None
                else [c.split("/")[0] for c in chamber_of_wafer]
            ),
            "chamber_id": chamber_of_wafer,
            "recipe_id": recipe,
            "run_time": wafers["fab_in_time"].to_numpy(),
        }
    )
    for k, v in out.items():
        df[k] = v

    # ── 시계열 생성 + 파생 피처 ★ ───────────────────────────────────────
    # 실제 FDC 시스템도 이렇게 동작한다. 장비가 **전체 시계열로부터** 요약값을
    # 계산해 올리고, 시계열 자체는 용량 때문에 일부만 보관한다. 그래서 여기서도
    # 전 웨이퍼의 시계열을 만들어 피처를 뽑고, 저장은 표본만 한다.
    traces: dict[str, np.ndarray] = {}
    t_sec = np.linspace(0.0, TRACE_SECONDS, TRACE_POINTS)
    spike_rules = {} if is_test_step else {
        pat: rule
        for pat, rule in SPIKE_CAUSE_RULES.items()
        if rule.step_id == step.step_id
    }
    drift_rules = {} if is_test_step else {
        pat: rule
        for pat, rule in DRIFT_CAUSE_RULES.items()
        if rule.step_id == step.step_id
    }


    for pname, base in trace_params.items():
        spec = step.param(pname)
        # 안정화 — 목표값의 3% 아래에서 출발해 수 초 안에 자리를 잡는다.
        #
        # 왜 0에서 올리지 않나 ★: 예전 구현은 `base * (1 - exp(-t/6))` 이라 0에서
        #     목표값까지 올라갔다. 세정조 온도가 웨이퍼마다 0℃에서 65℃로 오르는
        #     장비는 없다. 조는 이미 데워져 있고, 웨이퍼가 들어오면서 살짝 식었다가
        #     회복될 뿐이다. 0에서 올리면 상승분(65℃ ≈ 100σ)이 모든 변동을 압도해
        #     **그래프에서도 피처에서도 스파이크가 묻힌다.**
        settle = 1.0 - 0.03 * np.exp(-t_sec / 4.0)
        matrix = base[:, None] * settle[None, :] + rng.normal(
            0.0, spec.trace_noise, (n, TRACE_POINTS)
        )

        # 드리프트 — 처리 중 서서히 밀린다. 산포는 커지지만 '넘은 시간'은 0이다
        for pat, rule in drift_rules.items():
            pert = next(
                (p for p in rule.perturbations if p.param == pname and p.drift_sigma), None
            )
            if pert is None:
                continue
            hit = np.flatnonzero((pattern == pat) & has_signal & (mech == "drift"))
            if not len(hit):
                continue
            # 평균을 기준으로 **대칭**으로 흔든다: -D/2에서 시작해 +D/2로 끝난다.
            #
            # 왜 0에서 시작하지 않나 ★: 0 → +D로 밀면 60초 평균이 +D/2만큼 올라간다.
            #     그러면 스파이크와 비교할 때 평균만으로도 둘이 갈라져(2.49σ 차이)
            #     "모양을 구분할 수 있는가"라는 질문이 흐려진다. 한 웨이퍼 처리 중의
            #     교정 이탈은 순 오프셋이라기보다 흔들림에 가깝기도 하다.
            span = pert.drift_sigma * spec.sigma
            ramp = np.linspace(-span / 2.0, span / 2.0, TRACE_POINTS)
            matrix[hit] += ramp[None, :]

        # 순간 스파이크 — 평균은 거의 안 움직이지만 시계열에는 뚜렷이 남는다
        for pat, rule in spike_rules.items():
            pert = next(
                (p for p in rule.perturbations if p.param == pname and p.spike_sigma), None
            )
            if pert is None:
                continue
            hit = np.flatnonzero((pattern == pat) & has_signal & (mech == "spike"))
            if not len(hit):
                continue
            width = max(1, int(round(pert.spike_seconds / TRACE_SECONDS * TRACE_POINTS)))
            # 스파이크 시점은 웨이퍼마다 다르다 — 늘 같은 자리면 찾기가 비현실적으로 쉬워진다
            starts = rng.integers(TRACE_POINTS // 4, TRACE_POINTS - width, len(hit))
            for row, start in zip(hit, starts):
                matrix[row, start:start + width] += pert.spike_sigma * spec.sigma

        traces[pname] = matrix

        # ── 요약통계도 **시계열에서** 다시 계산한다 ★ ────────────────────
        # 왜: 앞의 루프는 요약값을 시계열과 따로 만들었다. 그러면 그래프에는
        #     68.9℃ 스파이크가 보이는데 `_max`는 65.1이라고 적혀 있는,
        #     **스스로 모순된 데이터**가 된다. 실제 장비는 시계열로부터 요약을
        #     계산해 올리므로 그 순서를 따르는 것이 맞다.
        #
        #     부수 효과: 스파이크가 `_std`와 `_max`를 조금은 움직인다. 그래서
        #     "요약통계로는 전혀 안 보인다"가 아니라 "거의 안 보인다"가 정확한
        #     표현이 되며, 채점 결과도 그만큼 정직해진다.
        steady = matrix[:, int(TRACE_POINTS * 0.4):]
        df[f"{pname}_mean"] = steady.mean(axis=1)
        df[f"{pname}_std"] = steady.std(axis=1)
        df[f"{pname}_min"] = steady.min(axis=1)
        df[f"{pname}_max"] = steady.max(axis=1)

        for suffix, values_ in trace_features(matrix, t_sec, sigma=spec.sigma).items():
            df[f"{pname}{suffix}"] = values_

    return df, traces


def _build_traces(
    step: ProcessStep,
    fdc: pd.DataFrame,
    traces: dict[str, np.ndarray],
    sample_idx: np.ndarray,
) -> pd.DataFrame:
    """이미 만들어 둔 시계열 행렬에서 표본 웨이퍼만 골라 저장 형태로 편다.

    왜 여기서 다시 만들지 않나 ★: 예전에는 저장용 시계열을 따로 생성했다. 그러면
        **화면에 보이는 시계열과 피처를 계산한 시계열이 다른 데이터**가 된다.
        "피처는 스파이크가 있다는데 그래프에는 없다"는 상황이 생기고, 그건 디버깅이
        거의 불가능하다. 같은 행렬을 쓰면 그런 어긋남이 원천적으로 없다.

    왜 일부만 저장하나: 전 웨이퍼 × 전 파라미터 × 60시점을 저장하면 용량이 폭증한다.
        요약·피처는 이미 전수로 계산했으므로, 저장분은 눈으로 확인하는 용도면 충분하다.
    """
    if not len(sample_idx) or not traces:
        return pd.DataFrame(columns=["wafer_id", "step_id", "param", "t_sec", "value"])

    wafer_ids = fdc["wafer_id"].to_numpy()[sample_idx]
    t_sec = np.linspace(0.0, TRACE_SECONDS, TRACE_POINTS)
    n_sample = len(sample_idx)

    frames = []
    for pname, matrix in traces.items():
        sampled = matrix[sample_idx]
        frames.append(
            pd.DataFrame(
                {
                    "wafer_id": np.repeat(wafer_ids, TRACE_POINTS),
                    "step_id": step.step_id,
                    "param": pname,
                    "t_sec": np.tile(t_sec, n_sample),
                    "value": sampled.reshape(-1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# 4. 공개 API
# ──────────────────────────────────────────────────────────────────────────


def simulate(
    n_wafers: int = 6_000,
    n_days: int = 120,
    seed: int = DEFAULT_SEED,
    difficulty: SimulationDifficulty = DEFAULT_DIFFICULTY,
    start_date: str = "2026-01-05",
    trace_sample_rate: float = 0.05,
) -> SimulationResult:
    """가상 팹의 전체 공정 이력과 FDC 데이터를 생성한다.

    Args:
        n_wafers: 생성할 웨이퍼 수
        n_days: 시뮬레이션 기간(일)
        seed: 난수 시드 (재현성)
        difficulty: 교락·위양성·위음성 등 난이도 설정
        start_date: 기간 시작일
        trace_sample_rate: 시계열 trace를 남길 웨이퍼 비율

    Returns:
        SimulationResult — wafer_master(맵 제외), fdc_summary, fdc_trace, ground_truth, 이상사건 목록

    Note:
        wafer_master의 die_total/die_pass/yield_pct는 여기서 채우지 않는다.
        맵 렌더링(synth_wafer)을 거쳐야 확정되므로 build_dataset이 채운다.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    # 1) 타임라인
    wafers = _build_lots(n_wafers, n_days, start, rng)
    lots = wafers["lot_id"].drop_duplicates().to_numpy()

    # 2) 라우팅 (교락 포함) + 검사 셀 배정
    routing, affinities = _build_routing(lots, rng, difficulty)
    test_routing = _build_test_routing(lots, rng)

    # EDS 검사 시각 — 검사 기인 이상은 이 시각을 기준으로 발생한다
    eds_time = (
        wafers["fab_in_time"]
        + pd.Timedelta(hours=STEP_INTERVAL_HOURS * len(PROCESS_STEPS) + EDS_DELAY_HOURS)
    )

    # 3) 이상 사건 → 패턴 배정 (공정 기인 + 검사 기인)
    pattern, severity, mechanism_arr, excursions = _assign_patterns(
        wafers, routing, test_routing, eds_time.to_numpy(), rng, difficulty
    )
    test_induced = mechanism_arr == "test"
    spike_induced = mechanism_arr == "spike"
    pattern_arr = pattern.to_numpy()
    severity_arr = severity.to_numpy()

    # 4) 위음성/위양성 플래그
    #    is_unexplained: 불량인데 FDC에 신호가 없다 → 원인 분석이 실패하는 게 정상
    #    is_false_positive: 정상인데 파라미터는 이탈 → 분석이 속으면 안 되는 케이스
    is_defect = pattern_arr != "none"
    is_unexplained = is_defect & (rng.random(n_wafers) < difficulty.unexplained_rate)
    has_signal = is_defect & ~is_unexplained
    is_false_positive = (~is_defect) & (rng.random(n_wafers) < difficulty.false_positive_rate)

    # 5) 스텝별 FDC 생성
    route_map = routing.set_index("lot_id")
    lot_of_wafer = wafers["lot_id"].to_numpy()

    n_trace = max(1, int(n_wafers * trace_sample_rate))
    trace_idx = rng.choice(n_wafers, size=min(n_trace, n_wafers), replace=False)

    fdc_parts, trace_parts = [], []
    for i, step in enumerate(PROCESS_STEPS):
        chamber_of_wafer = route_map[step.step_id].reindex(lot_of_wafer).to_numpy()

        step_wafers = wafers.copy()
        step_wafers["fab_in_time"] = wafers["fab_in_time"] + pd.Timedelta(
            hours=STEP_INTERVAL_HOURS * i
        )

        fdc, trace_params = _simulate_step_fdc(
            step,
            step_wafers,
            chamber_of_wafer,
            pattern_arr,
            severity_arr,
            has_signal,
            is_false_positive,
            rng,
            difficulty,
            mechanism=mechanism_arr,
        )
        fdc_parts.append(fdc)
        trace_parts.append(_build_traces(step, fdc, trace_params, trace_idx))

    # ── 5-A) 검사 스텝(EDS) 이력 ────────────────────────────────────────
    # 공정 스텝과 **같은 테이블에** 넣는다. 그래야 커미널리티·SHAP이 프로브 카드를
    # 공정 챔버와 나란히 놓고 경쟁시킬 수 있다. 후보에 없으면 1위가 될 수 없다.
    card_of_wafer = (
        test_routing.set_index("lot_id")["probe_card_id"].reindex(lot_of_wafer).to_numpy()
    )
    tester_of_wafer = (
        test_routing.set_index("lot_id")["tester_id"].reindex(lot_of_wafer).to_numpy()
    )
    touchdowns = _touchdown_counts(
        card_of_wafer, TEST_STEP.param("touchdown_count").nominal, rng
    )

    test_wafers = wafers.copy()
    test_wafers["fab_in_time"] = eds_time  # 검사 이력의 시각은 EDS 검사 시각이다

    test_fdc, test_trace_params = _simulate_step_fdc(
        TEST_STEP,
        test_wafers,
        card_of_wafer,
        pattern_arr,
        severity_arr,
        has_signal,
        is_false_positive,
        rng,
        difficulty,
        rules=TEST_CAUSE_RULES,
        equip_of_wafer=tester_of_wafer,
        override_values={"touchdown_count": touchdowns},
        mechanism=mechanism_arr,
    )
    fdc_parts.append(test_fdc)
    trace_parts.append(_build_traces(TEST_STEP, test_fdc, test_trace_params, trace_idx))

    fdc_summary = pd.concat(fdc_parts, ignore_index=True)
    # 빈 프레임을 걸러낸다 — trace_noise가 0인 스텝은 trace를 만들지 않으므로
    # 그대로 concat하면 dtype 추론 경고가 난다.
    trace_parts = [t for t in trace_parts if len(t)]
    fdc_trace = (
        pd.concat(trace_parts, ignore_index=True)
        if trace_parts
        else pd.DataFrame(columns=["wafer_id", "step_id", "param", "t_sec", "value"])
    )

    # 6) wafer_master (맵 통계는 build_dataset이 채운다)
    wafer_master = pd.DataFrame(
        {
            "wafer_id": wafers["wafer_id"],
            "lot_id": wafers["lot_id"],
            "slot_no": wafers["slot_no"].astype("int64"),
            "product": PRODUCT_NAME,
            "tech_node": TECH_NODE,
            "fab_in_time": wafers["fab_in_time"],
            "eds_time": eds_time,
            # 검사 설비 — 맵의 불량이 공정이 아니라 검사에서 왔을 가능성을 보려면
            # 웨이퍼마다 '어느 테스터·어느 카드로 쟀는지'가 남아 있어야 한다.
            "tester_id": tester_of_wafer,
            "probe_card_id": card_of_wafer,
            "probe_touchdown": touchdowns.astype("int64"),
            "pattern_label": pattern_arr,
            "data_source": "synthetic",
            "is_labeled": True,
        }
    )

    # 7) 정답지 — 분석 모델에는 절대 넣지 않고 검증(M4)에만 쓴다
    root_step, root_equip, root_params = [], [], []
    for pat, from_test in zip(pattern_arr, test_induced):
        rule = TEST_CAUSE_RULES[pat] if from_test else CAUSE_RULES[pat]
        root_step.append(rule.step_id)
        root_params.append([p.param for p in rule.perturbations] or None)
        root_equip.append(None)  # 아래에서 실제 라우팅으로 채운다

    ground_truth = pd.DataFrame(
        {
            "wafer_id": wafers["wafer_id"],
            "pattern_label": pattern_arr,
            "true_root_step": root_step,
            "true_root_params": root_params,
            "severity": severity_arr,
            "is_confounded": False,
            "is_unexplained": is_unexplained,
            "is_false_positive": is_false_positive,
            # 공정이 아니라 **검사 설비**가 원인인가. 같은 맵 패턴이 두 경로로
            # 생기므로, 분석이 둘을 가려내는지 채점하려면 정답이 필요하다.
            "is_test_induced": test_induced,
            # 이상의 **모양** — 지속형인가 순간 스파이크인가. 요약통계로 잡히는지가
            # 갈리므로, 피처 추가의 효과를 채점하려면 정답이 필요하다.
            "is_spike_induced": spike_induced,
            "cause_mechanism": mechanism_arr,
        }
    )

    # 원인 설비: 해당 웨이퍼가 원인 스텝에서 실제로 지나간 챔버
    equip_col = np.full(n_wafers, None, dtype=object)
    for step in PROCESS_STEPS:
        sel = ground_truth["true_root_step"].to_numpy() == step.step_id
        if sel.any():
            ch = route_map[step.step_id].reindex(lot_of_wafer).to_numpy()
            equip_col[sel] = ch[sel]
    # 검사 기인 웨이퍼의 '원인 설비'는 프로브 카드다
    equip_col[test_induced] = card_of_wafer[test_induced]
    ground_truth["true_root_equip"] = equip_col

    # 교락 플래그: 친화 쌍에 걸린 lot 표시
    confounded = np.zeros(n_wafers, dtype=bool)
    for step_a, ch_a, step_b, ch_b in affinities:
        a = route_map[step_a].reindex(lot_of_wafer).to_numpy() == ch_a
        b = route_map[step_b].reindex(lot_of_wafer).to_numpy() == ch_b
        confounded |= a & b
    ground_truth["is_confounded"] = confounded

    # 원인이 없는 패턴(none/Random)은 severity도 정의되지 않는다
    ground_truth.loc[ground_truth["true_root_step"].isna(), "severity"] = np.nan

    return SimulationResult(
        wafer_master=wafer_master,
        fdc_summary=fdc_summary,
        fdc_trace=fdc_trace,
        ground_truth=ground_truth,
        excursions=excursions,
        affinities=affinities,
    )
