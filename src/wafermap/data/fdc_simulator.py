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

from wafermap.config import (
    CAUSE_RULES,
    DEFAULT_DIFFICULTY,
    DEFAULT_SEED,
    PRODUCT_NAME,
    PROCESS_STEPS,
    STEPS_BY_ID,
    TECH_NODE,
    WAFERS_PER_LOT,
    WM811K_LABEL_COUNTS,
    ParamSpec,
    ProcessStep,
    SimulationDifficulty,
)

#: 스텝 간 평균 소요 시간(시간). 실제 팹의 사이클 타임을 단순화한 값이다.
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
    """

    excursion_id: str
    pattern: str
    step_id: str | None
    chamber_id: str | None
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    attack_rate: float


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


def _assign_patterns(
    wafers: pd.DataFrame,
    routing: pd.DataFrame,
    rng: np.random.Generator,
    difficulty: SimulationDifficulty,
) -> tuple[pd.Series, pd.Series, list[Excursion]]:
    """이상 사건을 만들고, 각 웨이퍼에 불량 패턴을 배정한다.

    어떻게:
        1. 패턴별 목표 장수를 계산한다.
        2. 목표에 도달할 때까지 이상 사건을 반복 생성한다.
           - 원인 스텝의 챔버 하나와 시간 창을 뽑는다.
           - 그 창 안에서 그 챔버를 지난, 아직 미배정 상태인 웨이퍼를 attack_rate 만큼 감염시킨다.
        3. Random 패턴은 원인 설비가 없으므로 시간·설비와 무관하게 흩뿌린다.
        4. 끝까지 배정되지 않은 웨이퍼는 'none'.

    Returns:
        (패턴 Series, 심각도 Series, 이상 사건 목록)
    """
    n = len(wafers)
    pattern = pd.Series(["none"] * n, index=wafers.index)
    severity = pd.Series(np.nan, index=wafers.index)
    excursions: list[Excursion] = []

    lot_to_row = wafers["lot_id"].to_numpy()
    t = wafers["fab_in_time"].to_numpy()
    t_min, t_max = wafers["fab_in_time"].min(), wafers["fab_in_time"].max()
    span_hours = (t_max - t_min).total_seconds() / 3600.0

    targets = _target_counts(n)
    # 희소 패턴부터 배정한다. 흔한 패턴이 웨이퍼를 먼저 차지해 버리면
    # Near-full(전체의 0.09%)처럼 드문 패턴이 목표 장수를 못 채운다.
    order = sorted(targets, key=lambda p: targets[p])

    # lot_id → 스텝별 챔버 조회용 매핑
    route_map = routing.set_index("lot_id")

    for pat in order:
        rule = CAUSE_RULES[pat]
        target = targets[pat]
        assigned = 0
        guard = 0

        while assigned < target and guard < 400:
            guard += 1
            lo, hi = difficulty.severity_range

            if rule.step_id is None:
                # 원인 설비 없음(Random) — 시간·챔버와 무관하게 산발 배정.
                # 커미널리티 분석에서 '유의한 설비 없음'이 정답이 되는 케이스다.
                free = np.where(pattern.to_numpy() == "none")[0]
                if not len(free):
                    break
                take = target - assigned
                pick = rng.choice(free, size=min(take, len(free)), replace=False)
                pattern.iloc[pick] = pat
                severity.iloc[pick] = rng.uniform(lo, hi, size=len(pick))
                assigned += len(pick)
                excursions.append(
                    Excursion(
                        excursion_id=f"EXC-{len(excursions) + 1:04d}",
                        pattern=pat,
                        step_id=None,
                        chamber_id=None,
                        t_start=t_min,
                        t_end=t_max,
                        attack_rate=float("nan"),
                    )
                )
                continue

            step = STEPS_BY_ID[rule.step_id]
            chamber = str(rng.choice(step.chamber_ids))

            # 이상 구간: 전체 기간의 3~12% 길이
            win_hours = span_hours * rng.uniform(0.03, 0.12)
            start_off = rng.uniform(0, max(span_hours - win_hours, 1e-6))
            t0 = t_min + pd.Timedelta(hours=float(start_off))
            t1 = t0 + pd.Timedelta(hours=float(win_hours))

            lot_chamber = route_map[step.step_id].reindex(lot_to_row).to_numpy()
            in_window = (t >= np.datetime64(t0)) & (t <= np.datetime64(t1))
            eligible = np.where(
                (lot_chamber == chamber) & in_window & (pattern.to_numpy() == "none")
            )[0]
            if not len(eligible):
                continue

            attack = rng.uniform(0.45, 0.85)
            hit = eligible[rng.random(len(eligible)) < attack]
            if not len(hit):
                continue
            hit = hit[: target - assigned]

            pattern.iloc[hit] = pat
            severity.iloc[hit] = rng.uniform(lo, hi, size=len(hit))
            assigned += len(hit)

            excursions.append(
                Excursion(
                    excursion_id=f"EXC-{len(excursions) + 1:04d}",
                    pattern=pat,
                    step_id=step.step_id,
                    chamber_id=chamber,
                    t_start=t0,
                    t_end=t1,
                    attack_rate=float(attack),
                )
            )

    return pattern, severity, excursions


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

    # 챔버별 처리 순서 — 소모품 사용량과 드리프트 계산에 쓴다
    order = np.zeros(n, dtype=np.int64)
    for c in chambers:
        sel = chamber_of_wafer == c
        order[sel] = np.arange(int(sel.sum()))

    # 이 스텝이 원인인 패턴 목록 (예: P050 → Center, Scratch)
    rules_here = {
        pat: rule for pat, rule in CAUSE_RULES.items() if rule.step_id == step.step_id
    }

    out: dict[str, np.ndarray] = {}
    trace_params: dict[str, np.ndarray] = {}

    for param in step.params:
        bias = _equipment_bias(chambers, param, rng, difficulty.equip_bias_sigma)
        bias_vec = np.array([bias[c] for c in chamber_of_wafer])

        if param.kind == "counter":
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

            affected = (pattern == pat) & has_signal
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
            "equip_id": [c.split("/")[0] for c in chamber_of_wafer],
            "chamber_id": chamber_of_wafer,
            "recipe_id": recipe,
            "run_time": wafers["fab_in_time"].to_numpy(),
        }
    )
    for k, v in out.items():
        df[k] = v
    return df, trace_params


def _build_traces(
    step: ProcessStep,
    fdc: pd.DataFrame,
    trace_params: dict[str, np.ndarray],
    sample_idx: np.ndarray,
    rng: np.random.Generator,
    n_points: int = 24,
) -> pd.DataFrame:
    """선택된 웨이퍼에 대해 파라미터 시계열(trace)을 생성한다.

    왜 일부만: 전 웨이퍼 × 전 파라미터 × 시점을 저장하면 용량이 폭증한다. 원인 분석
        화면에서 trace는 '의심 웨이퍼 몇 장을 눈으로 확인'하는 용도이므로 표본이면 충분하다.
    """
    if not len(sample_idx) or not trace_params:
        return pd.DataFrame(columns=["wafer_id", "step_id", "param", "t_sec", "value"])

    wafer_ids = fdc["wafer_id"].to_numpy()[sample_idx]
    rows = []
    for pname, values in trace_params.items():
        spec = step.param(pname)
        base = values[sample_idx]
        t = np.linspace(0.0, 60.0, n_points)  # 웨이퍼 1장 처리 시간을 60초로 정규화
        for i, wid in enumerate(wafer_ids):
            # 안정화 구간(초반 상승) + 정상 구간 노이즈
            ramp = 1.0 - np.exp(-t / 6.0)
            series = base[i] * ramp + rng.normal(0.0, spec.trace_noise, n_points)
            rows.append(
                pd.DataFrame(
                    {
                        "wafer_id": wid,
                        "step_id": step.step_id,
                        "param": pname,
                        "t_sec": t,
                        "value": series,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


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

    # 2) 라우팅 (교락 포함)
    routing, affinities = _build_routing(lots, rng, difficulty)

    # 3) 이상 사건 → 패턴 배정
    pattern, severity, excursions = _assign_patterns(wafers, routing, rng, difficulty)
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
        )
        fdc_parts.append(fdc)
        trace_parts.append(_build_traces(step, fdc, trace_params, trace_idx, rng))

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
    eds_time = wafers["fab_in_time"] + pd.Timedelta(
        hours=STEP_INTERVAL_HOURS * len(PROCESS_STEPS) + EDS_DELAY_HOURS
    )
    wafer_master = pd.DataFrame(
        {
            "wafer_id": wafers["wafer_id"],
            "lot_id": wafers["lot_id"],
            "slot_no": wafers["slot_no"].astype("int64"),
            "product": PRODUCT_NAME,
            "tech_node": TECH_NODE,
            "fab_in_time": wafers["fab_in_time"],
            "eds_time": eds_time,
            "pattern_label": pattern_arr,
            "data_source": "synthetic",
            "is_labeled": True,
        }
    )

    # 7) 정답지 — 분석 모델에는 절대 넣지 않고 검증(M4)에만 쓴다
    root_step, root_equip, root_params = [], [], []
    for pat in pattern_arr:
        rule = CAUSE_RULES[pat]
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
        }
    )

    # 원인 설비: 해당 웨이퍼가 원인 스텝에서 실제로 지나간 챔버
    equip_col = np.full(n_wafers, None, dtype=object)
    for step in PROCESS_STEPS:
        sel = ground_truth["true_root_step"].to_numpy() == step.step_id
        if sel.any():
            ch = route_map[step.step_id].reindex(lot_of_wafer).to_numpy()
            equip_col[sel] = ch[sel]
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
