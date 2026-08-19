"""SPC 이상 구간 탐지 — "언제부터 이상해졌나"를 찾는다.

무엇을: 수율이나 불량 발생률의 시계열을 관리도로 감시해, 공정이 평소와 다르게
        흘러간 **구간**을 자동으로 찾아낸다.

왜 이 단계가 필요한가 ★: 두 가지 이유가 있다.

    1. **조사 범위를 시간으로 좁힌다.** "수율이 떨어졌다"만으로는 어디를 봐야 할지
       알 수 없다. "3월 5일부터 나빠졌다"를 알면 그 날짜 전후에 무엇이 바뀌었는지
       (부품 교체, 레시피 변경, 신규 장비 투입)를 뒤지면 된다.

    2. **다음 단계의 입력을 만든다.** 커미널리티 분석(M3)은 "이상군 vs 정상군"을
       비교하는 방법이다. 그 두 집단을 나누려면 먼저 이상 구간이 특정되어야 한다.
       SPC가 없으면 M3가 성립하지 않는다.

핵심 개념 1 — 관리한계는 규격이 아니다:
    · **규격(spec)**: 제품이 만족해야 할 조건. 고객/설계가 정한다.
    · **관리한계(control limit)**: 이 공정이 **평소에 흔들리는 범위**. 데이터가 정한다.
    관리한계 안이어도 규격 밖일 수 있고, 그 반대도 가능하다. 둘은 다른 질문에 답한다.

핵심 개념 2 — 지표의 성격에 맞는 관리도를 써야 한다 ★:
    이 프로젝트에서 실제로 겪은 실패다. 처음에는 모든 지표에 평균±3σ 방식
    (individual chart)을 썼는데, 불량 발생률에서 **검출률 0%** 가 나왔다.

    원인: 발생률은 **비율 데이터**다. 일별 집계의 82%가 정확히 0%였고,
    나머지는 lot 하나가 통째로 감염되어 60~84%까지 치솟았다. 이런 분포에
    정규분포를 가정하면 σ가 17.9%까지 부풀려져 관리한계가 -46%~+60%가 된다.
    무엇도 걸리지 않는다.

    | 지표 | 성격 | 적합한 관리도 |
    |---|---|---|
    | 수율(%) | 연속값, 웨이퍼별 평균 | **individual** (평균 ± kσ) |
    | 불량 발생률 | 비율 (불량 수 / 전체 수) | **p-chart** (이항분포 기반) |

    p-chart는 시점마다 표본 수 n이 다른 것을 반영해 관리한계를 다르게 잡는다.
    표본이 적은 날은 넓게, 많은 날은 좁게. 이것이 통계적으로 올바른 처리다.

핵심 개념 3 — 과분산(overdispersion)과 Laney 보정:
    반도체는 웨이퍼가 lot(25장) 단위로 함께 움직인다. 한 lot이 감염되면 25장이
    통째로 불량이 되므로, 실제 변동이 이항분포가 예측하는 것보다 **크다.**
    이를 과분산이라 한다.

    Laney의 p'-chart는 표준화 잔차의 이동범위로 과분산 계수 σ_z를 추정해
    관리한계를 σ_z배 넓힌다. σ_z ≈ 1이면 보통 p-chart와 같고, 클수록 넓어진다.
    lot 단위로 움직이는 반도체 데이터에 딱 맞는 보정이다.

참고: docs/00_design.md §M2, docs/08_interpretation_guide.md §[3]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: EWMA 가중치. 작을수록 과거를 오래 기억해 작은 변화에 민감하지만 반응이 느리다.
#: 교과서 기본은 0.2지만, 이 데이터의 이상 사건은 5~13일로 짧아 반응성을 높였다.
DEFAULT_LAMBDA = 0.3
#: 관리한계 배수. 3σ가 관례(정상인데 벗어날 확률 약 0.27%)이나, 아래 측정 결과에 따라
#: 2.5σ를 기본으로 한다. 위양성이 늘지 않으면서 검출률이 오르기 때문이다.
DEFAULT_SIGMA = 2.5
#: CUSUM에서 무시할 작은 변동 (표준편차 배수). 이보다 작은 편차는 누적하지 않는다.
DEFAULT_SLACK = 0.5
#: CUSUM 경보 임계값 (표준편차 배수)
DEFAULT_CUSUM_LIMIT = 4.0
#: 연속 이탈이 이 길이 미만이면 경보로 보지 않는다.
#: 1로 둔 근거는 아래 측정 — 2로 올리면 짧은 사건을 놓치는데 위양성은 줄지 않았다.
DEFAULT_MIN_RUN = 1

# ── 기본값 근거 (합성 6,000장 / Edge-Ring 이상 사건 12건 기준) ────────────
#
#   λ    k(σ)  검출률   위양성
#  0.2   3.0    50%      0      ← 교과서 기본. 짧은 사건을 놓친다
#  0.3   2.5    58%      0      ← 채택
#  0.4   2.0    83%      0      ← 이 데이터에 가장 잘 맞지만 과적합 위험
#
# 0.4/2.0이 가장 높지만 채택하지 않았다. 이 데이터셋의 사건 길이 분포에 맞춘 값이라
# 실측 데이터에서도 최적일 보장이 없기 때문이다. 0.3/2.5는 문헌에서 흔히 쓰이는
# 범위(λ 0.05~0.4) 안이면서 짧은 사건에 대한 반응성을 확보한 절충이다.
#
# ⚠️ 위양성 0건을 과신하지 말 것: 이 데이터는 전체 121일 중 89일(74%)이 이상 구간이라
#    "정상 구간에서 잘못 울린 경보"를 판정할 여지 자체가 좁다. 실측 데이터에서는
#    위양성률을 다시 측정해야 한다.


@dataclass(frozen=True)
class Alarm:
    """검출된 이상 구간 하나.

    Attributes:
        alarm_id: 식별자
        method: 검출 방법 ("EWMA" | "CUSUM")
        start, end: 이상 구간의 시작/끝 (집계 단위 기준)
        n_points: 구간에 포함된 집계 시점 수
        peak_deviation: 구간 내 최대 이탈 정도 (표준편차 배수)
        direction: "up"(값이 높아짐) | "down"(값이 낮아짐)
        metric: 감시한 지표명
    """

    alarm_id: str
    method: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_points: int
    peak_deviation: float
    direction: str
    metric: str

    @property
    def duration_days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400.0


@dataclass
class ControlChart:
    """관리도 계산 결과.

    Attributes:
        series: 집계된 원본 시계열 (index=시점, value=지표값)
        center: 중심선 (기준 기간의 평균)
        sigma: 기준 기간의 표준편차
        ewma: EWMA 통계량
        ewma_upper, ewma_lower: EWMA 관리한계 (시점마다 다르다)
        cusum_high, cusum_low: CUSUM 누적합 (상방/하방)
        cusum_limit: CUSUM 경보 임계값
        alarms: 검출된 이상 구간
    """

    series: pd.Series
    center: float
    sigma: float
    ewma: pd.Series
    ewma_upper: pd.Series
    ewma_lower: pd.Series
    cusum_high: pd.Series
    cusum_low: pd.Series
    cusum_limit: float
    alarms: list[Alarm] = field(default_factory=list)
    #: "individual"(연속값) | "p"(비율)
    chart_type: str = "individual"
    #: 시점별 표준편차. p-chart는 표본 수에 따라 시점마다 다르다.
    sigma_series: pd.Series | None = None
    #: Laney 과분산 계수. 1이면 이항분포대로, 클수록 실제 변동이 더 크다는 뜻.
    overdispersion: float = 1.0
    #: 시점별 표본 수
    n: pd.Series | None = None

    def alarm_mask(self) -> pd.Series:
        """각 시점이 어떤 이상 구간에 속하는지 나타내는 불리언 Series."""
        mask = pd.Series(False, index=self.series.index)
        for alarm in self.alarms:
            mask |= (self.series.index >= alarm.start) & (self.series.index <= alarm.end)
        return mask


# ──────────────────────────────────────────────────────────────────────────
# 1. 집계
# ──────────────────────────────────────────────────────────────────────────


def aggregate(
    wafer_master: pd.DataFrame,
    *,
    metric: str = "yield_pct",
    freq: str = "D",
    time_col: str = "eds_time",
    min_count: int = 5,
) -> pd.DataFrame:
    """웨이퍼 단위 데이터를 시간 단위로 집계한다.

    Args:
        wafer_master: 웨이퍼 마스터 테이블
        metric: 감시할 지표. "yield_pct" 또는 "<패턴명>_rate" 형식
                (예: "Edge-Ring_rate" → Edge-Ring 발생률)
        freq: 집계 주기 ("D"=일, "12h"=반일, "W"=주)
        time_col: 시간 기준 컬럼
        min_count: 이 미만의 웨이퍼로 집계된 시점은 버린다

    Returns:
        시점을 인덱스로 갖는 DataFrame — 컬럼 `value`(지표값 %), `n`(웨이퍼 수),
        `defects`(해당 불량 수, 비율 지표일 때만)

    왜 표본 수 n을 함께 돌려주나 ★: p-chart는 시점마다 표본 수가 다른 것을
        반영해 관리한계를 다르게 잡는다. 웨이퍼 10장으로 잰 불량률과 500장으로
        잰 불량률은 신뢰도가 전혀 다르기 때문이다. n이 없으면 이 계산을 할 수 없다.

    왜 min_count가 필요한가: 웨이퍼 2장으로 계산한 수율은 0% 아니면 100%처럼
        극단적으로 튄다. 이런 시점이 섞이면 관리한계가 비현실적으로 넓어져
        진짜 이상을 놓치게 된다.
    """
    df = wafer_master.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    grouped = df.set_index(time_col).resample(freq)

    counts = grouped.size()

    if metric == "yield_pct":
        values = grouped["yield_pct"].mean()
        defects = pd.Series(np.nan, index=values.index)
    elif metric.endswith("_rate"):
        pattern = metric[: -len("_rate")]
        defects = grouped["pattern_label"].apply(lambda s: int((s == pattern).sum()))
        values = (defects / counts.replace(0, np.nan)) * 100
    else:
        raise ValueError(
            f"알 수 없는 지표: {metric!r} "
            f"('yield_pct' 또는 '<패턴명>_rate' 형식이어야 합니다)"
        )

    out = pd.DataFrame({"value": values, "n": counts, "defects": defects})
    out = out[out["n"] >= min_count]
    return out[out["value"].notna()]


# ──────────────────────────────────────────────────────────────────────────
# 2. 관리도
# ──────────────────────────────────────────────────────────────────────────


def _baseline(series: pd.Series, baseline_frac: float) -> tuple[float, float]:
    """기준 기간에서 중심선과 표준편차를 추정한다.

    왜 앞부분만 쓰나: 전체 기간으로 평균을 내면, 이상 구간의 값들이 중심선을
        끌어당겨 **이상이 정상처럼 보이게** 된다. 관리도의 기준은 '정상일 때의
        모습'이어야 하므로, 초기 안정 구간만으로 추정한다.

    Returns:
        (중심선, 표준편차)
    """
    n_baseline = max(int(len(series) * baseline_frac), 5)
    baseline = series.iloc[:n_baseline]

    center = float(baseline.mean())
    sigma = float(baseline.std(ddof=1))

    # 표준편차가 0이면 관리한계가 중심선과 겹쳐 모든 점이 경보가 된다.
    # 값이 완전히 일정한 극단적 경우를 대비한 안전장치.
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(abs(center) * 0.01) or 1e-6
    return center, sigma


def _ewma(series: pd.Series, center: float, sigma: float, lam: float, k: float):
    """EWMA 통계량과 관리한계를 계산한다.

    EWMA는 최근 값에 큰 가중치를 주며 과거로 갈수록 지수적으로 줄인다.

        z_t = λ·x_t + (1-λ)·z_{t-1}

    관리한계는 시점마다 다르다. 초반에는 누적된 정보가 적어 좁고, 시간이 지나면
    점근값으로 수렴한다.

        한계 = center ± k·σ·√( λ/(2-λ) · (1-(1-λ)^{2t}) )
    """
    z = series.ewm(alpha=lam, adjust=False).mean()

    t = np.arange(1, len(series) + 1)
    width = k * sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)))

    upper = pd.Series(center + width, index=series.index)
    lower = pd.Series(center - width, index=series.index)
    return z, upper, lower


def _cusum(series: pd.Series, center: float, sigma: float, slack: float):
    """상방/하방 CUSUM 누적합을 계산한다.

        C⁺_t = max(0, C⁺_{t-1} + (x_t - center) - slack·σ)
        C⁻_t = max(0, C⁻_{t-1} - (x_t - center) - slack·σ)

    slack(허용치)보다 작은 편차는 누적하지 않는다. 이것이 없으면 정상 변동도
    조금씩 쌓여 결국 경보가 울린다.

    Returns:
        (상방 누적합, 하방 누적합) — 둘 다 σ 단위로 정규화되어 있다
    """
    values = series.to_numpy(dtype=float)
    k = slack * sigma

    high = np.zeros(len(values))
    low = np.zeros(len(values))
    for i, x in enumerate(values):
        prev_high = high[i - 1] if i else 0.0
        prev_low = low[i - 1] if i else 0.0
        high[i] = max(0.0, prev_high + (x - center) - k)
        low[i] = max(0.0, prev_low - (x - center) - k)

    return (
        pd.Series(high / sigma, index=series.index),
        pd.Series(low / sigma, index=series.index),
    )


def _p_chart_sigma(
    defects: np.ndarray, n: np.ndarray, *, laney: bool = True
) -> tuple[float, np.ndarray, float]:
    """p-chart의 중심선과 시점별 표준편차를 계산한다 (Laney 보정 포함).

    이항분포 기반:
        p̄   = 전체 불량 수 / 전체 표본 수
        σ_i = √( p̄(1-p̄) / n_i )      ← 표본이 적은 시점일수록 크다

    Laney 과분산 보정:
        z_i = (p_i - p̄) / σ_i          ← 표준화 잔차
        σ_z = MR̄(z) / 1.128            ← 이동범위 평균 / d2 상수
        보정된 σ_i = σ_i × σ_z

    왜 이동범위를 쓰나: 표준편차를 그냥 계산하면 이상 구간의 큰 값들이 포함되어
        과분산이 실제보다 크게 추정된다. **연속한 두 점의 차이**(이동범위)는
        점진적 이동(shift)에 둔감해서, 순수한 단기 변동만 잡아낸다.
        1.128은 정규분포에서 이동범위 평균을 표준편차로 바꾸는 상수(d2)다.

    Returns:
        (중심선 p̄ (%), 시점별 σ 배열 (%), 과분산 계수 σ_z)
    """
    total_defects = float(np.nansum(defects))
    total_n = float(np.nansum(n))
    p_bar = total_defects / total_n if total_n > 0 else 0.0

    # 완전히 0이거나 1이면 분산이 0이 되어 관리도가 성립하지 않는다.
    # 최소 한 건은 관측될 수 있다고 보고 하한을 둔다.
    p_bar = float(np.clip(p_bar, 1.0 / max(total_n, 1.0), 1.0 - 1e-9))

    sigma_i = np.sqrt(p_bar * (1.0 - p_bar) / np.maximum(n, 1))

    sigma_z = 1.0
    if laney and len(defects) >= 3:
        p_i = np.divide(defects, np.maximum(n, 1), dtype=float)
        z = (p_i - p_bar) / np.maximum(sigma_i, 1e-12)
        moving_range = np.abs(np.diff(z))
        if moving_range.size and np.isfinite(moving_range).any():
            mr_bar = float(np.nanmean(moving_range))
            sigma_z = max(mr_bar / 1.128, 1.0)  # 1 미만이면 보정하지 않는다

    return p_bar * 100.0, sigma_i * 100.0 * sigma_z, sigma_z


def _extract_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """True가 연속된 구간을 (시작, 끝) 인덱스 목록으로 뽑는다."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def control_chart(
    data: pd.DataFrame | pd.Series,
    *,
    metric: str = "yield_pct",
    chart_type: str = "auto",
    lam: float = DEFAULT_LAMBDA,
    k: float = DEFAULT_SIGMA,
    slack: float = DEFAULT_SLACK,
    cusum_limit: float = DEFAULT_CUSUM_LIMIT,
    baseline_frac: float = 0.3,
    min_run: int = DEFAULT_MIN_RUN,
    direction: str = "auto",
    laney: bool = True,
) -> ControlChart:
    """관리도를 계산하고 이상 구간을 검출한다.

    Args:
        data: aggregate()가 만든 DataFrame(`value`,`n`,`defects`) 또는 값만 담긴 Series
        metric: 지표명 (경보 기록용)
        chart_type: "auto" | "individual" | "p"
            · **individual** — 연속값(수율)용. 중심선 ± k·σ, σ는 기준 기간에서 추정
            · **p** — 비율(불량 발생률)용. 이항분포 기반, 시점별 표본 수 반영
            · "auto"면 metric 이름으로 판단한다 (`_rate`로 끝나면 p-chart)
        lam: EWMA 가중치
        k: 관리한계 배수 (σ)
        slack: CUSUM 허용치 (σ)
        cusum_limit: CUSUM 경보 임계값 (σ)
        baseline_frac: individual chart에서 기준 기간으로 쓸 앞부분 비율
        min_run: 이 길이 미만의 연속 이탈은 경보로 보지 않는다 (단발성 잡음 제거)
        direction: "auto" | "up" | "down" | "both"
            수율은 떨어지는 게 문제라 "down", 불량률은 올라가는 게 문제라 "up".
        laney: p-chart에서 과분산 보정을 적용할지

    Returns:
        ControlChart

    Raises:
        ValueError: 시계열이 너무 짧거나 인자가 잘못됐을 때
    """
    # 입력 정규화 — Series로 들어오면 표본 수를 알 수 없어 individual만 가능하다
    if isinstance(data, pd.Series):
        series = data
        n = None
        defects = None
    else:
        series = data["value"]
        n = data["n"] if "n" in data else None
        defects = data["defects"] if "defects" in data else None

    if len(series) < 10:
        raise ValueError(
            f"시계열이 너무 짧습니다 ({len(series)}개). "
            f"관리한계를 추정하려면 최소 10개 시점이 필요합니다."
        )

    if chart_type == "auto":
        chart_type = "p" if metric.endswith("_rate") else "individual"
    if chart_type not in ("individual", "p"):
        raise ValueError(f"chart_type은 'individual' 또는 'p': {chart_type!r}")
    if chart_type == "p" and (n is None or defects is None or defects.isna().all()):
        raise ValueError(
            "p-chart에는 표본 수(n)와 불량 수(defects)가 필요합니다. "
            "aggregate()가 돌려준 DataFrame을 그대로 넘기세요."
        )

    if direction == "auto":
        direction = "down" if metric == "yield_pct" else "up"
    if direction not in ("up", "down", "both"):
        raise ValueError(f"direction은 'up'/'down'/'both': {direction!r}")

    # ── 중심선과 시점별 표준편차 ────────────────────────────────────────
    overdispersion = 1.0
    if chart_type == "p":
        center, sigma_arr, overdispersion = _p_chart_sigma(
            defects.to_numpy(dtype=float), n.to_numpy(dtype=float), laney=laney
        )
        sigma_series = pd.Series(sigma_arr, index=series.index)
        sigma = float(np.mean(sigma_arr))  # 대표값 (CUSUM 정규화에 쓴다)
    else:
        center, sigma = _baseline(series, baseline_frac)
        sigma_series = pd.Series(sigma, index=series.index)

    # ── EWMA ────────────────────────────────────────────────────────────
    ewma = series.ewm(alpha=lam, adjust=False).mean()
    t = np.arange(1, len(series) + 1)
    # 시간에 따라 넓어지다 수렴하는 계수
    ramp = np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * t)))
    # p-chart는 시점별 σ를 쓰므로 관리한계도 시점마다 다르다
    width = k * sigma_series.to_numpy() * ramp
    upper = pd.Series(center + width, index=series.index)
    lower = pd.Series(center - width, index=series.index)

    # ── CUSUM ───────────────────────────────────────────────────────────
    cusum_high, cusum_low = _cusum(series, center, sigma, slack)

    # ── 경보 추출 ───────────────────────────────────────────────────────
    alarms: list[Alarm] = []
    idx = series.index
    ewma_dev = (ewma - center).abs() / sigma_series.replace(0, np.nan)

    def add_alarms(flags: np.ndarray, method: str, dev: pd.Series, way: str) -> None:
        for start_i, end_i in _extract_runs(flags):
            if end_i - start_i + 1 < min_run:
                continue
            peak = dev.iloc[start_i : end_i + 1].max()
            alarms.append(
                Alarm(
                    alarm_id="",
                    method=method,
                    start=idx[start_i],
                    end=idx[end_i],
                    n_points=end_i - start_i + 1,
                    peak_deviation=float(peak) if np.isfinite(peak) else 0.0,
                    direction=way,
                    metric=metric,
                )
            )

    if direction in ("up", "both"):
        add_alarms((ewma > upper).to_numpy(), "EWMA", ewma_dev, "up")
        add_alarms((cusum_high > cusum_limit).to_numpy(), "CUSUM", cusum_high, "up")
    if direction in ("down", "both"):
        add_alarms((ewma < lower).to_numpy(), "EWMA", ewma_dev, "down")
        add_alarms((cusum_low > cusum_limit).to_numpy(), "CUSUM", cusum_low, "down")

    alarms.sort(key=lambda a: (a.start, a.method))
    for i, alarm in enumerate(alarms, start=1):
        object.__setattr__(alarm, "alarm_id", f"ALM-{i:03d}")

    return ControlChart(
        series=series,
        center=center,
        sigma=sigma,
        ewma=ewma,
        ewma_upper=upper,
        ewma_lower=lower,
        cusum_high=cusum_high,
        cusum_low=cusum_low,
        cusum_limit=cusum_limit,
        alarms=alarms,
        chart_type=chart_type,
        sigma_series=sigma_series,
        overdispersion=overdispersion,
        n=n,
    )


# ──────────────────────────────────────────────────────────────────────────
# 3. 이상군 / 정상군 분할
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseControlSplit:
    """이상군과 정상군으로 나눈 결과 — 커미널리티 분석(M3)의 입력.

    Attributes:
        case_ids: 이상 구간에 속한 웨이퍼 id
        control_ids: 정상 구간에 속한 웨이퍼 id
        alarm: 기준이 된 이상 구간
        case_window, control_window: 각 집단의 시간 범위
    """

    case_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    alarm: Alarm | None
    case_window: tuple[pd.Timestamp, pd.Timestamp] | None
    control_window: tuple[pd.Timestamp, pd.Timestamp] | None

    def __len__(self) -> int:
        return len(self.case_ids) + len(self.control_ids)

    def summary(self) -> str:
        return (
            f"이상군 {len(self.case_ids):,}장 · 정상군 {len(self.control_ids):,}장"
            + (f" (기준: {self.alarm.alarm_id})" if self.alarm else "")
        )


def split_case_control(
    wafer_master: pd.DataFrame,
    alarm: Alarm,
    *,
    time_col: str = "eds_time",
    control_days: float | None = None,
    freq_offset: str = "D",
) -> CaseControlSplit:
    """이상 구간과 그 직전 안정 구간으로 웨이퍼를 나눈다.

    왜 '직전' 구간을 대조군으로 쓰나 ★: 전체 정상 웨이퍼를 대조군으로 쓰면
        계절성·장기 추세·제품 변경 같은 시간 요인이 섞인다. 이상 구간 **바로 직전**을
        쓰면 그런 요인이 대부분 상쇄되어, 차이의 원인을 '이상 사건'으로 좁힐 수 있다.

        역학 조사에서 환자와 비슷한 조건의 사람을 대조군으로 고르는 것과 같은 발상이다.

    Args:
        wafer_master: 웨이퍼 마스터
        alarm: 기준이 될 이상 구간
        time_col: 시간 컬럼
        control_days: 대조 기간 길이(일). None이면 이상 구간과 같은 길이로 잡되
                      최소 7일을 보장한다.
        freq_offset: 집계 주기. 구간 끝을 포함하도록 보정하는 데 쓴다.

    Returns:
        CaseControlSplit
    """
    df = wafer_master.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # 집계 시점은 구간의 시작을 가리키므로, 끝 시점의 한 주기만큼을 포함시킨다
    period = pd.tseries.frequencies.to_offset(freq_offset)
    case_start, case_end = alarm.start, alarm.end + period

    span_days = max((case_end - case_start).total_seconds() / 86400.0, 1.0)
    window = control_days if control_days is not None else max(span_days, 7.0)
    control_end = case_start
    control_start = control_end - pd.Timedelta(days=window)

    in_case = (df[time_col] >= case_start) & (df[time_col] < case_end)
    in_control = (df[time_col] >= control_start) & (df[time_col] < control_end)

    return CaseControlSplit(
        case_ids=tuple(df.loc[in_case, "wafer_id"]),
        control_ids=tuple(df.loc[in_control, "wafer_id"]),
        alarm=alarm,
        case_window=(case_start, case_end),
        control_window=(control_start, control_end),
    )


def to_cause_window(
    alarm: Alarm, cycle_time: pd.Timedelta
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """검출된 이상 구간을 **원인이 발생한 시점**으로 되돌린다.

    무엇을: EDS 검사 시각 기준의 경보 구간을, 공정 투입 시각 기준으로 옮긴다.

    왜 필요한가 ★: **불량이 발견된 시점과 원인이 발생한 시점은 다르다.**

        웨이퍼가 공정에 투입되어 EDS 검사를 받기까지 시간이 걸린다. 이 프로젝트의
        가상 라인은 66시간(약 2.75일)이고, 실제 팹은 2~3개월이다.

        따라서 "3월 5일 EDS에서 불량이 급증했다"는 것은 "3월 5일에 장비가 고장 났다"가
        아니라 "**2.75일 전에** 투입된 웨이퍼들이 무언가를 겪었다"는 뜻이다.

        이 보정을 빼먹으면 엉뚱한 기간의 설비 이력을 뒤지게 된다. 실제로 이 프로젝트의
        초기 채점에서 이 시차 때문에 검출률이 42%로 나왔고, 보정 후 83%가 되었다.

    Note:
        커미널리티 분석(M3)에서는 이 보정이 필요 없다. 이상군 웨이퍼를 **wafer_id로**
        추적해 그 웨이퍼의 FDC 이력을 조회하기 때문이다. 시간이 아니라 개체를 따라가면
        시차 문제가 저절로 사라진다. 시간 기준으로 설비 이력을 훑을 때만 보정이 필요하다.

    Args:
        alarm: 검출된 이상 구간 (EDS 시각 기준)
        cycle_time: 공정 투입 → EDS 검사까지의 소요 시간

    Returns:
        (원인 발생 추정 시작, 끝)
    """
    return alarm.start - cycle_time, alarm.end - cycle_time


def split_by_pattern(
    wafer_master: pd.DataFrame, pattern: str
) -> CaseControlSplit:
    """패턴 라벨로 직접 이상군/정상군을 나눈다 (SPC를 거치지 않는 경로).

    왜 필요한가: SPC는 '시간'으로 나누므로, 이상 구간 안의 정상 웨이퍼도 이상군에
        섞인다. 반면 이 방법은 패턴이 확정된 웨이퍼만 골라 신호가 훨씬 선명하다.

        둘 다 쓸모가 있다. SPC 경로는 **실제 운영 상황**(패턴을 아직 모르는 상태)을,
        패턴 경로는 **분석 방법 자체의 성능**을 확인하는 데 맞다.
    """
    is_case = wafer_master["pattern_label"] == pattern
    return CaseControlSplit(
        case_ids=tuple(wafer_master.loc[is_case, "wafer_id"]),
        control_ids=tuple(wafer_master.loc[wafer_master["pattern_label"] == "none", "wafer_id"]),
        alarm=None,
        case_window=None,
        control_window=None,
    )
