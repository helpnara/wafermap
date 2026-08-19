"""커미널리티 분석 — "어느 설비가 범인인가"를 찾는다.

무엇을: 불량 웨이퍼들이 공통으로 거쳐 간 챔버를 통계적으로 찾아내고, 진짜 원인과
        "같이 흘렀을 뿐인" 설비를 갈라낸다.

발상은 역학 조사와 같다: 식중독 환자 50명이 생겼을 때 "이 사람들이 공통으로 먹은
음식이 뭐지?"를 찾는 것과 똑같다. 웨이퍼도 이력을 모아 놓고 공통점을 찾는다.

왜 이 단계가 중요한가 ★: **수백 개 공정, 수십 대 장비 중 어디를 봐야 할지 좁혀 준다.**
    이 단계 없이 FDC 파라미터 수백 개를 바로 뒤지면 시간이 너무 걸린다.
    그리고 조치 단위가 챔버이기 때문이기도 하다 — "식각 장비가 문제"로는 아무것도
    못 하지만 "4개 챔버 중 3번만 문제"면 그 챔버만 잠그면 된다.

반드시 다뤄야 할 두 가지 함정:

    ① **다중검정** — 챔버가 37개면 우연히 유의해 보이는 것이 반드시 나온다.
       유의수준 5%에서 37개를 검정하면 정상이어도 약 2개가 걸린다.
       → Benjamini-Hochberg FDR 보정으로 기준을 조정한다.

    ② **교락** — 진범과 늘 붙어 다닌 무고한 설비까지 유의하게 나온다.
       공장의 lot 배정에 규칙성이 있으면 특정 장비 조합이 함께 흐르기 때문이다.
       → Cochran–Mantel–Haenszel 층화 검정으로 갈라낸다.

    ③ **표본 단위** ★ — 검정 단위를 웨이퍼로 잡으면 표본이 25배로 부풀려진다.
       팹은 lot(25장) 단위로 설비를 배정하므로, 같은 lot의 25장은 **독립 관측이 아니다.**
       한 lot이 어느 챔버를 썼는지는 25번이 아니라 **한 번** 결정된 사건이다.

       웨이퍼 단위로 검정하면 이런 일이 벌어진다(이 프로젝트에서 실제로 관측):

           오즈비 582, p_adj = 8.6e-63

       표본이 부풀려져 **어떤 사소한 차이도 압도적으로 유의하게** 나온다.
       그러면 진짜 원인과 우연한 상관을 구분할 수 없다.
       → 기본 검정 단위를 **lot**으로 둔다.

참고: docs/00_design.md §M3, docs/08_interpretation_guide.md §[4]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: 유의성 판정 기준 (보정 후 p-value)
DEFAULT_ALPHA = 0.05
#: 검정에 포함할 챔버의 최소 통과 웨이퍼 수. 너무 적으면 통계가 무의미하다.
MIN_CHAMBER_WAFERS = 5


@dataclass(frozen=True)
class ChamberResult:
    """챔버 하나에 대한 검정 결과.

    Attributes:
        step_id, chamber_id, equip_id: 대상 설비
        case_through, case_total: 이상군 중 이 챔버를 지난 수 / 전체
        control_through, control_total: 정상군 중 이 챔버를 지난 수 / 전체
        odds_ratio: 오즈비 (몇 배 더 잘 일어나는가)
        ci_low, ci_high: 오즈비의 95% 신뢰구간
        p_value: Fisher 정확검정 p-value (보정 전)
        p_adj: BH-FDR 보정 후 p-value
        lift: 이상군 통과율 / 정상군 통과율
    """

    step_id: str
    chamber_id: str
    equip_id: str
    case_through: int
    case_total: int
    control_through: int
    control_total: int
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adj: float
    lift: float

    @property
    def case_rate(self) -> float:
        return self.case_through / self.case_total if self.case_total else 0.0

    @property
    def control_rate(self) -> float:
        return self.control_through / self.control_total if self.control_total else 0.0

    def is_significant(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """보정된 p-value 기준 유의성. **보정 전 p_value를 쓰지 말 것.**"""
        return self.p_adj < alpha

    def describe(self) -> str:
        """사람이 읽는 한 줄 요약 — 세 숫자를 함께 보여 준다."""
        return (
            f"{self.chamber_id} — 이상군 {self.case_through}/{self.case_total}"
            f"({self.case_rate:.1%}) vs 정상군 {self.control_through}/{self.control_total}"
            f"({self.control_rate:.1%}) · OR {self.odds_ratio:.1f}"
            f" (CI {self.ci_low:.1f}–{self.ci_high:.1f}) · p_adj={self.p_adj:.2e}"
        )


def _odds_ratio_ci(
    a: int, b: int, c: int, d: int, z: float = 1.96
) -> tuple[float, float, float]:
    """2×2 표에서 오즈비와 신뢰구간을 계산한다.

        표 구성:   이 챔버 통과   안 지남
        이상군          a            b
        정상군          c            d

    Haldane-Anscombe 보정: 셀에 0이 있으면 오즈비가 0이나 무한대가 된다.
    모든 셀에 0.5를 더해 이를 피한다. 표본이 적은 반도체 분석에서 흔히 필요하다.

    Returns:
        (오즈비, 신뢰구간 하한, 상한)
    """
    a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds = (a_ * d_) / (b_ * c_)

    # log(OR)의 표준오차 — Woolf 공식
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    log_or = np.log(odds)
    return float(odds), float(np.exp(log_or - z * se)), float(np.exp(log_or + z * se))


def to_lot_groups(
    wafer_master: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """웨이퍼 집단을 lot 집단으로 바꾼다.

    규칙:
        · 이상군 lot = 이상군 웨이퍼를 **한 장이라도** 포함한 lot
        · 정상군 lot = 정상군 웨이퍼를 포함하되 이상군 웨이퍼는 **하나도** 없는 lot

    왜 이렇게 나누나: 이상군 웨이퍼가 한 장이라도 있으면 그 lot은 문제의 설비를
        거쳤을 가능성이 있다. 반면 정상군은 순수해야 대조가 성립한다.

    Returns:
        (이상군 lot id들, 정상군 lot id들)
    """
    case_set, control_set = set(case_ids), set(control_ids)
    lot_of = wafer_master.set_index("wafer_id")["lot_id"]

    case_lots = set(lot_of.reindex(list(case_set)).dropna())
    control_lots = set(lot_of.reindex(list(control_set)).dropna()) - case_lots
    return tuple(sorted(case_lots)), tuple(sorted(control_lots))


def analyze(
    fdc_summary: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    *,
    wafer_master: pd.DataFrame | None = None,
    unit: str = "lot",
    alpha: float = DEFAULT_ALPHA,
    min_wafers: int = MIN_CHAMBER_WAFERS,
) -> pd.DataFrame:
    """모든 챔버에 대해 이상군/정상군 통과율을 비교한다.

    어떻게:
        1. 각 챔버마다 2×2 교차표를 만든다 (이상군/정상군 × 통과/미통과)
        2. Fisher 정확검정으로 p-value를 구한다
        3. 오즈비와 95% 신뢰구간을 계산한다
        4. **BH-FDR로 p-value를 보정한다** ← 이걸 빼면 결과를 믿을 수 없다

    왜 Fisher 정확검정인가: 카이제곱 검정은 기대빈도가 5 미만인 셀에서 근사가 깨진다.
        챔버가 37개면 표본이 잘게 쪼개져 저빈도 셀이 흔해진다. Fisher는 정확검정이라
        표본이 적어도 올바른 p-value를 준다. 계산이 무겁지만 정확성이 우선이다.

    Args:
        fdc_summary: FDC 요약 테이블 (wafer_id, step_id, chamber_id, equip_id 필요)
        case_ids: 이상군 웨이퍼 id
        control_ids: 정상군 웨이퍼 id
        wafer_master: lot 단위 검정에 필요한 wafer_id → lot_id 매핑
        unit: **"lot"(기본)** | "wafer" — 검정의 표본 단위.
            팹은 lot 단위로 설비를 배정하므로 lot이 통계적으로 올바른 단위다.
            "wafer"는 비교·학습 목적으로만 쓴다(표본이 25배 부풀려진다).
        alpha: 유의수준
        min_wafers: 이보다 적게 통과한 챔버는 검정에서 제외

    Returns:
        p_adj 오름차순으로 정렬된 결과 DataFrame

    Raises:
        ValueError: 두 집단이 비었거나 겹칠 때, 또는 lot 단위인데 wafer_master가 없을 때
    """
    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests

    if unit not in ("lot", "wafer"):
        raise ValueError(f"unit은 'lot' 또는 'wafer': {unit!r}")

    case_set, control_set = set(case_ids), set(control_ids)
    if not case_set or not control_set:
        raise ValueError(
            f"두 집단 모두 비어 있지 않아야 합니다 "
            f"(이상군 {len(case_set)}, 정상군 {len(control_set)})"
        )
    overlap = case_set & control_set
    if overlap:
        raise ValueError(
            f"이상군과 정상군이 {len(overlap)}장 겹칩니다. "
            f"같은 웨이퍼가 양쪽에 들어가면 검정이 무의미해집니다."
        )

    df = fdc_summary[fdc_summary["wafer_id"].isin(case_set | control_set)]
    df = df.assign(_is_case=df["wafer_id"].isin(case_set))

    if unit == "lot":
        if wafer_master is None:
            raise ValueError(
                "lot 단위 검정에는 wafer_master가 필요합니다 "
                "(wafer_id → lot_id 매핑). unit='wafer'로 바꾸거나 wafer_master를 넘기세요."
            )
        lot_of = wafer_master.set_index("wafer_id")["lot_id"]
        df = df.assign(_unit=df["wafer_id"].map(lot_of))
        # lot 하나가 한 스텝에서 쓴 챔버는 하나뿐이므로 중복을 제거한다.
        # 이상군 웨이퍼를 한 장이라도 포함하면 그 lot은 이상군으로 본다.
        df = (
            df.groupby(["step_id", "chamber_id", "equip_id", "_unit"], as_index=False)
            .agg(_is_case=("_is_case", "any"))
        )
        case_lots, control_lots = to_lot_groups(wafer_master, case_ids, control_ids)
        n_case, n_control = len(case_lots), len(control_lots)
        # 이상군 lot에 속한 웨이퍼가 정상군에도 있으면 그 lot은 이상군으로 흡수된다.
        df = df[df["_unit"].isin(set(case_lots) | set(control_lots))]
        df["_is_case"] = df["_unit"].isin(set(case_lots))
    else:
        df = df.assign(_unit=df["wafer_id"])
        n_case, n_control = len(case_set), len(control_set)
    rows: list[dict[str, object]] = []

    for (step_id, chamber_id), group in df.groupby(["step_id", "chamber_id"], sort=False):
        # 같은 단위(lot 또는 웨이퍼)가 중복 집계되지 않도록 유일화한다
        units = group.drop_duplicates("_unit")
        case_through = int(units["_is_case"].sum())
        control_through = int((~units["_is_case"]).sum())
        if case_through + control_through < min_wafers:
            continue

        # 2×2 표
        a, b = case_through, n_case - case_through
        c, d = control_through, n_control - control_through

        _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
        odds, ci_low, ci_high = _odds_ratio_ci(a, b, c, d)

        case_rate = a / n_case
        control_rate = c / n_control

        rows.append(
            {
                "step_id": step_id,
                "chamber_id": chamber_id,
                "equip_id": str(group["equip_id"].iloc[0]),
                "case_through": a,
                "case_total": n_case,
                "control_through": c,
                "control_total": n_control,
                "case_rate": case_rate,
                "control_rate": control_rate,
                "odds_ratio": odds,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "lift": case_rate / control_rate if control_rate > 0 else np.inf,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["step_id", "chamber_id", "equip_id", "case_through", "case_total",
                     "control_through", "control_total", "case_rate", "control_rate",
                     "odds_ratio", "ci_low", "ci_high", "p_value", "p_adj", "lift",
                     "significant"]
        )

    result = pd.DataFrame(rows)

    # ── 다중검정 보정 ★ ──────────────────────────────────────────────
    # 이 한 줄이 없으면 37개 챔버 중 우연히 유의해 보이는 것들이 그대로 보고된다.
    reject, p_adj, _, _ = multipletests(result["p_value"], alpha=alpha, method="fdr_bh")
    result["p_adj"] = p_adj
    result["significant"] = reject

    return result.sort_values(["p_adj", "p_value"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# 원인이 둘 이상일 때 — 껍질 벗기기(peeling)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CauseLayer:
    """섞여 있던 원인 하나를 벗겨 낸 결과.

    Attributes:
        rank: 몇 번째로 벗겨 낸 원인인가 (1부터)
        step_id, chamber_id, equip_id: 지목된 설비
        odds_ratio, p_adj: 그 층에서의 검정 결과
        n_case_before: 이 층을 분석할 때 남아 있던 불량 웨이퍼 수
        n_explained: 이 설비를 지난 불량 웨이퍼 수 (= 다음 층에서 제외됨)
        is_test_equipment: 공정 설비가 아니라 **검사 설비**인가
    """

    rank: int
    step_id: str
    chamber_id: str
    equip_id: str
    odds_ratio: float
    p_adj: float
    n_case_before: int
    n_explained: int
    is_test_equipment: bool

    def describe(self) -> str:
        kind = "검사" if self.is_test_equipment else "공정"
        return (
            f"{self.rank}층 [{kind}] {self.step_id} {self.chamber_id}  "
            f"OR {self.odds_ratio:6.2f}  p_adj {self.p_adj:.3g}  "
            f"불량 {self.n_explained}/{self.n_case_before}장 설명"
        )


def peel(
    fdc_summary: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    *,
    wafer_master: pd.DataFrame | None = None,
    unit: str = "lot",
    max_layers: int = 3,
    min_cases: int = 15,
    min_odds_ratio: float = 1.5,
) -> list[CauseLayer]:
    """원인을 하나씩 벗겨 내며 반복 분석한다 ★★.

    왜 필요한가: 커미널리티는 **불량군 전체가 한 가지 원인에서 왔다고 가정**한다.
        그런데 같은 맵 패턴이 두 경로로 생기면(식각 챔버 마모 / 프로브 카드 마모)
        불량군은 두 집단의 혼합이 된다. 그러면 각 원인은 자기 몫의 웨이퍼만
        설명하므로 **양쪽 신호가 다 같이 희석된다.** 실제로 이 프로젝트에서도
        Edge-Ring의 1위 OR이 6.4에서 3.2로 반토막 났고, Loc의 검사 기인 원인은
        상위 6위 안에 들지도 못했다.

    어떻게: 현업 엔지니어가 하는 것과 같다 — **1위를 찾고, 그 설비를 지난 불량
        웨이퍼를 빼고, 남은 것으로 다시 본다.** 첫 원인이 설명하던 웨이퍼가
        사라지면 가려져 있던 두 번째 원인이 드러난다.

    언제 멈추나:
        · 남은 불량이 `min_cases` 미만 — 통계적으로 더 볼 수 없다
        · 1위 OR이 `min_odds_ratio` 미만 — 남은 것은 신호가 아니라 잡음이다
        · `max_layers` 도달

    ⚠️ 한계 1 — **과잉 제거.** 모든 웨이퍼는 모든 스텝을 지나므로, 챔버 하나를
        벗겨 내면 그 챔버가 실제로 유발하지 않은 웨이퍼까지 대량으로 빠진다.
        스텝의 챔버가 적을수록 심하다. 실측: 포토(스캐너 3대)에서 2개 층을 벗기자
        불량이 125장 → 0장이 되어, 남아 있던 **검사 기인 원인(PC-05)이 드러날
        기회조차 사라졌다.** 챔버가 3대 이하인 스텝에서는 이 방법을 믿지 말 것.

    ⚠️ 한계 2: 벗겨 낸 층이 **진짜 독립된 원인이라는 보장은 없다.** 교락된 설비를
        1층으로 잘못 뽑으면 2층부터 전부 어긋난다. 각 층을 `stratified_test`로
        검증하고, 무엇보다 공정 엔지니어가 물리 기전으로 납득할 수 있어야 한다.

    → 원인이 **공정이냐 검사냐**를 가리는 것이 목적이라면 `by_axis()`를 쓰는 편이
      안전하다. 축을 나눠 각각의 1위를 보면 한쪽이 다른 쪽을 가릴 수 없다.

    Returns:
        벗겨 낸 순서대로의 `CauseLayer` 목록
    """
    from wafermap.config import TEST_STEP_ID

    remaining = list(case_ids)
    remaining_controls = list(control_ids)
    layers: list[CauseLayer] = []

    for rank in range(1, max_layers + 1):
        if len(remaining) < min_cases:
            break

        ranking = analyze(
            fdc_summary, remaining, remaining_controls,
            wafer_master=wafer_master, unit=unit,
        )
        if ranking.empty:
            break

        top = ranking.iloc[0]
        if float(top["odds_ratio"]) < min_odds_ratio:
            break

        # 이 설비를 지난 웨이퍼를 찾아 다음 층에서 제외한다.
        #
        # ★ 정상군에서도 똑같이 빼야 한다. 불량군에서만 빼면 남은 불량은 정의상
        #   그 스텝의 다른 챔버만 지났는데 정상군은 여전히 모든 챔버에 퍼져 있어,
        #   나머지 챔버의 OR이 기계적으로 부풀려진다. 양쪽을 함께 빼야 "그 설비를
        #   지나지 않은 웨이퍼들"이라는 **같은 조건**에서 비교하게 된다.
        #   (실측: Loc 2층 OR 15.95 → 8.47로 내려갔다.)
        through = set(
            fdc_summary[
                (fdc_summary["step_id"] == top["step_id"])
                & (fdc_summary["chamber_id"] == top["chamber_id"])
            ]["wafer_id"]
        )
        explained = set(remaining) & through

        layers.append(
            CauseLayer(
                rank=rank,
                step_id=str(top["step_id"]),
                chamber_id=str(top["chamber_id"]),
                equip_id=str(top["equip_id"]),
                odds_ratio=float(top["odds_ratio"]),
                p_adj=float(top["p_adj"]),
                n_case_before=len(remaining),
                n_explained=len(explained),
                is_test_equipment=str(top["step_id"]) == TEST_STEP_ID,
            )
        )

        if not explained:
            break
        remaining = [w for w in remaining if w not in through]
        remaining_controls = [w for w in remaining_controls if w not in through]

    return layers


def by_axis(
    fdc_summary: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    *,
    wafer_master: pd.DataFrame | None = None,
    unit: str = "lot",
) -> pd.DataFrame:
    """공정 설비와 검사 설비를 **각각** 1위까지 좁혀 나란히 보여준다 ★★.

    왜 축을 나누나: 단일 랭킹에서는 표본이 많은 쪽이 이긴다. Loc에서 실제로
        공정 기인 94장 / 검사 기인 31장이었는데, 통합 랭킹에서 진짜 검사 원인
        PC-05는 **43개 중 8위**로 밀렸다. 상위 5개만 보는 실무 관행에서는
        존재하지 않는 것과 같다. 축을 나누면 표본이 적은 쪽도 자기 축에서는
        1위로 올라와 **최소한 눈에 띈다.**

    왜 이것이 중요한가: 두 축의 조치가 완전히 다르다. 프로브 카드는 세정에 수 시간,
        공정 챔버 PM은 수 일이 걸린다. 검사 문제를 공정 탓으로 돌리면 멀쩡한 챔버를
        세우고도 불량이 계속된다. 최소한 **두 후보를 같이 놓고 비교**할 수 있어야 한다.

    Returns:
        축(process/test)별 1위 한 줄씩. `axis`, `axis_ko`, `rank_overall` 컬럼이 추가된다.
        해당 축에 후보가 없으면 그 행은 생략된다.
    """
    from wafermap.config import TEST_STEP_ID

    ranking = analyze(
        fdc_summary, case_ids, control_ids, wafer_master=wafer_master, unit=unit
    ).reset_index(drop=True)
    if ranking.empty:
        return ranking

    ranking["rank_overall"] = ranking.index + 1
    is_test = ranking["step_id"] == TEST_STEP_ID

    rows = []
    for axis, axis_ko, mask in (
        ("process", "공정 설비", ~is_test),
        ("test", "검사 설비", is_test),
    ):
        subset = ranking[mask]
        if subset.empty:
            continue
        top = subset.iloc[0].to_dict()
        top["axis"] = axis
        top["axis_ko"] = axis_ko
        rows.append(top)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# 교락 판별 — CMH 층화 검정
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StratifiedResult:
    """층화 검정 결과 — "이 설비가 진짜 원인인가, 같이 흘렀을 뿐인가".

    Attributes:
        target_step, target_chamber: 검정 대상
        stratify_step: 통제한(층으로 나눈) 스텝
        crude_or: 층화하지 않은 원래 오즈비
        adjusted_or: 층화 후 조정된 오즈비 (CMH)
        p_value: CMH 검정 p-value
        n_strata: 사용된 층의 수
        verdict: "원인 유지" | "교락 의심" | "판정 불가"
    """

    target_step: str
    target_chamber: str
    stratify_step: str
    crude_or: float
    adjusted_or: float
    p_value: float
    n_strata: int
    verdict: str

    def describe(self) -> str:
        return (
            f"{self.target_chamber} ({self.stratify_step} 고정): "
            f"OR {self.crude_or:.1f} → {self.adjusted_or:.1f} · "
            f"p={self.p_value:.2e} · {self.verdict}"
        )


def stratified_test(
    fdc_summary: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    target_step: str,
    target_chamber: str,
    stratify_step: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_per_stratum: int = 4,
    attenuation_threshold: float = 0.5,
) -> StratifiedResult:
    """다른 스텝의 설비를 고정한 채 대상 챔버의 효과가 남는지 검정한다 (CMH).

    무엇을 하나 ★: 아이스크림과 익사 사고의 예로 설명하면 —
        '더운 날'과 '선선한 날'로 나눈 뒤, 각 그룹 **안에서만** 다시 비교한다.
        더운 날들끼리만 봤을 때 관계가 사라진다면 아이스크림은 범인이 아니다.

        반도체에서는 이렇게 된다. CMP 챔버별로 나눈 뒤, 각 CMP 챔버를 쓴 웨이퍼들
        **안에서만** 식각 챔버의 효과를 본다. 효과가 남으면 식각이 진범이고,
        사라지면 CMP와의 동행 때문이었던 것이다.

    판정 기준:
        · 조정 OR이 원래 OR의 절반 미만으로 줄고 유의성도 사라짐 → **교락 의심**
        · 조정 OR이 유지되고 여전히 유의함 → **원인 유지**
        · 층이 너무 적거나 표본 부족 → **판정 불가**

    Args:
        fdc_summary: FDC 요약 테이블
        case_ids, control_ids: 이상군/정상군
        target_step, target_chamber: 검정할 대상 챔버
        stratify_step: 통제할 스텝 (이 스텝의 챔버별로 층을 나눈다)
        alpha: 유의수준
        min_per_stratum: 층 하나에 최소 이만큼 웨이퍼가 있어야 사용한다
        attenuation_threshold: OR이 이 비율 미만으로 줄면 교락으로 본다

    Returns:
        StratifiedResult
    """
    from statsmodels.stats.contingency_tables import StratifiedTable

    case_set, control_set = set(case_ids), set(control_ids)
    all_ids = case_set | control_set

    df = fdc_summary[fdc_summary["wafer_id"].isin(all_ids)]

    target = (
        df[df["step_id"] == target_step]
        .set_index("wafer_id")["chamber_id"]
        .eq(target_chamber)
    )
    strata = df[df["step_id"] == stratify_step].set_index("wafer_id")["chamber_id"]

    joined = pd.DataFrame({"through": target, "stratum": strata}).dropna()
    joined["is_case"] = joined.index.isin(case_set)

    # 층별 2×2 표를 쌓는다. statsmodels는 (2, 2, n_strata) 형태를 기대한다.
    tables: list[np.ndarray] = []
    for _, group in joined.groupby("stratum", sort=False):
        if len(group) < min_per_stratum:
            continue
        a = int((group["is_case"] & group["through"]).sum())
        b = int((group["is_case"] & ~group["through"]).sum())
        c = int((~group["is_case"] & group["through"]).sum())
        d = int((~group["is_case"] & ~group["through"]).sum())
        # 한 행이나 열이 통째로 0이면 그 층은 정보를 주지 못한다
        if (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
            continue
        tables.append(np.array([[a, b], [c, d]]))

    crude_a = int((joined["is_case"] & joined["through"]).sum())
    crude_b = int((joined["is_case"] & ~joined["through"]).sum())
    crude_c = int((~joined["is_case"] & joined["through"]).sum())
    crude_d = int((~joined["is_case"] & ~joined["through"]).sum())
    crude_or, _, _ = _odds_ratio_ci(crude_a, crude_b, crude_c, crude_d)

    if len(tables) < 2:
        return StratifiedResult(
            target_step=target_step,
            target_chamber=target_chamber,
            stratify_step=stratify_step,
            crude_or=crude_or,
            adjusted_or=float("nan"),
            p_value=float("nan"),
            n_strata=len(tables),
            verdict="판정 불가",
        )

    st = StratifiedTable(np.dstack(tables))
    adjusted_or = float(st.oddsratio_pooled)
    p_value = float(st.test_null_odds().pvalue)

    attenuated = adjusted_or < crude_or * attenuation_threshold
    still_significant = p_value < alpha
    if attenuated and not still_significant:
        verdict = "교락 의심"
    elif still_significant:
        verdict = "원인 유지"
    else:
        verdict = "교락 의심"

    return StratifiedResult(
        target_step=target_step,
        target_chamber=target_chamber,
        stratify_step=stratify_step,
        crude_or=crude_or,
        adjusted_or=adjusted_or,
        p_value=p_value,
        n_strata=len(tables),
        verdict=verdict,
    )


def screen_confounders(
    fdc_summary: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    ranking: pd.DataFrame,
    *,
    top_n: int = 5,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """상위 후보들을 서로 층화해 교락 여부를 판정한다.

    어떻게: 상위 N개 후보를 뽑아, 각 후보를 **다른 후보들로 층화**해 본다.
        진범이라면 어떤 설비를 고정해도 효과가 남고, 교락이라면 진범을 고정하는
        순간 효과가 사라진다.

    왜 상위만 보나: 모든 조합을 검정하면 그것 자체가 다중검정 문제를 만든다.
        유의하게 나온 소수 후보에만 적용하는 것이 표준 절차다.

    Returns:
        후보별 층화 결과 요약 DataFrame
    """
    significant = ranking[ranking["significant"]].head(top_n)
    if len(significant) < 2:
        return pd.DataFrame(
            columns=["step_id", "chamber_id", "crude_or", "min_adjusted_or",
                     "n_tests", "n_survived", "verdict"]
        )

    rows: list[dict[str, object]] = []
    for _, target in significant.iterrows():
        results: list[StratifiedResult] = []
        for _, other in significant.iterrows():
            if other["step_id"] == target["step_id"]:
                continue
            results.append(
                stratified_test(
                    fdc_summary,
                    case_ids,
                    control_ids,
                    str(target["step_id"]),
                    str(target["chamber_id"]),
                    str(other["step_id"]),
                    alpha=alpha,
                )
            )

        usable = [r for r in results if np.isfinite(r.adjusted_or)]
        n_survived = sum(1 for r in usable if r.verdict == "원인 유지")

        if not usable:
            verdict = "판정 불가"
        elif n_survived == len(usable):
            verdict = "원인 유지"
        elif n_survived == 0:
            verdict = "교락 의심"
        else:
            verdict = "부분 유지"

        rows.append(
            {
                "step_id": target["step_id"],
                "chamber_id": target["chamber_id"],
                "crude_or": float(target["odds_ratio"]),
                "min_adjusted_or": min((r.adjusted_or for r in usable), default=float("nan")),
                "n_tests": len(usable),
                "n_survived": n_survived,
                "verdict": verdict,
            }
        )

    return pd.DataFrame(rows)
