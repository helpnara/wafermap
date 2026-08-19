"""파라미터 기여도 해석 — SHAP 결과를 "조치 가능한 결론"으로 바꾼다.

무엇을: SHAP 기여도 랭킹에 세 가지 검증을 덧붙인다.
    1. **조치 가능성** — 조작할 수 있는 파라미터인가, 결과로 측정된 값인가
    2. **분포 차이** — 불량군과 정상군에서 실제로 값이 다른가 (모델과 독립적인 증거)
    3. **규격 대비 위치** — 관리 규격을 벗어났는가

왜 SHAP만으로 부족한가 ★:

    ① **SHAP은 상관을 보여 줄 뿐 인과를 증명하지 않는다.**
       기여도가 크다는 것은 "예측에 크게 쓰였다"이지 "원인이다"가 아니다.

    ② **계측값이 상위에 오는 함정.** 이 프로젝트에서 실제로 관측했다.
       Donut 불량에서 `thickness_sigma`(두께 산포)가 1위로 나왔는데,
       정답은 `susceptor_temp_mid_delta`(서셉터 온도 편차)였다.

       틀린 결과는 아니다. 온도가 튀면 두께 산포가 커지므로 물리적으로 이어져 있다.
       문제는 **두께 산포를 조작할 수 없다**는 것이다. 조치하려면 온도를 잡아야 한다.

           온도 이상 (원인, 조작 가능)  →  두께 산포 ↑ (결과, 측정만 가능)  →  Donut 불량
                    ↑ 여기를 고쳐야 함        ↑ SHAP은 여기가 더 강하다고 말함

       계측값은 원인에 더 가까운 위치에 있어 신호가 더 선명한 경우가 많다.
       그래서 **조치 가능한 파라미터만 따로 랭킹**하는 절차가 필요하다.

참고: docs/00_design.md §M4, docs/08_interpretation_guide.md §[5]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wafermap.config import PARAM_ROLE, STEPS_BY_ID, is_controllable
from wafermap.features.trace_feat import FEATURE_SUFFIXES as TRACE_FEATURE_SUFFIXES

#: 분포 차이가 '크다'고 볼 Cliff's delta 기준 (관례: 0.147 작음 / 0.33 중간 / 0.474 큼)
LARGE_EFFECT = 0.33


#: 컬럼명에서 떼어 낼 접미사. 요약통계 + 시계열 파생 피처(M5.5-②).
#: 긴 것부터 검사해야 `_n_excursions`가 `_std`류와 섞이지 않는다.
_COLUMN_SUFFIXES: tuple[str, ...] = tuple(
    sorted(("_mean", "_std", "_min", "_max") + TRACE_FEATURE_SUFFIXES, key=len, reverse=True)
)


def base_param_name(column: str) -> str:
    """`chamber_pressure_mean` → `chamber_pressure` 로 접미사를 떼어 낸다.

    시계열 파생 피처도 같은 파라미터를 가리킨다. `bath_temp_time_above`가
    `bath_temp`로 돌아와야 "정답 파라미터를 찾았는가"를 제대로 채점할 수 있다.
    """
    for suffix in _COLUMN_SUFFIXES:
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


@dataclass(frozen=True)
class ParamEvidence:
    """파라미터 하나에 대한 종합 증거.

    Attributes:
        column: FDC 컬럼명 (예: `chamber_pressure_mean`)
        param: 기본 파라미터명 (예: `chamber_pressure`)
        role: "control"(조작 가능) | "measurement"(결과 측정)
        shap_importance: 평균 |SHAP|
        shap_rank: SHAP 기준 순위 (1부터)
        direction: +1(값이 클수록 불량) | -1(작을수록) | 0(불명)
        case_mean, control_mean: 불량군/정상군 평균
        delta_sigma: 정상군 표준편차 단위의 차이
        cliffs_delta: 분포 겹침 정도 (-1~1, 절댓값이 클수록 잘 분리)
        ks_pvalue: 두 분포가 같은지 검정한 p-value
        spec_violation_rate: 불량군 중 규격을 벗어난 비율
    """

    column: str
    param: str
    role: str
    shap_importance: float
    shap_rank: int
    direction: int
    case_mean: float
    control_mean: float
    delta_sigma: float
    cliffs_delta: float
    ks_pvalue: float
    spec_violation_rate: float

    @property
    def is_controllable(self) -> bool:
        return self.role == "control"

    @property
    def has_distribution_evidence(self) -> bool:
        """모델과 무관하게 분포 자체가 다른가."""
        return abs(self.cliffs_delta) >= LARGE_EFFECT and self.ks_pvalue < 0.05

    def describe(self) -> str:
        arrow = "↑" if self.direction > 0 else ("↓" if self.direction < 0 else "·")
        tag = "조작가능" if self.is_controllable else "계측값"
        evidence = "✅" if self.has_distribution_evidence else "  "
        return (
            f"{evidence} {self.column:<30}[{tag}] "
            f"|SHAP|={self.shap_importance:.4f} {arrow} "
            f"Δ={self.delta_sigma:+.2f}σ  δ={self.cliffs_delta:+.2f}"
        )


def cliffs_delta(case: np.ndarray, control: np.ndarray, *, max_pairs: int = 200_000) -> float:
    """Cliff's delta — 두 분포가 얼마나 겹치지 않는지.

    정의: P(case > control) − P(case < control). 범위는 -1~1.

    왜 평균 차이 대신 이걸 쓰나: 평균 차이는 이상치 하나에 크게 흔들리고 단위에
        의존한다. Cliff's delta는 **순위만 보므로** 이상치에 강건하고 단위가 없다.
        FDC 파라미터는 단위가 제각각(psi, degC, sccm)이라 비교하려면 무단위 지표가 필요하다.

    Args:
        case, control: 두 집단의 값
        max_pairs: 비교할 최대 쌍 수. 초과하면 표본을 줄여 계산한다.
    """
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    if len(case) == 0 or len(control) == 0:
        return 0.0

    # 전수 비교가 너무 크면 표본을 줄인다 (O(n·m) 이라 금방 폭증한다)
    if len(case) * len(control) > max_pairs:
        rng = np.random.default_rng(0)
        limit = int(np.sqrt(max_pairs))
        if len(case) > limit:
            case = rng.choice(case, limit, replace=False)
        if len(control) > limit:
            control = rng.choice(control, limit, replace=False)

    greater = (case[:, None] > control[None, :]).sum()
    less = (case[:, None] < control[None, :]).sum()
    total = len(case) * len(control)
    return float((greater - less) / total)


def compare_distributions(
    fdc_summary: pd.DataFrame,
    step_id: str,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    columns: list[str],
    *,
    chamber_id: str | None = None,
) -> pd.DataFrame:
    """불량군/정상군의 파라미터 분포를 비교한다 (모델과 독립적인 증거).

    왜 모델과 따로 보나 ★: SHAP은 모델이 학습한 것을 설명한다. 모델이 과적합했거나
        엉뚱한 신호를 잡았다면 SHAP도 그것을 충실히 설명할 뿐이다.

        분포 비교는 **모델을 거치지 않고** 데이터를 직접 본다. 두 결과가 일치하면
        신뢰도가 크게 올라가고, 어긋나면 둘 중 하나를 의심해야 한다.

    Returns:
        컬럼별 (case_mean, control_mean, delta_sigma, cliffs_delta, ks_pvalue,
        spec_violation_rate)
    """
    from scipy.stats import ks_2samp

    df = fdc_summary[fdc_summary["step_id"] == step_id]
    if chamber_id is not None:
        df = df[df["chamber_id"] == chamber_id]

    case_set, control_set = set(case_ids), set(control_ids)
    case_df = df[df["wafer_id"].isin(case_set)]
    control_df = df[df["wafer_id"].isin(control_set)]

    step = STEPS_BY_ID.get(step_id)
    rows: list[dict[str, object]] = []

    for column in columns:
        if column not in df.columns:
            continue
        case_values = case_df[column].to_numpy(dtype=float)
        control_values = control_df[column].to_numpy(dtype=float)
        if len(case_values) < 3 or len(control_values) < 3:
            continue

        control_sd = float(np.nanstd(control_values, ddof=1))
        case_mean = float(np.nanmean(case_values))
        control_mean = float(np.nanmean(control_values))
        delta_sigma = (case_mean - control_mean) / control_sd if control_sd > 1e-12 else 0.0

        try:
            ks_p = float(ks_2samp(case_values, control_values).pvalue)
        except ValueError:
            ks_p = 1.0

        # 규격 위반율 — 불량군 중 spec을 벗어난 비율
        violation = np.nan
        base = base_param_name(column)
        if step is not None and column.endswith("_mean"):
            try:
                spec = step.param(base)
                violation = float(
                    np.mean((case_values < spec.spec_lo) | (case_values > spec.spec_hi))
                )
            except KeyError:
                pass

        rows.append(
            {
                "column": column,
                "param": base,
                "role": PARAM_ROLE.get(base, "control"),
                "case_mean": case_mean,
                "control_mean": control_mean,
                "delta_sigma": delta_sigma,
                "cliffs_delta": cliffs_delta(case_values, control_values),
                "ks_pvalue": ks_p,
                "spec_violation_rate": violation,
            }
        )

    return pd.DataFrame(rows)


def build_evidence(
    result,
    fdc_summary: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    *,
    chamber_id: str | None = None,
    top_n: int = 15,
) -> list[ParamEvidence]:
    """SHAP 결과에 분포 증거와 조치 가능성을 결합한다.

    Args:
        result: cause_model.fit()의 결과
        fdc_summary: FDC 요약 테이블
        case_ids, control_ids: 불량/정상 웨이퍼
        chamber_id: 층화한 챔버
        top_n: SHAP 상위 몇 개까지 검토할지

    Returns:
        ParamEvidence 목록 (SHAP 순위 순)
    """
    top_columns = list(result.shap_importance.head(top_n).index)
    dist = compare_distributions(
        fdc_summary, result.step_id, case_ids, control_ids, top_columns,
        chamber_id=chamber_id,
    ).set_index("column")

    evidence: list[ParamEvidence] = []
    for rank, column in enumerate(top_columns, start=1):
        if column not in dist.index:
            continue
        row = dist.loc[column]
        evidence.append(
            ParamEvidence(
                column=column,
                param=str(row["param"]),
                role=str(row["role"]),
                shap_importance=float(result.shap_importance[column]),
                shap_rank=rank,
                direction=int(result.direction.get(column, 0)),
                case_mean=float(row["case_mean"]),
                control_mean=float(row["control_mean"]),
                delta_sigma=float(row["delta_sigma"]),
                cliffs_delta=float(row["cliffs_delta"]),
                ks_pvalue=float(row["ks_pvalue"]),
                spec_violation_rate=float(row["spec_violation_rate"])
                if np.isfinite(row["spec_violation_rate"]) else float("nan"),
            )
        )
    return evidence


def actionable_ranking(evidence: list[ParamEvidence]) -> list[ParamEvidence]:
    """조치 가능한 파라미터만 남겨 다시 순위를 매긴다 ★.

    왜 필요한가: 계측값(measurement)은 공정의 **결과**라 조작할 수 없다.
        SHAP 1위가 계측값이면 "그래서 뭘 하면 되나?"에 답할 수 없다.

        예시 (이 프로젝트에서 실제로 나온 결과):
            SHAP 1위: thickness_sigma_mean   [계측값]  ← 조치 불가
            SHAP 2위: susceptor_temp_mid_delta_mean [조작가능] ← 여기를 고쳐야 함

        계측값이 상위라는 사실 자체는 유용한 정보다 — 그 계측값을 만들어 내는
        상류 파라미터를 찾으라는 신호이기 때문이다. 하지만 **개선안의 대상은
        조작 가능한 파라미터여야 한다.**

    Returns:
        조작 가능한 파라미터만, SHAP 순위를 유지한 채
    """
    return [e for e in evidence if e.is_controllable]


def summarize(evidence: list[ParamEvidence], *, top_n: int = 8) -> str:
    """사람이 읽는 요약 — 조치 가능성과 분포 증거를 함께 보여 준다."""
    lines = [
        f"  {'':<3}{'파라미터':<30}{'구분':<8}{'|SHAP|':>9}{'Δσ':>8}{'δ':>7}{'규격위반':>9}",
        "  " + "-" * 76,
    ]
    for e in evidence[:top_n]:
        tag = "조작가능" if e.is_controllable else "계측값"
        mark = "✅" if e.has_distribution_evidence else "  "
        arrow = "↑" if e.direction > 0 else ("↓" if e.direction < 0 else "·")
        violation = (
            f"{e.spec_violation_rate:>8.0%}" if np.isfinite(e.spec_violation_rate) else "       -"
        )
        lines.append(
            f"  {mark} {e.column:<30}{tag:<8}{e.shap_importance:>9.4f}"
            f"{e.delta_sigma:>+7.2f}{arrow}{e.cliffs_delta:>+7.2f}{violation}"
        )
    lines.append("")
    lines.append("  ✅ = 분포 차이도 크다 (모델과 독립적인 증거)")
    lines.append("  Δσ = 정상군 표준편차 단위 평균 차이 · δ = Cliff's delta (분포 분리도)")
    return "\n".join(lines)
