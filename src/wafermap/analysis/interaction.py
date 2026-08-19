"""공정 간 교호작용 분석 — M5.5-③ (첨부 도메인 문서 §14·15·16·21·37).

무엇을: 두 파라미터(또는 두 설비)의 **조합**이 만드는 불량을 찾는다.
왜:     지금까지의 분석은 전부 "어느 하나가 이탈했나"를 물었다. 그런데 실제 팹에서는
        **각각은 규격 안인데 조합이 문제인** 경우가 흔하다.

            증착이 두께 상한 쪽  +  CMP 제거량이 하한 쪽  →  중심부 잔막
            (규격 안)              (규격 안)                  (불량)

        이 구조에서는 단변량 SPC도, 규격 위반 검사도, 파라미터별 평균 비교도
        전부 조용하다. **주효과가 0이기 때문이다.**

핵심 도구는 두 가지다.

1. **2×2 분할표** (`pair_table`) — 두 파라미터를 각각 상/하로 나눠 네 칸의 불량률을 본다.
   교호작용이 있으면 대각선 두 칸만 높다. 이것이 가장 설명하기 쉬운 형태다.
2. **SHAP interaction values** (`shap_pairs`) — 모델이 학습한 상호작용 기여를 분해한다.
   변수가 많을 때 후보를 좁히는 용도.

⚠️ 이 분석은 **탐색**이다. 쌍의 수가 파라미터 수의 제곱으로 늘어나므로 우연히 유의해
   보이는 쌍이 반드시 나온다. 다중검정 보정을 하고, 물리적으로 말이 되는지 반드시 확인할 것.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from wafermap.config import STEPS_BY_ID, is_controllable

#: 2×2 분할표의 각 칸에 최소한 있어야 하는 웨이퍼 수.
#: 이보다 적으면 불량률이 0% 아니면 100%로 튀어 아무 의미가 없다.
MIN_CELL = 8

#: 교호작용 후보로 볼 최소 대비값(비율 차이의 차이). 아래면 잡음으로 본다.
#:
#: 왜 0.5%p인가: 이 값은 **전체 불량률에 맞춰야 한다.** 처음에 5%p로 잡았더니
#: 불량률이 0.87%인 데이터에서 대비가 1.9%p인 진짜 교호작용이 통째로 걸러졌다.
#: 불량률이 낮으면 대비도 작을 수밖에 없다. 데이터의 불량률이 크게 다르면
#: `hidden_from_main_effects(min_contrast=...)`로 조정할 것.
MIN_CONTRAST = 0.005


def cross_step_matrix(
    fdc_summary: pd.DataFrame,
    wafer_ids: list[str] | tuple[str, ...],
    *,
    steps: tuple[str, ...] | None = None,
    controllable_only: bool = True,
) -> pd.DataFrame:
    """여러 스텝의 파라미터를 웨이퍼 1행으로 합친 넓은 행렬을 만든다.

    왜 필요한가: FDC는 (웨이퍼 × 스텝) 긴 형태다. 스텝을 가로지르는 조합을 보려면
        한 웨이퍼의 증착 조건과 CMP 조건이 **같은 행**에 있어야 한다.

    컬럼명은 `P030.precursor_flow` 처럼 스텝을 접두로 붙인다. 스텝이 달라도 같은
    이름의 파라미터가 있기 때문이다(`chamber_pressure`는 식각에도 증착에도 있다).

    Args:
        fdc_summary: FDC 요약 테이블
        wafer_ids: 대상 웨이퍼
        steps: 포함할 스텝 (None이면 전부)
        controllable_only: 계측값을 뺄지. 조작할 수 없는 값의 조합은 조치로 이어지지 않는다

    Returns:
        wafer_id를 인덱스로 하는 넓은 행렬
    """
    wanted = set(wafer_ids)
    rows = fdc_summary[fdc_summary["wafer_id"].isin(wanted)]
    if steps is not None:
        rows = rows[rows["step_id"].isin(steps)]

    frames = []
    for step_id, group in rows.groupby("step_id"):
        cols = [
            c for c in group.columns
            if c.endswith("_mean")
            and (not controllable_only or is_controllable(c))
        ]
        if not cols:
            continue
        sub = group.set_index("wafer_id")[cols]
        sub.columns = [f"{step_id}.{c[:-len('_mean')]}" for c in cols]
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(axis=1, how="all")


@dataclass(frozen=True)
class PairResult:
    """파라미터 두 개의 조합 분석 결과.

    Attributes:
        param_a, param_b: 두 파라미터 (`P030.precursor_flow` 형식)
        rate_ll, rate_lh, rate_hl, rate_hh: 네 칸(저/고 조합)의 불량률
        n_ll, n_lh, n_hl, n_hh: 각 칸의 웨이퍼 수
        main_a, main_b: 각 파라미터 **단독**의 불량률 차이 (고 − 저)
        contrast: 교호작용 대비 = (hh − hl) − (lh − ll)
        p_value, p_adj: 비가법성 검정 p값과 보정값
    """

    param_a: str
    param_b: str
    rate_ll: float
    rate_lh: float
    rate_hl: float
    rate_hh: float
    n_ll: int
    n_lh: int
    n_hl: int
    n_hh: int
    main_a: float
    main_b: float
    contrast: float
    p_value: float
    p_adj: float = float("nan")

    def is_hidden(self, min_contrast: float = MIN_CONTRAST) -> bool:
        """주효과로는 안 보이는가 ★.

        교호작용 대비는 큰데 두 파라미터 각각의 단독 효과는 작은 경우.
        **단변량 분석이 구조적으로 놓치는 상황**이 정확히 이것이다.
        """
        return (
            abs(self.contrast) > min_contrast
            and abs(self.main_a) < abs(self.contrast) / 2
            and abs(self.main_b) < abs(self.contrast) / 2
        )

    @property
    def hidden_from_main_effects(self) -> bool:
        return self.is_hidden()

    def table(self) -> str:
        """2×2 분할표를 사람이 읽는 형태로."""
        short_a = self.param_a.split(".")[-1]
        short_b = self.param_b.split(".")[-1]
        return (
            f"                {short_b} 낮음      {short_b} 높음\n"
            f"  {short_a} 높음   {self.rate_hl:>6.1%} ({self.n_hl:>3}장)"
            f"   {self.rate_hh:>6.1%} ({self.n_hh:>3}장)\n"
            f"  {short_a} 낮음   {self.rate_ll:>6.1%} ({self.n_ll:>3}장)"
            f"   {self.rate_lh:>6.1%} ({self.n_lh:>3}장)"
        )

    def describe(self) -> str:
        mark = "★" if self.hidden_from_main_effects else " "
        return (
            f"{mark} {self.param_a} × {self.param_b}\n"
            f"    교호작용 대비 {self.contrast:+.1%}  "
            f"(단독 효과: {self.param_a.split('.')[-1]} {self.main_a:+.1%}, "
            f"{self.param_b.split('.')[-1]} {self.main_b:+.1%})  p_adj {self.p_adj:.3g}"
        )


def pair_table(
    matrix: pd.DataFrame,
    is_case: np.ndarray,
    param_a: str,
    param_b: str,
    *,
    min_cell: int = MIN_CELL,
) -> PairResult | None:
    """두 파라미터를 중앙값으로 나눠 2×2 불량률 표를 만든다.

    어떻게: 각 파라미터를 **정상군 중앙값** 기준으로 상/하로 나눈다.
        전체 중앙값을 쓰면 불량군이 한쪽에 몰려 있을 때 경계가 끌려간다.

    왜 이 표가 핵심인가 ★: 교호작용을 설명하는 가장 쉬운 형태다. 대각선 두 칸만
        불량률이 높으면 "둘이 같은 방향으로 치우칠 때만 문제"라는 뜻이고,
        한 줄만 높으면 그건 교호작용이 아니라 그 파라미터의 주효과다.
        회귀계수나 SHAP 값보다 공정 엔지니어를 설득하기 쉽다.

    Returns:
        PairResult. 어느 칸이든 표본이 `min_cell` 미만이면 None
    """
    from scipy.stats import chi2_contingency

    a = matrix[param_a].to_numpy(dtype=float)
    b = matrix[param_b].to_numpy(dtype=float)
    control = ~is_case
    if control.sum() < 4:
        return None

    cut_a, cut_b = np.median(a[control]), np.median(b[control])
    hi_a, hi_b = a > cut_a, b > cut_b

    cells = {
        "ll": ~hi_a & ~hi_b,
        "lh": ~hi_a & hi_b,
        "hl": hi_a & ~hi_b,
        "hh": hi_a & hi_b,
    }
    counts = {k: int(v.sum()) for k, v in cells.items()}
    if min(counts.values()) < min_cell:
        return None

    rates = {k: float(is_case[v].mean()) for k, v in cells.items()}

    # 비가법성 검정 — 2×2 표에서 관측 불량 수가 가법 모형과 다른가
    observed = np.array([
        [is_case[cells["ll"]].sum(), counts["ll"] - is_case[cells["ll"]].sum()],
        [is_case[cells["hh"]].sum(), counts["hh"] - is_case[cells["hh"]].sum()],
    ])
    off = np.array([
        [is_case[cells["lh"]].sum(), counts["lh"] - is_case[cells["lh"]].sum()],
        [is_case[cells["hl"]].sum(), counts["hl"] - is_case[cells["hl"]].sum()],
    ])
    # 대각선(같은 방향) vs 비대각선(엇갈린 방향)을 비교한다
    table = np.array([observed.sum(axis=0), off.sum(axis=0)])
    try:
        p_value = float(chi2_contingency(table)[1]) if table.min() >= 0 else 1.0
    except ValueError:
        p_value = 1.0

    return PairResult(
        param_a=param_a,
        param_b=param_b,
        rate_ll=rates["ll"], rate_lh=rates["lh"],
        rate_hl=rates["hl"], rate_hh=rates["hh"],
        n_ll=counts["ll"], n_lh=counts["lh"],
        n_hl=counts["hl"], n_hh=counts["hh"],
        main_a=(rates["hh"] + rates["hl"]) / 2 - (rates["lh"] + rates["ll"]) / 2,
        main_b=(rates["hh"] + rates["lh"]) / 2 - (rates["hl"] + rates["ll"]) / 2,
        contrast=(rates["hh"] - rates["hl"]) - (rates["lh"] - rates["ll"]),
        p_value=p_value,
    )


def screen_pairs(
    matrix: pd.DataFrame,
    is_case: np.ndarray,
    *,
    cross_step_only: bool = True,
    top_n: int = 20,
    min_cell: int = MIN_CELL,
) -> list[PairResult]:
    """모든 파라미터 쌍을 훑어 교호작용 후보를 찾는다.

    왜 다중검정 보정이 필수인가 ★★: 파라미터 40개면 쌍이 780개다. 유의수준 5%로
        검정하면 **아무 신호가 없어도 39개가 유의하게 나온다.** 보정 없이
        "교호작용을 찾았다"고 말하면 거의 확실히 잡음을 보고하는 것이다.

    Args:
        matrix: `cross_step_matrix()` 결과
        is_case: 불량 여부 (matrix의 행 순서와 같아야 한다)
        cross_step_only: 같은 스텝 안의 쌍은 제외. 같은 스텝 파라미터끼리는
            원래 강하게 상관돼 있어 교호작용처럼 보이기 쉽다
        top_n: 돌려줄 상위 개수
        min_cell: 2×2 각 칸의 최소 표본

    Returns:
        대비 절댓값 내림차순 상위 목록
    """
    columns = list(matrix.columns)
    results: list[PairResult] = []

    for col_a, col_b in combinations(columns, 2):
        if cross_step_only and col_a.split(".")[0] == col_b.split(".")[0]:
            continue
        result = pair_table(matrix, is_case, col_a, col_b, min_cell=min_cell)
        if result is not None:
            results.append(result)

    if not results:
        return []

    reject, p_adj, _, _ = multipletests(
        [r.p_value for r in results], alpha=0.05, method="fdr_bh"
    )
    results = [
        PairResult(**{**r.__dict__, "p_adj": float(adj)})
        for r, adj in zip(results, p_adj)
    ]
    results.sort(key=lambda r: abs(r.contrast), reverse=True)
    return results[:top_n]


# ──────────────────────────────────────────────────────────────────────────
# 설비 조합 — "B가 문제"가 아니라 "A → B 조합에서 증가한다"
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EquipPairResult:
    """앞 공정 챔버 × 뒤 공정 챔버 조합의 불량률.

    Attributes:
        step_a, chamber_a: 앞 공정
        step_b, chamber_b: 뒤 공정
        n_pair: 이 조합을 지난 웨이퍼 수
        rate_pair: 이 조합의 불량률
        rate_a_only: chamber_a를 지났지만 chamber_b는 아닌 웨이퍼의 불량률
        rate_b_only: chamber_b만
        rate_other: 둘 다 아닌 웨이퍼
        excess: 조합이 각각의 단독보다 얼마나 더 나쁜가
    """

    step_a: str
    chamber_a: str
    step_b: str
    chamber_b: str
    n_pair: int
    rate_pair: float
    rate_a_only: float
    rate_b_only: float
    rate_other: float
    excess: float

    def describe(self) -> str:
        return (
            f"{self.chamber_a} → {self.chamber_b}  "
            f"불량률 {self.rate_pair:.1%} ({self.n_pair}장)  "
            f"vs 단독 {self.rate_a_only:.1%}/{self.rate_b_only:.1%} · "
            f"그 외 {self.rate_other:.1%}  초과 {self.excess:+.1%}"
        )


def equipment_pairs(
    fdc_summary: pd.DataFrame,
    case_ids: list[str] | tuple[str, ...],
    all_ids: list[str] | tuple[str, ...],
    *,
    step_a: str,
    step_b: str,
    min_wafers: int = 25,
    top_n: int = 10,
) -> list[EquipPairResult]:
    """두 스텝의 챔버 조합별 불량률을 비교한다.

    왜 필요한가 (첨부 문서 §15 예시 3): 같은 후공정 장비라도 **어떤 선행 장비를
        거쳤느냐**에 따라 결과가 달라질 수 있다. 그러면 답은 "B가 문제다"가 아니라
        "**A → B 조합**에서 문제가 증가한다"이며, 조치도 달라진다 — B를 세우는
        대신 그 조합만 피하도록 디스패치 규칙을 바꾸면 된다.

    Returns:
        초과 불량률 내림차순 상위 목록
    """
    case_set, all_set = set(case_ids), set(all_ids)
    rows = fdc_summary[
        fdc_summary["step_id"].isin({step_a, step_b})
        & fdc_summary["wafer_id"].isin(all_set)
    ]
    wide = rows.pivot_table(
        index="wafer_id", columns="step_id", values="chamber_id", aggfunc="first"
    )
    if step_a not in wide.columns or step_b not in wide.columns:
        return []

    wide = wide.dropna(subset=[step_a, step_b])
    is_case = wide.index.isin(case_set)

    results = []
    for chamber_a in sorted(wide[step_a].unique()):
        for chamber_b in sorted(wide[step_b].unique()):
            in_a = (wide[step_a] == chamber_a).to_numpy()
            in_b = (wide[step_b] == chamber_b).to_numpy()
            pair = in_a & in_b
            if pair.sum() < min_wafers:
                continue

            def rate(mask: np.ndarray) -> float:
                return float(is_case[mask].mean()) if mask.sum() else float("nan")

            rate_pair = rate(pair)
            rate_a = rate(in_a & ~in_b)
            rate_b = rate(~in_a & in_b)
            # 각각 단독으로 지났을 때보다 조합에서 얼마나 더 나쁜가
            solo = np.nanmax([rate_a, rate_b])
            results.append(
                EquipPairResult(
                    step_a=step_a, chamber_a=chamber_a,
                    step_b=step_b, chamber_b=chamber_b,
                    n_pair=int(pair.sum()),
                    rate_pair=rate_pair,
                    rate_a_only=rate_a,
                    rate_b_only=rate_b,
                    rate_other=rate(~in_a & ~in_b),
                    excess=float(rate_pair - solo),
                )
            )

    results.sort(key=lambda r: r.excess, reverse=True)
    return results[:top_n]


# ──────────────────────────────────────────────────────────────────────────
# SHAP interaction — 모델이 학습한 상호작용
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ShapPairReport:
    """SHAP interaction 상위 쌍."""

    pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def describe(self, top_n: int = 5) -> str:
        lines = []
        for a, b, value in self.pairs[:top_n]:
            lines.append(f"  {a} × {b}   평균 |상호작용| {value:.4f}")
        return "\n".join(lines) or "  (상호작용 쌍 없음)"


def shap_pairs(
    matrix: pd.DataFrame,
    is_case: np.ndarray,
    *,
    top_n: int = 10,
    max_samples: int = 400,
    seed: int = 0,
) -> ShapPairReport:
    """TreeSHAP interaction values로 상호작용 쌍의 순위를 매긴다.

    ⚠️ 비용이 크다: 계산량이 피처 수의 **제곱**에 비례한다. 그래서 표본을 줄이고,
        피처도 미리 좁혀서 넣어야 한다. 이 함수는 후보를 좁히는 용도이고,
        확인은 `pair_table()`의 2×2 표로 하는 것이 낫다 — 설명이 쉽기 때문이다.
    """
    import lightgbm as lgb
    import shap

    rng = np.random.default_rng(seed)
    y = is_case.astype(int)
    n_case, n_control = int(y.sum()), int((1 - y).sum())
    if n_case < 10 or n_control < 10:
        raise ValueError(f"표본이 부족합니다 (불량 {n_case}, 정상 {n_control})")

    params = {
        "objective": "binary", "verbose": -1, "num_leaves": 15,
        "learning_rate": 0.05, "min_data_in_leaf": 10, "seed": seed,
        "scale_pos_weight": n_control / max(n_case, 1),
    }
    model = lgb.train(params, lgb.Dataset(matrix, label=y), num_boost_round=120)

    if len(matrix) > max_samples:
        idx = rng.choice(len(matrix), size=max_samples, replace=False)
        sample = matrix.iloc[idx]
    else:
        sample = matrix

    values = shap.TreeExplainer(model).shap_interaction_values(sample)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)

    strength = np.abs(values).mean(axis=0)
    np.fill_diagonal(strength, 0.0)  # 대각선은 주효과다

    pairs = []
    columns = list(matrix.columns)
    for i, j in combinations(range(len(columns)), 2):
        pairs.append((columns[i], columns[j], float(strength[i, j] * 2)))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return ShapPairReport(pairs=pairs[:top_n])


def summarize(results: list[PairResult], *, top_n: int = 5) -> str:
    """콘솔용 요약."""
    if not results:
        return "  교호작용 후보가 없습니다."
    lines = []
    for result in results[:top_n]:
        lines.append(result.describe())
    hidden = [r for r in results[:top_n] if r.hidden_from_main_effects]
    if hidden:
        lines.append("")
        lines.append(
            f"  ★ 표시 {len(hidden)}건은 **주효과로는 안 보이는** 쌍이다. "
            "단변량 분석으로는 찾을 수 없다."
        )
    return "\n".join(lines)
