"""원인 파라미터 규명 모델 — "그 챔버의 무엇이 문제인가"를 찾는다.

무엇을: M3가 특정한 스텝의 FDC 파라미터들로 "이 웨이퍼가 불량이 될까"를 예측하는
        모델을 학습하고, SHAP으로 각 파라미터의 기여도를 분해한다.

왜 이 단계가 필요한가 ★: **"어느 챔버"까지 알아도 아직 조치할 수 없다.**

    장비 전체를 세우면 생산이 멈춘다. 실제로 손을 대려면
    "이 챔버의 **어떤 파라미터가** 규격의 어느 방향으로 벗어났는가"까지 내려가야 한다.

        M3 결과: "ETCH-B/ch2가 유력하다"        → 아직 조치 불가
        M4 결과: "그 챔버의 chamber_pressure가   → 압력 설정 조정
                  47.2로 평소(45.0)보다 높다"

    그리고 현업에서는 **설명하지 못하는 모델은 채택되지 않는다.** 장비를 세우거나
    설정을 바꾸는 데는 비용이 들고, "AI가 그렇대요"로는 결재가 나지 않는다.
    그래서 예측 정확도보다 **기여도 분해**가 이 단계의 본체다.

⚠️ 가장 중요한 주의: **SHAP은 상관을 보여 줄 뿐 인과를 증명하지 않는다.**
    SHAP 값이 크다는 것은 "그 파라미터가 예측에 크게 기여했다"이지
    "그것이 원인이다"가 아니다. 물리적 타당성(§2.4 매핑 표)과 층화 분석으로
    반드시 교차 확인해야 한다.

참고: docs/00_design.md §M4, docs/08_interpretation_guide.md §[5]
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: FDC 요약 테이블에서 파라미터가 아닌 식별 컬럼
from wafermap.features.trace_feat import (
    FEATURE_SUFFIXES as TRACE_FEATURE_SUFFIXES,
)

ID_COLUMNS = ("wafer_id", "step_id", "step_name", "equip_id", "chamber_id",
              "recipe_id", "run_time")

#: 학습 기본 파라미터. 표본이 적고(수백 lot) 피처가 수십 개라 보수적으로 잡았다.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


@dataclass
class CauseModelResult:
    """원인 규명 결과.

    Attributes:
        step_id: 분석한 스텝
        pattern: 대상 불량 패턴
        feature_names: 사용한 파라미터 목록
        shap_importance: 파라미터별 평균 |SHAP| (기여도 랭킹)
        shap_values: (n_samples, n_features) SHAP 값 행렬
        X: 모델 입력 행렬
        y: 정답 라벨 (불량=1)
        auc: 교차검증 AUC — 모델이 신호를 잡았는지 보는 지표
        n_case, n_control: 표본 수
        direction: 파라미터별 이탈 방향 (+1=높을수록 불량, -1=낮을수록 불량)
        booster: 학습된 LightGBM 모델 — M5 반사실 시뮬레이션에서 재예측에 쓴다
        X_all, y_all: SHAP 샘플링 이전의 **전체** 표본. 반사실 시뮬레이션은
            모집단 분포에서 돌려야 하므로 층화 샘플링된 X를 쓰면 안 된다
    """

    step_id: str
    pattern: str
    feature_names: list[str]
    shap_importance: pd.Series
    shap_values: np.ndarray
    X: pd.DataFrame
    y: np.ndarray
    auc: float
    n_case: int
    n_control: int
    direction: dict[str, int] = field(default_factory=dict)
    booster: object | None = None
    X_all: pd.DataFrame | None = None
    y_all: np.ndarray | None = None

    @property
    def population(self) -> tuple[pd.DataFrame, np.ndarray]:
        """반사실 시뮬레이션에 쓸 모집단 (전체 표본이 있으면 그것을 쓴다).

        왜 구분하나 ★: `X`는 SHAP 계산 속도를 위해 **불량을 과대 표집한** 부분집합이다.
            여기서 불량률을 계산하면 실제보다 훨씬 높게 나온다. 기대효과를 추정할 때는
            반드시 원래 분포를 써야 한다.
        """
        if self.X_all is not None and self.y_all is not None:
            return self.X_all, self.y_all
        return self.X, self.y

    def top_features(self, n: int = 5) -> list[str]:
        """기여도 상위 n개 파라미터."""
        return list(self.shap_importance.head(n).index)

    def summary(self) -> str:
        lines = [
            f"스텝 {self.step_id} · 패턴 {self.pattern}",
            f"표본: 불량 {self.n_case} / 정상 {self.n_control} · AUC {self.auc:.3f}",
            "",
            f"  {'파라미터':<28}{'평균|SHAP|':>12}{'방향':>6}",
            "  " + "-" * 46,
        ]
        for name, value in self.shap_importance.head(8).items():
            arrow = "↑" if self.direction.get(name, 0) > 0 else "↓"
            lines.append(f"  {name:<28}{value:>12.4f}{arrow:>6}")
        return "\n".join(lines)


#: 기본 피처 접미사. 요약통계 중 대표값과 안정성만 쓴다.
DEFAULT_SUFFIXES: tuple[str, ...] = ("_mean", "_std")

#: 시계열 파생 피처까지 포함한 확장 집합 (M5.5-②)
TRACE_SUFFIXES: tuple[str, ...] = DEFAULT_SUFFIXES + TRACE_FEATURE_SUFFIXES


def parameter_columns(
    fdc_step: pd.DataFrame, suffixes: tuple[str, ...] = DEFAULT_SUFFIXES
) -> list[str]:
    """FDC 테이블에서 파라미터 컬럼만 골라낸다.

    왜 `_mean`만 쓰나: 한 스텝의 요약 통계는 `<param>_mean/_std/_min/_max` 4종이 있다.
        네 개를 다 넣으면 서로 강하게 상관되어 SHAP 기여도가 **분산된다.**
        예를 들어 압력이 원인일 때 `chamber_pressure_mean`과 `_max`가 기여를 반씩
        나눠 가지면, 둘 다 순위가 밀려 진짜 원인을 놓칠 수 있다.

        `_mean`이 대표성이 가장 높으므로 이것만 쓴다. `_std`는 '안정성'이라는
        다른 정보를 담으므로 함께 넣는다.

    Args:
        fdc_step: 한 스텝의 FDC 테이블
        suffixes: 쓸 피처 접미사. `TRACE_SUFFIXES`를 주면 시계열 파생 피처까지
            포함한다 — 순간 스파이크처럼 요약통계로는 안 보이는 이상을 잡으려면 필요하다.
    """
    cols = [
        c for c in fdc_step.columns
        if c not in ID_COLUMNS and c.endswith(suffixes)
    ]
    # 값이 하나뿐인 컬럼은 정보가 없다 (분할이 불가능해 모델이 무시한다)
    return [c for c in cols if fdc_step[c].nunique(dropna=True) > 1]


def build_matrix(
    fdc_summary: pd.DataFrame,
    step_id: str,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    *,
    chamber_id: str | None = None,
    suffixes: tuple[str, ...] = DEFAULT_SUFFIXES,
) -> tuple[pd.DataFrame, np.ndarray]:
    """특정 스텝의 FDC 파라미터로 학습 행렬을 만든다.

    Args:
        fdc_summary: FDC 요약 테이블
        step_id: 분석할 스텝
        case_ids, control_ids: 불량/정상 웨이퍼
        chamber_id: 특정 챔버로 한정할지. None이면 스텝 전체.
            **챔버를 고정하면 층화 분석이 된다** — 챔버 간 baseline 차이가 제거되어
            "이 챔버 안에서 어떤 파라미터가 문제인가"만 남는다.

    Returns:
        (X, y) — y는 불량=1
    """
    df = fdc_summary[fdc_summary["step_id"] == step_id]
    if chamber_id is not None:
        df = df[df["chamber_id"] == chamber_id]

    case_set, control_set = set(case_ids), set(control_ids)
    df = df[df["wafer_id"].isin(case_set | control_set)].copy()

    y = df["wafer_id"].isin(case_set).to_numpy().astype(int)
    X = df[parameter_columns(df, suffixes)].astype("float32")
    return X, y


def fit(
    fdc_summary: pd.DataFrame,
    step_id: str,
    pattern: str,
    case_ids: list[str] | tuple[str, ...],
    control_ids: list[str] | tuple[str, ...],
    *,
    chamber_id: str | None = None,
    n_splits: int = 5,
    seed: int = 0,
    num_boost_round: int = 200,
    params: dict[str, object] | None = None,
    max_background: int = 500,
    suffixes: tuple[str, ...] = DEFAULT_SUFFIXES,
) -> CauseModelResult:
    """원인 규명 모델을 학습하고 SHAP 기여도를 계산한다.

    어떻게:
        1. 교차검증으로 AUC를 구한다 (모델이 신호를 잡았는지 확인)
        2. 전체 데이터로 최종 모델을 학습한다
        3. TreeSHAP으로 파라미터별 기여도를 분해한다
        4. 이탈 방향(높을수록 불량인지 낮을수록인지)을 함께 계산한다

    왜 AUC를 먼저 보나 ★: **AUC가 0.5 근처면 SHAP을 볼 필요가 없다.**
        모델이 아무 신호도 못 잡았다는 뜻이므로, 그 위에서 계산한 기여도는
        노이즈를 분해한 것에 불과하다. 순서가 중요하다 — 모델이 먼저 유효해야
        해석이 의미를 갖는다.

    Args:
        fdc_summary: FDC 요약 테이블
        step_id: 분석할 스텝
        pattern: 대상 불량 패턴 (기록용)
        case_ids, control_ids: 불량/정상 웨이퍼
        chamber_id: 특정 챔버로 한정 (층화)
        n_splits: 교차검증 fold 수
        seed: 난수 시드
        num_boost_round: 부스팅 라운드
        params: LightGBM 파라미터
        max_background: SHAP 계산에 쓸 최대 표본 수 (속도 조절)
        suffixes: 쓸 피처 접미사 (`TRACE_SUFFIXES`면 시계열 파생 피처 포함)

    Returns:
        CauseModelResult

    Raises:
        ValueError: 표본이 너무 적거나 한쪽 클래스만 있을 때
    """
    import lightgbm as lgb
    import shap
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    X, y = build_matrix(
        fdc_summary, step_id, case_ids, control_ids,
        chamber_id=chamber_id, suffixes=suffixes,
    )

    n_case, n_control = int(y.sum()), int((1 - y).sum())
    if n_case < 10 or n_control < 10:
        raise ValueError(
            f"표본이 부족합니다 (불량 {n_case}, 정상 {n_control}). "
            f"각 10장 이상 필요합니다."
        )
    if X.empty or X.shape[1] == 0:
        raise ValueError(f"{step_id}에 사용할 수 있는 파라미터가 없습니다.")

    params = {**DEFAULT_PARAMS, **(params or {})}
    # 불균형 보정 — 정상이 압도적으로 많으면 모델이 전부 정상이라 답한다
    params["scale_pos_weight"] = n_control / max(n_case, 1)

    # ── 교차검증 AUC ────────────────────────────────────────────────────
    n_splits = min(n_splits, n_case, n_control)
    oof = np.zeros(len(y), dtype=float)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, valid_idx in skf.split(X, y):
            model = lgb.train(
                params,
                lgb.Dataset(X.iloc[train_idx], label=y[train_idx]),
                num_boost_round=num_boost_round,
            )
            oof[valid_idx] = model.predict(X.iloc[valid_idx])
        auc = float(roc_auc_score(y, oof))
    else:
        auc = float("nan")

    # ── 최종 모델 + SHAP ────────────────────────────────────────────────
    final = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=num_boost_round)

    # 표본이 많으면 SHAP 계산이 느려진다. 층화 샘플링으로 줄인다.
    if len(X) > max_background:
        rng = np.random.default_rng(seed)
        case_idx = np.flatnonzero(y == 1)
        control_idx = np.flatnonzero(y == 0)
        take_case = min(len(case_idx), max_background // 2)
        take_control = min(len(control_idx), max_background - take_case)
        sample = np.concatenate([
            rng.choice(case_idx, take_case, replace=False),
            rng.choice(control_idx, take_control, replace=False),
        ])
        X_shap, y_shap = X.iloc[sample], y[sample]
    else:
        X_shap, y_shap = X, y

    explainer = shap.TreeExplainer(final)
    shap_values = explainer.shap_values(X_shap)
    # 이진 분류에서 버전에 따라 (n, f) 또는 [음성, 양성] 리스트로 나온다
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # (n, f, 2) 형태
        shap_values = shap_values[:, :, 1]

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)

    # ── 이탈 방향 ───────────────────────────────────────────────────────
    # SHAP 값과 파라미터 값의 상관 부호로 판단한다.
    # 양수면 "값이 클수록 불량 확률이 올라간다"는 뜻이다.
    direction: dict[str, int] = {}
    for i, name in enumerate(X.columns):
        values = X_shap[name].to_numpy(dtype=float)
        if np.std(values) < 1e-12:
            direction[name] = 0
            continue
        corr = np.corrcoef(values, shap_values[:, i])[0, 1]
        direction[name] = int(np.sign(corr)) if np.isfinite(corr) else 0

    return CauseModelResult(
        step_id=step_id,
        pattern=pattern,
        feature_names=list(X.columns),
        shap_importance=importance,
        shap_values=shap_values,
        X=X_shap,
        y=y_shap,
        auc=auc,
        n_case=n_case,
        n_control=n_control,
        direction=direction,
        booster=final,
        X_all=X,
        y_all=y,
    )


def dependence_curve(
    result: CauseModelResult, feature: str, *, n_bins: int = 12
) -> pd.DataFrame:
    """파라미터 값 구간별 평균 SHAP 기여도 — "어느 값부터 위험해지나".

    무엇을: 파라미터를 구간으로 나눠 각 구간의 평균 SHAP을 계산한다.
        기여도가 급격히 올라가는 지점이 곧 **위험 임계값**이다.

    왜 중요한가 ★: 이것이 개선안의 직접적 근거다.
        "edge_ring_rf_hours가 350시간을 넘으면 기여도가 급증한다"는 것을 보이면
        **"PM 주기를 400시간에서 300시간으로 단축하자"** 는 구체적 제안이 나온다.

        기여도 순위만으로는 "무엇이 문제인가"까지만 알 수 있고,
        "어디까지 허용할 것인가"는 이 곡선에서 나온다.

    Returns:
        구간별 (value_mid, mean_shap, n_samples, case_rate)
    """
    if feature not in result.X.columns:
        raise KeyError(f"'{feature}'는 모델에 없는 파라미터입니다")

    col_idx = list(result.X.columns).index(feature)
    values = result.X[feature].to_numpy(dtype=float)
    shap_col = result.shap_values[:, col_idx]

    # 분위수 기반 구간 — 값 분포가 치우쳐 있어도 각 구간에 표본이 고르게 들어간다
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["value_mid", "mean_shap", "n_samples", "case_rate"])

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (values >= lo) & (values <= hi if hi == edges[-1] else values < hi)
        if sel.sum() < 3:
            continue
        rows.append(
            {
                "value_lo": float(lo),
                "value_hi": float(hi),
                "value_mid": float((lo + hi) / 2),
                "mean_shap": float(shap_col[sel].mean()),
                "n_samples": int(sel.sum()),
                "case_rate": float(result.y[sel].mean()),
            }
        )
    return pd.DataFrame(rows)


def explain_wafer(
    result: CauseModelResult, wafer_index: int, *, top_n: int = 5
) -> pd.DataFrame:
    """웨이퍼 한 장의 판정 근거를 분해한다 (local SHAP).

    현업에서 엔지니어에게 설명할 때 가장 설득력 있는 형태다.

        "이 웨이퍼는 edge_ring_rf_hours=412h가 불량 확률을 +0.38 올렸습니다"

    Returns:
        기여도 절댓값 순 상위 파라미터 (parameter, value, shap, direction)
    """
    if not 0 <= wafer_index < len(result.X):
        raise IndexError(f"wafer_index 범위 초과: {wafer_index} (0~{len(result.X) - 1})")

    contributions = result.shap_values[wafer_index]
    values = result.X.iloc[wafer_index]

    frame = pd.DataFrame(
        {
            "parameter": result.X.columns,
            "value": values.to_numpy(),
            "shap": contributions,
        }
    )
    frame["abs_shap"] = frame["shap"].abs()
    frame = frame.sort_values("abs_shap", ascending=False).head(top_n)
    frame["direction"] = np.where(frame["shap"] > 0, "불량 쪽", "정상 쪽")
    return frame.drop(columns="abs_shap").reset_index(drop=True)


def recommend_spec(
    result: CauseModelResult, feature: str, *, target_quantile: float = 0.5
) -> dict[str, float] | None:
    """SHAP 기여도가 낮은 구간을 찾아 권고 운전 구간을 제안한다.

    어떻게: dependence 곡선에서 평균 SHAP이 낮은(=불량 기여가 작은) 구간들을 모아
        그 값 범위를 권고안으로 낸다.

    ⚠️ 한계를 반드시 함께 보고할 것:
        · **관측된 범위 안에서만 유효하다.** 한 번도 본 적 없는 값에 대해서는
          모델이 근거 없이 외삽한다.
        · **상관 기반이다.** 이 구간으로 옮기면 정말 좋아진다는 인과 보장은 없다.

    Returns:
        {"low", "high", "current_min", "current_max", "coverage"} 또는 None
    """
    curve = dependence_curve(result, feature)
    if curve.empty:
        return None

    threshold = curve["mean_shap"].quantile(target_quantile)
    good = curve[curve["mean_shap"] <= threshold]
    if good.empty:
        return None

    return {
        "low": float(good["value_lo"].min()),
        "high": float(good["value_hi"].max()),
        "current_min": float(result.X[feature].min()),
        "current_max": float(result.X[feature].max()),
        "coverage": float(good["n_samples"].sum() / curve["n_samples"].sum()),
    }
