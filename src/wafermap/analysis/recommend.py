"""M5 — 개선안 도출과 기대효과 추정 (설계서 §3 M5).

M4까지 "무엇이 원인인가"에 답했다. 여기서는 **"그래서 무엇을 할 것인가"** 와
**"하면 얼마를 버는가"** 에 답한다. 분석이 보고서로 끝나지 않으려면 이 단계가 필요하다.

세 부분으로 나뉜다.

1. **조치안 템플릿** (`build_actions`) — 파라미터 성격에 따라 조치 종류가 달라진다.
   카운터(pad_life)는 PM 주기를 당기는 문제고, 연속 파라미터는 규격을 조이는 문제다.
2. **반사실 시뮬레이션** (`counterfactual`) — 권고 구간으로 값을 clip한 가상의 입력을
   모델에 다시 통과시켜 불량률이 얼마나 내려가는지 본다.
3. **ROI 추정** (`estimate_roi`) — 수율 향상분을 금액으로 환산한다. 가정치는
   전부 바꿀 수 있어야 하고, 무엇이 가정이고 무엇이 데이터인지 표시해야 한다.

⚠️ 이 모듈 전체에 걸린 가장 큰 한계: **상관을 인과로 바꾸지 못한다.**
   "이 구간으로 옮기면 불량이 준다"가 아니라 "이 구간에 있던 웨이퍼는 불량이 적었다"이다.
   확증은 DOE(실험계획)로 해야 하며, 이 분석은 **DOE 대상을 좁히는 도구**다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import pandas as pd

from wafermap.analysis.attribution import ParamEvidence, base_param_name
from wafermap.config import STEPS_BY_ID
from wafermap.models.cause_model import CauseModelResult, dependence_curve

# ── 조치안 ──────────────────────────────────────────────────────────────

ActionKind = Literal["pm_shortening", "spec_tightening", "chamber_matching", "sampling"]

#: 조치 종류별 표시 정보 (라벨, 아이콘, 기본 난이도)
ACTION_KINDS: dict[str, dict[str, str]] = {
    "pm_shortening": {
        "label": "PM 주기 단축",
        "icon": "🔧",
        "effort": "중",
        "why": "소모품이 수명 끝단에 갈수록 불량이 오르는 패턴이면, 규격을 조이는 게 아니라 "
               "교체 시점을 앞당기는 것이 답이다.",
    },
    "spec_tightening": {
        "label": "관리 규격 강화",
        "icon": "📐",
        "effort": "하",
        "why": "연속 파라미터는 운전 구간을 좁히는 것만으로 설비 정지 없이 개선할 수 있다. "
               "가장 값싼 조치다.",
    },
    "chamber_matching": {
        "label": "챔버 매칭",
        "icon": "⚖️",
        "effort": "상",
        "why": "같은 설비의 챔버끼리 baseline이 다르면 한 챔버만 계속 불량을 낸다. "
               "챔버 간 편차를 맞추는 작업이다.",
    },
    "sampling": {
        "label": "계측 샘플링 강화",
        "icon": "🔍",
        "effort": "하",
        "why": "원인이 확실하지 않을 때의 조치. 조치가 아니라 **관측을 늘려 다음 판단을 "
               "가능하게 하는 것**이 목적이다.",
    },
}

#: 이 값 미만의 AUC에서는 개선안을 '검증 필요'로 낮춰 표시한다
MIN_TRUSTWORTHY_AUC = 0.70


@dataclass(frozen=True)
class Action:
    """조치안 1건.

    Attributes:
        kind: 조치 종류 (`ACTION_KINDS` 키)
        pattern: 대상 불량 패턴
        step_id, step_name: 대상 스텝
        param: 대상 파라미터 (기본 이름, `_mean` 접미사 제거)
        column: FDC 컬럼명
        title: 화면에 띄울 한 줄 제목
        current: 현재 상태 서술
        proposed: 제안 서술
        rationale: 왜 이 조치인가 (근거 수치 포함)
        evidence: 근거 강도 라벨 ("강함" / "보통" / "약함 — 검증 필요")
        effort: 조치 난이도 ("하"/"중"/"상")
        spec: 권고 구간 dict (있으면)
        mode: 반사실 시뮬레이션 방식 — "clip"(규격 강화) / "wrap"(PM 주기 단축)
    """

    kind: str
    pattern: str
    step_id: str
    step_name: str
    param: str
    column: str
    title: str
    current: str
    proposed: str
    rationale: str
    evidence: str
    effort: str
    spec: dict[str, float] | None = None
    mode: str = "clip"

    @property
    def icon(self) -> str:
        return ACTION_KINDS[self.kind]["icon"]

    @property
    def kind_label(self) -> str:
        return ACTION_KINDS[self.kind]["label"]


def safe_band(
    result: CauseModelResult,
    column: str,
    *,
    max_move_fraction: float = 0.25,
) -> dict[str, float] | None:
    """**옮겨야 하는 웨이퍼 비율을 제한한** 권고 운전 구간을 찾는다.

    왜 M4의 `recommend_spec`을 그대로 쓰지 않나 ★★:
        `recommend_spec`은 SHAP이 낮은 하위 50% 구간을 권고안으로 낸다. 그러면
        정의상 **표본의 절반이 권고 구간 밖**에 놓인다. 파라미터 3개에 적용하면
        78%가 밖으로 나갔다 — 이건 규격 강화가 아니라 **공정 재설계**다.
        그런 안을 "규격을 조이면 됩니다"라고 보고하면 현장에서 즉시 반려된다.

        그래서 M5는 반대 방향으로 접근한다. **"움직여야 하는 웨이퍼를 25% 이내로
        묶는다"** 는 제약을 먼저 걸고, 그 안에서 가장 나쁜 구간만 잘라낸다.
        실행 가능성이 개선안의 전제이기 때문이다.

    어떻게:
        1. dependence 곡선에서 가장 좋은(평균 SHAP 최소) 구간을 씨앗으로 잡는다
        2. 임계값을 올려 가며 **인접한** 구간을 흡수한다 (구간은 연속이어야
           운전 규격이 된다 — 띄엄띄엄한 구간은 규격으로 쓸 수 없다)
        3. 덮는 표본이 `1 - max_move_fraction`에 도달하면 멈춘다

    Args:
        result: 학습된 원인 규명 모델
        column: 대상 파라미터 컬럼
        max_move_fraction: 권고 구간 밖에 남겨도 되는 표본 비율 상한

    Returns:
        {"low", "high", "current_min", "current_max", "coverage", "move_fraction"}
        또는 곡선을 못 만들면 None
    """
    curve = dependence_curve(result, column)
    if curve.empty:
        return None

    shap_means = curve["mean_shap"].to_numpy(dtype=float)
    counts = curve["n_samples"].to_numpy(dtype=float)
    total = counts.sum()
    if total <= 0:
        return None

    seed = int(np.argmin(shap_means))
    target_coverage = 1.0 - max_move_fraction
    best: tuple[int, int] | None = None

    for threshold in np.unique(shap_means):
        ok = shap_means <= threshold
        if not ok[seed]:
            continue
        lo = seed
        while lo - 1 >= 0 and ok[lo - 1]:
            lo -= 1
        hi = seed
        while hi + 1 < len(ok) and ok[hi + 1]:
            hi += 1
        best = (lo, hi)
        if counts[lo:hi + 1].sum() / total >= target_coverage:
            break

    if best is None:
        return None

    lo, hi = best
    coverage = float(counts[lo:hi + 1].sum() / total)
    observed = result.X[column]
    return {
        "low": float(curve.iloc[lo]["value_lo"]),
        "high": float(curve.iloc[hi]["value_hi"]),
        "current_min": float(observed.min()),
        "current_max": float(observed.max()),
        "coverage": coverage,
        "move_fraction": 1.0 - coverage,
    }


def _evidence_label(ev: ParamEvidence, auc: float) -> str:
    """근거 강도를 한 줄로 요약한다.

    왜 필요한가 ★: 모든 조치안이 같은 무게를 갖지 않는다. AUC가 낮거나 분포 증거가
        없는 조치안을 다른 것과 나란히 놓으면, 보는 사람이 **약한 근거를 강한 근거로
        착각한다.** M4에서 Scratch의 `pad_life`가 정확히 그 사례였다.
    """
    if not np.isnan(auc) and auc < MIN_TRUSTWORTHY_AUC:
        return "약함 — 모델 AUC 부족, 검증 필요"
    if not ev.has_distribution_evidence:
        return "약함 — SHAP 단독, 분포 증거 없음"
    if abs(ev.cliffs_delta) >= 0.474:  # large effect
        return "강함"
    return "보통"


def _counter_action(
    ev: ParamEvidence, result: CauseModelResult, step, spec_hi: float
) -> Action:
    """카운터 파라미터(누적 사용량)에 대한 PM 주기 단축 조치안."""
    X, _ = result.population
    values = X[ev.column].to_numpy(dtype=float)
    # 불량 웨이퍼가 몰린 하한을 찾아 그 앞에서 교체하도록 제안한다
    case_values = values[result.y_all == 1] if result.y_all is not None else values
    onset = float(np.percentile(case_values, 10)) if len(case_values) else float(values.max())
    proposed_limit = max(0.0, min(onset, spec_hi))

    return Action(
        kind="pm_shortening",
        pattern=result.pattern,
        step_id=result.step_id,
        step_name=step.name_ko,
        param=ev.param,
        column=ev.column,
        title=f"{ev.param} PM 주기 {spec_hi:,.0f} → {proposed_limit:,.0f}",
        current=f"PM 주기 상한 {spec_hi:,.0f}",
        proposed=f"{proposed_limit:,.0f}에서 교체 (약 {1 - proposed_limit / spec_hi:.0%} 단축)",
        rationale=(
            f"불량 웨이퍼의 90%가 {onset:,.0f} 이상에서 나왔다. "
            f"정상군 대비 {ev.delta_sigma:+.2f}σ, Cliff's δ {ev.cliffs_delta:+.2f}."
        ),
        evidence=_evidence_label(ev, result.auc),
        effort=ACTION_KINDS["pm_shortening"]["effort"],
        mode="wrap",
        spec={"low": 0.0, "high": proposed_limit,
              "current_min": float(values.min()), "current_max": float(values.max()),
              "coverage": float((values <= proposed_limit).mean())},
    )


def _spec_action(
    ev: ParamEvidence, result: CauseModelResult, step, max_move: float
) -> Action | None:
    """연속 파라미터에 대한 규격 강화 조치안."""
    spec = safe_band(result, ev.column, max_move_fraction=max_move)
    if spec is None:
        return None

    param_spec = None
    try:
        param_spec = step.param(ev.param)
    except (KeyError, ValueError):
        pass

    current = (
        f"규격 {param_spec.spec_lo:g} ~ {param_spec.spec_hi:g} {param_spec.unit}"
        if param_spec is not None
        else f"관측 범위 {spec['current_min']:.2f} ~ {spec['current_max']:.2f}"
    )
    return Action(
        kind="spec_tightening",
        pattern=result.pattern,
        step_id=result.step_id,
        step_name=step.name_ko,
        param=ev.param,
        column=ev.column,
        title=f"{ev.param} 운전 구간 {spec['low']:.2f} ~ {spec['high']:.2f}로 강화",
        current=current,
        proposed=(
            f"{spec['low']:.2f} ~ {spec['high']:.2f} "
            f"(현재 표본의 {spec['coverage']:.0%}는 이미 이 안에 있다 — "
            f"{spec['move_fraction']:.0%}만 조정하면 된다)"
        ),
        rationale=(
            f"정상군 대비 {ev.delta_sigma:+.2f}σ 이탈, Cliff's δ {ev.cliffs_delta:+.2f}, "
            f"규격 위반율 {ev.spec_violation_rate:.0%}. SHAP 기여도 {ev.shap_rank}위."
        ),
        evidence=_evidence_label(ev, result.auc),
        effort=ACTION_KINDS["spec_tightening"]["effort"],
        spec=spec,
    )


def build_actions(
    evidence: list[ParamEvidence],
    result: CauseModelResult,
    *,
    top_n: int = 3,
    chamber_id: str | None = None,
    move_budget: float = 0.25,
) -> list[Action]:
    """근거 목록에서 조치안을 생성한다.

    어떻게: 조치 가능한(계측값이 아닌) 파라미터만 골라, 파라미터 성격에 따라
        PM 단축 / 규격 강화로 나눈다. 그리고 상황에 따라 챔버 매칭·샘플링 조치를 덧붙인다.

    왜 계측값을 빼나: 파티클 수나 두께 산포는 **결과**지 조작 손잡이가 아니다.
        "파티클을 줄이시오"는 개선안이 아니라 문제의 재진술이다 (M4 발견 1).

    Args:
        evidence: `attribution.build_evidence()` 결과
        result: 학습된 원인 규명 모델
        top_n: 파라미터 조치안 최대 개수
        chamber_id: 층화에 쓴 챔버 (챔버 매칭 조치안 생성 조건)
        move_budget: 조치안 **전체**에서 값을 옮겨야 하는 웨이퍼 비율 상한

    Returns:
        조치안 목록. 근거가 강한 것부터 정렬된다.
    """
    step = STEPS_BY_ID[result.step_id]
    actions: list[Action] = []

    # ── 후보를 먼저 확정한다 ────────────────────────────────────────────
    candidates: list[tuple[ParamEvidence, object]] = []
    for ev in evidence:
        if len(candidates) >= top_n:
            break
        if not ev.is_controllable:
            continue
        if ev.column.endswith("_std"):
            # 산포 자체는 직접 설정하는 값이 아니다. 평균값 조치로 다뤄야 한다.
            continue
        try:
            param_spec = step.param(base_param_name(ev.column))
        except (KeyError, ValueError):
            param_spec = None
        candidates.append((ev, param_spec))

    # ── 이동 예산을 파라미터 수로 나눈다 ★ ──────────────────────────────
    # 왜: 파라미터마다 "25%까지 옮겨도 된다"고 하면 3개를 동시에 조이는 순간
    #     합쳐서 1-(0.75)^3 ≈ 58%가 움직여야 한다. 개별로는 온건해 보이는 안이
    #     합치면 공정 재설계가 되는 것이다. 예산은 **합계 기준**으로 잡아야 한다.
    n_continuous = sum(
        1 for _, spec in candidates if spec is None or spec.kind != "counter"
    )
    per_param_move = (
        1.0 - (1.0 - move_budget) ** (1.0 / n_continuous) if n_continuous else move_budget
    )

    for ev, param_spec in candidates:
        if param_spec is not None and param_spec.kind == "counter":
            actions.append(_counter_action(ev, result, step, param_spec.spec_hi))
        else:
            action = _spec_action(ev, result, step, per_param_move)
            if action is not None:
                actions.append(action)

    # ── 보조 조치안 ────────────────────────────────────────────────────
    if chamber_id:
        actions.append(Action(
            kind="chamber_matching",
            pattern=result.pattern, step_id=result.step_id, step_name=step.name_ko,
            param="-", column="-",
            title=f"{chamber_id} 챔버 매칭 점검",
            current=f"{chamber_id} 한 챔버에서만 {result.pattern} 불량이 집중",
            proposed="동일 스텝의 정상 챔버를 기준으로 baseline 재조정",
            rationale=(
                f"M3 커미널리티에서 {chamber_id}가 1위로 지목됐다. "
                f"같은 레시피를 쓰는데 한 챔버만 다르다면 설비 상태 차이다."
            ),
            evidence="보통",
            effort=ACTION_KINDS["chamber_matching"]["effort"],
        ))

    if np.isnan(result.auc) or result.auc < MIN_TRUSTWORTHY_AUC or result.n_case < 30:
        actions.append(Action(
            kind="sampling",
            pattern=result.pattern, step_id=result.step_id, step_name=step.name_ko,
            param="-", column="-",
            title="계측 샘플링 강화 (결론 보류)",
            current=f"불량 표본 {result.n_case}장 · 모델 AUC {result.auc:.3f}",
            proposed="해당 챔버 계측 빈도 상향 후 재분석",
            rationale=(
                "표본이 적거나 모델이 신호를 충분히 잡지 못했다. "
                "이 상태에서 규격을 조이면 **근거 없는 규제**가 된다. "
                "먼저 데이터를 더 모으는 것이 옳은 조치다."
            ),
            evidence="—",
            effort=ACTION_KINDS["sampling"]["effort"],
        ))
    return actions


# ── 반사실 시뮬레이션 ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Counterfactual:
    """권고 구간 적용 시 기대되는 불량률 변화.

    Attributes:
        baseline_score: 현행 모델 예측 평균 점수
        improved_score: clip 후 예측 평균 점수
        relative_reduction: 상대 감소율 (0~1)
        observed_defect_rate: 실제 관측 불량률
        expected_defect_rate: 상대 감소율을 관측 불량률에 적용한 값
        yield_gain_pp: 수율 향상분 (%p)
        n_wafers: 시뮬레이션 표본 수
        n_clipped: 실제로 값이 바뀐 웨이퍼 수
        clipped_columns: 적용한 파라미터 목록
        extrapolated: 권고 구간이 관측 범위를 벗어났는지 (외삽 경고)
        scope_fraction: 이 조치가 닿는 웨이퍼가 라인 전체에서 차지하는 비율.
            챔버 하나로 층화해 분석했다면 그 챔버의 물량 비중이다
    """

    baseline_score: float
    improved_score: float
    relative_reduction: float
    observed_defect_rate: float
    expected_defect_rate: float
    yield_gain_pp: float
    n_wafers: int
    n_clipped: int
    clipped_columns: list[str] = field(default_factory=list)
    extrapolated: bool = False
    scope_fraction: float = 1.0

    @property
    def line_yield_gain_pp(self) -> float:
        """**라인 전체 기준** 수율 향상분 ★★.

        왜 나누어 보나: `yield_gain_pp`는 분석한 모집단 — 대개 **문제 챔버 한 대** —
            안에서의 수치다. 그 챔버가 라인 물량의 12%만 처리한다면, 라인 전체
            수율은 그 비율만큼만 오른다. 이 구분을 안 하고 챔버 수치를 라인 물량에
            곱하면 기대효과가 몇 배로 부풀려진다. ROI에는 반드시 이 값을 쓴다.
        """
        return self.yield_gain_pp * self.scope_fraction

    @property
    def move_fraction(self) -> float:
        """권고 구간 밖에 있어 값을 옮겨야 하는 웨이퍼 비율."""
        return self.n_clipped / self.n_wafers if self.n_wafers else 0.0

    @property
    def feasibility(self) -> str:
        """실행 가능성 등급 — 옮겨야 하는 웨이퍼가 많을수록 실행이 어렵다."""
        if self.move_fraction <= 0.15:
            return "높음"
        if self.move_fraction <= 0.35:
            return "보통"
        return "낮음 — 사실상 공정 재설계"

    def caveats(self) -> list[str]:
        """이 수치와 **항상 같이** 보여야 하는 단서들."""
        notes = [
            "상관 기반 추정이다. 실제로 값을 옮겼을 때 같은 결과가 나온다는 인과 보장이 아니다.",
            "모델이 학습한 관측 범위 안에서만 유효하다.",
        ]
        if self.relative_reduction < 0:
            notes.append(
                f"❌ 제안한 조치가 예측 불량률을 오히려 {-self.relative_reduction:.0%} "
                "**올렸다.** 이 조치안은 채택하면 안 된다."
            )
        if self.n_clipped == 0:
            notes.append("⚠️ 권고 구간을 벗어난 웨이퍼가 없어 변화가 계산되지 않았다.")
        if self.extrapolated:
            notes.append("⚠️ 권고 구간이 관측 범위 밖까지 뻗어 있다. 외삽 구간이다.")
        if self.n_wafers < 200:
            notes.append(f"⚠️ 표본 {self.n_wafers}장은 적다. 추정 오차가 크다.")
        if self.scope_fraction < 1.0:
            notes.append(
                f"이 수치는 분석한 챔버 안에서의 값이다. 그 챔버가 라인 물량의 "
                f"{self.scope_fraction:.0%}를 처리하므로, **라인 전체 수율 향상분은 "
                f"+{self.line_yield_gain_pp:.2f}%p**다."
            )
        if self.move_fraction > 0.35:
            notes.append(
                f"⚠️ 표본의 {self.move_fraction:.0%}를 옮겨야 한다. 이 정도면 규격 강화가 아니라 "
                "**공정 재설계**이고, 기대효과도 과대평가됐을 가능성이 크다."
            )
        return notes


def counterfactual(
    result: CauseModelResult,
    clips: dict[str, tuple[float, float]],
    *,
    wrap_columns: tuple[str, ...] | list[str] = (),
    scope_fraction: float = 1.0,
) -> Counterfactual:
    """권고 구간으로 파라미터를 clip한 가상 입력의 예측 불량률을 계산한다.

    어떻게:
        1. 모집단 X를 복사해 지정한 컬럼을 [low, high]로 clip한다
        2. 학습된 모델로 clip 전/후를 각각 예측한다
        3. **상대 감소율**을 구해 실제 관측 불량률에 적용한다

    왜 상대값을 쓰나 ★★: 이 모델은 `scale_pos_weight`로 불균형을 보정해 학습했다.
        그래서 예측 확률의 **절대값은 실제 불량률과 다르다** (훨씬 크게 나온다).
        보정된 모델의 확률을 그대로 "예상 불량률 12%"라고 쓰면 완전히 틀린 숫자를
        보고하게 된다. 절대값은 버리고 **감소 비율만** 취해 실측 불량률에 적용한다.

    왜 clip과 wrap을 나누나 ★★: 카운터 파라미터(누적 사용량)에 clip을 쓰면
        **물리적으로 틀린 시뮬레이션**이 된다. pad_life 900인 웨이퍼를 상한 430으로
        clip하면 "430에서 처리된 웨이퍼"가 되는데, PM 주기를 430으로 당겼을 때
        실제로 일어나는 일은 그게 아니다. 카운터는 PM에서 0으로 리셋되므로
        900번째 웨이퍼는 새 주기의 900 mod 430 = 40번째가 된다. 즉 분포가
        **경계에 쌓이는 게 아니라 [0, 상한]으로 되접힌다.** clip을 쓰면 모든
        웨이퍼가 불량이 시작되는 경계값에 몰려 예측이 오히려 나빠진다.

    Args:
        result: 학습된 원인 규명 모델 (booster 필요)
        clips: {컬럼명: (하한, 상한)}
        wrap_columns: 카운터 파라미터 — clip 대신 나머지 연산으로 되접는다
        scope_fraction: 분석 모집단이 라인 전체에서 차지하는 물량 비중 (0~1)

    Returns:
        Counterfactual

    Raises:
        ValueError: 모델이 없거나 clips가 비었을 때
        KeyError: 모르는 컬럼을 지정했을 때
    """
    if result.booster is None:
        raise ValueError("학습된 모델이 없습니다 (cause_model.fit 결과를 넘기세요).")
    if not clips:
        raise ValueError("적용할 권고 구간이 없습니다.")

    X, y = result.population
    unknown = set(clips) - set(X.columns)
    if unknown:
        raise KeyError(f"모르는 파라미터: {sorted(unknown)}")

    X_cf = X.copy()
    wrap = set(wrap_columns)
    extrapolated = False
    for column, (low, high) in clips.items():
        observed = X[column]
        if low < observed.min() - 1e-9 or high > observed.max() + 1e-9:
            extrapolated = True
        if column in wrap and high > low:
            # PM 주기 단축 — 상한을 넘는 값은 새 주기로 되접힌다
            span = high - low
            X_cf[column] = np.where(
                observed > high, low + np.mod(observed - low, span), observed
            )
        else:
            X_cf[column] = observed.clip(lower=low, upper=high)

    changed = (X_cf[list(clips)] != X[list(clips)]).any(axis=1)

    base = np.asarray(result.booster.predict(X), dtype=float)
    improved = np.asarray(result.booster.predict(X_cf), dtype=float)

    base_mean, improved_mean = float(base.mean()), float(improved.mean())
    # ⚠️ 음수(=조치가 오히려 나쁘게 만든 경우)를 0으로 깎지 않는다. 깎으면
    #    "효과 없음"으로 보이지만 실제로는 **역효과**이고, 그건 반드시 드러나야 한다.
    reduction = (base_mean - improved_mean) / base_mean if base_mean > 0 else 0.0
    reduction = float(min(reduction, 1.0))

    observed_rate = float(y.mean())
    expected_rate = observed_rate * (1.0 - reduction)

    return Counterfactual(
        baseline_score=base_mean,
        improved_score=improved_mean,
        relative_reduction=reduction,
        observed_defect_rate=observed_rate,
        expected_defect_rate=expected_rate,
        yield_gain_pp=(observed_rate - expected_rate) * 100.0,
        n_wafers=len(X),
        n_clipped=int(changed.sum()),
        clipped_columns=list(clips),
        extrapolated=extrapolated,
        scope_fraction=float(scope_fraction),
    )


def counterfactual_from_actions(
    result: CauseModelResult, actions: list[Action], *, scope_fraction: float = 1.0
) -> Counterfactual | None:
    """조치안 목록에서 권고 구간을 뽑아 그대로 시뮬레이션한다."""
    usable = [a for a in actions if a.spec is not None and a.column in result.X.columns]
    clips = {a.column: (a.spec["low"], a.spec["high"]) for a in usable}
    if not clips:
        return None
    wraps = tuple(a.column for a in usable if a.mode == "wrap")
    return counterfactual(result, clips, wrap_columns=wraps, scope_fraction=scope_fraction)


# ── ROI ─────────────────────────────────────────────────────────────────

#: 가정치의 성격 배지 — 무엇이 데이터고 무엇이 추정인지 화면에서 구분한다 (설계서 §3 M5.4)
ASSUMPTION_BADGES: dict[str, tuple[str, str]] = {
    "wafer_value_krw": ("가정", "공정원가 + 양품 매출 기여분 추정치. 사내 원가자료로 교체 권장"),
    "monthly_wafers": ("가정", "중소 팹 라인 규모 가정"),
    "adoption_rate": ("가정", "조치가 실제로 적용·유지되는 비율. 보수적으로 잡는다"),
    "implementation_cost_krw": ("가정", "PM 추가·엔지니어 공수 등 1회성 비용"),
    "annual_running_cost_krw": ("가정", "PM 주기 단축에 따른 소모품·다운타임 증가분"),
    "yield_gain_pp": ("데이터 기반", "M5 반사실 시뮬레이션 결과값"),
}


@dataclass(frozen=True)
class RoiAssumptions:
    """ROI 계산 가정치 — **전부 화면에서 바꿀 수 있어야 한다** (설계서 §3 M5.4).

    왜 기본값을 박아 두지 않고 노출하나: 이 숫자들은 회사·라인마다 자릿수가 다르다.
        고정해 두면 "그 숫자 어디서 나왔냐"는 질문 하나에 분석 전체의 신뢰가 무너진다.
        기본값은 출발점일 뿐이고, 보는 사람이 자기 숫자를 넣어볼 수 있어야 한다.
    """

    wafer_value_krw: float = 5_000_000.0
    monthly_wafers: int = 3_000
    adoption_rate: float = 0.8
    implementation_cost_krw: float = 30_000_000.0
    annual_running_cost_krw: float = 60_000_000.0

    def __post_init__(self) -> None:
        if self.wafer_value_krw <= 0:
            raise ValueError("웨이퍼 단가는 0보다 커야 합니다")
        if self.monthly_wafers <= 0:
            raise ValueError("월 투입량은 0보다 커야 합니다")
        if not 0.0 < self.adoption_rate <= 1.0:
            raise ValueError("적용률은 0 초과 1 이하여야 합니다")


@dataclass(frozen=True)
class Roi:
    """ROI 추정 결과.

    Attributes:
        yield_gain_pp: 입력한 수율 향상분 (%p)
        effective_gain_pp: 적용률을 반영한 실효 향상분 (%p)
        recovered_wafers_year: 연간 추가 양품 웨이퍼 수
        gross_saving_krw: 연간 총 효익
        net_saving_krw: 비용을 뺀 연간 순효익
        payback_months: 투자 회수 기간 (개월). 순효익이 0 이하면 None
        assumptions: 사용한 가정치
    """

    yield_gain_pp: float
    effective_gain_pp: float
    recovered_wafers_year: float
    gross_saving_krw: float
    net_saving_krw: float
    payback_months: float | None
    assumptions: RoiAssumptions

    def format_krw(self, value: float) -> str:
        """억/만원 단위로 읽기 쉽게 바꾼다."""
        if abs(value) >= 100_000_000:
            return f"{value / 100_000_000:,.1f}억원"
        if abs(value) >= 10_000:
            return f"{value / 10_000:,.0f}만원"
        return f"{value:,.0f}원"


def estimate_roi(yield_gain_pp: float, assumptions: RoiAssumptions | None = None) -> Roi:
    """수율 향상분을 연간 금액으로 환산한다.

    계산식 (설계서 §3 M5.4):
        연간 추가 양품 = 월 투입량 × 12 × 실효 수율 향상분
        총 효익 = 연간 추가 양품 × 웨이퍼 단가
        순효익 = 총 효익 − 연간 운영비 (1회성 비용은 회수기간에만 반영)

    왜 적용률을 곱하나: 조치안이 100% 지켜지는 라인은 없다. 규격을 조여도 일부 lot은
        기존대로 흐르고, PM 주기를 당겨도 일정이 밀린다. 보수적으로 잡지 않으면
        기대효과가 부풀려진다.

    Args:
        yield_gain_pp: 수율 향상분 (%p). 음수는 0으로 처리한다
        assumptions: 가정치 (생략 시 기본값)

    Returns:
        Roi
    """
    assumptions = assumptions or RoiAssumptions()
    gain_pp = max(0.0, float(yield_gain_pp))
    effective_pp = gain_pp * assumptions.adoption_rate

    annual_wafers = assumptions.monthly_wafers * 12
    recovered = annual_wafers * effective_pp / 100.0
    gross = recovered * assumptions.wafer_value_krw
    net = gross - assumptions.annual_running_cost_krw

    payback = None
    if net > 0:
        payback = assumptions.implementation_cost_krw / (net / 12.0)

    return Roi(
        yield_gain_pp=gain_pp,
        effective_gain_pp=effective_pp,
        recovered_wafers_year=recovered,
        gross_saving_krw=gross,
        net_saving_krw=net,
        payback_months=payback,
        assumptions=assumptions,
    )


def roi_sensitivity(
    yield_gain_pp: float,
    assumptions: RoiAssumptions | None = None,
    *,
    field_name: str = "wafer_value_krw",
    factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """가정치 하나를 배수로 흔들며 순효익이 어떻게 변하는지 본다.

    왜 필요한가 ★: 단일 숫자로 제시한 ROI는 **가정치가 틀리면 통째로 틀린다.**
        어떤 가정에 결과가 민감한지 보여 주면, 보는 사람이 "그 가정만 확인하면
        되는구나"를 알 수 있다. 이것이 가정을 숨기지 않는 방식이다.
    """
    assumptions = assumptions or RoiAssumptions()
    base_value = getattr(assumptions, field_name)
    if not isinstance(base_value, (int, float)):
        raise TypeError(f"{field_name}은 수치 가정치가 아닙니다")

    rows = []
    for factor in factors:
        value = type(base_value)(base_value * factor)
        roi = estimate_roi(yield_gain_pp, replace(assumptions, **{field_name: value}))
        rows.append({
            "factor": factor,
            "value": value,
            "net_saving_krw": roi.net_saving_krw,
            "payback_months": roi.payback_months,
        })
    return pd.DataFrame(rows)


def summarize(actions: list[Action], cf: Counterfactual | None, roi: Roi | None) -> str:
    """콘솔용 요약 문자열."""
    lines = [f"조치안 {len(actions)}건"]
    for i, a in enumerate(actions, 1):
        lines.append(f"  {i}. {a.icon} [{a.kind_label}] {a.title}")
        lines.append(f"     근거({a.evidence}): {a.rationale}")

    if cf is not None:
        lines += [
            "",
            f"기대효과: 불량률 {cf.observed_defect_rate:.2%} → {cf.expected_defect_rate:.2%} "
            f"(상대 {cf.relative_reduction:+.0%}, 챔버 수율 {cf.yield_gain_pp:+.2f}%p "
            f"· 라인 수율 {cf.line_yield_gain_pp:+.2f}%p)",
            f"  표본 {cf.n_wafers}장 중 {cf.n_clipped}장이 권고 구간 밖",
        ]
        for note in cf.caveats():
            lines.append(f"  · {note}")

    if roi is not None:
        lines += [
            "",
            f"ROI: 연간 순효익 {roi.format_krw(roi.net_saving_krw)} "
            f"(추가 양품 {roi.recovered_wafers_year:,.0f}장)",
        ]
        if roi.payback_months is not None:
            lines.append(f"  투자 회수 {roi.payback_months:.1f}개월")
        else:
            lines.append("  ⚠️ 순효익이 운영비를 넘지 못한다 — 조치 자체를 재검토해야 한다")
    return "\n".join(lines)
