"""분석 채점 — "우리 분석이 맞았는가"를 정답지와 대조해 정량화한다.

무엇을: 커미널리티 분석이 지목한 설비를, 시뮬레이터가 실제로 심어 둔 원인과
        비교해 적중률을 계산한다.

왜 이것이 이 프로젝트의 핵심인가 ★:
    보통의 원인 분석은 "이게 원인인 것 같다"까지만 말할 수 있다. 정답을 모르기
    때문이다. 그래서 분석 방법이 좋은지 나쁜지 판단할 근거가 없다.

    이 프로젝트는 공정 데이터를 시뮬레이터로 만들었으므로 **정답을 안다.**
    따라서 "원인 스텝 Top-1 적중률 100%" 같은 정량 지표를 제시할 수 있다.

    면접에서 **"그 분석이 맞다는 걸 어떻게 아나요?"** 라는 질문에 답할 수 있다는 뜻이다.

⚠️ 이 모듈은 검증 전용이다. 분석 파이프라인이 여기 있는 정보를 입력으로 쓰면
   시험 중에 답안지를 보는 것과 같아진다.

참고: docs/00_design.md §M4, docs/08_interpretation_guide.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternScore:
    """패턴 하나에 대한 채점 결과.

    Attributes:
        pattern: 불량 패턴명
        n_case_units: 이상군 표본 수 (lot 또는 웨이퍼)
        true_step: 정답 원인 스텝
        true_chambers: 정답 원인 챔버들
        top1_chamber, top1_step: 분석이 1순위로 지목한 것
        top1_chamber_hit, top1_step_hit: 1순위 적중 여부
        top3_chamber_hit: 상위 3개 안에 정답이 있는가
        n_significant: 유의 판정된 챔버 수
        top1_odds_ratio, top1_p_adj: 1순위의 통계량
    """

    pattern: str
    n_case_units: int
    true_step: str | None
    true_chambers: tuple[str, ...]
    top1_chamber: str
    top1_step: str
    top1_chamber_hit: bool
    top1_step_hit: bool
    top3_chamber_hit: bool
    n_significant: int
    top1_odds_ratio: float
    top1_p_adj: float

    def describe(self) -> str:
        c1 = "✅" if self.top1_chamber_hit else "❌"
        s1 = "✅" if self.top1_step_hit else "❌"
        c3 = "✅" if self.top3_chamber_hit else "❌"
        return (
            f"{self.pattern:<11} 정답 {self.true_step}  "
            f"Top1챔버 {c1}  Top1스텝 {s1}  Top3 {c3}  "
            f"유의 {self.n_significant:>2}개  1위={self.top1_chamber} "
            f"(OR {self.top1_odds_ratio:.1f}, p_adj {self.top1_p_adj:.1e})"
        )


@dataclass
class ScoreReport:
    """전체 채점 요약."""

    scores: list[PatternScore] = field(default_factory=list)
    unit: str = "lot"

    @property
    def top1_chamber_rate(self) -> float:
        return _rate([s.top1_chamber_hit for s in self.scores])

    @property
    def top1_step_rate(self) -> float:
        return _rate([s.top1_step_hit for s in self.scores])

    @property
    def top3_chamber_rate(self) -> float:
        return _rate([s.top3_chamber_hit for s in self.scores])

    def summary(self) -> str:
        n = len(self.scores)
        if not n:
            return "채점할 패턴이 없습니다."
        lines = [
            f"검정 단위: {self.unit}   대상 패턴: {n}종",
            "",
            f"  원인 스텝 Top-1 적중률 : {self.top1_step_rate:.0%} "
            f"({sum(s.top1_step_hit for s in self.scores)}/{n})",
            f"  원인 챔버 Top-1 적중률 : {self.top1_chamber_rate:.0%} "
            f"({sum(s.top1_chamber_hit for s in self.scores)}/{n})",
            f"  원인 챔버 Top-3 적중률 : {self.top3_chamber_rate:.0%} "
            f"({sum(s.top3_chamber_hit for s in self.scores)}/{n})",
        ]
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pattern": s.pattern,
                    "n_case_units": s.n_case_units,
                    "true_step": s.true_step,
                    "top1_chamber": s.top1_chamber,
                    "top1_step": s.top1_step,
                    "top1_chamber_hit": s.top1_chamber_hit,
                    "top1_step_hit": s.top1_step_hit,
                    "top3_chamber_hit": s.top3_chamber_hit,
                    "n_significant": s.n_significant,
                    "top1_odds_ratio": s.top1_odds_ratio,
                    "top1_p_adj": s.top1_p_adj,
                }
                for s in self.scores
            ]
        )


def _rate(flags: list[bool]) -> float:
    return float(np.mean(flags)) if flags else 0.0


def true_causes(ground_truth: pd.DataFrame, pattern: str) -> tuple[str | None, tuple[str, ...]]:
    """정답지에서 해당 패턴의 원인 스텝과 챔버를 꺼낸다.

    Returns:
        (원인 스텝, 원인 챔버들). 원인이 없는 패턴(none/Random)은 (None, ())
    """
    rows = ground_truth[ground_truth["pattern_label"] == pattern]
    if rows.empty:
        return None, ()

    steps = rows["true_root_step"].dropna()
    if steps.empty:
        return None, ()

    chambers = rows["true_root_equip"].dropna()
    return str(steps.iloc[0]), tuple(sorted(set(chambers)))


def score_pattern(
    ranking: pd.DataFrame,
    ground_truth: pd.DataFrame,
    pattern: str,
    n_case_units: int,
) -> PatternScore | None:
    """커미널리티 랭킹 하나를 채점한다.

    Args:
        ranking: commonality.analyze()의 결과
        ground_truth: 정답지
        pattern: 대상 패턴
        n_case_units: 이상군 표본 수 (기록용)

    Returns:
        PatternScore. 정답이 없는 패턴이거나 랭킹이 비면 None
    """
    if ranking.empty:
        return None

    true_step, true_chambers = true_causes(ground_truth, pattern)
    if true_step is None:
        return None

    top = ranking.iloc[0]
    top3 = set(ranking.head(3)["chamber_id"])

    return PatternScore(
        pattern=pattern,
        n_case_units=n_case_units,
        true_step=true_step,
        true_chambers=true_chambers,
        top1_chamber=str(top["chamber_id"]),
        top1_step=str(top["step_id"]),
        top1_chamber_hit=str(top["chamber_id"]) in true_chambers,
        top1_step_hit=str(top["step_id"]) == true_step,
        top3_chamber_hit=bool(top3 & set(true_chambers)),
        n_significant=int(ranking["significant"].sum()),
        top1_odds_ratio=float(top["odds_ratio"]),
        top1_p_adj=float(top["p_adj"]),
    )


def score_all(
    wafer_master: pd.DataFrame,
    fdc_summary: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    patterns: list[str] | None = None,
    unit: str = "lot",
    min_case: int = 15,
) -> ScoreReport:
    """여러 패턴에 대해 커미널리티 분석을 돌리고 한꺼번에 채점한다.

    Args:
        wafer_master, fdc_summary, ground_truth: 데이터
        patterns: 채점할 패턴들. None이면 원인이 있는 패턴 전부
        unit: 검정 단위 ("lot" 권장)
        min_case: 이상군이 이보다 적으면 건너뛴다 (통계가 무의미)

    Returns:
        ScoreReport
    """
    from wafermap.analysis import commonality, spc

    if patterns is None:
        patterns = sorted(
            ground_truth.loc[ground_truth["true_root_step"].notna(), "pattern_label"].unique()
        )

    report = ScoreReport(unit=unit)
    for pattern in patterns:
        split = spc.split_by_pattern(wafer_master, pattern)
        if len(split.case_ids) < min_case:
            continue

        ranking = commonality.analyze(
            fdc_summary,
            split.case_ids,
            split.control_ids,
            wafer_master=wafer_master,
            unit=unit,
        )
        score = score_pattern(ranking, ground_truth, pattern, len(split.case_ids))
        if score is not None:
            report.scores.append(score)

    return report


def score_spc_detection(
    alarms: list,
    excursions: pd.DataFrame,
    pattern: str,
    cycle_time: pd.Timedelta,
    period: pd.Timedelta = pd.Timedelta(days=1),
) -> dict[str, float]:
    """SPC가 실제 이상 사건을 검출했는지 채점한다.

    ⚠️ **사이클 타임 보정이 필수다.** 이상 사건의 시각은 공정 투입 기준이고,
       SPC 경보는 EDS 검사 기준이라 그 차이만큼 밀려 있다. 보정을 빼먹으면
       검출률이 실제보다 낮게 나온다(이 프로젝트에서 42% → 83%로 바뀌었다).

    Args:
        alarms: control_chart()가 검출한 경보들
        excursions: 정답 이상 사건 목록
        pattern: 대상 패턴
        cycle_time: 공정 투입 → EDS 검사 소요 시간
        period: 집계 주기 (경보 끝 시점 보정용)

    Returns:
        {"n_truth", "n_alarms", "n_detected", "detection_rate", "n_false_positive"}
    """
    truth = excursions[excursions["pattern"] == pattern]
    if truth.empty:
        return {
            "n_truth": 0, "n_alarms": len(alarms), "n_detected": 0,
            "detection_rate": 0.0, "n_false_positive": len(alarms),
        }

    def overlaps(alarm, row) -> bool:
        start = row["t_start"] + cycle_time
        end = row["t_end"] + cycle_time
        return not (alarm.end + period < start or alarm.start > end)

    detected = sum(1 for _, row in truth.iterrows() if any(overlaps(a, row) for a in alarms))
    false_positive = sum(
        1 for a in alarms if not any(overlaps(a, row) for _, row in truth.iterrows())
    )

    return {
        "n_truth": len(truth),
        "n_alarms": len(alarms),
        "n_detected": detected,
        "detection_rate": detected / len(truth),
        "n_false_positive": false_positive,
    }


# ──────────────────────────────────────────────────────────────────────────
# M4 — 원인 파라미터 채점
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParamScore:
    """패턴 하나의 원인 파라미터 채점 결과.

    Attributes:
        pattern: 불량 패턴
        step_id: 분석한 스텝
        chamber_id: 층화한 챔버 (None이면 스텝 전체)
        auc: 모델의 교차검증 AUC
        true_params: 정답 파라미터
        ranked_params: 분석이 매긴 순위 (조치 가능한 것만)
        top1_hit, top3_hit, top5_hit: 상위 n개 안에 정답이 있는가
        n_case: 이상군 표본 수
        top1_is_measurement: 필터 전 1위가 계측값이었는가
    """

    pattern: str
    step_id: str
    chamber_id: str | None
    auc: float
    true_params: tuple[str, ...]
    ranked_params: tuple[str, ...]
    top1_hit: bool
    top3_hit: bool
    top5_hit: bool
    n_case: int
    top1_is_measurement: bool

    def describe(self) -> str:
        marks = "".join(
            "✅" if hit else "❌" for hit in (self.top1_hit, self.top3_hit, self.top5_hit)
        )
        note = " (원본 1위는 계측값)" if self.top1_is_measurement else ""
        return (
            f"{self.pattern:<11} {self.step_id}  AUC {self.auc:.3f}  "
            f"Top1/3/5 {marks}  1위={self.ranked_params[0] if self.ranked_params else '-'}{note}"
        )


@dataclass
class ParamReport:
    """M4 전체 채점 요약."""

    scores: list[ParamScore] = field(default_factory=list)

    @property
    def top1_rate(self) -> float:
        return _rate([s.top1_hit for s in self.scores])

    @property
    def top3_rate(self) -> float:
        return _rate([s.top3_hit for s in self.scores])

    @property
    def top5_rate(self) -> float:
        return _rate([s.top5_hit for s in self.scores])

    @property
    def mean_auc(self) -> float:
        values = [s.auc for s in self.scores if np.isfinite(s.auc)]
        return float(np.mean(values)) if values else float("nan")

    def summary(self) -> str:
        n = len(self.scores)
        if not n:
            return "채점할 패턴이 없습니다."
        return "\n".join(
            [
                f"대상 패턴: {n}종   평균 AUC: {self.mean_auc:.3f}",
                "",
                f"  원인 파라미터 Top-1 적중률 : {self.top1_rate:.0%} "
                f"({sum(s.top1_hit for s in self.scores)}/{n})",
                f"  원인 파라미터 Top-3 적중률 : {self.top3_rate:.0%} "
                f"({sum(s.top3_hit for s in self.scores)}/{n})",
                f"  원인 파라미터 Top-5 포함률 : {self.top5_rate:.0%} "
                f"({sum(s.top5_hit for s in self.scores)}/{n})",
            ]
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pattern": s.pattern,
                    "step_id": s.step_id,
                    "chamber_id": s.chamber_id,
                    "auc": s.auc,
                    "n_case": s.n_case,
                    "top1_hit": s.top1_hit,
                    "top3_hit": s.top3_hit,
                    "top5_hit": s.top5_hit,
                    "top1_is_measurement": s.top1_is_measurement,
                    "true_params": ", ".join(s.true_params),
                    "top5_ranked": ", ".join(s.ranked_params[:5]),
                }
                for s in self.scores
            ]
        )


def score_parameters(
    wafer_master: pd.DataFrame,
    fdc_summary: pd.DataFrame,
    *,
    patterns: list[str] | None = None,
    stratify_by_chamber: bool = True,
    actionable_only: bool = True,
    min_case: int = 15,
) -> ParamReport:
    """원인 파라미터 규명을 돌리고 정답지와 대조해 채점한다.

    Args:
        wafer_master, fdc_summary: 데이터
        patterns: 채점할 패턴들. None이면 원인이 있는 패턴 전부
        stratify_by_chamber: M3가 찾은 진범 챔버로 층화할지.
            층화하면 챔버 간 baseline 차이가 제거되어 AUC가 올라간다.
        actionable_only: 조치 가능한 파라미터만 순위에 넣을지(§attribution).
            계측값은 조작할 수 없어 개선안의 대상이 될 수 없다.
        min_case: 이상군이 이보다 적으면 건너뛴다

    Returns:
        ParamReport
    """
    from wafermap.analysis import attribution, commonality, spc
    from wafermap.config import CAUSE_RULES
    from wafermap.models import cause_model

    if patterns is None:
        patterns = [p for p, r in CAUSE_RULES.items() if r.step_id is not None]

    report = ParamReport()
    for pattern in sorted(patterns):
        rule = CAUSE_RULES.get(pattern)
        if rule is None or rule.step_id is None:
            continue

        split = spc.split_by_pattern(wafer_master, pattern)
        if len(split.case_ids) < min_case:
            continue

        chamber = None
        if stratify_by_chamber:
            ranking = commonality.analyze(
                fdc_summary, split.case_ids, split.control_ids,
                wafer_master=wafer_master, unit="lot",
            )
            same_step = ranking[ranking["step_id"] == rule.step_id]
            if not same_step.empty:
                chamber = str(same_step.iloc[0]["chamber_id"])

        try:
            result = cause_model.fit(
                fdc_summary, rule.step_id, pattern,
                split.case_ids, split.control_ids, chamber_id=chamber,
            )
        except ValueError:
            continue

        evidence = attribution.build_evidence(
            result, fdc_summary, split.case_ids, split.control_ids, chamber_id=chamber
        )
        if not evidence:
            continue

        top1_is_measurement = not evidence[0].is_controllable
        ranked = attribution.actionable_ranking(evidence) if actionable_only else evidence
        ranked_params = tuple(e.param for e in ranked)

        true_params = tuple(p.param for p in rule.perturbations)
        report.scores.append(
            ParamScore(
                pattern=pattern,
                step_id=rule.step_id,
                chamber_id=chamber,
                auc=result.auc,
                true_params=true_params,
                ranked_params=ranked_params,
                top1_hit=bool(ranked_params[:1]) and ranked_params[0] in true_params,
                top3_hit=bool(set(ranked_params[:3]) & set(true_params)),
                top5_hit=bool(set(ranked_params[:5]) & set(true_params)),
                n_case=result.n_case,
                top1_is_measurement=top1_is_measurement,
            )
        )

    return report


# ──────────────────────────────────────────────────────────────────────────
# M5.5-① 검사 설비 축 채점 — "공정이냐 검사냐"를 가려내는가
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AxisScore:
    """패턴 1종에 대해 두 축의 1위가 정답과 맞는지.

    Attributes:
        pattern: 대상 패턴
        n_process, n_test: 공정 기인 / 검사 기인 불량 장수 (정답지)
        truth_process, truth_test: 각 축의 정답 설비 (없으면 None)
        found_process, found_test: 각 축에서 분석이 지목한 1위
        rank_test_overall: 검사 축 1위가 **통합 랭킹**에서 몇 위였는가
        or_process, or_test: 각 축 1위의 오즈비
    """

    pattern: str
    n_process: int
    n_test: int
    truth_process: str | None
    truth_test: str | None
    found_process: str | None
    found_test: str | None
    rank_test_overall: int | None
    or_process: float
    or_test: float

    @property
    def process_hit(self) -> bool | None:
        if self.truth_process is None:
            return None
        return self.found_process == self.truth_process

    @property
    def test_hit(self) -> bool | None:
        if self.truth_test is None:
            return None
        return self.found_test == self.truth_test

    @property
    def hidden_in_combined(self) -> bool:
        """검사 원인이 통합 랭킹 상위 5위 밖으로 밀렸는가.

        왜 5위인가: 현업에서 커미널리티 결과를 볼 때 상위 몇 개만 확인한다.
            정답이 9위에 있으면 **찾았다고 말할 수 없다.**
        """
        return (
            self.truth_test is not None
            and self.rank_test_overall is not None
            and self.rank_test_overall > 5
        )

    def describe(self) -> str:
        def mark(hit: bool | None) -> str:
            return "—" if hit is None else ("✅" if hit else "❌")

        parts = [
            f"{self.pattern:<11}",
            f"공정 {mark(self.process_hit)} {str(self.found_process or '—'):<12}"
            f"OR {self.or_process:5.2f}",
            f"검사 {mark(self.test_hit)} {str(self.found_test or '—'):<7}"
            f"OR {self.or_test:5.2f}",
            f"(불량 공정 {self.n_process} / 검사 {self.n_test})",
        ]
        line = "  ".join(parts)
        if self.hidden_in_combined:
            line += f"  ⚠️ 통합 {self.rank_test_overall}위 — 축을 안 나눴으면 못 봤다"
        return line


@dataclass
class AxisReport:
    """검사 설비 축 채점 묶음."""

    scores: list[AxisScore] = field(default_factory=list)

    def summary(self) -> str:
        with_test = [s for s in self.scores if s.truth_test is not None]
        with_proc = [s for s in self.scores if s.truth_process is not None]
        hidden = [s for s in with_test if s.hidden_in_combined]

        lines = [f"대상 패턴: {len(self.scores)}종 (검사 기인이 섞인 패턴 {len(with_test)}종)", ""]
        if with_proc:
            hit = _rate([bool(s.process_hit) for s in with_proc])
            lines.append(
                f"  공정 설비 Top-1 적중률 : {hit:.0%} "
                f"({sum(bool(s.process_hit) for s in with_proc)}/{len(with_proc)})"
            )
        if with_test:
            hit = _rate([bool(s.test_hit) for s in with_test])
            lines.append(
                f"  검사 설비 Top-1 적중률 : {hit:.0%} "
                f"({sum(bool(s.test_hit) for s in with_test)}/{len(with_test)})"
            )
            lines.append(
                f"  통합 랭킹에 묻힌 건수  : {len(hidden)}/{len(with_test)} "
                f"— 축을 나누지 않았다면 놓쳤을 원인"
            )

        # ── 위양성 위험 ★ 정직하게 같이 보고한다 ────────────────────────
        # 검사 축은 **검사 원인이 없어도 무언가를 1위로 내놓는다.** 그 OR이 진짜
        # 원인이 있을 때보다 오히려 클 수 있다면, OR만 보고 "검사가 문제"라고
        # 판단해서는 안 된다는 뜻이다.
        without_test = [
            s for s in self.scores if s.truth_test is None and np.isfinite(s.or_test)
        ]
        real = [s.or_test for s in with_test if np.isfinite(s.or_test)]
        fake = [s.or_test for s in without_test]
        if real and fake:
            lines += [
                "",
                f"  ⚠️ 검사 축 OR — 진짜 원인 있을 때 평균 {np.mean(real):.2f} "
                f"vs 없을 때 평균 {np.mean(fake):.2f}",
            ]
            if np.mean(fake) >= np.mean(real):
                lines.append(
                    "     **OR만으로는 검사 기인 여부를 판정할 수 없다.** 축은 후보를"
                )
                lines.append(
                    "     보여줄 뿐이며, 확정은 재검사(다른 카드로 재측정)로 해야 한다."
                )
        return "\n".join(lines)


def score_equipment_axis(
    wafer_master: pd.DataFrame,
    fdc_summary: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    patterns: list[str] | None = None,
    unit: str = "lot",
    min_case: int = 15,
) -> AxisReport:
    """검사 설비 축을 넣었을 때 분석이 두 원인을 갈라내는지 채점한다 ★.

    왜 이 채점이 필요한가: 검사 설비를 데이터에 넣기만 하고 끝내면, 정말 도움이
        되는지 알 수 없다. 검사 기인 불량이 섞인 패턴에서 **프로브 카드를 지목하는가**,
        그리고 축을 나누지 않았다면 **놓쳤을 것인가**를 수치로 확인한다.

    Returns:
        AxisReport
    """
    from wafermap.analysis import commonality, spc

    if patterns is None:
        patterns = sorted(
            ground_truth.loc[ground_truth["true_root_step"].notna(), "pattern_label"].unique()
        )

    report = AxisReport()
    for pattern in patterns:
        sub = ground_truth[ground_truth["pattern_label"] == pattern]
        if len(sub) < min_case:
            continue

        from_test = sub["is_test_induced"].to_numpy()

        def _mode(frame: pd.DataFrame) -> str | None:
            values = frame["true_root_equip"].dropna()
            return str(values.mode().iloc[0]) if len(values) else None

        truth_process = _mode(sub[~from_test]) if (~from_test).any() else None
        truth_test = _mode(sub[from_test]) if from_test.any() else None

        split = spc.split_by_pattern(wafer_master, pattern)
        axes = commonality.by_axis(
            fdc_summary, split.case_ids, split.control_ids,
            wafer_master=wafer_master, unit=unit,
        )
        if axes.empty:
            continue

        def _pick(axis: str) -> dict:
            row = axes[axes["axis"] == axis]
            return row.iloc[0].to_dict() if len(row) else {}

        proc, test = _pick("process"), _pick("test")
        report.scores.append(
            AxisScore(
                pattern=pattern,
                n_process=int((~from_test).sum()),
                n_test=int(from_test.sum()),
                truth_process=truth_process,
                truth_test=truth_test,
                found_process=proc.get("chamber_id"),
                found_test=test.get("chamber_id"),
                rank_test_overall=(
                    int(test["rank_overall"]) if "rank_overall" in test else None
                ),
                or_process=float(proc.get("odds_ratio", float("nan"))),
                or_test=float(test.get("odds_ratio", float("nan"))),
            )
        )
    return report
