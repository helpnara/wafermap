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


def true_steps(ground_truth: pd.DataFrame, pattern: str) -> tuple[str, ...]:
    """이 패턴의 원인이 될 수 있는 **모든** 스텝.

    왜 하나가 아닌가 ★: 같은 패턴이 여러 경로로 생긴다(공정/검사/스파이크/드리프트).
        Edge-Ring은 식각 챔버 마모로도, 프로브 카드 마모로도 나온다. 예전 구현은
        정답지의 **첫 행**에서 스텝을 꺼냈는데, 그러면 그 웨이퍼가 어느 경로였는지에
        따라 정답이 바뀌었다. 실제로 Edge-Ring의 정답이 T010으로 잡혀, 식각 챔버를
        올바르게 지목한 분석이 오답 처리됐다.
    """
    rows = ground_truth[ground_truth["pattern_label"] == pattern]
    steps = rows["true_root_step"].dropna()
    return tuple(sorted(set(steps.astype(str))))


def true_causes(ground_truth: pd.DataFrame, pattern: str) -> tuple[str | None, tuple[str, ...]]:
    """정답지에서 해당 패턴의 대표 원인 스텝과 원인 챔버들을 꺼낸다.

    대표 스텝은 **가장 많은 웨이퍼를 설명하는** 스텝이다. 여러 스텝이 정답일 수
    있으므로, 적중 판정에는 `true_steps()`를 함께 쓴다.

    Returns:
        (대표 원인 스텝, 원인 챔버들). 원인이 없는 패턴(none/Random)은 (None, ())
    """
    rows = ground_truth[ground_truth["pattern_label"] == pattern]
    if rows.empty:
        return None, ()

    steps = rows["true_root_step"].dropna()
    if steps.empty:
        return None, ()

    chambers = rows["true_root_equip"].dropna()
    return str(steps.mode().iloc[0]), tuple(sorted(set(chambers)))


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
        # 여러 경로가 정답일 수 있으므로 **집합** 포함으로 판정한다
        top1_step_hit=str(top["step_id"]) in true_steps(ground_truth, pattern),
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


# ──────────────────────────────────────────────────────────────────────────
# M5.5-② 시계열 파생 피처 채점 — 요약통계가 놓치는 것을 잡는가
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShapeScore:
    """이상의 **모양**을 구분할 수 있는가 — 스파이크 vs 드리프트.

    왜 이 형태로 재나 ★★: 처음에는 "요약통계로는 스파이크를 아예 못 본다"를 재려
        했다. 그런데 요약값을 시계열에서 제대로 계산하자 `std`가 스파이크를 잘
        잡았다(AUC 0.89). 첨부 문서가 말한 것은 **평균**이고, 그건 실제로 거의
        안 움직인다(0.35σ). 산포까지 못 본다는 것은 사실이 아니었다.

        그래서 질문을 바꿨다. 스파이크와 드리프트는 산포가 같도록 맞춰 두면
        **요약통계에서 사실상 구분되지 않는다.** 그런데 원인도 조치도 전혀 다르다.
        시계열 피처의 진짜 값어치는 검출이 아니라 **무엇이 일어났는지 말해 주는 것**이다.

    Attributes:
        param: 대상 파라미터
        n_spike, n_drift: 두 기전의 웨이퍼 수
        best_summary, auc_summary: 요약통계 중 가장 잘 구분한 컬럼과 AUC
        best_trace, auc_trace: 시계열 피처 중 가장 잘 구분한 컬럼과 AUC
        mean_gap_sigma, std_gap_sigma: 두 기전의 평균·산포 차이 (σ 단위)
    """

    param: str
    n_spike: int
    n_drift: int
    best_summary: str
    auc_summary: float
    best_trace: str
    auc_trace: float
    mean_gap_sigma: float
    std_gap_sigma: float

    def describe(self) -> str:
        return (
            f"{self.param:<12} 스파이크 {self.n_spike}장 vs 드리프트 {self.n_drift}장\n"
            f"    두 기전의 요약값 차이 : 평균 {self.mean_gap_sigma:.2f}σ · "
            f"산포 {self.std_gap_sigma:.2f}σ  (맞춰 둔 값)\n"
            f"    요약통계 최고 구분력  : AUC {self.auc_summary:.3f}  ({self.best_summary})\n"
            f"    시계열 피처 최고      : AUC {self.auc_trace:.3f}  ({self.best_trace})"
        )


@dataclass(frozen=True)
class TraceFeatureScore:
    """기전별로 요약통계만 썼을 때와 시계열 피처를 더했을 때를 비교한다.

    Attributes:
        pattern: 대상 패턴
        mechanism: 이상의 모양 ("process"=지속형 / "spike"=순간형)
        n_case: 해당 기전의 불량 장수
        auc_summary, auc_trace: 두 조건의 판별 AUC
        top_summary, top_trace: 두 조건의 SHAP 1위 파라미터
        truth_params: 정답 파라미터 목록
        n_features_summary, n_features_trace: 쓴 피처 수
    """

    pattern: str
    mechanism: str
    n_case: int
    auc_summary: float
    auc_trace: float
    top_summary: str | None
    top_trace: str | None
    truth_params: tuple[str, ...]
    n_features_summary: int
    n_features_trace: int

    @staticmethod
    def _hit(column: str | None, truth: tuple[str, ...]) -> bool:
        if not column:
            return False
        from wafermap.analysis.attribution import base_param_name

        return base_param_name(column) in truth

    @property
    def hit_summary(self) -> bool:
        return self._hit(self.top_summary, self.truth_params)

    @property
    def hit_trace(self) -> bool:
        return self._hit(self.top_trace, self.truth_params)

    @property
    def auc_gain(self) -> float:
        return self.auc_trace - self.auc_summary

    def describe(self) -> str:
        def mark(hit: bool) -> str:
            return "✅" if hit else "❌"

        return (
            f"{self.pattern:<10}{self.mechanism:<9}불량 {self.n_case:>3}장  "
            f"요약만 AUC {self.auc_summary:.3f} {mark(self.hit_summary)}"
            f"{str(self.top_summary or '—'):<24}"
            f"→ +trace AUC {self.auc_trace:.3f} {mark(self.hit_trace)}"
            f"{str(self.top_trace or '—')}"
        )


@dataclass
class TraceFeatureReport:
    scores: list[TraceFeatureScore] = field(default_factory=list)
    shapes: list[ShapeScore] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"대상: {len(self.scores)}건 (패턴 × 기전)", ""]
        for mechanism in ("process", "spike"):
            group = [s for s in self.scores if s.mechanism == mechanism]
            if not group:
                continue
            label = "지속형" if mechanism == "process" else "순간 스파이크형"
            hit_s = _rate([s.hit_summary for s in group])
            hit_t = _rate([s.hit_trace for s in group])
            auc_s = float(np.mean([s.auc_summary for s in group]))
            auc_t = float(np.mean([s.auc_trace for s in group]))
            lines += [
                f"  [{label}]",
                f"    요약통계만    : AUC {auc_s:.3f}  원인 파라미터 적중 {hit_s:.0%}",
                f"    + trace 피처  : AUC {auc_t:.3f}  원인 파라미터 적중 {hit_t:.0%}",
                f"    AUC 변화      : {auc_t - auc_s:+.3f}",
                "",
            ]

        if self.shapes:
            lines.append("  [이상의 모양 구분 — 스파이크 vs 드리프트]")
            for shape in self.shapes:
                lines.append("    " + shape.describe().replace("\n", "\n    "))
            gain = float(np.mean([s.auc_trace - s.auc_summary for s in self.shapes]))
            lines += [
                "",
                f"    구분력 향상: {gain:+.3f}",
                "",
                "  📖 검출력 차이는 크지 않다. 진짜 차이는 **해석**이다.",
                "     `min`이 낮다는 사실은 무엇을 고쳐야 하는지 말해 주지 않는다.",
                "     `time_above 2.2초 · n_excursions 1`은 순간 과열이라 인터락을 걸라는 뜻이고,",
                "     `slope 0.079`는 서서히 밀린다는 뜻이라 센서 교정을 하라는 뜻이다.",
                "     조치가 갈리는 지점이 여기다.",
            ]
        return "\n".join(lines)


def score_trace_features(
    wafer_master: pd.DataFrame,
    fdc_summary: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    min_case: int = 20,
) -> TraceFeatureReport:
    """기전별로 시계열 피처의 효과를 채점한다 ★.

    왜 기전을 나눠서 보나: 전체 평균만 보면 효과가 희석된다. 시계열 피처는
        **순간 스파이크형에만** 도움이 되고 지속형에는 영향이 없어야 정상이다.
        지속형에서도 AUC가 올라간다면 그건 피처를 늘려 과적합된 것일 수 있다.

    Returns:
        TraceFeatureReport
    """
    from wafermap.analysis import attribution
    from wafermap.config import CAUSE_RULES, SPIKE_CAUSE_RULES
    from wafermap.models import cause_model

    report = TraceFeatureReport()
    control = list(ground_truth.loc[ground_truth["pattern_label"] == "none", "wafer_id"])

    for pattern in sorted(SPIKE_CAUSE_RULES):
        sub = ground_truth[ground_truth["pattern_label"] == pattern]
        for mechanism, rules in (("process", CAUSE_RULES), ("spike", SPIKE_CAUSE_RULES)):
            case = list(sub.loc[sub["cause_mechanism"] == mechanism, "wafer_id"])
            if len(case) < min_case:
                continue
            rule = rules[pattern]
            truth = tuple(p.param for p in rule.perturbations)

            fitted = {}
            for key, suffixes in (
                ("summary", cause_model.DEFAULT_SUFFIXES),
                ("trace", cause_model.TRACE_SUFFIXES),
            ):
                result = cause_model.fit(
                    fdc_summary, rule.step_id, pattern, case, control, suffixes=suffixes
                )
                evidence = attribution.build_evidence(
                    result, fdc_summary, case, control, top_n=3
                )
                fitted[key] = (result, evidence[0].column if evidence else None)

            report.scores.append(
                TraceFeatureScore(
                    pattern=pattern,
                    mechanism=mechanism,
                    n_case=len(case),
                    auc_summary=fitted["summary"][0].auc,
                    auc_trace=fitted["trace"][0].auc,
                    top_summary=fitted["summary"][1],
                    top_trace=fitted["trace"][1],
                    truth_params=truth,
                    n_features_summary=len(fitted["summary"][0].feature_names),
                    n_features_trace=len(fitted["trace"][0].feature_names),
                )
            )

        report.shapes.extend(_score_shape(fdc_summary, sub, pattern, min_case=min_case))
    return report


def _score_shape(
    fdc_summary: pd.DataFrame,
    sub: pd.DataFrame,
    pattern: str,
    *,
    min_case: int,
) -> list[ShapeScore]:
    """스파이크와 드리프트를 갈라낼 수 있는지 **원인 파라미터 하나로만** 잰다.

    왜 파라미터를 하나로 묶나 ★: 스텝의 전 파라미터를 넣고 모델을 돌리면 AUC가
        0.98까지 나온다. 그런데 그건 모양을 구분한 게 아니다. 두 기전은 서로 다른
        이상 사건에서 나왔으므로 **챔버·시간대가 달라** 무관한 파라미터로도 갈린다.
        "시계열 모양을 구분할 수 있는가"를 물으려면 그 파라미터만 봐야 한다.

    왜 단변량 AUC인가: 컬럼 하나하나가 얼마나 구분하는지 보여야 "어느 피처가
        일을 하는지"가 드러난다. 모델 AUC 한 숫자로는 알 수 없다.
    """
    from sklearn.metrics import roc_auc_score

    from wafermap.config import DRIFT_CAUSE_RULES
    from wafermap.features.trace_feat import FEATURE_SUFFIXES

    rule = DRIFT_CAUSE_RULES.get(pattern)
    if rule is None:
        return []
    param = rule.perturbations[0].param

    spike = set(sub.loc[sub["cause_mechanism"] == "spike", "wafer_id"])
    drift = set(sub.loc[sub["cause_mechanism"] == "drift", "wafer_id"])
    if len(spike) < min_case or len(drift) < min_case:
        return []

    step = fdc_summary[fdc_summary["step_id"] == rule.step_id]
    rows = step[step["wafer_id"].isin(spike | drift)]
    y = rows["wafer_id"].isin(spike).astype(int).to_numpy()

    best = {"summary": (0.0, ""), "trace": (0.0, "")}
    for column in [c for c in rows.columns if c.startswith(f"{param}_")]:
        values = rows[column].to_numpy(dtype=float)
        # 방향을 모르므로 뒤집은 쪽도 본다 (구분력의 절대 크기가 관심사다)
        auc = max(roc_auc_score(y, values), roc_auc_score(y, -values))
        kind = "trace" if column.endswith(FEATURE_SUFFIXES) else "summary"
        if auc > best[kind][0]:
            best[kind] = (auc, column)

    def gap(suffix: str) -> float:
        a = rows.loc[rows["wafer_id"].isin(spike), f"{param}{suffix}"]
        b = rows.loc[rows["wafer_id"].isin(drift), f"{param}{suffix}"]
        pooled = float(np.sqrt((a.var() + b.var()) / 2)) or 1.0
        return float(abs(a.mean() - b.mean()) / pooled)

    return [
        ShapeScore(
            param=param,
            n_spike=len(spike),
            n_drift=len(drift),
            best_summary=best["summary"][1],
            auc_summary=best["summary"][0],
            best_trace=best["trace"][1],
            auc_trace=best["trace"][0],
            mean_gap_sigma=gap("_mean"),
            std_gap_sigma=gap("_std"),
        )
    ]
