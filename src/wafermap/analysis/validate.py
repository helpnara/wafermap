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
