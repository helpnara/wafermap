"""측정된 마일스톤 성과 — 화면과 문서가 같은 숫자를 쓰도록 한곳에 모은다.

왜 코드에 두나: 개요 화면과 `docs/04_results.md`가 서로 다른 숫자를 말하면
    어느 쪽을 믿어야 할지 알 수 없다. 출처를 한곳으로 모으고, **어떻게 측정했는지**를
    숫자 옆에 붙여 둔다.

왜 하드코딩인가: 시드 반복 채점은 한 번에 수 분이 걸린다. 화면을 열 때마다 돌릴 수
    없으므로 측정값을 적어 둔다. 대신 **재현 명령을 함께 적어** 누구든 다시 잴 수
    있게 한다. 숫자를 갱신할 때는 반드시 그 명령으로 다시 재고 여기를 고칠 것.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    """측정된 지표 하나.

    Attributes:
        label: 화면에 띄울 이름
        value: 대표값 문자열
        detail: 측정 조건 (시드 수 등) — 없으면 단일 실행
        robust: 시드를 바꿔도 흔들리지 않는가. False면 화면에 주의 표시를 붙인다
    """

    label: str
    value: str
    detail: str = ""
    robust: bool = True


@dataclass(frozen=True)
class Milestone:
    """마일스톤 하나의 요약."""

    key: str
    title: str
    question: str
    method: str
    metrics: tuple[Metric, ...]
    reproduce: str = ""


#: 측정 결과. 갱신 시 `reproduce` 명령으로 다시 재고 docs/04_results.md도 함께 고칠 것.
MILESTONES: tuple[Milestone, ...] = (
    Milestone(
        key="M2",
        title="패턴 분류",
        question="어떤 불량 패턴인가?",
        method="기하·Radon 피처 99종 + LightGBM (CNN 대조군)",
        metrics=(
            Metric("macro-F1", "0.973", "5-fold 교차검증"),
            Metric("CNN 대조군", "0.946", "103배 느림"),
        ),
        reproduce="python scripts/train_pattern_model.py",
    ),
    Milestone(
        key="M3",
        title="이상 구간·원인 설비",
        question="언제, 어느 설비에서 시작됐나?",
        method="p-chart SPC(Laney 보정) + 커미널리티(Fisher·OR·BH-FDR·CMH)",
        metrics=(
            Metric("원인 챔버 Top-3", "90% ± 12%", "시드 8개"),
            Metric("원인 챔버 Top-1", "67% ± 17%", "시드 8개", robust=False),
        ),
        reproduce="python scripts/run_seed_study.py --n-seeds 8",
    ),
    Milestone(
        key="M4",
        title="원인 파라미터",
        question="그 설비의 무엇이 문제인가?",
        method="LightGBM + SHAP, 분포 증거 대조, 조치 가능성 필터",
        metrics=(
            Metric("파라미터 Top-3", "100% ± 0%", "시드 4개"),
            Metric("파라미터 Top-1", "75% ± 8%", "시드 4개", robust=False),
        ),
        reproduce="python scripts/run_seed_study.py --milestone m4 --n-seeds 4",
    ),
    Milestone(
        key="M5",
        title="개선안·기대효과",
        question="무엇을 바꾸고, 얼마를 버는가?",
        method="권고 운전 구간 + 반사실 시뮬레이션 + ROI(가정치 조정 가능)",
        metrics=(
            Metric("이동 예산 제약", "20%", "실행 가능성 반영"),
            Metric("기대효과 기준", "라인 환산", "챔버 수율과 병기"),
        ),
        reproduce="python scripts/run_recommend.py --pattern Donut",
    ),
    Milestone(
        key="M5.5-①",
        title="공정 vs 검사",
        question="이 불량, 공정에서 온 게 맞나?",
        method="프로브 카드·테스터를 후보에 넣고 축을 나눠 비교",
        metrics=(
            Metric("검사 설비 Top-1", "2/2", "검사 기인이 섞인 2개 패턴"),
            Metric("공정 설비 Top-1", "6/6", "회귀 없음"),
        ),
        reproduce="python scripts/run_equipment_axis.py",
    ),
    Milestone(
        key="M5.5-②",
        title="순간 이상",
        question="평균은 정상인데 순간 튀지 않았나?",
        method="시계열 파생 피처 6종 (slope·time_above·n_excursions 등)",
        metrics=(
            Metric("스파이크↔드리프트 구분", "0.901 → 0.969", "단변량 AUC"),
            Metric("3초 스파이크의 평균 이동", "0.35σ", "평균으로는 안 보인다"),
        ),
        reproduce="python scripts/run_trace_features.py --show-trace",
    ),
    Milestone(
        key="M5.5-③",
        title="공정 간 교호작용",
        question="각각은 규격 안인데 조합이 문제 아닌가?",
        method="2×2 분할표 + 설비 조합(A→B) + SHAP interaction",
        metrics=(
            Metric("설비 조합 Top-1", "1위", "초과 불량률 +36%p"),
            Metric("파라미터 쌍 (★필터)", "1위", "주효과는 0.12σ"),
        ),
        reproduce="python scripts/run_interaction.py",
    ),
)

#: 파이프라인 단계 — 개요 화면의 흐름도에 쓴다
PIPELINE: tuple[tuple[str, str, str], ...] = (
    ("🗺️", "웨이퍼 맵", "die별 합격/불합격을 공간으로"),
    ("🏷️", "패턴 분류", "9종 중 어느 형상인가"),
    ("📉", "이상 탐지", "언제부터 늘었나"),
    ("🏭", "원인 설비", "어느 챔버·어느 카드인가"),
    ("🔬", "원인 파라미터", "그 설비의 무엇이"),
    ("🎯", "개선안·ROI", "무엇을 바꾸고 얼마를 버나"),
)
