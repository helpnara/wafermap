"""프로젝트 전역 설정 — 경로, 제품 프로파일, 공정 플로우, 패턴-원인 매핑.

무엇을: 시뮬레이터·분석·앱이 공유하는 **단일 진실 공급원(single source of truth)** 이다.
        공정 스텝/설비/파라미터 규격과 "어떤 불량 패턴이 어떤 공정 때문에 생기는가"라는
        도메인 지식을 데이터 구조로 표현한다.
어떻게: frozen dataclass로 규격을 선언하고 모듈 상수로 노출한다. 값이 아니라 **구조**로
        정의했기 때문에, 파라미터를 추가해도 시뮬레이터/피처/분석 코드를 고칠 필요가 없다.
왜:    이런 상수를 각 모듈에 흩어 두면 "CMP down_force 규격이 어디 기준이냐"가 곧바로
        불일치를 만든다. 특히 §2.4의 패턴-원인 매핑은 시뮬레이터가 '심는 정답'인 동시에
        분석 모델이 '다시 찾아내야 할 대상'이라, 한 곳에서만 정의해야 검증이 성립한다.

참고: docs/00_design.md §2.4(패턴-원인 매핑), §2.5(공정 플로우)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# 1. 경로
# ──────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # LSWMD.pkl 등 원본 (git 제외)
INTERIM_DIR = DATA_DIR / "interim"      # 중간 산출물 (git 제외)
PROCESSED_DIR = DATA_DIR / "processed"  # 분석용 parquet (git 제외)
SAMPLE_DIR = DATA_DIR / "sample"        # 클라우드 배포용 소형 샘플 (git 포함)

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
LEARNING_NOTES_DIR = DOCS_DIR / "learning_notes"

WM811K_FILENAME = "LSWMD.pkl"


def processed_dir(source: str) -> Path:
    """데이터 소스별 산출물 디렉터리를 돌려준다.

    왜: 합성/실측 결과가 같은 폴더에 섞이면 어느 쪽 성능인지 추적이 불가능해진다.
        소스별로 물리적으로 분리해 두면 앱의 소스 토글(§4.7)이 캐시 키만으로 안전해진다.
    """
    if source not in ("synthetic", "real"):
        raise ValueError(f"source는 'synthetic' 또는 'real'이어야 합니다: {source!r}")
    return PROCESSED_DIR / source


# ──────────────────────────────────────────────────────────────────────────
# 2. 제품 프로파일 (가상 DRAM) — 설계서 §2.5.1
# ──────────────────────────────────────────────────────────────────────────

PRODUCT_NAME = "DDR5-16Gb"
TECH_NODE = "1z-nm"
WAFER_SIZE_MM = 300
WAFERS_PER_LOT = 25

#: EDS Bin 코드 정의. 맵 자체는 pass/fail 이진이며, Bin은 탐색 화면의 참고 정보다(§1.1).
#: 실측 WM-811K에는 Bin 정보가 없으므로 합성 데이터에서만 채워지고 UI에 '합성 예시' 배지를 붙인다.
BIN_CODES: dict[int, str] = {
    1: "Good",
    2: "Open/Short",
    3: "Leakage",
    4: "Cell Fail",
    5: "Refresh/Retention Fail",  # DRAM 특유 — capacitor 전하 유지 불량
    6: "Speed Fail",
}

# 웨이퍼맵 셀 값 (WM-811K 원본 규약과 동일하게 맞춘다)
DIE_NONE = 0  # 웨이퍼 밖 (die 없음)
DIE_PASS = 1
DIE_FAIL = 2


# ──────────────────────────────────────────────────────────────────────────
# 3. 불량 패턴 — 설계서 §2.4
# ──────────────────────────────────────────────────────────────────────────

#: WM-811K 9종 라벨. 문자열은 원본 데이터셋 표기를 그대로 따른다(실데이터 전환 시 매핑 불필요).
PATTERN_LABELS: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)

#: WM-811K 라벨 실측 분포(총 172,950장). 합성 데이터도 이 비율을 그대로 재현해
#: 극심한 클래스 불균형(none 147k vs Near-full 149)이라는 실제 난이도를 유지한다(§M1 불균형 처리).
WM811K_LABEL_COUNTS: dict[str, int] = {
    "none": 147_431,
    "Edge-Ring": 9_680,
    "Edge-Loc": 5_189,
    "Center": 4_294,
    "Loc": 3_593,
    "Scratch": 1_193,
    "Random": 866,
    "Donut": 555,
    "Near-full": 149,
}


# ──────────────────────────────────────────────────────────────────────────
# 4. 공정 파라미터 규격
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParamSpec:
    """FDC 파라미터 1개의 규격.

    Attributes:
        name: 파라미터명 (fdc_summary 컬럼 접두사가 된다)
        unit: 단위 (UI 표시용)
        nominal: 목표값(target). ``kind="counter"`` 이면 PM 주기 상한을 뜻한다.
        sigma: 정상 상태의 웨이퍼 간 변동 표준편차
        spec_lo, spec_hi: 관리 규격 하한/상한 (SPC 위반 판정 및 개선안 도출 기준)
        kind: ``"gaussian"`` = 목표값 주변 정규분포 / ``"counter"`` = PM 때 0으로 리셋되는
              소모품 누적 사용량(edge ring RF 시간, CMP 패드 수명 등)
        role: ``"control"`` = 엔지니어가 **직접 조작**할 수 있는 설정값 /
              ``"measurement"`` = 공정의 **결과로 측정**되는 값

              왜 이 구분이 필요한가 ★: 원인 분석에서 계측값이 상위에 오는 일이 흔하다.
              예를 들어 증착 온도가 튀면 두께 산포(thickness_sigma)가 커지고, 그 산포가
              불량과 더 강한 상관을 보인다. 하지만 **두께 산포는 조작할 수 없다.**
              조치하려면 온도를 잡아야 한다.

              계측값은 "원인의 결과"이므로, 조치 가능한 파라미터와 반드시 구분해야
              엉뚱한 개선안이 나오지 않는다.
        trace_noise: 웨이퍼 1장 처리 중 시계열 변동폭 (fdc_trace 생성에 사용)
    """

    name: str
    unit: str
    nominal: float
    sigma: float
    spec_lo: float
    spec_hi: float
    kind: str = "gaussian"
    role: str = "control"
    trace_noise: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("gaussian", "counter"):
            raise ValueError(f"kind는 'gaussian' 또는 'counter': {self.kind!r}")
        if self.role not in ("control", "measurement"):
            raise ValueError(f"role은 'control' 또는 'measurement': {self.role!r}")
        if self.spec_lo >= self.spec_hi:
            raise ValueError(f"{self.name}: spec_lo < spec_hi 여야 합니다")


@dataclass(frozen=True)
class ProcessStep:
    """공정 스텝 1개 = 설비군 + 파라미터 규격."""

    step_id: str
    name_ko: str
    name_en: str
    equipments: tuple[str, ...]
    chambers_per_equip: int
    params: tuple[ParamSpec, ...]
    note: str = ""
    #: 검정 단위를 직접 지정할 때 쓴다(비우면 설비×챔버로 자동 생성).
    #: EDS 검사 스텝은 단위가 챔버가 아니라 **프로브 카드**라서 이 필드가 필요하다.
    explicit_units: tuple[str, ...] = ()

    @property
    def chamber_ids(self) -> tuple[str, ...]:
        """`ETCH-B/ch3` 형태의 전체 챔버 식별자 목록.

        왜: 커미널리티 분석(M3)의 검정 단위가 설비가 아니라 **챔버**다. 같은 설비라도
            특정 챔버만 문제인 경우가 현업에서 흔하므로, 식별자를 처음부터 챔버 단위로 만든다.

        검사 스텝은 예외다: 소모품인 **프로브 카드**가 테스터 사이를 옮겨 다니므로
        "설비 안의 챔버"라는 계층이 성립하지 않는다. 그래서 단위를 직접 지정한다.
        """
        if self.explicit_units:
            return self.explicit_units
        return tuple(
            f"{eq}/ch{c}"
            for eq in self.equipments
            for c in range(1, self.chambers_per_equip + 1)
        )

    def param(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{self.step_id}에 '{name}' 파라미터가 없습니다")


# ── 공정 플로우 정의 (설계서 §2.5.2) ─────────────────────────────────────
# 수치는 공개 문헌 수준의 일반적 값을 참고한 **가상 규격**이다. 실제 팹 데이터가 아니며
# docs/03_simulator_spec.md에 가정을 전부 공개한다.

PROCESS_STEPS: tuple[ProcessStep, ...] = (
    ProcessStep(
        step_id="P010",
        name_ko="포토",
        name_en="Photo / Litho",
        equipments=("SCAN-01", "SCAN-02", "SCAN-03"),
        chambers_per_equip=1,
        params=(
            ParamSpec("focus_offset", "nm", 0.0, 8.0, -40.0, 40.0, trace_noise=3.0),
            ParamSpec("exposure_dose", "mJ/cm2", 30.0, 0.30, 28.0, 32.0, trace_noise=0.1),
            ParamSpec("overlay_x", "nm", 0.0, 1.5, -6.0, 6.0),
            ParamSpec("overlay_y", "nm", 0.0, 1.5, -6.0, 6.0),
            ParamSpec("bake_temp", "degC", 110.0, 0.40, 108.0, 112.0, trace_noise=0.2),
            ParamSpec("ebr_width", "mm", 2.0, 0.08, 1.7, 2.3),
            ParamSpec("defect_adder_count", "ea", 3.0, 1.8, 0.0, 15.0, role="measurement"),
        ),
    ),
    ProcessStep(
        step_id="P020",
        name_ko="식각",
        name_en="Etch",
        equipments=("ETCH-A", "ETCH-B", "ETCH-C"),
        chambers_per_equip=4,
        params=(
            ParamSpec("rf_power", "W", 1500.0, 12.0, 1440.0, 1560.0, trace_noise=6.0),
            ParamSpec("chamber_pressure", "mTorr", 45.0, 0.80, 42.0, 48.0, trace_noise=0.4),
            ParamSpec("cf4_flow", "sccm", 120.0, 1.5, 114.0, 126.0, trace_noise=0.8),
            ParamSpec("o2_flow", "sccm", 20.0, 0.60, 18.0, 22.0, trace_noise=0.3),
            ParamSpec("electrode_temp", "degC", 60.0, 0.70, 57.0, 63.0, trace_noise=0.3),
            ParamSpec("endpoint_time", "s", 95.0, 2.0, 88.0, 102.0, role="measurement"),
            # 소모품 수명: PM 주기 400시간, 마모될수록 엣지 식각률이 떨어진다(§2.4 Edge-Ring)
            ParamSpec("edge_ring_rf_hours", "h", 400.0, 0.0, 0.0, 400.0, kind="counter"),
        ),
    ),
    ProcessStep(
        step_id="P030",
        name_ko="박막증착",
        name_en="Thin Film / CVD",
        equipments=("CVD-01", "CVD-02"),
        chambers_per_equip=2,
        params=(
            ParamSpec("dep_temp", "degC", 620.0, 2.0, 612.0, 628.0, trace_noise=1.0),
            ParamSpec("precursor_flow", "sccm", 350.0, 5.0, 335.0, 365.0, trace_noise=2.5),
            ParamSpec("chamber_pressure", "Torr", 2.5, 0.05, 2.3, 2.7, trace_noise=0.02),
            ParamSpec("thickness_mean", "A", 450.0, 6.0, 430.0, 470.0, role="measurement"),
            ParamSpec("thickness_sigma", "A", 8.0, 1.2, 0.0, 14.0, role="measurement"),
            # 중간 반경대 온도 편차 → Donut 패턴의 물리적 기전(§2.4)
            ParamSpec("susceptor_temp_mid_delta", "degC", 0.0, 0.50, -2.5, 2.5),
        ),
    ),
    ProcessStep(
        step_id="P040",
        name_ko="이온주입",
        name_en="Implant",
        equipments=("IMP-01", "IMP-02"),
        chambers_per_equip=1,
        params=(
            ParamSpec("dose", "1e13/cm2", 3.0, 0.03, 2.85, 3.15),
            ParamSpec("energy", "keV", 40.0, 0.30, 38.5, 41.5),
            ParamSpec("beam_current", "mA", 5.0, 0.08, 4.7, 5.3, trace_noise=0.04),
            ParamSpec("tilt_angle", "deg", 7.0, 0.10, 6.6, 7.4),
        ),
    ),
    ProcessStep(
        step_id="P045",
        name_ko="캐패시터 형성",
        name_en="Capacitor (Storage Node) Formation",
        equipments=("CAP-01", "CAP-02"),
        chambers_per_equip=2,
        params=(
            ParamSpec("dielectric_thickness", "A", 65.0, 1.2, 60.0, 70.0),
            ParamSpec("node_profile_cd", "nm", 38.0, 0.80, 35.0, 41.0),
            ParamSpec("anneal_temp", "degC", 550.0, 3.0, 540.0, 560.0, trace_noise=1.5),
        ),
        note=(
            "distractor 스텝 — 어떤 불량 패턴과도 인과관계가 없다. "
            "커미널리티 분석이 우연한 상관을 걸러내는지 확인하는 장치(§2.4)."
        ),
    ),
    ProcessStep(
        step_id="P050",
        name_ko="화학적기계연마",
        name_en="CMP",
        equipments=("CMP-01", "CMP-02"),
        chambers_per_equip=4,  # head 1~4
        params=(
            ParamSpec("down_force", "psi", 3.5, 0.09, 3.0, 4.0, trace_noise=0.04),
            ParamSpec("platen_speed", "rpm", 60.0, 1.2, 55.0, 65.0, trace_noise=0.6),
            ParamSpec("slurry_flow", "ml/min", 200.0, 5.0, 180.0, 220.0, trace_noise=2.5),
            ParamSpec("conditioning_time", "s", 30.0, 1.0, 26.0, 34.0),
            ParamSpec("removal_rate", "A/min", 2800.0, 60.0, 2600.0, 3000.0, role="measurement"),
            ParamSpec("center_zone_pressure", "psi", 3.2, 0.10, 2.8, 3.6, trace_noise=0.05),
            ParamSpec("slurry_particle_count", "ea", 4.0, 2.0, 0.0, 18.0, role="measurement"),
            # 소모품 수명: 패드 1200장 주기. 마모될수록 중심부 제거율이 튄다(§2.4 Center)
            ParamSpec("pad_life", "wafers", 1200.0, 0.0, 0.0, 1200.0, kind="counter"),
        ),
    ),
    ProcessStep(
        step_id="P060",
        name_ko="세정",
        name_en="Cleaning",
        equipments=("CLN-01", "CLN-02"),
        chambers_per_equip=1,
        params=(
            ParamSpec("chem_conc", "%", 2.0, 0.05, 1.8, 2.2),
            ParamSpec("bath_temp", "degC", 65.0, 0.60, 62.0, 68.0, trace_noise=0.3),
            ParamSpec("di_resistivity", "Mohm-cm", 18.2, 0.15, 17.5, 18.5),
            ParamSpec("particle_count", "ea", 5.0, 2.5, 0.0, 20.0, role="measurement"),
            ParamSpec("chuck_edge_temp_dev", "degC", 0.0, 0.30, -1.5, 1.5),
        ),
    ),
    ProcessStep(
        step_id="P070",
        name_ko="확산·열처리",
        name_en="Diffusion / Anneal",
        equipments=("DIFF-01",),
        chambers_per_equip=1,
        params=(
            ParamSpec("furnace_temp_z1", "degC", 800.0, 1.5, 793.0, 807.0, trace_noise=0.8),
            ParamSpec("furnace_temp_z2", "degC", 800.0, 1.5, 793.0, 807.0, trace_noise=0.8),
            ParamSpec("furnace_temp_z3", "degC", 800.0, 1.5, 793.0, 807.0, trace_noise=0.8),
            ParamSpec("ramp_rate", "degC/min", 10.0, 0.20, 9.0, 11.0),
            ParamSpec("o2_flow", "slm", 8.0, 0.15, 7.5, 8.5),
        ),
    ),
    ProcessStep(
        step_id="P080",
        name_ko="인라인 계측",
        name_en="Inline Metrology",
        equipments=("MET-01",),
        chambers_per_equip=1,
        params=(
            ParamSpec("cd_mean", "nm", 38.0, 0.60, 35.5, 40.5, role="measurement"),
            ParamSpec("cd_sigma", "nm", 1.2, 0.20, 0.0, 2.2, role="measurement"),
            ParamSpec("thickness", "A", 450.0, 5.0, 430.0, 470.0, role="measurement"),
            ParamSpec("overlay_residual", "nm", 0.0, 1.0, -4.0, 4.0, role="measurement"),
        ),
    ),
)

# ──────────────────────────────────────────────────────────────────────────
# 4-A. EDS 검사 설비 (설계서 §1 · 첨부 도메인 문서 §3)
#
# 왜 검사 설비를 데이터 모델에 넣나 ★★:
#     웨이퍼 맵의 불량이 **공정에서 왔다는 보장이 없다.** 프로브 카드 니들이 마모되면
#     엣지 die에서 접촉이 먼저 끊겨 Edge-Ring과 **똑같이 생긴 맵**이 나온다. 그런데
#     이건 식각 챔버를 아무리 손봐도 없어지지 않는다 — 카드를 세정·교체해야 한다.
#
#     검사 설비 축이 데이터에 없으면 분석은 **구조적으로 항상 공정을 범인으로 지목한다.**
#     후보 목록에 없는 것은 1위가 될 수 없기 때문이다. 이건 모델 성능 문제가 아니라
#     데이터 설계 문제이고, 현업에서는 멀쩡한 챔버를 세우는 오진으로 이어진다.
# ──────────────────────────────────────────────────────────────────────────

#: 검사 스텝 식별자 — 공정 스텝(P0xx)과 구분되도록 T 접두를 쓴다
TEST_STEP_ID = "T010"

#: ATE(테스터). 웨이퍼는 lot 단위로 한 테스터에 배정된다.
TESTERS: tuple[str, ...] = ("ATE-01", "ATE-02", "ATE-03", "ATE-04")

#: 프로브 카드. **소모품이며 테스터 사이를 옮겨 다닌다** — 그래서 검정 단위가 된다.
PROBE_CARDS: tuple[str, ...] = ("PC-01", "PC-02", "PC-03", "PC-04", "PC-05", "PC-06")

#: 동시 측정 site 수. die 수 ÷ 이 값 ≈ 웨이퍼 1장당 터치다운 횟수.
PROBE_PARALLELISM = 16

#: 웨이퍼 1장을 다 찍는 데 필요한 터치다운 횟수.
#: 300mm 웨이퍼에 die 1,584개(합성 기하 기준) ÷ 16 site = 99회.
TOUCHDOWNS_PER_WAFER = 99

#: 프로브 카드 세정 주기(터치다운 횟수). 이 값을 넘기면 니들 오염으로 접촉 저항이 오른다.
PROBE_CARD_PM_TOUCHDOWNS = 250_000

TEST_STEP = ProcessStep(
    step_id=TEST_STEP_ID,
    name_ko="EDS 검사",
    name_en="EDS / Wafer Test",
    equipments=TESTERS,
    chambers_per_equip=1,
    # 검정 단위는 테스터가 아니라 프로브 카드다 (위 설명 참조)
    explicit_units=PROBE_CARDS,
    params=(
        # 누적 터치다운 — 카드 마모의 대리 지표. PM에서 0으로 리셋되는 카운터다.
        ParamSpec("touchdown_count", "회", float(PROBE_CARD_PM_TOUCHDOWNS), 0.0,
                  0.0, float(PROBE_CARD_PM_TOUCHDOWNS), kind="counter"),
        # 오버드라이브 — 니들을 pad에 얼마나 눌러 넣는가. 작으면 엣지에서 접촉이 끊긴다.
        ParamSpec("probe_overdrive", "um", 75.0, 2.5, 60.0, 90.0, trace_noise=1.0),
        # 접촉 저항 — 마모의 **결과**다. 조작 손잡이가 아니라 계측값.
        ParamSpec("contact_resistance", "ohm", 0.85, 0.06, 0.0, 1.50,
                  role="measurement"),
        # 프로브 마크 위치 오차 — 정렬이 틀어지면 특정 die에서만 접촉이 빗나간다.
        ParamSpec("probe_mark_offset", "um", 0.0, 1.2, -6.0, 6.0, trace_noise=0.5),
        # 척 온도 — 측정 조건. 흔들리면 전기 특성 판정이 흔들린다.
        ParamSpec("chuck_temp", "degC", 25.0, 0.35, 23.0, 27.0, trace_noise=0.15),
        # 테스트 시간 — 재측정이 늘면 길어진다. 계측값.
        ParamSpec("test_time", "s", 210.0, 6.0, 180.0, 260.0, role="measurement"),
    ),
    note="맵의 불량이 공정이 아니라 검사에서 왔을 가능성을 판별하기 위한 축",
)

#: 스텝 조회표 — 공정 스텝 + 검사 스텝.
#: 검사 스텝은 `PROCESS_STEPS`에 넣지 **않는다.** 라우팅·사이클타임 계산은 팹 공정만
#: 대상으로 하며, 검사는 그 뒤에 한 번 일어나는 별도 단계이기 때문이다.
STEPS_BY_ID: dict[str, ProcessStep] = {s.step_id: s for s in PROCESS_STEPS}
STEPS_BY_ID[TEST_STEP.step_id] = TEST_STEP

#: 파라미터명 → 역할("control" | "measurement"). 원인 분석에서 조치 가능한 것만
#: 골라내는 데 쓴다(§M4). 같은 이름이 여러 스텝에 있으면 역할은 동일하다고 본다.
PARAM_ROLE: dict[str, str] = {
    param.name: param.role
    for step in (*PROCESS_STEPS, TEST_STEP)
    for param in step.params
}


def is_controllable(param_name: str) -> bool:
    """엔지니어가 직접 조작할 수 있는 파라미터인가.

    `chamber_pressure_mean` 처럼 접미사가 붙은 컬럼명도 받아들인다.
    """
    base = param_name
    for suffix in ("_mean", "_std", "_min", "_max"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return PARAM_ROLE.get(base, "control") == "control"


# ──────────────────────────────────────────────────────────────────────────
# 5. 패턴 → 원인 공정 매핑 (설계서 §2.4) — 시뮬레이터의 인과 규칙
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParamPerturbation:
    """원인 파라미터 1개에 가할 섭동.

    Attributes:
        param: 파라미터명
        shift_sigma: 정상 sigma의 몇 배만큼 밀 것인가 (부호가 방향)
        counter_ratio: ``kind="counter"`` 파라미터일 때, PM 주기 대비 사용률
                       (예: 0.9 = 수명의 90%까지 마모된 상태)
        spike_sigma: **순간 스파이크**의 크기 (정상 sigma의 배수). 지속적인 평균
            이동이 아니라 처리 중 짧은 순간만 튀는 이상이다.
        spike_seconds: 스파이크 지속 시간(초). 웨이퍼 1장 처리 시간 60초 기준.
        drift_sigma: 처리 중 **서서히 밀리는** 양(정상 sigma의 배수). 시작과 끝의 차이다.

    왜 스파이크를 따로 두나 ★: 60초 중 3초만 +6σ 튀면 **평균은 0.3σ밖에 안 움직인다.**
        요약통계(mean/std/min/max)만 보는 분석은 이 이상을 구조적으로 놓친다.
        그런데 순간적인 과열·과압이 품질을 망치는 일은 실제로 흔하다.
    """

    param: str
    shift_sigma: float = 0.0
    counter_ratio: float | None = None
    spike_sigma: float = 0.0
    spike_seconds: float = 0.0
    drift_sigma: float = 0.0


@dataclass(frozen=True)
class CauseRule:
    """불량 패턴 1종의 인과 규칙.

    Attributes:
        pattern: 패턴명
        step_id: 원인 공정 스텝
        mechanism: 물리적 기전 설명 (UI·학습노트에 그대로 노출)
        perturbations: 해당 스텝에서 이탈시킬 파라미터들
    """

    pattern: str
    step_id: str | None
    mechanism: str
    perturbations: tuple[ParamPerturbation, ...] = field(default_factory=tuple)


CAUSE_RULES: dict[str, CauseRule] = {
    "Center": CauseRule(
        pattern="Center",
        step_id="P050",
        mechanism="반경방향 제거율 프로파일이 중심에서 이탈 — 패드 마모 + 과도한 하중으로 중심부 과연마",
        perturbations=(
            ParamPerturbation("down_force", shift_sigma=+4.0),
            ParamPerturbation("center_zone_pressure", shift_sigma=+3.5),
            ParamPerturbation("pad_life", counter_ratio=0.92),
        ),
    ),
    "Donut": CauseRule(
        pattern="Donut",
        step_id="P030",
        mechanism="서셉터 중간 반경대 온도 불균일로 증착 두께가 링 형태로 이탈",
        perturbations=(
            ParamPerturbation("susceptor_temp_mid_delta", shift_sigma=+4.0),
            ParamPerturbation("precursor_flow", shift_sigma=-2.5),
        ),
    ),
    "Edge-Ring": CauseRule(
        pattern="Edge-Ring",
        step_id="P020",
        mechanism="챔버 edge ring 마모로 엣지 영역 플라즈마 시스가 변형 → 최외곽 식각률 저하",
        perturbations=(
            ParamPerturbation("edge_ring_rf_hours", counter_ratio=0.95),
            ParamPerturbation("chamber_pressure", shift_sigma=+3.0),
            ParamPerturbation("o2_flow", shift_sigma=-2.0),
        ),
    ),
    "Edge-Loc": CauseRule(
        pattern="Edge-Loc",
        step_id="P060",
        mechanism="척 엣지 온도 국부 편차 및 파티클 부착 — 웨이퍼 핸들링 접촉부에 집중",
        perturbations=(
            ParamPerturbation("chuck_edge_temp_dev", shift_sigma=+3.5),
            ParamPerturbation("particle_count", shift_sigma=+3.0),
        ),
    ),
    "Loc": CauseRule(
        pattern="Loc",
        step_id="P010",
        mechanism="국부 디포커스 및 파티클 낙하로 특정 영역 패턴 전사 실패",
        perturbations=(
            ParamPerturbation("focus_offset", shift_sigma=+4.0),
            ParamPerturbation("defect_adder_count", shift_sigma=+3.0),
        ),
    ),
    "Scratch": CauseRule(
        pattern="Scratch",
        step_id="P050",
        mechanism="슬러리 공급 부족 + 패드 컨디셔닝 미흡으로 이물이 끼여 선형 스크래치 발생",
        perturbations=(
            ParamPerturbation("slurry_flow", shift_sigma=-3.5),
            ParamPerturbation("conditioning_time", shift_sigma=-3.0),
            ParamPerturbation("slurry_particle_count", shift_sigma=+3.5),
        ),
    ),
    "Near-full": CauseRule(
        pattern="Near-full",
        step_id="P040",
        mechanism="이온주입 도즈 대폭 이탈(레시피 오적용 수준) — 웨이퍼 전면 소자 특성 붕괴",
        perturbations=(
            ParamPerturbation("dose", shift_sigma=+8.0),
            ParamPerturbation("beam_current", shift_sigma=-5.0),
        ),
    ),
    # Random은 특정 설비 원인이 아니라 라인 전반의 파티클 baseline 상승으로 모델링한다.
    # → 커미널리티 분석에서 '유의한 설비 없음'이 나와야 정상이며, 이것이 위양성 검증 소재가 된다.
    "Random": CauseRule(
        pattern="Random",
        step_id=None,
        mechanism="특정 설비에 귀속되지 않는 랜덤 파티클/미세 오염 — 원인 설비 없음이 정답",
    ),
    "none": CauseRule(
        pattern="none",
        step_id=None,
        mechanism="정상 웨이퍼",
    ),
}


# ── 검사 기인 불량 (설계서 §2.4-A) ──────────────────────────────────────
#
# 같은 맵 패턴을 만드는 **두 번째 경로**다. 공정 규칙(CAUSE_RULES)과 나란히 존재하며,
# 분석은 둘 중 어느 쪽인지 가려낼 수 있어야 한다.
#
# 왜 이 구분이 실무에서 중요한가:
#   · 오진 비용이 비대칭이다. 검사 문제를 공정 탓으로 돌리면 **멀쩡한 챔버를 세우고도**
#     불량이 계속된다. 반대로 공정 문제를 카드 탓으로 돌리면 카드만 갈다 시간을 버린다.
#   · 조치 리드타임이 다르다. 프로브 카드 세정은 수 시간, 챔버 PM은 수 일이다.
#   · 재검사로 확인이 가능하다 — 같은 웨이퍼를 다른 카드로 다시 측정했을 때 결과가
#     달라지면 검사 기인이다. 분석은 그 재검사 대상을 좁혀 주는 역할을 한다.

TEST_CAUSE_RULES: dict[str, CauseRule] = {
    "Edge-Ring": CauseRule(
        pattern="Edge-Ring",
        step_id=TEST_STEP_ID,
        mechanism=(
            "프로브 카드 니들 오염·마모로 접촉 저항 상승 → 평탄도가 나쁜 최외곽 die부터 "
            "접촉이 끊겨 링 모양 fail. 식각 기인 Edge-Ring과 맵으로는 구분되지 않는다"
        ),
        perturbations=(
            ParamPerturbation("touchdown_count", counter_ratio=0.93),
            ParamPerturbation("contact_resistance", shift_sigma=+3.5),
            ParamPerturbation("probe_overdrive", shift_sigma=-3.0),
        ),
    ),
    "Loc": CauseRule(
        pattern="Loc",
        step_id=TEST_STEP_ID,
        mechanism=(
            "프로브 카드 정렬 틀어짐으로 특정 site의 니들이 pad를 빗나감 → 같은 카드로 "
            "측정한 웨이퍼마다 **같은 위치**에서 국부 fail. 포토 기인 Loc과 구분해야 한다"
        ),
        perturbations=(
            ParamPerturbation("probe_mark_offset", shift_sigma=+4.0),
            ParamPerturbation("contact_resistance", shift_sigma=+2.0),
            ParamPerturbation("test_time", shift_sigma=+2.5),
        ),
    ),
}

# ── 스파이크 기인 불량 (설계서 §2.4-B · 첨부 도메인 문서 §12) ──────────
#
# 세 번째 경로다. 같은 스텝·같은 패턴이지만 **이상의 모양**이 다르다.
#   · 지속형: 파라미터 평균이 통째로 이탈 → 요약통계로 잡힌다
#   · 스파이크형: 60초 중 2~3초만 튄다 → **평균이 거의 안 움직인다**
#
# 첨부 문서 §12가 든 예가 정확히 이것이다:
#     정상 500℃인데 30~31초 구간만 540℃ → 평균은 정상에 가깝게 보인다
#
# 이 경로를 데이터에 넣어야 "요약통계만으로는 부족하다"를 **주장이 아니라
# 측정으로** 보일 수 있다.

SPIKE_CAUSE_RULES: dict[str, CauseRule] = {
    "Edge-Loc": CauseRule(
        pattern="Edge-Loc",
        step_id="P060",
        mechanism=(
            "세정조 히터 순간 과열(2~3초) — 웨이퍼 가장자리 메니스커스 접촉부에서만 "
            "자연산화막이 과도하게 제거돼 국부 불량. 60초 평균으로는 거의 보이지 않는다"
        ),
        perturbations=(
            ParamPerturbation("bath_temp", spike_sigma=+7.0, spike_seconds=3.0),
        ),
    ),
}

#: 각 패턴에서 **스파이크 기인이 차지하는 비율** (지속형 몫에서 떼어 낸다).
SPIKE_INDUCED_SHARE: dict[str, float] = {
    # 왜 20%인가: 처음에는 30%로 잡았다가 드리프트 25%까지 더하니 Edge-Loc의
    # 지속형 몫이 45%로 줄어, P060 챔버 2대에 3개 경로가 흩어지면서 M3의 원인
    # 챔버 적중률이 100% → 83%로 떨어졌다. **기존 성능을 깨뜨리는 것은 개선이
    # 아니다.** 시연에 필요한 최소 표본(각 30장 이상)은 유지하면서 지속형이
    # 다수로 남도록 낮춰 잡았다.
    "Edge-Loc": 0.20,
}


# ── 드리프트 기인 불량 — 스파이크와 구분되는가를 시험하는 대조군 ★★ ──────
#
# 왜 이 경로를 따로 두나: 스파이크와 드리프트는 **요약통계에서 똑같이 보인다.**
# 둘 다 60초 산포(std)를 키우기 때문이다. 그런데 원인도 조치도 전혀 다르다.
#
#   스파이크 → 순간 과열. 히터 제어기·인터락 문제. 조치: 인터락 설정
#   드리프트 → 서서히 밀림. 센서 교정 이탈·소모품 마모. 조치: 교정/교체
#
# "std가 높다"까지만 알면 어느 쪽인지 모른 채 엉뚱한 조치를 하게 된다.
# 시계열 피처(`_time_above` vs `_slope`)가 정말 값어치가 있는지는
# **이 둘을 갈라내는가**로 판가름난다.

DRIFT_CAUSE_RULES: dict[str, CauseRule] = {
    "Edge-Loc": CauseRule(
        pattern="Edge-Loc",
        step_id="P060",
        mechanism=(
            "세정조 온도 센서 교정 이탈 — 처리 60초 동안 온도가 서서히 밀려 "
            "후반부 웨이퍼 가장자리에서 식각량이 누적 초과. 스파이크와 산포는 비슷하다"
        ),
        perturbations=(
            # 8.6σ는 임의로 고른 값이 아니다. 스파이크(7σ × 3초)와 **60초 산포가
            # 같아지도록** 역산했다. 크기를 다르게 두면 std만으로도 둘이 갈라져
            # "시계열 피처가 필요한가"라는 질문 자체가 성립하지 않는다.
            #   스파이크 분산 ≈ p(1-p)·Δ²  (p = 3/60초 구간 비율)
            #   드리프트 분산 ≈ (구간 내 이동량)² / 12
            ParamPerturbation("bath_temp", drift_sigma=+8.6),
        ),
    ),
}

#: 각 패턴에서 드리프트 기인이 차지하는 비율.
DRIFT_INDUCED_SHARE: dict[str, float] = {
    "Edge-Loc": 0.18,
}

# ── 공정 간 교호작용 (설계서 §2.4-C · 첨부 도메인 문서 §14·15·37) ──────
#
# 네 번째 경로. 앞의 세 경로와 결정적으로 다른 점이 있다.
#
#     **어느 파라미터도 혼자서는 규격을 벗어나지 않는다.**
#
# 증착이 두께 상한 쪽이고 CMP 제거량이 하한 쪽이면, 각각은 규격 안이지만
# 최종 잔막이 이탈한다. 반대 조합(얇게 증착 + 강하게 연마)도 마찬가지다.
# 그래서 단변량 SPC도, 규격 위반 검사도, 파라미터별 평균 비교도 아무것도 못 잡는다.
#
# 첨부 문서 §16이 지적한 바로 그 구조다:
#     "Temperature 자체 → 영향 작음 / Pressure 자체 → 영향 작음
#      Temperature × Pressure → 불량률 급격히 증가"
#
# 방향을 웨이퍼마다 뒤집어(두껍+약함 / 얇음+강함) **주변부 분포의 평균이 정상군과
# 같도록** 만든다. 그래야 "주효과로는 안 보인다"가 진짜가 된다.


@dataclass(frozen=True)
class InteractionRule:
    """두 공정 파라미터의 **조합**이 만드는 불량.

    Attributes:
        pattern: 유발되는 불량 패턴
        step_a, param_a: 첫 번째 파라미터 (앞 공정)
        step_b, param_b: 두 번째 파라미터 (뒤 공정)
        sign_b: 두 파라미터의 상대 방향. -1이면 "A는 올라가고 B는 내려갈 때" 위험하다
        equip_pair: 이 조합이 자주 나오는 (앞 챔버, 뒤 챔버). 설비 조합 분석의 정답
        shift_range: 각 파라미터를 밀 크기(σ). **규격 안에 머물도록** 작게 잡는다
        mechanism: 물리적 기전
    """

    pattern: str
    step_a: str
    param_a: str
    step_b: str
    param_b: str
    sign_b: float
    equip_pair: tuple[str, str]
    mechanism: str
    shift_range: tuple[float, float] = (1.3, 1.9)


INTERACTION_CAUSE_RULES: dict[str, InteractionRule] = {
    "Center": InteractionRule(
        pattern="Center",
        step_a="P030",
        param_a="precursor_flow",
        step_b="P050",
        param_b="down_force",
        sign_b=-1.0,
        equip_pair=("CVD-02/ch1", "CMP-02/ch3"),
        mechanism=(
            "증착이 두께 상한 쪽인데 CMP 제거량이 하한 쪽이면 중심부에 잔막이 남는다. "
            "반대 조합(얇은 증착 + 강한 연마)이면 중심부가 과연마된다. "
            "**각 파라미터는 규격 안에 있어 단독으로는 이상으로 잡히지 않는다**"
        ),
    ),
}

#: 각 패턴에서 교호작용 기인이 차지하는 비율.
INTERACTION_INDUCED_SHARE: dict[str, float] = {
    "Center": 0.30,
}

#: 각 패턴에서 **검사 기인이 차지하는 비율**.
#: 왜 비율로 두나: 패턴별 총 장수는 WM-811K 실측 분포를 따라야 한다(§M1). 검사 기인을
#: 위에 얹으면 그 분포가 깨지므로, 정해진 장수를 두 경로가 **나눠 갖게** 한다.
TEST_INDUCED_SHARE: dict[str, float] = {
    "Edge-Ring": 0.30,
    "Loc": 0.25,
}


# ──────────────────────────────────────────────────────────────────────────
# 6. 시뮬레이션 난이도 설정 — 설계서 §2.2 원칙 3
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SimulationDifficulty:
    """분석이 자명해지지 않도록 넣는 교란 요소.

    왜: 원인 파라미터만 깔끔하게 이탈시키면 SHAP이 100% 맞히고, 그건 분석 역량을 증명하지
        못한다. 현실에서는 ① 정상인데 파라미터가 흔들리고(위양성) ② 불량인데 신호가 없고
        (위음성) ③ 설비마다 baseline이 다르고 ④ 설비가 노후화된다. 이걸 전부 넣는다.

    Attributes:
        false_positive_rate: 정상 웨이퍼인데 원인 파라미터가 이탈한 비율
        unexplained_rate: 불량 웨이퍼인데 FDC에 신호가 없는 비율 (계측 못 한 원인)
        confound_rate: 원인 설비와 다른 설비가 함께 흘러가 교락을 만드는 비율
        equip_bias_sigma: 설비/챔버별 고유 baseline 편차 (sigma 배수)
        drift_sigma_per_pm: PM 주기 동안 누적되는 드리프트 크기 (sigma 배수)
        severity_range: 패턴별 섭동 강도의 웨이퍼 간 산포 (배수 범위)
    """

    false_positive_rate: float = 0.06
    unexplained_rate: float = 0.10
    confound_rate: float = 0.25
    equip_bias_sigma: float = 0.8
    drift_sigma_per_pm: float = 1.2
    severity_range: tuple[float, float] = (0.55, 1.35)


DEFAULT_DIFFICULTY = SimulationDifficulty()

#: 재현성을 위한 기본 난수 시드. 시뮬레이터·분할·모델 학습이 모두 이 값을 파생해 쓴다.
DEFAULT_SEED = 20260817


# ──────────────────────────────────────────────────────────────────────────
# 7. 시각화 팔레트 — 설계서 §4.8 (색약 친화)
# ──────────────────────────────────────────────────────────────────────────

#: red/green 대신 blue/orange를 쓴다. 적록색약에서도 구분되고, 명도 차이도 충분하다.
COLOR_PASS = "#4C78A8"   # blue
COLOR_FAIL = "#F58518"   # orange
COLOR_NO_DIE = "#E8E8E8"  # light gray
