"""데이터 스키마 정의와 검증 — 설계서 §2.6.

무엇을: processed 계층 4개 테이블(wafer_master / fdc_summary / fdc_trace / ground_truth)의
        컬럼·타입·제약을 선언하고, DataFrame이 그 규격을 만족하는지 검사한다.
어떻게: TableSchema에 컬럼 목록을 선언하고 ``validate()`` 가 누락/타입/널/값범위를 확인한다.
왜:    합성 데이터와 실측 데이터가 **같은 스키마**로 나와야 앱의 소스 토글(§2.3)이 성립한다.
        스키마를 코드로 강제하지 않으면 "합성에서는 되는데 실측에서는 컬럼이 없다" 같은
        문제가 파이프라인 한참 뒤에서 터진다. 생성 직후에 즉시 걸러내는 게 훨씬 싸다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wafermap.config import PATTERN_LABELS


@dataclass(frozen=True)
class Column:
    """컬럼 1개의 규격.

    Attributes:
        name: 컬럼명
        dtype: pandas dtype 계열 — "int" | "float" | "str" | "datetime" | "bool" | "list"
        nullable: 결측 허용 여부
        allowed: 허용 값 집합 (범주형 컬럼에만 사용)
        min_value, max_value: 수치 범위 제약
    """

    name: str
    dtype: str
    nullable: bool = False
    allowed: frozenset[str] | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class TableSchema:
    """테이블 1개의 규격."""

    name: str
    columns: tuple[Column, ...]
    #: 이 컬럼 조합이 유일해야 한다 (중복 행 방지)
    unique_key: tuple[str, ...] = ()
    #: 선언되지 않은 컬럼을 허용할지. FDC는 파라미터가 스텝마다 달라 True로 둔다.
    allow_extra: bool = False

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


# ──────────────────────────────────────────────────────────────────────────
# 테이블 정의
# ──────────────────────────────────────────────────────────────────────────

WAFER_MASTER = TableSchema(
    name="wafer_master",
    unique_key=("wafer_id",),
    columns=(
        Column("wafer_id", "str"),
        Column("lot_id", "str"),
        Column("slot_no", "int", min_value=1, max_value=25),
        Column("product", "str"),
        Column("tech_node", "str"),
        Column("fab_in_time", "datetime"),
        Column("eds_time", "datetime"),
        # 검사 설비 — 불량이 공정이 아니라 검사에서 왔을 가능성을 판별하는 축(§2.5-A).
        # 실측 WM-811K에는 없는 정보라 nullable로 둔다.
        Column("tester_id", "str", nullable=True),
        Column("probe_card_id", "str", nullable=True),
        Column("probe_touchdown", "int", nullable=True, min_value=0),
        Column("die_total", "int", min_value=1),
        Column("die_pass", "int", min_value=0),
        Column("yield_pct", "float", min_value=0.0, max_value=100.0),
        Column("pattern_label", "str", allowed=frozenset(PATTERN_LABELS)),
        Column("data_source", "str", allowed=frozenset({"real", "synthetic"})),
        # 미라벨 데이터 확장(백로그 §11)을 위해 1차부터 자리를 잡아 둔다.
        Column("is_labeled", "bool"),
    ),
)

FDC_SUMMARY = TableSchema(
    name="fdc_summary",
    unique_key=("wafer_id", "step_id"),
    # 파라미터 컬럼(<param>_mean 등)은 스텝마다 다르므로 추가 컬럼을 허용한다.
    allow_extra=True,
    columns=(
        Column("wafer_id", "str"),
        Column("step_id", "str"),
        Column("step_name", "str"),
        Column("equip_id", "str"),
        Column("chamber_id", "str"),
        Column("recipe_id", "str"),
        Column("run_time", "datetime"),
    ),
)

FDC_TRACE = TableSchema(
    name="fdc_trace",
    unique_key=("wafer_id", "step_id", "param", "t_sec"),
    columns=(
        Column("wafer_id", "str"),
        Column("step_id", "str"),
        Column("param", "str"),
        Column("t_sec", "float", min_value=0.0),
        Column("value", "float"),
    ),
)

GROUND_TRUTH = TableSchema(
    name="ground_truth",
    unique_key=("wafer_id",),
    columns=(
        Column("wafer_id", "str"),
        Column("pattern_label", "str", allowed=frozenset(PATTERN_LABELS)),
        # 원인이 없는 패턴(none/Random)은 결측이 정상이다.
        Column("true_root_step", "str", nullable=True),
        Column("true_root_equip", "str", nullable=True),
        Column("true_root_params", "list", nullable=True),
        Column("severity", "float", nullable=True, min_value=0.0),
        Column("is_confounded", "bool"),
        # 공정이 아니라 검사 설비가 원인인가 — 같은 맵 패턴의 두 번째 경로
        Column("is_test_induced", "bool"),
        Column("is_unexplained", "bool"),
        Column("is_false_positive", "bool"),
    ),
)

ALL_SCHEMAS: dict[str, TableSchema] = {
    s.name: s for s in (WAFER_MASTER, FDC_SUMMARY, FDC_TRACE, GROUND_TRUTH)
}


# ──────────────────────────────────────────────────────────────────────────
# 검증
# ──────────────────────────────────────────────────────────────────────────


class SchemaError(ValueError):
    """스키마 위반. 어떤 테이블의 어떤 컬럼이 왜 틀렸는지를 메시지에 담는다."""


def _dtype_ok(series: pd.Series, dtype: str) -> bool:
    """pandas dtype이 선언한 계열에 부합하는지 확인한다.

    왜 느슨하게 보는가: int64/int32, str/object 처럼 같은 의미인데 표기가 갈리는 경우가 많다.
    엄격히 비교하면 parquet 왕복(round-trip)만으로 실패해서, 계열 단위로만 확인한다.
    """
    kind = series.dtype.kind  # 'i','u','f','O','b','M' 등
    if dtype == "int":
        return kind in "iu"
    if dtype == "float":
        return kind in "fiu"  # 정수도 float 컬럼으로 허용
    if dtype == "str":
        return kind in "OU"
    if dtype == "bool":
        return kind == "b"
    if dtype == "datetime":
        return kind == "M"
    if dtype == "list":
        return kind == "O"
    raise ValueError(f"알 수 없는 dtype 선언: {dtype!r}")


def validate(df: pd.DataFrame, schema: TableSchema, *, strict: bool = True) -> list[str]:
    """DataFrame이 스키마를 만족하는지 검사한다.

    Args:
        df: 검사할 데이터프레임
        schema: 기대 스키마
        strict: True면 위반 시 SchemaError를 던지고, False면 문제 목록만 돌려준다.

    Returns:
        위반 사항 문자열 목록 (문제가 없으면 빈 리스트)

    Raises:
        SchemaError: ``strict=True`` 이고 위반이 있을 때
    """
    problems: list[str] = []

    # 1) 컬럼 존재 여부
    missing = [c.name for c in schema.columns if c.name not in df.columns]
    if missing:
        problems.append(f"누락 컬럼: {missing}")

    if not schema.allow_extra:
        extra = [c for c in df.columns if c not in schema.column_names]
        if extra:
            problems.append(f"정의되지 않은 컬럼: {extra}")

    # 2) 컬럼별 타입·널·범위·허용값
    for col in schema.columns:
        if col.name not in df.columns:
            continue
        s = df[col.name]

        if not _dtype_ok(s, col.dtype):
            problems.append(f"{col.name}: dtype '{s.dtype}'가 '{col.dtype}' 계열이 아님")

        n_null = int(s.isna().sum())
        if not col.nullable and n_null:
            problems.append(f"{col.name}: 결측 {n_null}건 (허용 안 함)")

        non_null = s.dropna()
        if col.allowed is not None and len(non_null):
            bad = set(non_null.unique()) - set(col.allowed)
            if bad:
                problems.append(f"{col.name}: 허용되지 않은 값 {sorted(bad)[:5]}")

        if col.dtype in ("int", "float") and len(non_null):
            if col.min_value is not None and float(non_null.min()) < col.min_value:
                problems.append(f"{col.name}: 최솟값 {non_null.min()} < {col.min_value}")
            if col.max_value is not None and float(non_null.max()) > col.max_value:
                problems.append(f"{col.name}: 최댓값 {non_null.max()} > {col.max_value}")

    # 3) 유일키 중복
    if schema.unique_key and all(k in df.columns for k in schema.unique_key):
        n_dup = int(df.duplicated(subset=list(schema.unique_key)).sum())
        if n_dup:
            problems.append(f"유일키 {schema.unique_key} 중복 {n_dup}건")

    if problems and strict:
        raise SchemaError(f"[{schema.name}] 스키마 위반:\n  - " + "\n  - ".join(problems))
    return problems


def validate_consistency(
    wafer_master: pd.DataFrame,
    fdc_summary: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    strict: bool = True,
) -> list[str]:
    """테이블 간 참조 무결성을 검사한다.

    무엇을: wafer_id 집합이 세 테이블에서 일치하는지, die_pass ≤ die_total 인지 확인한다.
    왜:    조인 키가 어긋나면 커미널리티 분석(M3)이 조용히 표본을 잃는다. 검정 결과가
           틀리는 게 아니라 **표본이 줄어든 채로 그럴듯한 p-value가 나오는** 게 더 위험하다.
    """
    problems: list[str] = []

    wm_ids = set(wafer_master["wafer_id"])
    fdc_ids = set(fdc_summary["wafer_id"])
    gt_ids = set(ground_truth["wafer_id"])

    if wm_ids != gt_ids:
        problems.append(
            f"wafer_master↔ground_truth wafer_id 불일치: "
            f"master만 {len(wm_ids - gt_ids)}건, gt만 {len(gt_ids - wm_ids)}건"
        )
    if not fdc_ids <= wm_ids:
        problems.append(f"fdc_summary에 master에 없는 wafer_id {len(fdc_ids - wm_ids)}건")
    if wm_ids - fdc_ids:
        problems.append(f"FDC 데이터가 없는 wafer {len(wm_ids - fdc_ids)}건")

    bad_yield = wafer_master[wafer_master["die_pass"] > wafer_master["die_total"]]
    if len(bad_yield):
        problems.append(f"die_pass > die_total 인 웨이퍼 {len(bad_yield)}건")

    if problems and strict:
        raise SchemaError("테이블 간 정합성 위반:\n  - " + "\n  - ".join(problems))
    return problems
