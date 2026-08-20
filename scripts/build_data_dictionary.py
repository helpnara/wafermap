#!/usr/bin/env python3
"""`docs/02_data_dictionary.md`를 코드에서 생성한다.

왜 손으로 안 쓰나 ★: 데이터 사전은 컬럼이 하나 늘 때마다 낡는다. 손으로 쓴 문서는
    코드와 조용히 어긋나고, 어긋난 것을 아무도 모른다. 이 프로젝트는 화면의 데이터
    사전도 `schema.py`를 직접 읽어 그리는데, 문서만 손으로 쓰면 같은 문제가 문서에
    남는다. 그래서 문서도 생성물로 둔다.

    바꿔야 할 것은 이 스크립트가 아니라 `schema.py`의 `Column.note`와
    `config.py`의 `ParamSpec`이다.

사용법:
    python scripts/build_data_dictionary.py
    python scripts/build_data_dictionary.py --check   # 최신인지만 확인 (CI용)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap import config as C  # noqa: E402
from wafermap.data import schema  # noqa: E402

OUT_PATH = C.DOCS_DIR / "02_data_dictionary.md"

DTYPE_LABEL = {
    "str": "문자", "int": "정수", "float": "실수",
    "datetime": "시각", "bool": "참/거짓", "list": "목록",
}

ROLE_LABEL = {
    "control": "제어 — 사람이 설정값을 바꿀 수 있다",
    "measurement": "계측 — 결과를 재는 값이라 직접 조작할 수 없다",
}

KIND_LABEL = {
    "gaussian": "정규",
    "counter": "카운터(단조 증가)",
}

#: 각 표가 왜 존재하는지. 화면(`views/data_overview.py`)과 같은 문구를 쓴다.
TABLE_PURPOSE = {
    "wafer_master": (
        "웨이퍼 1장 = 1행. 모든 분석의 **표본 단위**이자 다른 표를 잇는 축이다. "
        "수율·패턴 라벨·검사 설비가 여기 모인다."
    ),
    "fdc_summary": (
        "웨이퍼 × 스텝 = 1행. 센서 파형을 평균·표준편차·최소·최대로 압축한 표다. "
        "실제 팹의 FDC 시스템이 내보내는 형태가 이것이라 분석의 기본 입력이 된다."
    ),
    "fdc_trace": (
        "웨이퍼 × 스텝 × 파라미터 × 시각 = 1행. 압축하기 전의 **원파형**이다. "
        "3초짜리 스파이크는 60초 평균을 0.35σ밖에 못 움직여 요약통계에서 묻힌다 — "
        "그 경우를 잡으려면 파형이 남아 있어야 한다."
    ),
    "ground_truth": (
        "웨이퍼 1장 = 1행. 시뮬레이터가 **어디에 무엇을 심었는지**의 정답지다. "
        "분석 모델에는 절대 넣지 않고 채점에만 쓴다. **실데이터에는 이 표가 없다.**"
    ),
}


def _constraint(col: schema.Column) -> str:
    parts: list[str] = []
    if col.allowed is not None:
        vals = sorted(col.allowed)
        shown = ", ".join(f"`{v}`" for v in vals[:4])
        parts.append(f"{shown}{' …' if len(vals) > 4 else ''} ({len(vals)}종)")
    lo, hi = col.min_value, col.max_value
    if lo is not None and hi is not None:
        parts.append(f"{lo:g} ~ {hi:g}")
    elif lo is not None:
        parts.append(f"{lo:g} 이상")
    elif hi is not None:
        parts.append(f"{hi:g} 이하")
    return " · ".join(parts) or "—"


def _units_of(step: C.ProcessStep) -> tuple[str, ...]:
    """이 스텝의 '설비/챔버' 단위 목록. 검사 스텝은 프로브 카드가 단위다."""
    if step.explicit_units:
        return step.explicit_units
    return tuple(
        f"{equip}/ch{i + 1}"
        for equip in step.equipments
        for i in range(step.chambers_per_equip)
    )


def _table_section(name: str) -> list[str]:
    table = schema.ALL_SCHEMAS[name]
    lines = [f"### `{name}`", "", TABLE_PURPOSE[name], ""]
    key = " + ".join(f"`{k}`" for k in table.unique_key) or "—"
    lines.append(f"- **고유키**: {key}")
    if table.allow_extra:
        lines.append(
            "- **추가 컬럼 허용**: 파라미터 컬럼이 스텝마다 다르므로 스키마에 고정할 수 없다"
        )
    lines += ["", "| 컬럼 | 형 | 결측 | 허용 범위 | 무엇이고 왜 필요한가 |",
              "|---|---|---|---|---|"]
    for col in table.columns:
        lines.append(
            f"| `{col.name}` | {DTYPE_LABEL.get(col.dtype, col.dtype)} "
            f"| {'허용' if col.nullable else '불가'} | {_constraint(col)} | {col.note} |"
        )
    lines.append("")
    return lines


def _step_section(step: C.ProcessStep) -> list[str]:
    units = _units_of(step)
    lines = [f"### {step.step_id} {step.name_ko} — {step.name_en}", ""]
    if step.note:
        lines += [f"> {step.note}", ""]
    lines.append(
        f"- **설비** {len(step.equipments)}대: "
        + ", ".join(f"`{e}`" for e in step.equipments)
    )
    if step.explicit_units:
        lines.append(f"- **분석 단위** {len(units)}개 (설비가 아니라 프로브 카드): "
                     + ", ".join(f"`{u}`" for u in units))
    else:
        lines.append(
            f"- **분석 단위** {len(units)}개 = 설비 {len(step.equipments)}대 "
            f"× 챔버 {step.chambers_per_equip}개"
        )
    lines += ["", "| 파라미터 | 단위 | 중심값 | σ | 규격 | 종류 | 역할 |",
              "|---|---|---:|---:|---|---|---|"]
    for p in step.params:
        spec = f"{p.spec_lo:g} ~ {p.spec_hi:g}"
        role = "제어" if p.role == "control" else "계측"
        lines.append(
            f"| `{p.name}` | {p.unit} | {p.nominal:g} | {p.sigma:g} | {spec} "
            f"| {KIND_LABEL.get(p.kind, p.kind)} | {role} |"
        )
    lines.append("")
    return lines


def _synthetic_counts() -> dict[str, int]:
    """생성된 데이터셋이 있으면 실제 라벨 분포를 읽는다.

    없으면 빈 값을 돌려준다 — 문서 생성이 데이터셋 유무에 묶이면 안 된다.
    """
    try:
        from wafermap.data import loader
        if not loader.is_built("synthetic"):
            return {}
        return loader.load_wafer_master("synthetic")["pattern_label"].value_counts().to_dict()
    except Exception:
        return {}


def render() -> str:
    n_control = sum(1 for r in C.PARAM_ROLE.values() if r == "control")
    n_meas = sum(1 for r in C.PARAM_ROLE.values() if r == "measurement")

    lines: list[str] = [
        "# 02. 데이터 사전",
        "",
        "> ⚠️ **이 문서는 생성물이다.** 손으로 고치지 말 것 — 다음 생성 때 덮어쓰인다.",
        "> ",
        "> ```bash",
        "> python scripts/build_data_dictionary.py",
        "> ```",
        "> ",
        "> 내용을 바꾸려면 `src/wafermap/data/schema.py`의 `Column.note` 또는",
        "> `src/wafermap/config.py`의 `ParamSpec`을 고친다. 문서를 손으로 쓰면 코드와",
        "> 조용히 어긋나고, 어긋난 것을 아무도 모른다.",
        "",
        "---",
        "",
        "## 1. 표 4개",
        "",
        "| 표 | 1행이 무엇인가 | 행 수(합성 기준) |",
        "|---|---|---|",
        "| `wafer_master` | 웨이퍼 1장 | 6,000 |",
        "| `fdc_summary` | 웨이퍼 × 스텝 | 6,000 × 10 |",
        "| `fdc_trace` | 웨이퍼 × 스텝 × 파라미터 × 시각 | 표본 추출 |",
        "| `ground_truth` | 웨이퍼 1장 (정답지) | 6,000 |",
        "",
        "이 밖에 `die_map.npz`(웨이퍼별 2차원 배열)와 `excursions.parquet`(이상 구간 기록)이 있다.",
        "",
    ]
    for name in schema.ALL_SCHEMAS:
        lines += _table_section(name)

    lines += [
        "---",
        "",
        "## 2. 공정 스텝과 FDC 파라미터",
        "",
        f"공정 {len(C.PROCESS_STEPS)}개 + 검사 1개. "
        f"파라미터는 제어 {n_control}종 · 계측 {n_meas}종이다.",
        "",
        "**제어와 계측을 왜 나누나 ★**: 개선안을 낼 때 결정적이다. "
        "`removal_rate`(제거율)가 원인으로 지목돼도 그건 **결과를 잰 값**이라 "
        "\"제거율을 낮춰라\"는 조치가 성립하지 않는다. 실제로 돌릴 수 있는 손잡이는 "
        "`down_force`·`platen_speed` 같은 제어 파라미터뿐이다. "
        "`recommend.py`가 계측 파라미터를 조치 대상에서 빼는 근거가 이 구분이다.",
        "",
    ]
    for step in C.PROCESS_STEPS:
        lines += _step_section(step)

    lines += ["---", "", "## 3. 검사 스텝", "",
              "공정이 아니라 **검사**가 원인일 가능성을 가르기 위한 축이다. "
              "프로브 카드 니들이 마모되면 Edge-Ring과 똑같이 생긴 맵이 나오는데, "
              "조치는 전혀 다르다 — 카드 세정은 몇 시간, 챔버 PM은 며칠이다.", ""]
    lines += _step_section(C.TEST_STEP)

    lines += [
        f"- 테스터 {len(C.TESTERS)}대: " + ", ".join(f"`{t}`" for t in C.TESTERS),
        f"- 프로브 카드 {len(C.PROBE_CARDS)}장: " + ", ".join(f"`{p}`" for p in C.PROBE_CARDS),
        f"- 동시 측정 {C.PROBE_PARALLELISM} site · 웨이퍼당 터치다운 {C.TOUCHDOWNS_PER_WAFER}회",
        f"- 프로브 카드 PM 주기 {C.PROBE_CARD_PM_TOUCHDOWNS:,} 터치다운",
        "",
        "---",
        "",
        "## 4. 값 규약",
        "",
        "### die_map 배열",
        "",
        "| 값 | 뜻 |",
        "|---:|---|",
        f"| {C.DIE_NONE} | die 없음 (웨이퍼 밖 또는 가장자리 배제영역) |",
        f"| {C.DIE_PASS} | 합격 |",
        f"| {C.DIE_FAIL} | 불합격 |",
        "",
        "WM-811K 원본과 같은 규약이라 실측/합성이 한 코드 경로를 탄다.",
        "",
        "### 불량 패턴 라벨",
        "",
        "| 라벨 | 합성(6,000장) | WM-811K 실측(라벨 있는 172,950장) |",
        "|---|---:|---:|",
    ]
    syn = _synthetic_counts()
    for label in C.PATTERN_LABELS:
        real = C.WM811K_LABEL_COUNTS.get(label, 0)
        n = f"{syn[label]:,}" if label in syn else "—"
        lines.append(f"| {label} | {n} | {real:,} |")
    lines += [
        "",
        "실측 분포가 극단적으로 치우쳐 있다 — `none`이 85%이고 `Near-full`은 149장뿐이다. "
        "정확도(accuracy)가 아니라 **macro-F1**을 쓰는 이유가 이것이다. "
        "\"전부 정상\"이라고만 답해도 정확도 85%가 나온다.",
        "",
        "### Bin 코드",
        "",
        "| 코드 | 뜻 |",
        "|---:|---|",
    ]
    for code, name in C.BIN_CODES.items():
        lines.append(f"| {code} | {name} |")
    lines += [
        "",
        "> 현재 데이터는 die별 합격/불합격(0/1)까지만 만든다. Bin 코드는 정의만 해 두고 "
        "아직 부여하지 않는다 — 백로그 M5.5-④. 실제 EDS는 die마다 어떤 항목에서 "
        "떨어졌는지를 Bin으로 남기고, 같은 불량률이라도 Bin 구성이 다르면 원인이 다르다.",
        "",
        "---",
        "",
        "## 5. 단위를 헷갈리지 말 것 ★",
        "",
        "이 프로젝트에서 실제로 틀렸던 것들이다.",
        "",
        "| 헷갈리는 짝 | 무엇이 달랐나 |",
        "|---|---|",
        "| 웨이퍼 수율 ↔ 라인 수율 | 한 챔버의 수율 개선분 +1.13%p가 라인 전체로는 +0.26%p였다. "
        "그 챔버가 물량의 23%만 처리하기 때문이다 |",
        f"| 웨이퍼 ↔ lot | 같은 lot의 {C.WAFERS_PER_LOT}장은 함께 같은 설비를 지나므로 "
        "**독립 관측이 아니다.** 웨이퍼 단위로 검정하면 표본이 25배 부풀려져 오즈비가 "
        "비현실적으로 커진다 |",
        "| 공정 시각 ↔ EDS 시각 | 사이클 타임만큼 어긋난다. EDS 시각으로 관리도를 그리고 "
        "그 날짜의 공정 이력을 찾으면 엉뚱한 날을 본다 |",
        "| die ↔ 웨이퍼 | 불량률 1.5%가 die 기준인지 웨이퍼 기준인지에 따라 뜻이 완전히 다르다 |",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="데이터 사전 문서 생성")
    parser.add_argument("--check", action="store_true",
                        help="파일이 최신인지만 확인하고 쓰지 않는다 (CI용)")
    args = parser.parse_args()

    content = render()

    if args.check:
        if not OUT_PATH.exists():
            print(f"❌ {OUT_PATH} 가 없습니다.", file=sys.stderr)
            return 1
        if OUT_PATH.read_text(encoding="utf-8") != content:
            print(
                f"❌ {OUT_PATH} 가 코드와 어긋났습니다.\n"
                f"   python scripts/build_data_dictionary.py 를 실행해 갱신하세요.",
                file=sys.stderr,
            )
            return 1
        print(f"✅ {OUT_PATH} 최신")
        return 0

    OUT_PATH.write_text(content, encoding="utf-8")
    n_cols = sum(len(t.columns) for t in schema.ALL_SCHEMAS.values())
    n_params = sum(len(s.params) for s in (*C.PROCESS_STEPS, C.TEST_STEP))
    print(f"💾 {OUT_PATH}")
    print(f"   표 {len(schema.ALL_SCHEMAS)}개 · 컬럼 {n_cols}개 · 파라미터 {n_params}종")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
