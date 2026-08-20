"""화면 1 — 데이터 개요 (설계서 §4.1).

이 화면이 답하는 것:
    1. 데이터가 **어떤 단위로 쌓여 있나** — lot → wafer → die 계층
    2. 각 표에 **무슨 컬럼이 있고 왜 필요한가** — 데이터 사전
    3. 그 데이터를 **믿어도 되나** — 결측·범위 이탈·규격 위반 요약

왜 이 화면을 맨 앞에 두나: 분석 결과를 먼저 본 사람은 "이 숫자가 어느 단위로 센
    것인가"를 되묻게 된다. 반도체 데이터는 단위 하나로 결론이 바뀐다 — 챔버 수율과
    라인 수율이 4배 넘게 차이 났던 것이 이 프로젝트에서 실제로 있었던 일이다.
    표본이 웨이퍼인지 다이인지 lot인지를 먼저 못 박고 시작해야 한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wafermap.config import PROCESS_STEPS, TEST_STEP
from wafermap.data import loader, schema
from wafermap.data.synth_wafer import (
    DIE_HEIGHT_MM,
    DIE_WIDTH_MM,
    EDGE_EXCLUSION_MM,
    WAFER_SIZE_MM,
)
from wafermap.ui import layout, theme

SOURCE = "synthetic"

#: 데이터 사전에 보여 줄 표와, 그 표가 존재하는 이유.
TABLE_NOTES: dict[str, str] = {
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
        "분석 모델에는 절대 넣지 않고 채점에만 쓴다. 실데이터에는 이 표가 없다."
    ),
}

#: 데이터 사전에서 dtype을 사람 말로 바꾼다.
DTYPE_LABEL = {
    "str": "문자", "int": "정수", "float": "실수",
    "datetime": "시각", "bool": "참/거짓", "list": "목록",
}


@st.cache_data(show_spinner="웨이퍼 이력을 불러오는 중…")
def _master(source: str) -> pd.DataFrame:
    return loader.load_wafer_master(source)


@st.cache_data(show_spinner="정답지를 불러오는 중…")
def _truth(source: str) -> pd.DataFrame:
    return loader.load_ground_truth(source)


@st.cache_data(show_spinner="FDC 요약을 불러오는 중…")
def _fdc(source: str) -> pd.DataFrame:
    return loader.load_fdc_summary(source)


def _why_this_matters() -> None:
    st.markdown(
        '<div class="wm-why"><b>단위를 먼저 못 박는다</b><br>'
        '"불량률 1.5%"는 그 자체로 아무 뜻이 없다. 다이 기준인지 웨이퍼 기준인지, '
        '한 챔버 안인지 라인 전체인지에 따라 같은 사건이 <b>4배 넘게</b> 달라 보인다. '
        '이 화면은 그래서 숫자보다 <b>세는 단위</b>를 먼저 보여 준다.</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1) 계층
# ──────────────────────────────────────────────────────────────────────────

def _hierarchy(master: pd.DataFrame) -> None:
    st.subheader("데이터가 쌓이는 단위")

    n_lots = master["lot_id"].nunique()
    n_wafers = len(master)
    die_per = int(master["die_total"].iloc[0])
    n_die = int(master["die_total"].sum())
    n_steps = len(PROCESS_STEPS)

    layout.metric_grid([
        ("lot (묶음)", f"{n_lots:,}", "카세트 1개 = 웨이퍼 25장"),
        ("wafer (웨이퍼)", f"{n_wafers:,}", "분석의 표본 단위"),
        ("die (칩)", f"{n_die:,}", f"웨이퍼당 {die_per:,}개"),
        ("공정 스텝", f"{n_steps} + 1", f"{n_steps}개 공정 + {TEST_STEP.step_id} 검사"),
    ])

    # 계층을 글로 쓰면 "그래서 몇 대 몇인가"가 안 보인다. 배수를 함께 적는다.
    st.markdown(
        f'<div class="wm-card" style="margin-top:.6rem">'
        f'<div class="wm-row" style="font-size:.95rem">'
        f'<b>lot {n_lots:,}개</b> '
        f'<span style="color:{theme.MUTED}">— 웨이퍼 25장을 한 카세트에 넣어 같이 흘린다. '
        f'같은 설비를 지났을 확률이 높아 커미널리티의 기본 층이 된다</span></div>'
        f'<div class="wm-row" style="margin-left:1.2rem">'
        f'└ <b>wafer {n_wafers:,}장</b> '
        f'<span style="color:{theme.MUTED}">— 수율·패턴 라벨이 붙는 단위. '
        f'이 프로젝트의 모든 성능 수치는 <b>웨이퍼 기준</b>이다</span></div>'
        f'<div class="wm-row" style="margin-left:2.4rem">'
        f'└ <b>die {n_die:,}개</b> '
        f'<span style="color:{theme.MUTED}">— 합격/불합격이 실제로 찍히는 최소 단위. '
        f'맵 한 장이 이 {die_per:,}개의 격자다</span></div>'
        f'<div class="wm-row" style="margin-left:2.4rem;color:{theme.MUTED}">'
        f'　 웨이퍼 {WAFER_SIZE_MM:.0f}mm · 다이 {DIE_WIDTH_MM}×{DIE_HEIGHT_MM}mm · '
        f'가장자리 {EDGE_EXCLUSION_MM}mm 제외</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "가장자리를 왜 빼나: 웨이퍼 테두리는 코팅·식각이 균일하지 않아 어차피 못 쓴다. "
        "실제 팹도 이 영역은 처음부터 제품에서 제외하므로(edge exclusion) "
        "수율 분모에 넣지 않는다."
    )


# ──────────────────────────────────────────────────────────────────────────
# 2) 데이터 사전
# ──────────────────────────────────────────────────────────────────────────

def _constraint(col: schema.Column) -> str:
    """규격 제약을 한 칸에 적는다."""
    parts: list[str] = []
    if col.allowed is not None:
        vals = sorted(col.allowed)
        shown = ", ".join(vals[:4])
        parts.append(f"{shown}{' …' if len(vals) > 4 else ''} ({len(vals)}종)")
    if col.min_value is not None or col.max_value is not None:
        lo = "" if col.min_value is None else f"{col.min_value:g}"
        hi = "" if col.max_value is None else f"{col.max_value:g}"
        parts.append(f"{lo} ~ {hi}")
    return " · ".join(parts) or "—"


def _dict_table(table: schema.TableSchema) -> str:
    """스키마 1개를 HTML 표로 그린다.

    왜 `st.dataframe`이 아닌가: 이 표는 설명문 자체가 내용인데, 데이터프레임 위젯은
        셀을 줄바꿈하지 않고 잘라 버린다. 가장 중요한 열이 "…"로 끝나면 표를 둔
        이유가 사라진다.
    """
    rows = "".join(
        f'<tr><td class="n">{c.name}</td>'
        f'<td class="s">{DTYPE_LABEL.get(c.dtype, c.dtype)}</td>'
        f'<td class="s">{"허용" if c.nullable else "불가"}</td>'
        f'<td class="s">{_constraint(c)}</td>'
        f'<td>{theme.html(c.note)}</td></tr>'
        for c in table.columns
    )
    return (
        '<table class="wm-dict"><thead><tr>'
        '<th>컬럼</th><th>형</th><th>결측</th><th>허용 범위</th>'
        '<th>무엇이고 왜 필요한가</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _dictionary(master: pd.DataFrame, fdc: pd.DataFrame, truth: pd.DataFrame) -> None:
    st.subheader("데이터 사전")
    st.caption(
        "이 표는 `src/wafermap/data/schema.py`를 그대로 읽어 그린다. "
        "설명을 화면에 따로 적어 두면 스키마가 바뀔 때 둘이 어긋나기 때문이다."
    )

    frames = {"wafer_master": master, "fdc_summary": fdc, "ground_truth": truth}
    names = list(schema.ALL_SCHEMAS)
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        table = schema.ALL_SCHEMAS[name]
        df = frames.get(name)
        with tab:
            st.markdown(
                f'<div class="wm-note">{theme.html(TABLE_NOTES[name])}</div>',
                unsafe_allow_html=True,
            )
            key = " + ".join(table.unique_key) or "—"
            extra = " · 파라미터 컬럼 추가 허용" if table.allow_extra else ""
            rows_txt = f"{len(df):,}행" if df is not None else "화면에서는 미조회(대용량)"
            st.caption(f"고유키 `{key}` · {rows_txt}{extra}")

            st.markdown(_dict_table(table), unsafe_allow_html=True)

            if name == "fdc_summary":
                # 선언된 7개 외에 파라미터 컬럼이 스텝마다 다르게 붙는다. 그 규칙을 밝힌다.
                n_declared = len(table.columns)
                st.caption(
                    f"실제 컬럼은 {len(fdc.columns):,}개다. 선언된 {n_declared}개 외에는 "
                    f"모두 `<파라미터>_mean/_std/_min/_max` 형태로, 스텝마다 다른 센서가 "
                    f"붙기 때문에 스키마에 고정할 수 없다."
                )


# ──────────────────────────────────────────────────────────────────────────
# 3) 품질 점검
# ──────────────────────────────────────────────────────────────────────────

def _quality(master: pd.DataFrame, truth: pd.DataFrame) -> None:
    st.subheader("이 데이터를 믿어도 되나")

    issues = schema.validate(master, schema.WAFER_MASTER, strict=False)
    issues += schema.validate(truth, schema.GROUND_TRUTH, strict=False)

    if issues:
        st.error("스키마 위반 " + str(len(issues)) + "건")
        for msg in issues[:10]:
            st.markdown(f"- {msg}")
    else:
        st.success(
            "스키마 검증 통과 — 형·결측·허용값·범위 제약을 wafer_master와 "
            "ground_truth 전 행이 만족한다."
        )

    left, right = st.columns(2) if not layout.is_mobile() else (st.container(), st.container())

    with left:
        st.markdown("**결측 현황**")
        # 결측은 '있다/없다'보다 '허용된 결측인가'가 중요하다. 두 개를 나란히 둔다.
        rows = []
        for table, df in (("wafer_master", master), ("ground_truth", truth)):
            spec = {c.name: c for c in schema.ALL_SCHEMAS[table].columns}
            for col in df.columns:
                n_null = int(df[col].isna().sum())
                if n_null == 0:
                    continue
                rows.append({
                    "표": table,
                    "컬럼": col,
                    "결측": n_null,
                    # `%.1f%%` 서식은 값을 100배 하지 않는다. 비율 그대로 넘기면 0.9%로 찍힌다.
                    "비율": 100 * n_null / len(df),
                    # 좁은 화면에서 열이 잘리므로 판정은 짧게 쓴다.
                    "판정": ("정상" if spec.get(col) and spec[col].nullable
                             else "⚠️ 위반"),
                })
        if rows:
            st.dataframe(
                pd.DataFrame(rows).sort_values("결측", ascending=False),
                hide_index=True, width="stretch",
                column_config={"비율": st.column_config.NumberColumn(format="%.1f%%")},
            )
            st.caption(
                "`true_root_*`의 결측은 정상이다 — 원인을 심지 않은 정상 웨이퍼에는 "
                "적을 원인이 없다. 결측 자체가 '이 웨이퍼는 정상'이라는 정보다."
            )
        else:
            st.info("결측 없음")

    with right:
        st.markdown("**수율 분포**")
        _yield_figure(master)

    st.divider()
    _label_balance(master, truth)


def _yield_figure(master: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    normal = master.loc[master["pattern_label"] == "none", "yield_pct"]
    defect = master.loc[master["pattern_label"] != "none", "yield_pct"]

    fig = go.Figure()
    fig.add_histogram(x=normal, name="정상(none)", nbinsx=60,
                      marker_color=theme.GOOD, opacity=0.75)
    fig.add_histogram(x=defect, name="패턴 있음", nbinsx=60,
                      marker_color=theme.WARN, opacity=0.75)
    fig.update_layout(
        barmode="overlay", height=260, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="수율 (%)", yaxis_title="웨이퍼 수",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    # 모바일에서 모드바가 범례를 덮는다. 읽기만 하는 그림이라 도구는 끈다.
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    gap = normal.mean() - defect.mean()
    st.caption(
        f"정상 평균 {normal.mean():.2f}% · 패턴 있음 평균 {defect.mean():.2f}% "
        f"(차이 {gap:.2f}%p). 두 분포가 **겹친다**는 점이 중요하다 — 수율 숫자만으로는 "
        f"패턴 유무를 가를 수 없어서 맵의 *모양*을 봐야 하는 것이다."
    )


def _label_balance(master: pd.DataFrame, truth: pd.DataFrame) -> None:
    st.markdown("**라벨 균형과 원인 경로**")

    counts = master["pattern_label"].value_counts()
    mech = truth.groupby("pattern_label")["cause_mechanism"].value_counts().unstack(fill_value=0)

    rows = []
    for label, n in counts.items():
        row = {"패턴": label, "웨이퍼": int(n), "비율": 100 * n / len(master)}
        if label in mech.index:
            paths = [f"{k} {v}" for k, v in mech.loc[label].items() if v > 0]
            row["원인 경로"] = " · ".join(paths)
        else:
            row["원인 경로"] = "—"
        rows.append(row)

    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        height=(len(rows) + 1) * 35 + 3,
        column_config={"비율": st.column_config.NumberColumn(format="%.2f%%")},
    )

    rare = counts[counts < 30]
    warn = (
        f"표본이 30장 미만인 패턴: {', '.join(rare.index)} — 이 패턴들의 성능 수치는 "
        f"신뢰구간이 매우 넓다. 우연히 몇 장 맞고 틀리는 것으로 순위가 뒤집힌다."
        if len(rare) else "모든 패턴이 30장 이상이다."
    )
    st.markdown(
        f'<div class="wm-note">{theme.html(warn)}</div>', unsafe_allow_html=True)

    st.caption(
        "같은 패턴이라도 원인 경로가 여러 개다. Edge-Ring은 공정(식각)에서도, "
        "검사(프로브 카드)에서도 나온다 — 맵만 보고 공정을 고치러 가면 헛수고다. "
        "이 열이 분석이 어려운 이유 자체를 보여 준다."
    )


# ──────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("데이터 개요")

    try:
        master = _master(SOURCE)
        truth = _truth(SOURCE)
        fdc = _fdc(SOURCE)
    except FileNotFoundError as exc:
        layout.missing_artifact(exc, what="데이터 개요")
        return

    _why_this_matters()
    _hierarchy(master)
    st.divider()
    _dictionary(master, fdc, truth)
    st.divider()
    _quality(master, truth)

    st.divider()
    st.markdown(
        '<div class="wm-why"><b>이 데이터는 합성이다</b><br>'
        '시뮬레이터가 원인을 <b>심어서</b> 만든 데이터이므로 정답지가 있고, 그래서 '
        '분석 방법을 채점할 수 있다. 대신 실측보다 신호가 깨끗하다 — 결측·센서 드리프트·'
        '레시피 변경이 섞여 있지 않다. '
        '<span style="color:#4b5563">여기서 나온 성능은 모두 <b>상한선</b>으로 읽어야 하고, '
        '실데이터(WM-811K) 재측정이 M7에 남아 있다.</span></div>',
        unsafe_allow_html=True,
    )


render()
