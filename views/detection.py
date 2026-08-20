"""화면 4 — 이상공정 탐지 (설계서 §4.5).

이 화면이 답하는 것:
    1. 불량이 **언제부터** 늘었나 — 관리도
    2. 그 신호가 진짜인가 — 관리한계와 이탈 정도
    3. 그럼 **언제 투입된 웨이퍼**를 봐야 하나 — 사이클 타임 보정

SPC는 계산이 가볍다(집계 + 관리도). 그래서 사전 계산 없이 화면에서 바로 돌린다.
파라미터를 바꿔 가며 관리도가 어떻게 달라지는지 직접 만져 보는 것이 학습에 낫다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from wafermap.analysis import spc
from wafermap.config import CAUSE_RULES, PROCESS_STEPS
from wafermap.data.fdc_simulator import EDS_DELAY_HOURS, STEP_INTERVAL_HOURS
from wafermap.data import loader
from wafermap.ui import layout, theme

SOURCE = "synthetic"

#: 공정 투입 → EDS 검사까지 걸리는 시간.
#: 시뮬레이터의 상수에서 계산한다 — 값을 화면에 베껴 쓰면 시뮬레이터를 바꿨을 때 어긋난다.
CYCLE_TIME = pd.Timedelta(
    hours=EDS_DELAY_HOURS + STEP_INTERVAL_HOURS * len(PROCESS_STEPS)
)


@st.cache_data(show_spinner="웨이퍼 이력을 불러오는 중…")
def _master(source: str) -> pd.DataFrame:
    return loader.load_wafer_master(source)


@st.cache_data(show_spinner=False)
def _excursions(source: str) -> pd.DataFrame:
    return loader.load_excursions(source)


@st.cache_data(show_spinner="관리도를 계산하는 중…")
def _chart(
    source: str, pattern: str, freq: str, lam: float, k: float, min_run: int
) -> spc.ControlChart:
    master = _master(source)
    metric = f"{pattern}_rate"
    data = spc.aggregate(master, metric=metric, freq=freq)
    return spc.control_chart(data, metric=metric, lam=lam, k=k, min_run=min_run)


def _chart_figure(chart: spc.ControlChart, pattern: str) -> None:
    import plotly.graph_objects as go

    # `spc.aggregate`는 비율을 **이미 퍼센트 단위(0~100)** 로 돌려준다.
    # 여기서 다시 100을 곱하면 축이 10,000%까지 벌어진다 — 실제로 그랬다.
    series, upper, lower, ewma = (
        chart.series, chart.ewma_upper, chart.ewma_lower, chart.ewma
    )

    fig = go.Figure()
    # 이상 구간 음영 — 먼저 그려야 선 아래로 깔린다
    for alarm in chart.alarms:
        fig.add_vrect(
            x0=alarm.start, x1=alarm.end,
            fillcolor=theme.BAD, opacity=0.10, line_width=0,
        )
    fig.add_trace(go.Scatter(
        x=series.index, y=upper, mode="lines", name="관리한계",
        line=dict(color=theme.MUTED, width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=lower, mode="lines", name="관리한계", showlegend=False,
        line=dict(color=theme.MUTED, width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(107,114,128,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=series, mode="markers", name=f"{pattern} 발생률",
        marker=dict(color="#c7dbf5", size=5),
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=ewma, mode="lines", name="EWMA",
        line=dict(color=theme.GOOD, width=2),
    ))
    fig.add_hline(
        y=chart.center, line=dict(color=theme.MUTED, width=1),
        annotation_text="중심선", annotation_position="right",
    )

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", yaxis_title=f"{pattern} 발생률 (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=theme.BORDER)
    fig.update_yaxes(gridcolor=theme.BORDER, ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": not layout.is_mobile()})


def _cusum_figure(chart: spc.ControlChart) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart.cusum_high.index, y=chart.cusum_high, mode="lines",
        name="상방 누적합", line=dict(color=theme.BAD, width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=chart.cusum_low.index, y=-chart.cusum_low, mode="lines",
        name="하방 누적합", line=dict(color=theme.GOOD, width=1.6),
    ))
    fig.add_hline(y=chart.cusum_limit, line=dict(color=theme.MUTED, dash="dot"))
    fig.add_hline(y=-chart.cusum_limit, line=dict(color=theme.MUTED, dash="dot"))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white", yaxis_title="누적합 (σ)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    fig.update_xaxes(gridcolor=theme.BORDER)
    fig.update_yaxes(gridcolor=theme.BORDER)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(
        "CUSUM은 **작지만 지속적인 이동**에 강하다. 한 점이 크게 튀는 것보다 "
        "여러 점이 조금씩 같은 방향으로 밀릴 때 EWMA보다 빨리 반응한다."
    )


def _alarm_table(chart: spc.ControlChart, truth: pd.DataFrame, pattern: str) -> None:
    if not chart.alarms:
        st.info(
            "검출된 이상 구간이 없습니다. 민감도(λ·k)를 낮추거나 집계 단위를 "
            "바꿔 보세요. — **아무것도 안 나오는 것도 결과**입니다."
        )
        return

    rows = []
    for alarm in chart.alarms:
        cause_start, cause_end = spc.to_cause_window(alarm, CYCLE_TIME)
        rows.append({
            "방법": alarm.method,
            "EDS 검출 구간": f"{alarm.start:%m-%d} ~ {alarm.end:%m-%d}",
            "→ 공정 투입 구간": f"{cause_start:%m-%d} ~ {cause_end:%m-%d}",
            "일수": round(alarm.duration_days, 1),
            "최대 이탈": alarm.peak_deviation,
            "방향": "↑ 증가" if alarm.direction == "up" else "↓ 감소",
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={"최대 이탈": st.column_config.NumberColumn(format="%.1fσ")},
    )

    st.markdown(
        '<div class="wm-note"><b>왜 두 개의 구간을 같이 보여주나</b> ★<br>'
        '"3월 5일 EDS에서 불량 급증"은 <b>"3월 5일에 장비가 고장 났다"가 아니다.</b> '
        f'이 라인의 사이클 타임은 {CYCLE_TIME.total_seconds() / 3600:.0f}시간이므로, '
        '<b>2.75일 전에 투입된 웨이퍼들</b>이 무언가를 겪었다는 뜻이다.<br>'
        '이 보정을 빼먹으면 엉뚱한 기간의 설비 이력을 뒤지게 된다 — '
        '실제로 이 프로젝트에서 검출률이 42%에서 83%로 올라간 것이 이 보정 덕분이다.</div>',
        unsafe_allow_html=True,
    )

    # 정답지와 대조
    if not truth.empty:
        st.markdown("##### 정답지와 대조")
        # 경보를 공정 투입 시각으로 되돌린 구간들
        windows = [spc.to_cause_window(a, CYCLE_TIME) for a in chart.alarms]
        hit = 0
        for _, row in truth.iterrows():
            covered = any(
                not (end < row["t_start"] or start > row["t_end"])
                for start, end in windows
            )
            hit += covered
            st.markdown(
                f'<div class="wm-row">{"✅" if covered else "❌"} '
                f'{row["t_start"]:%m-%d} ~ {row["t_end"]:%m-%d} · '
                f'{row["chamber_id"] or "설비 없음"}</div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"심어 둔 이상 사건 {len(truth)}건 중 **{hit}건**을 덮었다. "
            "정답지는 시뮬레이터만 알고 있으며 분석에는 쓰지 않는다."
        )


def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("이상공정 탐지")

    try:
        master = _master(SOURCE)
    except FileNotFoundError as exc:
        layout.missing_artifact(exc, what="이상공정 탐지")
        return

    st.markdown(
        '<div class="wm-why"><b>왜 관리도인가</b><br>'
        '불량률은 매일 조금씩 흔들린다. <b>어디까지가 정상 변동이고 어디부터가 사건인가</b>를 '
        '눈대중으로 정하면 사람마다 답이 달라진다. 관리도는 그 경계를 통계로 그어 준다.<br>'
        '<span style="color:#4b5563">그리고 이 화면의 목적은 "며칠에 문제가 있었나"가 아니라 '
        '<b>"어느 기간의 설비 이력을 뒤져야 하나"</b>를 정하는 것이다.</span></div>',
        unsafe_allow_html=True,
    )

    patterns = sorted(p for p, r in CAUSE_RULES.items() if r.step_id)
    mobile = layout.is_mobile()
    cols = st.columns(1 if mobile else 4)
    with cols[0]:
        pattern = st.selectbox("불량 패턴", patterns,
                               index=patterns.index("Edge-Ring") if "Edge-Ring" in patterns else 0)
    with cols[1 if not mobile else 0]:
        freq = st.selectbox("집계 단위", ["D", "2D", "W"],
                            format_func={"D": "1일", "2D": "2일", "W": "1주"}.get)
    with cols[2 if not mobile else 0]:
        lam = st.slider("EWMA λ (민감도)", 0.05, 1.0, spc.DEFAULT_LAMBDA, 0.05,
                        help="작을수록 과거를 오래 기억한다 — 작은 이동에 민감해지지만 반응이 느리다")
    with cols[3 if not mobile else 0]:
        k = st.slider("관리한계 k (σ 배수)", 1.5, 4.0, spc.DEFAULT_SIGMA, 0.1,
                      help="작을수록 잘 잡지만 위양성도 늘어난다")

    chart = _chart(SOURCE, pattern, freq, lam, k, spc.DEFAULT_MIN_RUN)

    st.divider()
    layout.metric_grid([
        ("중심선", f"{chart.center:.2f}%", f"{len(chart.series)}개 시점"),
        ("관리도 종류", "p-chart" if chart.chart_type == "p" else "individual",
         "비율 지표에는 p-chart" if chart.chart_type == "p" else None),
        ("과분산 계수", f"{chart.overdispersion:.2f}×",
         "1보다 크면 이항분포보다 실제 변동이 크다"),
        ("검출 구간", f"{len(chart.alarms)}건", "EWMA + CUSUM"),
    ])

    if chart.overdispersion > 1.3:
        st.markdown(
            f'<div class="wm-note"><b>과분산 {chart.overdispersion:.2f}배</b> — '
            'lot 하나가 감염되면 25장이 함께 불량이 되기 때문이다. 이항분포를 그대로 쓰면 '
            '관리한계가 너무 좁아 <b>위양성이 폭증한다.</b> '
            'Laney p′-chart 보정으로 한계를 그만큼 넓혔다.</div>',
            unsafe_allow_html=True,
        )

    _chart_figure(chart, pattern)
    st.caption(
        "점은 일별 발생률, 파란 선은 EWMA(지수가중이동평균), 회색 띠가 관리한계다. "
        "**EWMA는 과거를 기억한다** — 한 점이 튀는 것보다 여러 점이 같은 방향으로 "
        "밀릴 때 반응하므로, 작고 지속적인 이동을 잡는 데 유리하다."
    )

    with st.expander("CUSUM 누적합 — 다른 방식으로 같은 것을 본다"):
        _cusum_figure(chart)

    st.divider()
    st.subheader("검출된 이상 구간")
    truth = _excursions(SOURCE)
    truth = truth[truth["pattern"] == pattern]
    _alarm_table(chart, truth, pattern)

    st.divider()
    st.markdown(
        '<div class="wm-note"><b>한계</b><br>'
        '이 데이터는 전체 121일 중 89일(74%)이 이상 구간이라 <b>위양성을 판정할 여지가 좁다.</b> '
        '"위양성 0건"이라는 결과를 성능으로 인용하면 안 된다. '
        '그리고 λ·k는 이 데이터의 사건 길이(5~13일)에 맞춘 절충이며, '
        '실측 데이터에서는 다시 조정해야 한다.</div>',
        unsafe_allow_html=True,
    )


render()
