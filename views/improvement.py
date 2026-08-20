"""화면 6 — 개선방안 및 기대효과 (설계서 §4.5, §3 M5).

이 화면이 답하는 질문:
    1. 무엇을 할 것인가? (조치안)
    2. 하면 얼마나 좋아지는가? (반사실 시뮬레이션)
    3. 그게 돈으로 얼마인가? (ROI — 가정치는 사용자가 바꾼다)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wafermap.analysis import recommend
from wafermap.ui import artifacts, layout, theme

SOURCE = "synthetic"


@st.cache_data(show_spinner="분석 결과를 불러오는 중…")
def _load(source: str) -> dict:
    return artifacts.load_recommendations(source)


def _why_this_matters() -> None:
    st.markdown(
        '<div class="wm-why"><b>왜 이 화면이 필요한가</b><br>'
        '원인을 찾는 것까지는 분석이고, <b>무엇을 얼마나 바꿀지</b>를 정하는 것이 개선이다. '
        '챔버까지만 특정하면 할 수 있는 조치는 "그 챔버를 세우는 것"뿐인데, 팹에서 챔버 하나를 '
        '세우면 그 스텝의 처리량이 통째로 줄어든다. 파라미터 운전 구간까지 내려와야 '
        '<b>설비를 세우지 않고</b> 레시피 조정만으로 해결할 수 있다.</div>',
        unsafe_allow_html=True,
    )


def _action_card(action: dict) -> None:
    color, mark = theme.evidence_style(action["evidence"])
    kind = recommend.ACTION_KINDS[action["kind"]]

    # f-string 안에서 같은 따옴표를 중첩할 수 없으므로(파이썬 3.11) 미리 꺼내 둔다
    title = f'{kind["icon"]} {action["title"]}'
    badges = "".join([
        theme.badge(kind["label"], theme.GOOD),
        theme.badge(f'근거 {mark} {action["evidence"]}', color),
        theme.badge(f'난이도 {action["effort"]}', theme.MUTED),
    ])
    current, proposed = theme.html(action["current"]), theme.html(action["proposed"])
    rationale = theme.html(action["rationale"])

    st.markdown(
        f'<div class="wm-card">'
        f'<h4>{title}</h4>{badges}'
        f'<div class="wm-row" style="margin-top:.5rem"><b>현재</b> · {current}</div>'
        f'<div class="wm-row"><b>제안</b> · {proposed}</div>'
        f'<div class="wm-row" style="color:#4b5563"><b>근거</b> · {rationale}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _dependence_chart(curve: dict, column: str, spec: dict | None) -> None:
    """구간별 불량률 — 개선안의 근거 그래프.

    왜 x축을 실제 값이 아니라 **구간 순번**으로 두나 ★: 구간은 분위수로 나뉘어
        폭이 제각각이다(꼬리 구간이 훨씬 넓다). 실제 값을 축으로 쓰면 표본이 몰린
        중앙 구간들이 서로 겹쳐 뭉개지고, 넓은 꼬리 구간 하나만 크게 보인다.
        각 구간이 **같은 표본 수**를 담고 있으므로 등간격으로 그리는 것이 옳다.
    """
    import plotly.graph_objects as go

    frame = pd.DataFrame(curve)
    rates = frame["case_rate"] * 100
    idx = list(range(len(frame)))

    # 권고 구간 안에 들어가는 구간 번호
    in_band = [
        bool(spec) and frame.iloc[i]["value_lo"] >= spec["low"] - 1e-9
        and frame.iloc[i]["value_hi"] <= spec["high"] + 1e-9
        for i in idx
    ]
    colors = [theme.GOOD if ok else theme.BAD for ok in in_band]

    fig = go.Figure(
        go.Bar(
            x=idx, y=rates, marker_color=colors, width=0.78,
            customdata=frame[["value_lo", "value_hi", "n_samples"]].to_numpy(),
            hovertemplate=(
                "구간 %{customdata[0]:.2f} ~ %{customdata[1]:.2f}<br>"
                "불량률 %{y:.1f}%<br>표본 %{customdata[2]:.0f}장<extra></extra>"
            ),
        )
    )
    if any(in_band):
        first, last = in_band.index(True), len(in_band) - 1 - in_band[::-1].index(True)
        fig.add_vrect(
            x0=first - 0.5, x1=last + 0.5, fillcolor=theme.GOOD, opacity=0.09,
            line_width=0, annotation_text="권고 운전 구간", annotation_position="top left",
        )

    ticks = [f'{row["value_lo"]:.2f}' for _, row in frame.iterrows()]
    ticks.append(f'{frame.iloc[-1]["value_hi"]:.2f}')
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=34, b=10),
        xaxis_title=f"{column} (구간별 · 각 구간 표본 수 동일)",
        yaxis_title="구간별 불량률 (%)",
        plot_bgcolor="white", showlegend=False, bargap=0.12,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=idx, ticktext=ticks[:-1],
        tickangle=-45, gridcolor=theme.BORDER,
    )
    fig.update_yaxes(gridcolor=theme.BORDER, ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)


def _roi_inputs() -> recommend.RoiAssumptions:
    """ROI 가정치 입력 — 전부 바꿀 수 있어야 한다 (설계서 §3 M5.4)."""
    st.markdown("##### 가정치")
    st.caption(
        "이 숫자들은 회사·라인마다 자릿수가 다르다. 고정해 두면 "
        "\"그 숫자 어디서 나왔냐\"는 질문 하나에 분석 전체의 신뢰가 무너진다."
    )

    if st.button("기본값으로 재설정", width="stretch"):
        for key in ("wafer_value", "monthly_wafers", "adoption", "running_cost", "impl_cost"):
            st.session_state.pop(key, None)
        st.rerun()

    defaults = recommend.RoiAssumptions()
    value = st.number_input(
        "웨이퍼 1장당 가치 (만원)", 50, 5_000,
        int(defaults.wafer_value_krw / 10_000), step=50, key="wafer_value",
    )
    kind_badge, note = recommend.ASSUMPTION_BADGES["wafer_value_krw"]
    st.markdown(
        f'<div class="wm-assume">🏷️ {kind_badge} — {note}</div>',
        unsafe_allow_html=True,
    )
    monthly = st.number_input(
        "월 투입 웨이퍼 (장)", 100, 100_000, defaults.monthly_wafers, step=500,
        key="monthly_wafers",
    )
    st.markdown(
        f'<div class="wm-assume">🏷️ {recommend.ASSUMPTION_BADGES["monthly_wafers"][1]}</div>'.replace(
            "🏷️ ", "🏷️ 가정 — "),
        unsafe_allow_html=True,
    )
    adoption = st.slider(
        "조치 적용률", 0.1, 1.0, defaults.adoption_rate, 0.05, key="adoption",
        help="조치안이 100% 지켜지는 라인은 없다. 보수적으로 잡지 않으면 기대효과가 부풀려진다.",
    )
    running = st.number_input(
        "연간 운영비 증가 (만원)", 0, 500_000,
        int(defaults.annual_running_cost_krw / 10_000), step=500, key="running_cost",
        help="PM 주기 단축에 따른 소모품·다운타임 증가분",
    )
    impl = st.number_input(
        "1회성 도입 비용 (만원)", 0, 500_000,
        int(defaults.implementation_cost_krw / 10_000), step=500, key="impl_cost",
    )
    return recommend.RoiAssumptions(
        wafer_value_krw=value * 10_000,
        monthly_wafers=int(monthly),
        adoption_rate=adoption,
        annual_running_cost_krw=running * 10_000,
        implementation_cost_krw=impl * 10_000,
    )


def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("개선방안 및 기대효과")

    try:
        data = _load(SOURCE)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    patterns = list(data["patterns"])
    if not patterns:
        st.warning("분석된 패턴이 없습니다.")
        return

    _why_this_matters()

    pattern = st.selectbox("불량 패턴", patterns, index=patterns.index("Donut")
                           if "Donut" in patterns else 0)
    item = data["patterns"][pattern]
    cf = item["counterfactual"]

    # ── 요약 지표 ───────────────────────────────────────────────────────
    layout.metric_grid([
        ("원인 스텝", f"{item['step_id']} {item['step_name']}", item["mechanism"]),
        ("원인 챔버", item["chamber_id"] or "—", "M3 커미널리티 1위"),
        ("모델 AUC", f"{item['auc']:.3f}",
         "0.7 미만이면 SHAP 해석을 신뢰할 수 없다"),
        ("불량 / 정상 표본",
         f"{item['n_case']:,} / {item['n_control']:,}",
         "불량이 적으면 순위는 나와도 규격을 정할 수는 없다"),
    ])

    if item["auc"] < recommend.MIN_TRUSTWORTHY_AUC:
        st.markdown(
            '<div class="wm-note">⚠️ AUC가 0.7 미만이다. 모델이 신호를 제대로 잡지 못했으므로 '
            '아래 조치안은 노이즈를 분해한 결과일 수 있다.</div>', unsafe_allow_html=True)

    st.divider()

    with layout.responsive_split(("조치안", "기대효과 · ROI"), ratio=(1.15, 1.0)) as (left, right):
        # ── 좌: 조치안 ──────────────────────────────────────────────────
        with left:
            st.subheader("조치안")
            st.caption(
                f"{item['step_id']} {item['step_name']} · {item['chamber_id']} 기준 · "
                f"계측값은 조작할 수 없어 조치 대상에서 제외했다"
            )
            for action in item["actions"]:
                _action_card(action)

            spec_actions = [a for a in item["actions"] if a["spec"] and a["column"] in item["curves"]]
            if spec_actions:
                st.subheader("개선안 근거")
                target = st.radio(
                    "파라미터", [a["column"] for a in spec_actions],
                    horizontal=not layout.is_mobile(), label_visibility="collapsed",
                )
                chosen = next(a for a in spec_actions if a["column"] == target)
                _dependence_chart(item["curves"][target], target, chosen["spec"])
                st.caption(
                    "막대는 구간별 실제 불량률이다. 파란 띠가 권고 운전 구간. "
                    "선형 반응이 아니라 **문턱**이 있다는 점이 핵심이다 — "
                    "그 문턱 앞에서 관리하면 된다."
                )

        # ── 우: 기대효과 + ROI ──────────────────────────────────────────
        with right:
            st.subheader("기대효과")
            if cf is None:
                st.info("이 패턴은 시뮬레이션할 권고 구간이 없습니다.")
            else:
                layout.metric_grid([
                    ("현재 불량률", f"{cf['observed_defect_rate']:.2%}", "분석 챔버 기준"),
                    ("개선 후(추정)", f"{cf['expected_defect_rate']:.2%}",
                     f"상대 {cf['relative_reduction']:+.0%}"),
                ], desktop_cols=2)
                layout.metric_grid([
                    ("챔버 수율", f"{cf['yield_gain_pp']:+.2f}%p", "분석 챔버 안에서의 값"),
                    ("라인 수율", f"{cf['line_yield_gain_pp']:+.2f}%p",
                     f"이 챔버가 라인 물량의 {cf['scope_fraction']:.0%}를 처리한다"),
                ], desktop_cols=2)

                feasibility = cf["feasibility"]
                n_wafers, n_clipped = cf["n_wafers"], cf["n_clipped"]
                move = cf["move_fraction"]
                st.markdown(
                    f'<div class="wm-row">실행 가능성 <b>{feasibility}</b> · '
                    f'표본 {n_wafers}장 중 {n_clipped}장({move:.0%})을 옮겨야 한다</div>',
                    unsafe_allow_html=True,
                )
                for note in cf["caveats"]:
                    st.markdown(
                        f'<div class="wm-note">{theme.html(note)}</div>', unsafe_allow_html=True)

            st.divider()
            st.subheader("ROI")

            if cf is None:
                st.caption("기대효과가 없어 ROI를 계산할 수 없습니다.")
            else:
                assumptions = _roi_inputs()
                roi = recommend.estimate_roi(cf["line_yield_gain_pp"], assumptions)

                st.markdown("##### 결과")
                layout.metric_grid([
                    ("연간 순효익", roi.format_krw(roi.net_saving_krw),
                     "총 효익 − 연간 운영비"),
                    ("추가 양품", f"{roi.recovered_wafers_year:,.0f}장/년", None),
                ], desktop_cols=2)
                layout.metric_grid([
                    ("연간 총 효익", roi.format_krw(roi.gross_saving_krw), None),
                    ("투자 회수",
                     f"{roi.payback_months:.1f}개월" if roi.payback_months else "회수 불가",
                     "1회성 도입 비용 ÷ 월 순효익"),
                ], desktop_cols=2)

                if roi.payback_months is None:
                    st.markdown(
                        '<div class="wm-note">순효익이 운영비를 넘지 못한다. '
                        '조치 자체를 재검토해야 한다.</div>', unsafe_allow_html=True)

                with st.expander("가정치 민감도 — 이 결론이 얼마나 흔들리는가"):
                    st.caption(
                        "단일 숫자로 제시한 ROI는 **가정치가 틀리면 통째로 틀린다.** "
                        "어떤 가정에 결과가 민감한지 보여야 인용하는 사람이 무엇을 "
                        "확인해야 하는지 안다."
                    )
                    table = recommend.roi_sensitivity(
                        cf["line_yield_gain_pp"], assumptions, field_name="wafer_value_krw"
                    )
                    st.dataframe(
                        pd.DataFrame({
                            "웨이퍼 단가": [f"{v / 10_000:,.0f}만원" for v in table["value"]],
                            "배수": [f"{f:.2f}×" for f in table["factor"]],
                            "연간 순효익": [roi.format_krw(v) for v in table["net_saving_krw"]],
                            "회수(개월)": [
                                f"{v:.1f}" if pd.notna(v) else "회수 불가"
                                for v in table["payback_months"]
                            ],
                        }),
                        hide_index=True, width="stretch",
                    )

    st.divider()
    st.markdown(
        '<div class="wm-note"><b>이 결과를 인용할 때</b><br>'
        '모든 수치는 <b>상관 기반</b>이다. "이 구간으로 옮기면 좋아진다"는 인과 보장이 아니라 '
        '"이 구간에 있던 웨이퍼는 불량이 적었다"이다. 확증은 DOE(실험계획)로 해야 하며, '
        '이 분석은 <b>DOE 대상을 좁히는 도구</b>다. '
        '그리고 AI가 원인을 확정하는 것이 아니라 <b>원인 후보와 근거를 제공하고, '
        '공정 전문가가 최종 검증</b>한다.</div>',
        unsafe_allow_html=True,
    )


render()
