"""화면 5 — 원인 분석 (설계서 §4.5).

이 프로젝트에서 가장 밀도가 높은 화면이다. 네 가지 질문에 순서대로 답한다.

    1. 어느 설비인가            — 커미널리티 랭킹 (M3)
    2. 공정인가 검사인가        — 축 비교 (M5.5-①)
    3. 그 설비의 무엇인가       — SHAP + 분포 증거 (M4, M5.5-②)
    4. 조합이 문제는 아닌가     — 2×2 분할표 (M5.5-③)

화면 순서가 곧 **분석의 순서**다. 위에서 아래로 읽으면 후보가 좁혀진다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wafermap.ui import artifacts, layout, theme

SOURCE = "synthetic"

MECHANISM_LABELS = {
    "process": "공정 기인", "test": "검사 기인", "spike": "순간 스파이크",
    "drift": "드리프트", "interaction": "교호작용",
}


@st.cache_data(show_spinner="분석 결과를 불러오는 중…")
def _load(source: str) -> dict:
    return artifacts.load_rootcause(source)


def _ranking_table(rows: list[dict]) -> None:
    frame = pd.DataFrame([
        {
            "순위": r["rank"],
            "축": "검사" if r["is_test"] else "공정",
            "스텝": f"{r['step_id']} {r['step_name']}",
            "설비": r["chamber_id"],
            "오즈비": r["odds_ratio"],
            # Streamlit의 "%.0f%%" 포맷은 값을 100배 해 주지 않는다.
            # 비율(0~1)을 그대로 넘기면 전부 0%로 찍힌다.
            "불량군 통과율": (r["case_rate"] or 0) * 100,
            "정상군 통과율": (r["control_rate"] or 0) * 100,
            "p_adj": r["p_adj"],
        }
        for r in rows
    ])
    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config={
            "오즈비": st.column_config.NumberColumn(format="%.2f"),
            "불량군 통과율": st.column_config.NumberColumn(format="%.0f%%"),
            "정상군 통과율": st.column_config.NumberColumn(format="%.0f%%"),
            "p_adj": st.column_config.NumberColumn(format="%.3g"),
        },
    )


def _axis_block(axes: list[dict], truth: dict) -> None:
    if not axes:
        st.info("축 비교 결과가 없습니다.")
        return
    for axis in axes:
        buried = axis["rank_overall"] > 5
        color = theme.WARN if buried else theme.GOOD
        note = (
            f'<div class="wm-row" style="color:{theme.WARN}">'
            f'⚠️ 통합 랭킹 {axis["rank_overall"]}위 — 축을 나누지 않았다면 못 봤다</div>'
            if buried else
            f'<div class="wm-row" style="color:{theme.MUTED}">통합 랭킹 {axis["rank_overall"]}위</div>'
        )
        st.markdown(
            f'<div class="wm-card">'
            f'{theme.badge(axis["axis_ko"], color)}'
            f'<h4 style="margin-top:.4rem">{axis["chamber_id"]}</h4>'
            f'<div class="wm-row">{axis["step_id"]} · 오즈비 <b>{axis["odds_ratio"]:.2f}</b></div>'
            f'{note}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="wm-note">검사 축은 <b>검사 원인이 없어도 무언가를 1위로 내놓는다.</b> '
        '오즈비만 보고 "검사가 문제"라고 판단해서는 안 된다. 확정하려면 '
        '<b>다른 프로브 카드로 재측정</b>해야 하며, 이 분석의 역할은 그 재측정 대상을 좁히는 것이다.</div>',
        unsafe_allow_html=True,
    )


def _evidence_table(evidence: list[dict]) -> None:
    frame = pd.DataFrame([
        {
            "": "✅" if e["has_distribution_evidence"] else "",
            "파라미터": e["column"],
            "구분": "조작가능" if e["is_controllable"] else "계측값",
            "|SHAP|": e["shap_importance"],
            "Δσ": e["delta_sigma"],
            "Cliff's δ": e["cliffs_delta"],
            "규격위반": (e["spec_violation_rate"] or 0) * 100,
        }
        for e in evidence
    ])
    st.dataframe(
        frame, hide_index=True, width="stretch",
        column_config={
            "|SHAP|": st.column_config.NumberColumn(format="%.3f"),
            "Δσ": st.column_config.NumberColumn(format="%+.2f"),
            "Cliff's δ": st.column_config.NumberColumn(format="%+.2f"),
            "규격위반": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    st.caption(
        "✅는 **모델과 무관한 분포 증거**가 SHAP과 같은 방향이라는 뜻이다. "
        "SHAP 1위인데 ✅가 없으면 의심해야 한다 — 실제로 Scratch에서 `pad_life`가 "
        "그랬고, 원인은 카운터가 시각의 대리 변수로 작동한 것이었다."
    )


def _interaction_block(inter: dict) -> None:
    st.caption(inter["mechanism"])

    equips = inter.get("equip_pairs") or []
    if equips:
        top = equips[0]
        st.markdown(
            f'<div class="wm-card">'
            f'<h4>{top["chamber_a"]} → {top["chamber_b"]}</h4>'
            f'<div class="wm-row">조합 불량률 <b>{top["rate_pair"]:.1%}</b> ({top["n_pair"]}장)</div>'
            f'<div class="wm-row">각 설비 단독 — {top["rate_a_only"]:.1%} / {top["rate_b_only"]:.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="wm-note">각 설비는 <b>단독으로는 문제가 없다.</b> '
            '"저 챔버가 문제"라고 보고했다면 멀쩡한 설비를 세우고 처리량만 잃었을 것이다. '
            '조치는 <b>디스패치 규칙 변경</b> — 그 조합만 피하면 된다.</div>',
            unsafe_allow_html=True,
        )

    pairs = inter.get("pairs") or []
    if not pairs:
        st.info("주효과로 안 보이는 파라미터 쌍이 없습니다.")
        return

    st.markdown("##### 파라미터 조합 — 각각은 규격 안인데 조합이 문제")
    pair = pairs[0]
    short_a = pair["param_a"].split(".")[-1]
    short_b = pair["param_b"].split(".")[-1]
    cells = pair["cells"]

    def cell(key: str) -> str:
        rate, n = cells[key]
        strong = rate >= max(c[0] for c in cells.values()) * 0.8
        color = theme.BAD if strong else theme.MUTED
        return (
            f'<td style="text-align:center;padding:.5rem .8rem;border:1px solid {theme.BORDER}">'
            f'<span style="color:{color};font-weight:700;font-size:1.05rem">{rate:.1%}</span>'
            f'<br><span style="font-size:.72rem;color:{theme.MUTED}">{n:,}장</span></td>'
        )

    st.markdown(
        f'<table style="border-collapse:collapse;margin:.3rem 0 .6rem 0">'
        f'<tr><td></td>'
        f'<th style="padding:.3rem .8rem;font-size:.8rem">{short_b} 낮음</th>'
        f'<th style="padding:.3rem .8rem;font-size:.8rem">{short_b} 높음</th></tr>'
        f'<tr><th style="padding:.3rem .6rem;font-size:.8rem;text-align:right">{short_a} 높음</th>'
        f'{cell("hl")}{cell("hh")}</tr>'
        f'<tr><th style="padding:.3rem .6rem;font-size:.8rem;text-align:right">{short_a} 낮음</th>'
        f'{cell("ll")}{cell("lh")}</tr></table>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="wm-row">교호작용 대비 <b>{pair["contrast"]:+.2%}</b> · '
        f'각 파라미터 단독 효과 {pair["main_a"]:+.2%} / {pair["main_b"]:+.2%}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "**대각선 두 칸만 높다** → 같은 방향으로 치우칠 때만 문제라는 뜻이다. "
        "단독 효과가 거의 0이라 단변량 분석(SPC·규격 검사·평균 비교)으로는 "
        "이 쌍을 영원히 못 찾는다."
    )


def render() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title("원인 분석")

    try:
        data = _load(SOURCE)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    patterns = list(data["patterns"])
    if not patterns:
        st.warning("분석된 패턴이 없습니다.")
        return

    default = patterns.index("Edge-Ring") if "Edge-Ring" in patterns else 0
    pattern = st.selectbox("불량 패턴", patterns, index=default)
    item = data["patterns"][pattern]
    truth = item["truth"]

    st.markdown(
        '<div class="wm-why"><b>화면 순서가 곧 분석 순서다</b><br>'
        '① 어느 설비인가 → ② 공정인가 검사인가 → ③ 그 설비의 무엇인가 → ④ 조합이 문제는 아닌가.<br>'
        '<span style="color:#4b5563">위에서 아래로 읽으면 후보가 좁혀진다. '
        '각 단계는 앞 단계의 답을 입력으로 받는다.</span></div>',
        unsafe_allow_html=True,
    )

    mech = " · ".join(
        f"{MECHANISM_LABELS.get(k, k)} {v}장"
        for k, v in sorted(item["mechanisms"].items(), key=lambda kv: -kv[1])
        if k != "none"
    )
    layout.metric_grid([
        ("불량 / 정상", f"{item['n_case']:,} / {item['n_control']:,}", None),
        ("정답 스텝", f"{truth['process_step']} {truth['process_step_name']}",
         "정답지 — 채점에만 쓴다"),
        ("모델 AUC", f"{item['model']['auc']:.3f}" if item["model"].get("auc") else "—",
         item["model"].get("error") or "0.7 미만이면 SHAP 해석을 믿지 말 것"),
        ("원인 경로 구성", mech or "—", "같은 패턴, 다른 원인"),
    ])
    st.caption(f"물리적 기전(정답지): {truth['mechanism']}")

    st.divider()
    st.subheader("① 어느 설비인가 — 커미널리티 랭킹")
    st.caption(
        "lot 단위로 검정한다. 같은 lot의 25장은 함께 같은 설비를 지나므로 "
        "**독립 관측이 아니다.** 웨이퍼 단위로 세면 표본이 25배로 부풀려져 "
        "오즈비가 비현실적으로 커진다."
    )
    _ranking_table(item["ranking"])

    st.divider()
    st.subheader("② 공정인가 검사인가")
    st.caption(
        "프로브 카드 니들이 마모돼도 Edge-Ring과 똑같이 생긴 맵이 나온다. "
        "그런데 조치가 다르다 — 카드 세정은 수 시간, 챔버 PM은 수 일이다."
    )
    _axis_block(item["axes"], truth)

    if item.get("layers"):
        with st.expander("원인이 둘 이상일 때 — 껍질 벗기기"):
            for layer in item["layers"]:
                kind = "검사" if layer["is_test"] else "공정"
                st.markdown(
                    f'<div class="wm-row">{layer["rank"]}층 [{kind}] '
                    f'<b>{layer["chamber_id"]}</b> · OR {layer["odds_ratio"]:.2f} · '
                    f'불량 {layer["n_explained"]}/{layer["n_case_before"]}장 설명</div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                "1위를 찾고 그 설비를 지난 웨이퍼를 뺀 뒤 다시 본다. "
                "⚠️ 모든 웨이퍼가 모든 스텝을 지나므로 **무관한 웨이퍼까지 대량으로 빠진다.** "
                "챔버가 적은 스텝일수록 심하니 축 비교(②)를 더 신뢰할 것."
            )

    st.divider()
    st.subheader("③ 그 설비의 무엇인가 — SHAP + 분포 증거")
    if not item["evidence"]:
        st.info(item["model"].get("error", "파라미터 분석 결과가 없습니다."))
    else:
        chamber = item["model"].get("chamber_id")
        st.caption(
            f"{chamber} 로 층화해 학습했다. 챔버를 고정하면 챔버 간 baseline 차이가 "
            "제거되어 '이 챔버 안에서 무엇이 문제인가'만 남는다."
        )
        _evidence_table(item["evidence"])

    if item.get("interaction"):
        st.divider()
        st.subheader("④ 조합이 문제는 아닌가 — 교호작용")
        _interaction_block(item["interaction"])

    st.divider()
    st.markdown(
        '<div class="wm-note"><b>이 결과를 인용할 때</b><br>'
        '모든 수치는 <b>상관 기반</b>이며, 시드에 따라 크게 달라진다 '
        '(원인 챔버 Top-1은 67% ± 17%). AI는 원인을 확정하지 않는다 — '
        '<b>원인 후보와 근거를 제시하고 공정 전문가가 최종 검증</b>한다. '
        '확증은 DOE(실험계획)로 해야 하며, 이 분석은 그 대상을 좁히는 도구다.</div>',
        unsafe_allow_html=True,
    )


render()
