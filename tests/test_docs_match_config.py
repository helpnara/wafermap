"""문서에 적은 숫자가 코드와 맞는지 (설계서 M7).

왜 필요한가 ★: `docs/03_simulator_spec.md`는 손으로 쓴 문서인데 코드의 숫자를
    인용한다. 설정값을 바꾸면 문서는 조용히 거짓말이 된다 — 그리고 문서가 틀린 것은
    코드가 틀린 것보다 발견이 늦다. 아무도 실행해 보지 않기 때문이다.

    `docs/02_data_dictionary.md`는 아예 생성물이라 `--check`로 확인한다.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from wafermap import config as C
from wafermap.data import fdc_simulator as sim

SPEC = (C.DOCS_DIR / "03_simulator_spec.md").read_text(encoding="utf-8")
DOMAIN = (C.DOCS_DIR / "01_domain_eds.md").read_text(encoding="utf-8")


def _cause(rules, pattern):
    return rules[pattern]


@pytest.mark.parametrize("text", [
    # 시간 구조 — 사이클 타임은 관리도 해석의 핵심이라 특히 틀리면 안 된다
    f"스텝 간격 **{sim.STEP_INTERVAL_HOURS:g}시간**",
    f"EDS까지 **{sim.EDS_DELAY_HOURS:g}시간**",
    f"사이클 타임 {int(sim.EDS_DELAY_HOURS + sim.STEP_INTERVAL_HOURS * len(C.PROCESS_STEPS))}시간",
    # 원파형 해상도
    f"`TRACE_POINTS = {sim.TRACE_POINTS}`",
    f"`TRACE_SECONDS = {sim.TRACE_SECONDS:.1f}`",
    # lot 구성
    f"× {C.WAFERS_PER_LOT}장",
])
def test_spec_quotes_match_code(text):
    assert text in SPEC, f"03_simulator_spec.md 와 코드가 어긋남: {text!r}"


def test_spike_numbers_match():
    """스파이크 규칙의 σ와 초는 M5.5-② 주장의 근거라 정확해야 한다."""
    pert = _cause(C.SPIKE_CAUSE_RULES, "Edge-Loc").perturbations[0]
    assert f"**{pert.spike_sigma:g}σ만큼 {pert.spike_seconds:g}초간**" in SPEC


def test_drift_sigma_matches():
    """8.6이라는 역산값 — 이 숫자가 왜 그것인지가 문서의 요점이다."""
    pert = _cause(C.DRIFT_CAUSE_RULES, "Edge-Loc").perturbations[0]
    assert f"**{pert.drift_sigma:g}σ 폭으로**" in SPEC


def test_interaction_range_matches():
    rule = _cause(C.INTERACTION_CAUSE_RULES, "Center")
    lo, hi = rule.shift_range
    assert f"**{lo:g}~{hi:g}σ**" in SPEC
    assert f"`{rule.equip_pair[0]}` × `{rule.equip_pair[1]}`" in SPEC


def test_difficulty_table_matches():
    """현실의 지저분함 설정 — 표의 값이 코드와 같아야 한다."""
    d = C.DEFAULT_DIFFICULTY
    for value in (d.false_positive_rate, d.unexplained_rate, d.confound_rate,
                  d.equip_bias_sigma, d.drift_sigma_per_pm):
        assert f"| {value:.2f} |" in SPEC or f"| {value:g} |" in SPEC, (
            f"난이도 설정 {value} 가 문서 표에 없다"
        )


def test_mechanism_shares_match():
    """경로별 혼입 비율 — 어떤 패턴에 얼마나 섞는지."""
    pairs = [
        ("Edge-Ring", C.TEST_INDUCED_SHARE), ("Loc", C.TEST_INDUCED_SHARE),
        ("Edge-Loc", C.SPIKE_INDUCED_SHARE), ("Edge-Loc", C.DRIFT_INDUCED_SHARE),
        ("Center", C.INTERACTION_INDUCED_SHARE),
    ]
    for pattern, share in pairs:
        pct = f"{pattern} {share[pattern] * 100:.0f}%"
        assert pct in SPEC, f"혼입 비율이 문서와 다름: {pct}"


def test_distractor_step_is_documented():
    """P045는 우연한 상관을 걸러내는지 확인하는 장치다 — 문서에 남아야 한다."""
    distractor = [s for s in C.PROCESS_STEPS if s.note and "distractor" in s.note]
    assert distractor, "distractor 스텝이 config에서 사라졌다"
    assert distractor[0].step_id in DOMAIN


def test_die_map_values_documented():
    """맵 값 규약은 실측/합성이 한 코드 경로를 타는 근거라 문서에 명시돼야 한다."""
    for value in (C.DIE_NONE, C.DIE_PASS, C.DIE_FAIL):
        assert f"| {value} |" in (C.DOCS_DIR / "02_data_dictionary.md").read_text(
            encoding="utf-8")


def test_generated_dictionary_is_up_to_date():
    """02는 생성물이다 — 코드를 고치고 다시 생성하지 않으면 실패한다."""
    result = subprocess.run(
        [sys.executable, str(C.PROJECT_ROOT / "scripts" / "build_data_dictionary.py"),
         "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "docs/02_data_dictionary.md 가 코드와 어긋났다. "
        "python scripts/build_data_dictionary.py 로 갱신할 것.\n" + result.stderr
    )
