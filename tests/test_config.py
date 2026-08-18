"""config의 도메인 상수가 서로 모순되지 않는지 검사한다.

왜 이 테스트가 중요한가: CAUSE_RULES는 '문자열로 파라미터를 지목'하는 구조라,
오타가 나도 파이썬은 아무 말을 하지 않는다. 시뮬레이터가 조용히 섭동을 건너뛰고,
몇 단계 뒤 "원인 분석 정확도가 왜 낮지?"로 나타난다. 여기서 즉시 잡는다.
"""

from __future__ import annotations

import pytest

from wafermap import config as C


def test_process_steps_have_unique_ids():
    ids = [s.step_id for s in C.PROCESS_STEPS]
    assert len(ids) == len(set(ids))


def test_chamber_ids_are_unique_and_well_formed():
    all_chambers = [c for s in C.PROCESS_STEPS for c in s.chamber_ids]
    assert len(all_chambers) == len(set(all_chambers)), "챔버 식별자가 중복됨"
    assert all("/ch" in c for c in all_chambers), "챔버 식별자는 'EQUIP/chN' 형식이어야 함"


def test_param_specs_are_valid():
    for step in C.PROCESS_STEPS:
        names = [p.name for p in step.params]
        assert len(names) == len(set(names)), f"{step.step_id}에 중복 파라미터"
        for p in step.params:
            assert p.spec_lo < p.spec_hi
            if p.kind == "gaussian":
                # 목표값이 규격 밖이면 정상 웨이퍼가 전부 규격 위반이 된다
                assert p.spec_lo <= p.nominal <= p.spec_hi, f"{step.step_id}.{p.name}"
                assert p.sigma > 0


def test_cause_rules_reference_existing_steps_and_params():
    """인과 규칙이 실재하는 스텝·파라미터를 가리키는지 확인한다."""
    for pattern, rule in C.CAUSE_RULES.items():
        assert pattern in C.PATTERN_LABELS
        if rule.step_id is None:
            assert not rule.perturbations, f"{pattern}: 원인 스텝이 없는데 섭동이 정의됨"
            continue

        assert rule.step_id in C.STEPS_BY_ID, f"{pattern}: 없는 스텝 {rule.step_id}"
        step = C.STEPS_BY_ID[rule.step_id]
        param_names = {p.name for p in step.params}
        for pert in rule.perturbations:
            assert pert.param in param_names, (
                f"{pattern}: {rule.step_id}에 '{pert.param}' 파라미터가 없음"
            )


def test_counter_perturbations_use_counter_ratio():
    """counter형 파라미터는 shift_sigma가 아니라 counter_ratio로 조작해야 한다.

    왜: counter형은 sigma=0이라 shift_sigma를 곱해도 값이 변하지 않는다.
        즉 섭동이 조용히 무효가 된다.
    """
    for rule in C.CAUSE_RULES.values():
        if rule.step_id is None:
            continue
        step = C.STEPS_BY_ID[rule.step_id]
        for pert in rule.perturbations:
            if step.param(pert.param).kind == "counter":
                assert pert.counter_ratio is not None, (
                    f"{rule.pattern}.{pert.param}: counter형인데 counter_ratio가 없음"
                )
                assert 0.0 < pert.counter_ratio <= 1.0


def test_distractor_step_has_no_cause_rule():
    """P045(캐패시터)는 어떤 패턴의 원인도 아니어야 한다 — 설계서 §2.4."""
    causes = {r.step_id for r in C.CAUSE_RULES.values()}
    assert "P045" not in causes


def test_label_counts_match_wm811k_total():
    assert sum(C.WM811K_LABEL_COUNTS.values()) == 172_950
    assert set(C.WM811K_LABEL_COUNTS) == set(C.PATTERN_LABELS)


def test_processed_dir_rejects_unknown_source():
    with pytest.raises(ValueError):
        C.processed_dir("bogus")
