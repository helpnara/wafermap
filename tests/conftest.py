"""pytest 공통 설정 — src 레이아웃을 import 경로에 추가한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.data.fdc_simulator import simulate  # noqa: E402
from wafermap.data.synth_wafer import build_geometry  # noqa: E402


@pytest.fixture(scope="session")
def geom():
    """웨이퍼 기하 — 생성 비용이 있어 세션 단위로 재사용한다."""
    return build_geometry()


@pytest.fixture(scope="session")
def sim():
    """작은 시뮬레이션 결과 — 여러 테스트가 공유한다."""
    return simulate(n_wafers=1_500, n_days=60, seed=1234)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
