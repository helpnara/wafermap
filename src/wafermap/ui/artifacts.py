"""앱이 읽는 사전 계산 아티팩트 로더 (설계서 §4.6).

왜 앱에서 계산하지 않나: SHAP 계산은 패턴 하나에 20~30초가 걸린다. 사용자가
    슬라이더를 움직일 때마다 그만큼 기다리게 만들 수는 없다. 무거운 계산은
    `scripts/build_recommendations.py`에서 한 번 하고, 앱은 결과만 읽는다.
    실시간으로 다시 계산하는 것은 **가정치가 바뀌면 즉시 달라져야 하는 ROI뿐**이다.
"""

from __future__ import annotations

import json
from pathlib import Path

from wafermap.config import processed_dir

RECOMMENDATIONS = "recommendations.json"


def recommendations_path(source: str = "synthetic") -> Path:
    return processed_dir(source) / RECOMMENDATIONS


def has_recommendations(source: str = "synthetic") -> bool:
    return recommendations_path(source).exists()


def load_recommendations(source: str = "synthetic") -> dict:
    """M5 아티팩트를 읽는다.

    Raises:
        FileNotFoundError: 아티팩트가 없을 때 — 생성 명령을 안내한다
    """
    path = recommendations_path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다.\n"
            f"먼저 실행하세요: python scripts/build_recommendations.py --source {source}"
        )
    return json.loads(path.read_text(encoding="utf-8"))
