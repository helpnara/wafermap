# 06. 실측 데이터(WM-811K) 전환 절차

이 문서는 **M2 패턴 분류를 실측 데이터로 다시 재는** 절차다.
합성 데이터로 낸 macro-F1 0.9728이 실제로는 얼마인지 확인하는 것이 목적이다.

> **왜 로컬에서만 하나**
> 원본 `LSWMD.pkl`이 약 2GB다. 원격 개발 환경은 세션마다 디스크를 새로 받으므로
> 매번 2GB를 내려받는 것이 비효율적이고, 컨테이너가 회수되면 사라진다.
> 한 번 받아 두고 쓰는 편이 나은 종류의 데이터다.

---

## 0. 무엇이 실측이고 무엇이 아닌가 ★

먼저 이것부터 못 박아야 한다. 헷갈리면 결과를 과대 해석하게 된다.

| 데이터 | 출처 | 정답지 |
|---|---|---|
| 웨이퍼 맵 | **실측** — WM-811K 원본 | — |
| 패턴 라벨 | **실측** — 사람이 붙인 것 | ✅ 이것이 M2의 정답지다 |
| FDC 센서 이력 | 합성 — 시뮬레이터 | — |
| 원인 설비·파라미터 | 합성 — 시뮬레이터가 심은 값 | ⚠️ 실측 정답지가 아니다 |

**WM-811K에는 FDC 데이터가 없다.** 공개 데이터셋이 맵과 라벨만 담고 있어서,
"어느 챔버가 원인이었는가"는 세상에 존재하지 않는 정보다.
`build_dataset.py --source real`은 실측 맵에 합성 공정 이력을 **라벨을 보고 짝지어**
붙인다. 그래서:

- **M2는 진짜 재측정이다.** 맵도 라벨도 실측이므로 점수가 정직하게 나온다.
- **M3~M5는 재측정이 아니다.** 원인 정답지가 여전히 심어 둔 값이므로,
  여기서 얻는 것은 "맵이 지저분해져 M2가 틀리기 시작할 때 하류가 어디까지 버티는가"라는
  **강건성 관찰**이지 실측 성능이 아니다.

포트폴리오에 쓸 때 이 구분을 흐리면 안 된다.

---

## 1. 사전 준비

```bash
git pull
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # 393 passed 확인
```

`pytest`가 통과하지 않으면 여기서 멈추고 원인을 먼저 잡는다.
피처 층이 깨진 상태로 실측을 돌리면 나온 숫자가 무엇 때문인지 알 수 없다.

---

## 2. 원본 내려받기

```bash
python scripts/download_wm811k.py
```

- 받는 것: `data/raw/LSWMD.pkl` (약 2GB)
- Kaggle 계정이 필요할 수 있다. 스크립트가 안내하는 대로 하거나,
  수동으로 받아 `data/raw/LSWMD.pkl`에 두어도 된다.
- 811,457장 중 **라벨이 있는 172,950장만** 쓴다 (`labeled_only=True`).

확인:

```bash
python -c "from wafermap.data import wm811k; print(wm811k.label_distribution())"
```

---

## 3. 실측 파이프라인 실행

```bash
python scripts/build_dataset.py  --source real --max-wafers 30000
python scripts/build_features.py --source real
python scripts/train_pattern_model.py --source real
```

**`--max-wafers`를 왜 두나**: 172,950장 전부를 쓰면 피처 추출에만 수십 분이 걸리고
메모리도 크게 든다. 먼저 30,000장으로 파이프라인이 끝까지 도는지 확인한 뒤,
문제가 없으면 늘린다.

> ⚠️ **합성 쪽도 다시 만들어야 한다.**
> 피처 코드를 고친 뒤라면 `models/synthetic`이 옛 피처로 학습된 상태일 수 있다.
> 그 상태로 비교하면 피처가 달라 대조가 성립하지 않는다
> (`compare_sources.py`가 이 경우를 잡아 준다).
> ```bash
> python scripts/build_features.py
> python scripts/train_pattern_model.py
> ```

---

## 4. 대조

```bash
python scripts/compare_sources.py --json docs/m2_real_vs_synthetic.json
```

클래스별 F1을 낙폭 순으로 보여 준다. 읽을 때 볼 것:

1. **macro-F1이 얼마나 떨어지는가** — 이것이 "합성은 상한선"이라는 주장의 검증이다.
2. **어느 클래스가 무너지는가** — 경계가 애매한 Edge-Loc↔Edge-Ring, Loc↔Scratch가
   특히 위험하다. 합성에서는 규칙으로 그려 또렷했던 것들이다.
3. **표본 수를 함께 보라** — F1이 낮은데 표본이 60장이면 "어렵다"가 아니라
   "모른다"이다. 두 가지를 구분하지 않으면 잘못된 결론이 나온다.
4. **거의 안 떨어졌다면 의심하라** — 30,000장을 무작위로 뽑으면 `none`이 대부분이라
   쉬운 문제가 된다. `--max-wafers`를 늘려 다시 본다.

---

## 5. 실측에서만 드러날 수 있는 함정 ★

합성 데이터는 맵 크기가 **전부 34×67로 고정**이다. 그래서 "맵 크기에만 반응하는 피처"가
섞여 있어도 그 값이 상수가 되어 모델이 무시한다 — **합성으로는 볼 수 없는 결함**이다.
실측은 맵 크기가 제각각이라 같은 피처가 강력한 지름길로 바뀐다.

실제로 이 준비 과정에서 세 건을 찾아 고쳤다:

| 무엇 | 왜 문제였나 | 조치 |
|---|---|---|
| `radon_mean_*` 20개 | 어느 각도에서 봐도 투영의 합은 불량 개수와 같다. 그래서 열평균은 정의상 `불량수/검출기길이`이고, 불량수로 나누면 **맵 크기 하나만** 남았다. 패턴 정보가 0이었다 | 삭제. 정규화로 살릴 수 있는 값이 아니었다 |
| Hu 모멘트·`fail_anisotropy` | Hu 모멘트는 이동·회전·크기에 불변이지만 **종횡비에는 불변이 아니다.** 격자가 2:1로 늘어난 상태에서 계산해 "웨이퍼가 길다"가 모양에 섞였다 | 등방 좌표로 다시 표본화한 뒤 계산 |
| `edge_inner_ratio` | 안쪽 링에 불량이 0이면 `1e-6`으로 나뉘어 값이 12,070까지 폭주했다. 맵이 작을수록 자주 터진다 | 상한 50으로 자름 |

`tests/test_resolution_invariance.py`가 이 부류를 회귀 방지로 잡는다.
**새 피처를 추가하면 이 테스트를 먼저 돌려 볼 것.**

### 아직 남아 있는 것 (실측에서 확인할 목록)

아래는 정규화로 고칠 수 없는 이산화 잔여분이다. 래스터 격자 위에서 연결성분과 둘레를
세는 이상, 격자가 성기면 이웃 덩어리가 붙고 둘레는 계단만큼 길어진다.

- `hu5` — 고차 모멘트라 수치적으로 불안정
- `largest_cluster_perimeter_ratio` — 래스터 둘레의 계단 효과
- `mean_cluster_size_norm`, `cluster_count_norm` — 성긴 격자에서 덩어리가 합쳐진다
- `radon_peak_angle` — 각도 자체라 분산이 크다

**실측 학습 후 이 피처들의 중요도를 확인하라.** 상위권에 올라오면
맵 크기를 학습했을 가능성을 의심해야 한다:

```bash
python -c "
from wafermap.models import pattern_lgbm
m, cols = pattern_lgbm.load('real')
print(pattern_lgbm.feature_importance(m, cols, 20).to_string())"
```

---

## 6. 결과 기록

측정이 끝나면 다음을 갱신한다. 숫자만 바꾸지 말고 **왜 그렇게 나왔는지**를 함께 적는다.

- `docs/04_results.md` — M7 절에 실측 측정 결과
- `src/wafermap/ui/milestones.py` — 화면이 읽는 지표
- `views/validation.py`의 `OPEN_LIMITS` — "합성 데이터 상한" 항목을 측정된 값으로 교체
- `README.md` — 대표 수치

---

## 자주 나오는 문제

**`DatasetNotBuiltError`** — 해당 소스의 `build_dataset.py`를 안 돌린 것이다.
앱 화면에서도 같은 상황이면 필요한 명령을 그대로 띄워 준다.

**메모리 부족** — `--max-wafers`를 줄인다. 30,000장이 대략 2GB 정도를 쓴다.

**피처 수가 99가 아니다** — 피처 코드를 고친 뒤 한쪽만 다시 만든 것이다.
양쪽 모두 `build_features.py`부터 다시 돌린다.

**실측 macro-F1이 합성보다 높다** — 표본 추출이 편향됐을 가능성이 크다.
`none`이 압도적으로 많아 macro 평균이 쉬운 클래스에 끌린 경우다. 클래스별 표를 보라.
