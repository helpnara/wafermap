---
module: src/wafermap/data/fdc_simulator.py
milestone: M1
difficulty: 중급
concepts: [이상사건 주도 생성, 교락(confounding), 위양성/위음성, 톱니파 드리프트, 정답지(ground truth), 판다스 벡터화]
prerequisites: [synth_wafer]
estimated_minutes: 25
---

# FDC 시뮬레이터 — 가상 팹의 공정 이력 만들기

## 이 코드는 무엇을 하나

**입력**: 웨이퍼 수, 기간(일), 난수 시드, 난이도 설정
**출력**: 네 개의 표

| 표 | 내용 | 행 수 (6,000장 기준) |
|---|---|---|
| `wafer_master` | 웨이퍼별 lot/시각/불량 패턴 | 6,000 |
| `fdc_summary` | (웨이퍼 × 스텝)별 설비·챔버·파라미터 195개 | 54,000 |
| `fdc_trace` | 일부 웨이퍼의 파라미터 시계열 | 151,200 |
| `ground_truth` | **정답지** — 진짜 원인이 뭐였는지 | 6,000 |

핵심은 마지막 `ground_truth`다. 우리가 직접 원인을 심었으니 정답을 안다.
나중에 분석 모델이 그 원인을 다시 찾아내는지 **채점**할 수 있다.

---

## 핵심 로직

### 가장 중요한 아이디어: 이상 사건(excursion) 주도 생성

가장 쉬운 방법은 웨이퍼마다 주사위를 굴려 패턴을 정하는 것이다. **그런데 그러면 안 된다.**

```
❌ 웨이퍼별 독립 추출              ✅ 이상 사건 주도 (채택)
불량이 시간·설비에 고르게 흩어짐    "ETCH-B/ch2가 2/18~3/02에 이상했다"
   ↓                                  ↓
SPC 관리도에 잡힐 이상 구간 없음    그 기간·그 챔버 웨이퍼가 불량
커미널리티에서 유의한 설비 없음        ↓
   ↓                               SPC가 구간을 잡고
분석 파이프라인 전체가 검증 불가     커미널리티가 챔버를 찾아냄
```

실제 팹의 불량은 **항상 특정 설비·특정 기간에 몰려서** 발생한다.
그 구조를 재현하지 않으면 M2~M4에서 찾을 것 자체가 존재하지 않는다.

`_assign_patterns`가 이 일을 한다.

```python
step = STEPS_BY_ID[rule.step_id]
chamber = str(rng.choice(step.chamber_ids))
...
win_hours = span_hours * rng.uniform(0.03, 0.12)
...
eligible = np.where(
    (lot_chamber == chamber) & in_window & (pattern.to_numpy() == "none")
)[0]
...
hit = eligible[rng.random(len(eligible)) < attack]
```

- `step` / `chamber` — 이 패턴의 원인 스텝에서 챔버 하나를 "범인"으로 지목
- `win_hours` — 전체 기간의 3~12% 길이로 이상 구간을 잡는다
- `eligible` — 그 챔버를 지났고 + 그 기간이었고 + 아직 미배정인 웨이퍼
- `hit` — 그중 `attack_rate` 비율만 실제 불량이 된다 (전부는 아니다)

"그 챔버를 지나갔고 + 그 기간이었고 + 아직 미배정인" 웨이퍼를 골라 감염시킨다.

### 파라미터 값 만들기

```
값 = 목표값 + 챔버 baseline 편차 + PM 주기 드리프트 + 정상 산포 + 인과 섭동
```

네 개의 항이 각각 다른 현실을 표현한다.

| 항 | 무엇을 표현하나 | 왜 필요한가 |
|---|---|---|
| 챔버 baseline 편차 | 같은 모델 설비도 챔버마다 미세하게 다름 | 분석이 절대값이 아니라 **평소 대비 변화**를 봐야 하게 만든다 |
| PM 주기 드리프트 | 정비 후 시간이 갈수록 서서히 이탈 | 트렌드 분석과 "PM 주기 단축" 개선안의 근거 |
| 정상 산포 | 웨이퍼 간 자연 변동 | 신호 대 잡음비를 현실적으로 |
| 인과 섭동 | 진짜 원인 | 분석이 찾아내야 할 대상 |

### 소모품 수명은 톱니파

`edge_ring_rf_hours`(식각 챔버 부품 사용시간), `pad_life`(CMP 패드 수명)는
정규분포가 아니다. **쓸수록 증가하다가 정비 때 0으로 리셋**된다.

```python
usage = (order % period).astype(float)
```

`%`(나머지) 하나로 톱니파가 만들어진다. `order`는 그 챔버에서 몇 번째로 처리된
웨이퍼인지이고, `period`가 PM 주기다.

```
사용량 ▲
       │    ╱│    ╱│    ╱│
       │  ╱  │  ╱  │  ╱  │      ← 정비할 때마다 0으로
       │╱    │╱    │╱    │
       └──────────────────▶ 시간
```

### 정답지 기록

| 컬럼 | 의미 |
|---|---|
| `true_root_step` | 진짜 원인 스텝 |
| `true_root_equip` | 진짜 원인 챔버 |
| `true_root_params` | 진짜 원인 파라미터 목록 |
| `severity` | 이탈 강도 (맵 생성기와 공유) |
| `is_confounded` | 교락에 걸렸는가 |
| `is_unexplained` | 불량인데 FDC 신호가 없는가 |
| `is_false_positive` | 정상인데 파라미터가 튀었는가 |

⚠️ 이 표는 **분석 모델 입력에 절대 넣으면 안 된다.** 채점용지를 시험 중에 보는 것과 같다.
`loader.load_ground_truth()`의 docstring에도 경고를 달아 뒀다.

---

## 왜 이렇게 만들었나 — 설계 선택과 대안

### 선택 1: 왜 일부러 어렵게 만드나?

원인 파라미터만 깔끔하게 이탈시키면 SHAP이 100% 맞힌다. 그런데 그건
**분석 역량을 증명하지 못한다.** 그래서 세 가지 교란을 의도적으로 넣는다.

| 교란 | 비율 | 현실에서의 의미 |
|---|---|---|
| `is_unexplained` | 10% | 불량인데 FDC에 신호가 없음 — 계측하지 못한 원인 |
| `is_false_positive` | 6% | 정상인데 파라미터가 이탈 — 분석이 속으면 안 되는 함정 |
| `is_confounded` | — | 진범과 늘 같이 흐른 무고한 설비 |

실제로 이 설계 덕분에, Center 패턴에 `down_force`를 **+4.0σ**로 설정했는데
관측되는 차이는 **+1.97σ**로 감쇠된다. 신호는 있지만 자명하지는 않은 상태다.

### 선택 2: 교락(confounding)을 어떻게 만드나? ★ 핵심

이게 이 시뮬레이터에서 가장 정교한 부분이다.

```python
hit = routing[sa.step_id] == ch_a
move = hit & (rng.random(n_lots) < difficulty.confound_rate * 3.0)
routing[sb.step_id] = np.where(move, ch_b, routing[sb.step_id])
```

"ETCH-B/ch3를 쓴 lot은 CMP-01/ch2도 자주 쓴다"는 편향을 심는다.
그러면 데이터에서 이런 일이 벌어진다.

```
진범:  ETCH-B/ch3  → Edge-Ring 불량률 22%  ← 진짜 원인
무고:  CMP-01/ch2  → Edge-Ring 불량률 19%  ← 같이 흘렀을 뿐
```

단순 교차표로는 둘 다 유의하게 나온다. M3의 **CMH 층화 검정**이 이 둘을
갈라낼 수 있어야 하고, 그러려면 데이터에 교락이 실제로 존재해야 한다.

### 선택 3: 왜 P045(캐패시터)에는 원인을 안 넣었나?

`P045`는 **distractor(미끼) 스텝**이다. 어떤 패턴의 원인도 아니다.
검증해 보면 이런 결과가 나온다.

```
P045 챔버별 불량률: CAP-01/ch1:17.2%  CAP-01/ch2:13.5%  CAP-02/ch1:17.4%  CAP-02/ch2:9.2%
```

최대 17.4% vs 최소 9.2%로 거의 2배 차이가 난다! 원인이 아닌데도 그럴듯한
상관이 보인다. 이게 바로 **다중검정 보정(BH-FDR)이 필요한 이유**다.
37개 챔버를 동시에 검정하면 우연히 유의해 보이는 게 반드시 나온다.

### 선택 4: 왜 lot 단위로 챔버를 배정하나?

```python
routing[step.step_id] = rng.choice(chambers, size=n_lots)
```

`size=n_lots` — 웨이퍼 수가 아니라 **lot 수**만큼 뽑는다.

웨이퍼 단위가 아니라 lot(25장) 단위다. 실제 팹이 그렇게 돌아가기 때문이기도 하지만,
통계적으로 더 중요한 이유가 있다. 웨이퍼마다 다른 챔버로 흩어지면 **표본이 25배로
부풀려져** 커미널리티 검정의 p-value가 비현실적으로 작게 나온다.
실질적인 표본 단위는 lot이라는 사실을 데이터 구조가 지켜 줘야 한다.

---

## 확인 문제

**1.** `_assign_patterns`에서 희소 패턴(Near-full)부터 배정하는 이유는?

<details><summary>정답</summary>

```python
order = sorted(targets, key=lambda p: targets[p])
```

목표 장수가 적은 패턴부터 정렬한다.

흔한 패턴(Edge-Ring 336장)이 먼저 웨이퍼를 차지해 버리면, 배정 조건인
`pattern == "none"`(미배정)을 만족하는 웨이퍼가 줄어든다. 그러면 Near-full(5장)처럼
드문 패턴이 조건에 맞는 웨이퍼를 못 찾아 목표 장수를 채우지 못할 수 있다.
</details>

**2.** `severity`가 맵 생성기와 FDC 시뮬레이터 양쪽에서 쓰이는 이유는?

<details><summary>정답</summary>

**두 데이터를 잇는 유일한 연결고리**이기 때문이다. 같은 severity를 공유해야
"파라미터가 심하게 튄 웨이퍼일수록 맵도 심하다"는 상관이 생긴다.
실제로 검증하면 severity ↔ 불량률 상관이 r = +0.41~+0.66으로 나온다.
이 연결이 없으면 M4의 원인 분석 모델이 학습할 신호 자체가 존재하지 않는다.
</details>

**3.** `test_unexplained_wafers_lack_signal` 테스트는 무엇을 방지하는가?

<details><summary>정답</summary>

플래그만 세우고 값은 그대로 흔들어 놓는 버그를 잡는다. 정답지에는 "설명 불가"라고
적혀 있는데 데이터에는 신호가 남아 있으면, 모델이 그 웨이퍼의 원인을 맞혀 버린다.
그러면 M4의 검증 지표가 실제 성능을 반영하지 못한다.
`has_signal = is_defect & ~is_unexplained` 로 섭동 자체를 건너뛰는 게 맞는 구현이다.
</details>

**4.** 시뮬레이터 결과가 매번 같으려면 무엇이 보장되어야 하는가?

<details><summary>정답</summary>

`rng = np.random.default_rng(seed)` 하나를 만들어 **모든 하위 함수에 같은 객체를
넘기는 것**, 그리고 난수를 뽑는 **순서가 항상 같은 것**이다.
조건문 때문에 어떤 실행에서만 난수를 더 뽑으면 이후 값이 전부 어긋난다.
`test_simulation_is_reproducible`이 이를 검사한다.
</details>

---

## 더 알아보기

- **교락(confounding)**: 통계학에서 인과 추론을 어렵게 만드는 대표적 문제.
  M3의 Cochran–Mantel–Haenszel 검정이 이를 다룬다 → `commonality.md` (M3에서 작성 예정)
- **다중검정 문제**: 37개를 동시에 검정하면 유의수준 5%에서 약 2개는 우연히 유의하다.
  Benjamini-Hochberg FDR 보정이 필요한 이유
- **판다스 벡터화**: `np.where(targets, values + shift, values)` 처럼 반복문 없이
  조건부 갱신하는 방식. 6,000장 × 195컬럼을 2초에 처리하는 비결
- 관련 모듈: [`synth_wafer.md`](synth_wafer.md) — 여기서 정한 패턴·severity로 맵을 그린다
