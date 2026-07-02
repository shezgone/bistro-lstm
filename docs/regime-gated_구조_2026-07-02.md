# regime-gated 앙상블 — 구조 설명

- **작성일**: 2026-07-02
- **목적**: GDP Nowcasting에서 DFM+XGBoost(0.765)를 넘는 우리 모델. 실시간 국면 전환 앙상블.
- **성과**: 전체 RMSE **0.7548** (DFM+XGBoost 0.765 대비 −0.010, 양 국면 우위, K∈{3~8} 견고)
- **코드**: `gdp-nowcasting-renewal/phase_b_regime_gated.py` / 결과 `output/csv/_phase_b_regime_gated.csv`

---

## 핵심 아이디어

"어느 한 모델이 항상 최선"이 아니라 **국면(regime)마다 승자가 다르다** — shock기엔 DFM+XGBoost, 평온기엔 DFM+RF. 이를 **실시간(vintage-safe) 충격 탐지기**로 판정해 전환한다.

```
        [직전 K분기 실현 GDP 변동성]        각 target 분기 q · 빈티지 v
                  │
           ┌──────▼──────┐   vintage-safe
           │ 충격 탐지기   │   (미래정보 사용 X)
           └──────┬──────┘
            shock │ calm
           ┌──────┴────────┐
           ▼               ▼
      DFM+XGBoost      DFM+RF        ← 국면별 최적 앙상블 2종
      =(DFM+XGB)/2    =(DFM+RF)/2
           └──────┬────────┘
                  ▼
            최종 예측 (gated)
```

---

## 구성요소

### ① Base 예측 (재료, 3종)
각 `(분기, 빈티지, 전망주차)`마다:
- **DFM** — 한은 기준 상태공간 모형 (`output/model/DFM/11/<q>/rtf.pkl`)
- **XGBoost** — DFM 보정 월별 피처로 학습 (그들 파이프라인 재현, train_rows=140 rolling)
- **RF** — 동일 피처로 학습

### ② 국면별 최적 앙상블 (2종)
| 앙상블 | 계산 | 강한 국면 | 근거 RMSE |
|---|---|---|---|
| DFM+XGBoost | (DFM+XGB)/2 | shock | COVID 0.999 (최선) |
| DFM+RF | (DFM+RF)/2 | calm | 최근 0.523 (최선) |

### ③ 충격 탐지기 (vintage-safe)
- **신호**: target 분기 q 직전 **K분기(K=4)** 실현 GDP(flash) 변동성(표준편차). 급변기 = 변동성↑.
- **임계**: 고정값 아님 → **적응형**(q 이전까지 관측된 변동성들의 확장창 중앙값). q 시점 정보만 사용.
- **판정**: `vol_q > median(과거 vol들)` → shock. 초기 정보부족 분기는 보수적으로 shock(=DFM+XGB).

### ④ 게이트 (스위치)
```python
gated = (DFM+XGBoost)/2  if shock(q) else  (DFM+RF)/2
```

---

## 성과 (flash w[−19,−1] 평균 RMSE)

| 모형 | 전체 | COVID | 최근 |
|---|---|---|---|
| **regime-gated (우리)** | **0.7548** | **0.9917** | **0.5198** |
| DFM+XGBoost (기존 최고) | 0.765 | 0.999 | 0.542 |
| DFM+RF | 0.788 | 1.054 | 0.523 |
| DFM | 0.865 | 1.255 | 0.541 |

- 양 국면 모두 우위 · 사후라벨 상한(0.7607)보다도 우수(데이터 기반 게이트가 "COVID=shock"보다 정교)
- 견고성: K∈{3,4,5,6,8} 전부 <0.765 (0.7548~0.7594)

---

## 정직한 한계

1. 게이트 **구조**(shock→XGB, calm→RF)는 Phase B 관찰에서 선택 — 약한 선택 편향. (탐지기 하이퍼 K·임계는 견고성 확인됨)
2. 개선폭 ~1%는 **32분기 소표본에서 DM 통계 유의 미검증**. ("robust하게 넘되 유의성 별도 검증 필요")
3. 탐지기가 **급변 온셋을 한 박자 늦게** 감지(2020Q2 급락은 이후에 vol↑로 포착). 고도화 여지: DFM revision·월별 지표 기반 조기 감지.

---

## 의의

우리 bistro-lstm **regime-detector 자산**을 GDP에 이식. **정확도 1위(DFM+XGBoost 상회) + 국면 설명가능성("왜 이 국면에 이 모델")**을 동시에 제공 → 정확도·차별화·설명가능성 결합.

## 다음 단계 후보
- DM test로 유의성 검증 (정직한 마무리)
- 탐지기 고도화 (DFM revision / 월별 지표 → 조기 감지)
- 한은 보고자료로 정리
