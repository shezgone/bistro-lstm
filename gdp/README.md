# GDP Nowcasting 협업 — 분석 코드

한국은행 GDP Nowcasting 시스템 고도화 협업에서 수행한 재현·확장 실험 코드입니다.
(데이터·모델 아티팩트는 기밀이므로 포함하지 않으며, 코드는 한은 소스 저장소
`gdp-nowcasting-renewal` 클론 위에서 실행됩니다.)

## 실행 전제
- 한은 소스 저장소(`changhoon0807/gdp-nowcasting-renewal`) 클론
- 별도 전달받은 빈티지 데이터 + DFM/LSTM 산출물을 `data/`, `output/model/`에 배치
- 위 스크립트를 저장소 루트에 두고 실행 (py3.11 + numpy/pandas/scikit-learn/xgboost/torch)

## 스크립트

| 파일 | 역할 |
|---|---|
| `phase_b_harness.py` | 공통 채점 하네스 (그들과 동일 flash w[-19,-1] 평균 RMSE) |
| `phase_b_fair.py` | 우리 신경망(MLP)을 그들 backtest에 주입 — XGBoost와 동일 데이터/walk-forward |
| `phase_b_transformer.py` / `phase_b_seq.py` | AttnLSTM(비시퀀스/시퀀스) 예측 실험 |
| `phase_b_foundation.py` | Chronos·Moirai zero-shot |
| `phase_b_ensemble_search.py` | 앙상블 조합 탐색 |
| `phase_b_regime_gated.py` | **국면전환 앙상블 + 실시간 충격탐지기 (제안 모델)** |
| `phase_b_regime_gated_v2.py` | **v2: 반등(REBOUND) 국면 추가 3-arm** — 직전분기 flash<0 & 심리 저점통과 → DFM 단독(보정 OFF). `--strict`는 flash 발표 후 주차만 발동 |
| `phase_b_llm.py` | Claude LLM(Fable 5/Opus 4.8) zero-shot 나우캐스트 비교 (2025H1, API 키 필요) |
| `phase_b_tf_detector.py` | Transformer 기반 충격탐지기 실험 |
| `phase_b_tabpfn.py` | TabPFN(PFN, in-context 학습) 공정 주입 — 비-트리 ML 최고(DFM+TabPFN 0.815) but DFM+XGB 미달 |
| `phase_b_ncde.py` | NCDENow-style 근사(PCA요인+euler CDE) — 기각 (DFM+NCDE 0.932 > DFM 0.865) |
| `phase_b_soft_gate.py` | soft gate·AE 탐지기 타당성 (soft 0.7226 ≈ v2 hard) |
| `phase_b_score_new.py` | TabPFN·NCDE 통합 채점 |
| `phase_b_compare.py` | 종합 비교표 |
| `make_bok_report.py` / `report.html` | 중간결과 보고서 생성 |

## 핵심 결과 (flash w[-19,-1] 평균 RMSE)

| 모형 | 전체 32분기 |
|---|---|
| **regime-gated v2 (3-arm, 반등 국면 추가)** | **0.722** (strict 0.738) |
| regime-gated v1 (2-arm) | 0.755 |
| DFM+XGBoost (기존 최고) | 0.765 |
| DFM (기준선) | 0.865 |

> v2는 유망 후보 단계 (반등 arm 사후 선택·6분기 소표본·DM 미검증).
> 세부: `../docs/regime-gated_v2_3arm_구조_2026-07-09.md`, `../docs/regime-gated_v2_3arm_1p_2026-07-09.pptx`, 메일 초안 `../docs/메일_한은_regime-gated_v2_2026-07-09.md`

- 재현: DFM·DFM+XGBoost를 소수점 수준으로 재현
- 개선: 국면전환 앙상블로 기존 최고치 상회 (K∈{3–8} 견고)
- 딥러닝 단독(Transformer·Foundation): 소표본 특성상 정확도 개선 미확인

세부는 `../docs/GDP_모델_평가표_2026-07-02.md`, `../docs/regime-gated_구조_2026-07-02.md` 참조.
