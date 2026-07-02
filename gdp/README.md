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
| `phase_b_tf_detector.py` | Transformer 기반 충격탐지기 실험 |
| `phase_b_compare.py` | 종합 비교표 |
| `make_bok_report.py` / `report.html` | 중간결과 보고서 생성 |

## 핵심 결과 (flash w[-19,-1] 평균 RMSE)

| 모형 | 전체 32분기 |
|---|---|
| **regime-gated (제안)** | **0.755** |
| DFM+XGBoost (기존 최고) | 0.765 |
| DFM (기준선) | 0.865 |

- 재현: DFM·DFM+XGBoost를 소수점 수준으로 재현
- 개선: 국면전환 앙상블로 기존 최고치 상회 (K∈{3–8} 견고)
- 딥러닝 단독(Transformer·Foundation): 소표본 특성상 정확도 개선 미확인

세부는 `../docs/GDP_모델_평가표_2026-07-02.md`, `../docs/regime-gated_구조_2026-07-02.md` 참조.
