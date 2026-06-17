# Graph Report - .  (2026-06-10)

## Corpus Check
- 79 files · ~65,164 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1247 nodes · 1838 edges · 79 communities (65 shown, 14 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 107 edges (avg confidence: 0.51)
- Token cost: 29,716 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Agent Experiment Framework|Agent Experiment Framework]]
- [[_COMMUNITY_Economic Regime Eras|Economic Regime Eras]]
- [[_COMMUNITY_Dashboard & Results Loading|Dashboard & Results Loading]]
- [[_COMMUNITY_Claude Rolling Results|Claude Rolling Results]]
- [[_COMMUNITY_Baseline Forecasts 2025|Baseline Forecasts 2025]]
- [[_COMMUNITY_Regime Detector Signals|Regime Detector Signals]]
- [[_COMMUNITY_Blinded Claude Results|Blinded Claude Results]]
- [[_COMMUNITY_Paper Ablation Results|Paper Ablation Results]]
- [[_COMMUNITY_CSI Forced Results|CSI Forced Results]]
- [[_COMMUNITY_Extra Ablation Results|Extra Ablation Results]]
- [[_COMMUNITY_Transformer Architecture|Transformer Architecture]]
- [[_COMMUNITY_Google Trends Aug Results|Google Trends Aug Results]]
- [[_COMMUNITY_Blinded HCX Results|Blinded HCX Results]]
- [[_COMMUNITY_CSI Covariate Results|CSI Covariate Results]]
- [[_COMMUNITY_Few-shot Results|Few-shot Results]]
- [[_COMMUNITY_HCX Rolling Results|HCX Rolling Results]]
- [[_COMMUNITY_LSTM Training Pipeline|LSTM Training Pipeline]]
- [[_COMMUNITY_Causal Narrative Analysis|Causal Narrative Analysis]]
- [[_COMMUNITY_Project Overview Concepts|Project Overview Concepts]]
- [[_COMMUNITY_Torch CSI Benchmarks|Torch CSI Benchmarks]]
- [[_COMMUNITY_Format Ablation Results|Format Ablation Results]]
- [[_COMMUNITY_PyTorch Rolling Runners|PyTorch Rolling Runners]]
- [[_COMMUNITY_LSTM Runner Pipeline|LSTM Runner Pipeline]]
- [[_COMMUNITY_Feature Importance XAI|Feature Importance XAI]]
- [[_COMMUNITY_HCX Value by Regime|HCX Value by Regime]]
- [[_COMMUNITY_Training & Evaluation|Training & Evaluation]]
- [[_COMMUNITY_Regime Detector Backtest|Regime Detector Backtest]]
- [[_COMMUNITY_PDF Report Export|PDF Report Export]]
- [[_COMMUNITY_Ablation Study Runner|Ablation Study Runner]]
- [[_COMMUNITY_Torch Model Results|Torch Model Results]]
- [[_COMMUNITY_HCX Value Quantification|HCX Value Quantification]]
- [[_COMMUNITY_Foundation Model Runner|Foundation Model Runner]]
- [[_COMMUNITY_Data Collection FREDBIS|Data Collection FRED/BIS]]
- [[_COMMUNITY_Stress Test Results|Stress Test Results]]
- [[_COMMUNITY_Chronos Evaluation|Chronos Evaluation]]
- [[_COMMUNITY_Moirai Evaluation|Moirai Evaluation]]
- [[_COMMUNITY_Post-hoc Aggregation Results|Post-hoc Aggregation Results]]
- [[_COMMUNITY_Post-hoc Ensemble Results|Post-hoc Ensemble Results]]
- [[_COMMUNITY_2025 Actuals|2025 Actuals]]
- [[_COMMUNITY_HCX Rolling Script|HCX Rolling Script]]
- [[_COMMUNITY_Format Ablation Script|Format Ablation Script]]
- [[_COMMUNITY_Blinded Evaluation Script|Blinded Evaluation Script]]
- [[_COMMUNITY_Few-shot Script|Few-shot Script]]
- [[_COMMUNITY_Aggregation Methods Code|Aggregation Methods Code]]
- [[_COMMUNITY_Per-seed Ensemble Code|Per-seed Ensemble Code]]
- [[_COMMUNITY_Blinded Claude Script|Blinded Claude Script]]
- [[_COMMUNITY_Extra Ablations Script|Extra Ablations Script]]
- [[_COMMUNITY_Google Trends Script|Google Trends Script]]
- [[_COMMUNITY_Claude Rolling Script|Claude Rolling Script]]
- [[_COMMUNITY_CSI Forced Script|CSI Forced Script]]
- [[_COMMUNITY_CSI Rolling Script|CSI Rolling Script]]
- [[_COMMUNITY_Paper Ablations Script|Paper Ablations Script]]
- [[_COMMUNITY_Stress Test Script|Stress Test Script]]
- [[_COMMUNITY_Consensus Aggregation|Consensus Aggregation]]
- [[_COMMUNITY_Claude Results Metadata|Claude Results Metadata]]
- [[_COMMUNITY_HCX Seed Ablation|HCX Seed Ablation]]
- [[_COMMUNITY_HCX-32B Ablation Results|HCX-32B Ablation Results]]
- [[_COMMUNITY_Bias-corrected Mean|Bias-corrected Mean]]
- [[_COMMUNITY_Ensemble 3070|Ensemble 30/70]]
- [[_COMMUNITY_Ensemble 5050|Ensemble 50/50]]
- [[_COMMUNITY_Ensemble 7030|Ensemble 70/30]]
- [[_COMMUNITY_Max Seed Aggregation|Max Seed Aggregation]]
- [[_COMMUNITY_Mean Aggregation|Mean Aggregation]]
- [[_COMMUNITY_Median Aggregation|Median Aggregation]]
- [[_COMMUNITY_Min Seed Aggregation|Min Seed Aggregation]]
- [[_COMMUNITY_Trend12 Baseline|Trend12 Baseline]]
- [[_COMMUNITY_Trimmed Mean Aggregation|Trimmed Mean Aggregation]]
- [[_COMMUNITY_Claude Settings|Claude Settings]]
- [[_COMMUNITY_Chronos Results|Chronos Results]]
- [[_COMMUNITY_Moirai Results|Moirai Results]]
- [[_COMMUNITY_Ensemble Decomposition|Ensemble Decomposition]]
- [[_COMMUNITY_HCX Clean Format|HCX Clean Format]]
- [[_COMMUNITY_Augmented Panel Builder|Augmented Panel Builder]]
- [[_COMMUNITY_Cognition Lessons|Cognition Lessons]]
- [[_COMMUNITY_ECOS CSI Fetcher|ECOS CSI Fetcher]]
- [[_COMMUNITY_HCX Covariate Forecast|HCX Covariate Forecast]]
- [[_COMMUNITY_HCX Univariate Forecast|HCX Univariate Forecast]]
- [[_COMMUNITY_HCX No-Think Variant|HCX No-Think Variant]]
- [[_COMMUNITY_XAI Importance Concepts|XAI Importance Concepts]]

## God Nodes (most connected - your core abstractions)
1. `results` - 49 edges
2. `LSTMConfig` - 43 edges
3. `AttentionLSTMForecaster` - 42 edges
4. `ZScoreNormalizer` - 39 edges
5. `Candidate` - 22 edges
6. `train_model()` - 21 edges
7. `results` - 17 edges
8. `results` - 17 edges
9. `hcx_results` - 17 edges
10. `run_training()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `device` --uses--> `LSTMConfig`  [INFERRED]
  ablation_study.py → lstm_core.py
- `device` --uses--> `AttentionLSTMForecaster`  [INFERRED]
  ablation_study.py → lstm_model.py
- `device` --uses--> `ZScoreNormalizer`  [INFERRED]
  ablation_study.py → preprocessing_util.py
- `ndarray` --uses--> `LSTMConfig`  [INFERRED]
  ablation_study.py → lstm_core.py
- `ndarray` --uses--> `AttentionLSTMForecaster`  [INFERRED]
  ablation_study.py → lstm_model.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Economic Narrative Analysis Pipeline (CF + Jacobian + Pathway -> Narrative)** — readme_counterfactual_analysis, readme_jacobian_lag_analysis, readme_pathway_decomposition, readme_narrative_generation [EXTRACTED 1.00]
- **XAI Methods Suite (6 types)** — readme_cross_variable_attention, readme_integrated_gradients, readme_permutation_importance, readme_ablation_study, readme_counterfactual_analysis, readme_narrative_generation [EXTRACTED 1.00]
- **Korean CPI Forecasting Benchmark (Task-specific vs Foundation Models)** — readme_task_specific_transformer, readme_attentionlstmforecaster, readme_chronos_2, readme_bistro_moirai, readme_timesfm, readme_ar1_baseline [EXTRACTED 1.00]

## Communities (79 total, 14 thin omitted)

### Community 0 - "Agent Experiment Framework"
Cohesion: 0.06
Nodes (68): analyze(), _extract_json(), _format_history_block(), _format_lessons_block(), _format_round_block(), Lesson, Analyzer agent — Claude Opus distills transferable lessons from the latest round, _build_messages() (+60 more)

### Community 1 - "Economic Regime Eras"
Cohesion: 0.05
Nodes (58): first_flag, flagged, period_len, rate, first_flag, flagged, period_len, rate (+50 more)

### Community 2 - "Dashboard & Results Loading"
Cohesion: 0.07
Nodes (39): load_all_data(), BISTRO-LSTM Streamlit Dashboard ================================ 8개 탭: 핵심결론 + 트레, load_narrative_results(), compare_forecasts(), compare_variable_rankings(), compute_metrics(), load_bistro_ablation(), load_bistro_results() (+31 more)

### Community 3 - "Claude Rolling Results"
Cohesion: 0.04
Nodes (49): results, claude-opus-4-5|2025-05|cov, claude-opus-4-5|2025-05|univar, claude-opus-4-5|2025-06|cov, claude-opus-4-5|2025-06|univar, claude-opus-4-5|2025-07|cov, claude-opus-4-5|2025-07|univar, claude-opus-4-5|2025-08|cov (+41 more)

### Community 4 - "Baseline Forecasts 2025"
Cohesion: 0.04
Nodes (49): AR(1), BoK linear, MA12, RW, Trend12, AR(1), BoK linear, MA12 (+41 more)

### Community 5 - "Regime Detector Signals"
Cohesion: 0.05
Nodes (41): auc, fpr_at_opt, j_at_opt, n_shock, n_stable, n_total, optimal_threshold, tpr_at_opt (+33 more)

### Community 6 - "Blinded Claude Results"
Cohesion: 0.07
Nodes (39): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+31 more)

### Community 7 - "Paper Ablation Results"
Cohesion: 0.08
Nodes (34): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+26 more)

### Community 8 - "CSI Forced Results"
Cohesion: 0.06
Nodes (33): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+25 more)

### Community 9 - "Extra Ablation Results"
Cohesion: 0.09
Nodes (33): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+25 more)

### Community 10 - "Transformer Architecture"
Cohesion: 0.10
Nodes (16): Tensor, 최근 forward의 attention weights 반환. (B, N, N), Temporal Attention Decoder.     Learnable forecast queries가 LSTM hidden states에, Per-variable linear projection.     각 변수를 독립적인 learned embedding으로 변환.      Inpu, 최근 forward의 temporal attention weights. (B, pred_len, seq_len), Forward pass.          Parameters         ----------         x : (batch, seq_len, 예측 + 불확실성 추정.          Returns         -------         dict with keys:, Cross-variable attention per timestep.     각 시점에서 변수 간 상호작용을 학습.      Input:  (b (+8 more)

### Community 11 - "Google Trends Aug Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 12 - "Blinded HCX Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 13 - "CSI Covariate Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 14 - "Few-shot Results"
Cohesion: 0.06
Nodes (32): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+24 more)

### Community 15 - "HCX Rolling Results"
Cohesion: 0.06
Nodes (31): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+23 more)

### Community 16 - "LSTM Training Pipeline"
Cohesion: 0.12
Nodes (22): DataLoader, LSTMConfig, LSTM 모델 설정.      Parameters     ----------     variates    : 변수 이름 리스트 (순서 = 입력, EarlyStopping, evaluate(), optuna_objective(), AttentionLSTMForecaster, device (+14 more)

### Community 17 - "Causal Narrative Analysis"
Cohesion: 0.15
Nodes (25): counterfactual_analysis(), generate_narrative(), get_active_channels(), get_variable_channel(), jacobian_lag_analysis(), main(), pathway_decomposition(), AttentionLSTMForecaster (+17 more)

### Community 18 - "Project Overview Concepts"
Cohesion: 0.09
Nodes (26): AR(1) Statistical Baseline, AttentionLSTMForecaster, BISTRO-LSTM Project, BISTRO (Moirai 91M), BISTRO-XAI (Related Project), Chronos-2 (120M, Zero-shot), Counterfactual Analysis, Cross-variable Attention (XAI) (+18 more)

### Community 19 - "Torch CSI Benchmarks"
Cohesion: 0.08
Nodes (24): LSTM-base (18), mae, n_params, name, rmse, rows, LSTM-csi (18+CSI), mae (+16 more)

### Community 20 - "Format Ablation Results"
Cohesion: 0.08
Nodes (23): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+15 more)

### Community 21 - "PyTorch Rolling Runners"
Cohesion: 0.14
Nodes (14): BISTRO-LSTM Model — Attention-Augmented Stacked LSTM ===========================, set_seed(), prepare_walk_forward_splits(), DataFrame, Walk-forward CV용 데이터 분할.      Parameters     ----------     df : 전체 학습 데이터 (Peri, Per-variable z-score 정규화.     학습 세트 기준으로 mean/std 계산, 테스트 세트에 동일 적용., ZScoreNormalizer, LSTM/Transformer rolling 2025 with vs without BoK CSI sentiment column.  4 model (+6 more)

### Community 22 - "LSTM Runner Pipeline"
Cohesion: 0.17
Nodes (18): main(), AttentionLSTMForecaster, DataFrame, device, LSTMConfig, ZScoreNormalizer, BISTRO-LSTM Runner — 2-Stage Inference Pipeline ================================, Stage 2: 선택된 변수로 최종 학습 + 예측. (+10 more)

### Community 23 - "Feature Importance XAI"
Cohesion: 0.21
Nodes (16): compute_all_importance(), compute_gradient_importance(), compute_permutation_importance(), extract_attention_importance(), AttentionLSTMForecaster, device, ndarray, Tensor (+8 more)

### Community 24 - "HCX Value by Regime"
Cohesion: 0.11
Nodes (17): main_per_target, main_summary, non-flagged, shock-flagged, hcx_rmse, mean_delta_err, n, p (+9 more)

### Community 25 - "Training & Evaluation"
Cohesion: 0.16
Nodes (15): ar1_forecast(), PeriodIndex, Series, Utility: AR(1) baseline forecast. bistro-xai의 inference_util.py와 동일., Simple AR(1) forecast: y_t = c + phi * y_{t-1}      Parameters     ----------, load_stage1_screening(), Stage 1 전체 변수 스크리닝 결과 로딩., Optuna 하이퍼파라미터 튜닝.      Returns     -------     dict with best_params, best_valu (+7 more)

### Community 26 - "Regime Detector Backtest"
Cohesion: 0.24
Nodes (15): compute_trend12_errors(), coverage_in_periods(), cusum_two_sided(), evaluate_detector(), mad_around_ma(), main(), multi_signal(), ndarray (+7 more)

### Community 27 - "PDF Report Export"
Cohesion: 0.22
Nodes (8): generate(), load_all(), make_2023_chart(), make_2024_chart(), make_seed_chart(), CPI Forecasting Benchmark - PDF Report v1 ======================================, LSTM vs Transformer seed 비교 차트., Report

### Community 28 - "Ablation Study Runner"
Cohesion: 0.27
Nodes (12): main(), device, ndarray, Ablation Study for BISTRO-LSTM =============================== bistro-xai의 ablat, Leave-one-out ablation study.      Parameters     ----------     base_vars : Sta, Incremental addition: 중요도 순으로 변수를 하나씩 추가.      Returns     -------     dict with, 주어진 변수 서브셋으로 학습 후 2023 OOS RMSE/MAE 반환., run_ablation() (+4 more)

### Community 29 - "Torch Model Results"
Cohesion: 0.15
Nodes (12): LSTM, mae, n_params, name, rmse, rows, Transformer, mae (+4 more)

### Community 30 - "HCX Value Quantification"
Cohesion: 0.31
Nodes (12): annotate_baselines(), load_2022_h1_round0(), load_2023_2024_blinded(), load_2025_forced_cot(), main(), multi_signal(), DataFrame, ndarray (+4 more)

### Community 31 - "Foundation Model Runner"
Cohesion: 0.26
Nodes (12): load_data(), main(), DataFrame, ndarray, Foundation Model Inference for Korean CPI Forecasting ==========================, Sundial zero-shot inference via HuggingFace transformers., CPI 데이터 로딩 (monthly)., Chronos-2 zero-shot inference. (+4 more)

### Community 32 - "Data Collection FRED/BIS"
Cohesion: 0.27
Nodes (11): build_daily_panel(), build_panel(), build_tournament_daily_panel(), download_all_fred(), download_fred(), load_bis_data(), DataFrame, Macro Variable Data Collector for BISTRO-XAI Feature Selection ================= (+3 more)

### Community 33 - "Stress Test Results"
Cohesion: 0.17
Nodes (11): actuals, 2023, 2024, forecasts, 2023, 2024, n_seeds, rationales (+3 more)

### Community 34 - "Chronos Evaluation"
Cohesion: 0.26
Nodes (11): evaluate(), load_series(), main(), origins(), pick_device(), predict_one(), ndarray, Period (+3 more)

### Community 35 - "Moirai Evaluation"
Cohesion: 0.29
Nodes (11): build_predictor(), evaluate(), forecast_one(), load_panel(), main(), origins(), pick_device(), DataFrame (+3 more)

### Community 36 - "Post-hoc Aggregation Results"
Cohesion: 0.20
Nodes (9): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+1 more)

### Community 37 - "Post-hoc Ensemble Results"
Cohesion: 0.20
Nodes (9): ar1_rmse, ensemble_30_per_seed_rmses, ensemble_50_per_seed_rmses, ensemble_70_per_seed_rmses, hcx_per_seed_rmses, median_ensemble_rmse, optimal_rmses, optimal_weights_per_seed (+1 more)

### Community 38 - "2025 Actuals"
Cohesion: 0.22
Nodes (9): actuals, 2025-05, 2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11 (+1 more)

### Community 39 - "HCX Rolling Script"
Cohesion: 0.28
Nodes (8): baseline_forecasts(), build_messages(), hcx_call(), main(), Period, Rolling 1-step-ahead forecast for 2025-05 to 2025-12 (post-LLM-cutoff OOS).  Mod, Build prompt for 1-step-ahead forecast given data through origin month., Compute deterministic baseline forecasts for 1-step-ahead from origin.

### Community 40 - "Format Ablation Script"
Cohesion: 0.31
Nodes (8): build_messages(), hcx_call(), main(), panel_to_tsv_with_units(), DataFrame, Period, HCX rolling 2025 format ablation: minimal header changes vs current TSV.  Tests, TSV with 'Date' header + units in column names. No other changes.

### Community 41 - "Blinded Evaluation Script"
Cohesion: 0.36
Nodes (7): build_messages(), hcx_call(), main(), panel_blinded_tsv(), Period, HCX rolling 2025 BLINDED evaluation — strip identifying info to test contaminati, Convert to blinded TSV: var_N column names, t=N row labels.

### Community 42 - "Few-shot Script"
Cohesion: 0.36
Nodes (7): build_messages(), hcx_call(), main(), panel_to_tsv(), DataFrame, Period, HCX rolling 2025 OOS with few-shot examples + forced CoT.  For each origin t: -

### Community 45 - "Blinded Claude Script"
Cohesion: 0.38
Nodes (5): build_messages(), claude_call(), panel_blinded_tsv(), Period, Claude (Opus 4.7 + Sonnet 4.5) BLINDED rolling 2025 — contamination evidence.  K

### Community 46 - "Extra Ablations Script"
Cohesion: 0.43
Nodes (6): build_messages(), hcx_call(), main(), panel_tsv(), Period, HCX rolling 2025 — extra ablations (literature-grounded).  Modes: 1. wrong_cpi:

### Community 47 - "Google Trends Script"
Cohesion: 0.40
Nodes (4): build_messages(), hcx_call(), Period, HCX rolling 2025 OOS: cov_base (18) vs cov_aug (18 + 5 Google Trends).  Direct a

### Community 48 - "Claude Rolling Script"
Cohesion: 0.40
Nodes (4): build_messages(), claude_call(), Period, Rolling 1-step-ahead forecast for 2025-05 to 2025-12 with Claude models.  Identi

### Community 49 - "CSI Forced Script"
Cohesion: 0.40
Nodes (4): build_messages(), hcx_call(), Period, HCX rolling 2025 with CSI + FORCED attention prompt.  Re-uses cov_base results f

### Community 50 - "CSI Rolling Script"
Cohesion: 0.40
Nodes (4): build_messages(), hcx_call(), Period, HCX rolling 2025 OOS: cov_base (18) vs cov_csi (18 + 1 BoK CSI).  Drop Google Tr

### Community 51 - "Paper Ablations Script"
Cohesion: 0.40
Nodes (4): build_messages(), hcx_call(), Period, HCX ablations grounded in literature.  Modes (system msg variants only — Minimal

### Community 52 - "Stress Test Script"
Cohesion: 0.47
Nodes (4): build_messages(), hcx_call(), panel_blinded_tsv(), HCX blinded 12-step forecast for 2023 + 2024 (shock + transition periods).  Test

### Community 53 - "Consensus Aggregation"
Cohesion: 0.40
Nodes (5): mae, preds, rmse, methods, Consensus (3-closest-to-median)

### Community 54 - "Claude Results Metadata"
Cohesion: 0.40
Nodes (4): ctx_len, n_seeds, targets, temperature

### Community 55 - "HCX Seed Ablation"
Cohesion: 0.50
Nodes (3): build_messages(), call_once(), HCX-32B-Think n-seed ablation: (cov/univar) x (think on/off) x (2023/2024).

### Community 56 - "HCX-32B Ablation Results"
Cohesion: 0.50
Nodes (3): n_per_cell, rows, temperature

### Community 57 - "Bias-corrected Mean"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Bias-corrected mean (TTA-style)

### Community 58 - "Ensemble 30/70"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Ensemble HCX 30% + Trend12 70%

### Community 59 - "Ensemble 50/50"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Ensemble HCX 50% + Trend12 50%

### Community 60 - "Ensemble 70/30"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Ensemble HCX 70% + Trend12 30%

### Community 61 - "Max Seed Aggregation"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Max seed (cherry-pick pessimistic)

### Community 62 - "Mean Aggregation"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Mean (current default, baseline)

### Community 63 - "Median Aggregation"
Cohesion: 0.50
Nodes (4): mae, preds, rmse, Median (Self-Consistency aggreg.)

### Community 64 - "Min Seed Aggregation"
Cohesion: 0.50
Nodes (4): Min seed (cherry-pick optimistic), mae, preds, rmse

### Community 65 - "Trend12 Baseline"
Cohesion: 0.50
Nodes (4): Trend12 baseline alone, mae, preds, rmse

### Community 66 - "Trimmed Mean Aggregation"
Cohesion: 0.50
Nodes (4): Trimmed mean (k=1, drop min+max), mae, preds, rmse

## Knowledge Gaps
- **569 isolated node(s):** `allow`, `ndarray`, `lessons`, `rows`, `n_per_cell` (+564 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LSTMConfig` connect `LSTM Training Pipeline` to `Dashboard & Results Loading`, `Causal Narrative Analysis`, `PyTorch Rolling Runners`, `LSTM Runner Pipeline`, `Training & Evaluation`, `Ablation Study Runner`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Why does `Report` connect `PDF Report Export` to `Dashboard & Results Loading`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **Why does `baselines` connect `Baseline Forecasts 2025` to `HCX Rolling Results`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Are the 22 inferred relationships involving `LSTMConfig` (e.g. with `device` and `ndarray`) actually correct?**
  _`LSTMConfig` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `AttentionLSTMForecaster` (e.g. with `device` and `ndarray`) actually correct?**
  _`AttentionLSTMForecaster` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `ZScoreNormalizer` (e.g. with `device` and `ndarray`) actually correct?**
  _`ZScoreNormalizer` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `Candidate` (e.g. with `Candidate` and `DataFrame`) actually correct?**
  _`Candidate` has 11 INFERRED edges - model-reasoned connections that need verification._