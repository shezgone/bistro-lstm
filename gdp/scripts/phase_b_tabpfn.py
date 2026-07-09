"""Phase B: TabPFN(v2, Prior-Fitted Network)을 그들 backtest에 공정 주입.
XGBoost와 동일 (X,y)·140행·walk-forward. 소표본 특화 in-context 학습기 —
대상 데이터로 gradient 학습을 하지 않음 (Hoo et al. arXiv:2501.02945 참조).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pathlib import Path

import gdp_nowcasting.ml_models as MM
import gdp_nowcasting.hyperparameter_tuning as HT
import gdp_nowcasting.pipeline_ai_ml as PL
from gdp_nowcasting.ml_models import SklearnNowcastModel, ModelSpec
from gdp_nowcasting.pipeline_ai_ml import AIRunContext, BacktestConfig, run_tree_ensemble_backtest
from gdp_nowcasting.ml_preprocessing import TabularFeatureConfig
from gdp_nowcasting.hyperparameter_tuning import TuningConfig
import phase_b_harness as H
from tabpfn import TabPFNRegressor

N_EST = int(os.environ.get("TABPFN_NEST", "4"))

def make_est():
    return TabPFNRegressor(device="cpu", n_estimators=N_EST, random_state=0,
                           ignore_pretraining_limits=True)

_orig = MM.create_model
def cm(model_name, model_params=None):
    spec = model_name if isinstance(model_name, ModelSpec) else None
    name = (spec.model_name if spec else model_name).lower()
    if name == "our_tabpfn":
        return SklearnNowcastModel(model_name="our_tabpfn", estimator=make_est())
    return _orig(model_name, model_params)
MM.create_model = cm; PL.create_model = cm
_og = HT.get_default_parameter_grid
HT.get_default_parameter_grid = lambda n: {} if str(n).lower() == "our_tabpfn" else _og(n)

repo = os.getcwd(); os.environ["GDP_NOWCAST_USE_S3"] = "0"; os.environ["GDP_NOWCAST_LOCAL_PATH"] = repo
ctx = AIRunContext(specfile=str(Path(repo)/"data/meta/Spec_kim_34var_transformchange_named.xlsx"),
                   input_data=str(Path(repo)/"data/vintages"), repo_root=Path(repo),
                   forecasting_week="2026-01-23", sample_yrs=12, dfm_suffix="11")
fc = TabularFeatureConfig(lookback_months=1)      # phase_b_fair(our_mlp)·XGBoost와 동일
tuning = TuningConfig(min_train_size=8, max_candidates=10)
QSUB = os.environ.get("QSUB")
quarters = QSUB.split(",") if QSUB else [str(p) for p in pd.period_range("2018Q1", "2025Q4", freq="Q")]
print(f"[run] our_tabpfn (n_estimators={N_EST}) quarters={len(quarters)}", flush=True)

res = run_tree_ensemble_backtest(ctx, feature_config=fc, linear_specs=[],
    tree_specs=[ModelSpec("our_tabpfn", {}, model_id="our_tabpfn")],
    backtest_config=BacktestConfig(tuning_mode="quarter_batch",
        quarter_batch_selection_mode="fvintage_window", min_train_rows=8, verbose=True),
    tuning_config=tuning, dfm_snapshot_root=Path(repo)/"output/model/DFM/11",
    gdp_release_file=str(Path(repo)/"data/GDP_releases.pkl"), quarters=quarters, news_config=None)

res.predictions.to_csv("output/csv/_phase_b_tabpfn_predictions.csv", index=False)
sc = H.score(res.predictions)
print("\n=== our_tabpfn (공정 주입) flash w[-19,-1] avg RMSE ===")
print(sc[sc.index.str.contains("tabpfn")].to_string())
print("[참고] DFM+XGBoost 0.765 / DFM+our_mlp 0.820 / DFM 0.865 / 3-arm v2 0.722")
print("PHASE_B_TABPFN_DONE")
