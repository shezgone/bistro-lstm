"""Phase B 공정 재실험: 우리 신경망(torch MLP)을 그들 backtest에 model_spec으로 주입.
→ XGBoost와 동일 (X,y)·140행·walk-forward·튜닝·DFM+ML 앙상블 로직으로 학습/평가.
create_model monkeypatch로 'our_mlp' 지원. tree_specs에 추가.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import gdp_nowcasting.ml_models as MM
import gdp_nowcasting.hyperparameter_tuning as HT
import gdp_nowcasting.pipeline_ai_ml as PL
from gdp_nowcasting.ml_models import (SklearnNowcastModel, ModelSpec,
    get_linear_kernel_model_specs, get_tree_model_specs)
from gdp_nowcasting.pipeline_ai_ml import AIRunContext, BacktestConfig, NewsDecompositionConfig, run_tree_ensemble_backtest
from gdp_nowcasting.ml_preprocessing import TabularFeatureConfig
from gdp_nowcasting.hyperparameter_tuning import TuningConfig
import phase_b_harness as H


class TorchMLP(BaseEstimator, RegressorMixin):
    """소형 MLP (우리 신경망 접근). 입력은 파이프라인의 imputer+scaler 통과 후."""
    def __init__(self, hidden=32, epochs=300, lr=1e-3, wd=1e-3, dropout=0.2, seeds=3):
        self.hidden=hidden; self.epochs=epochs; self.lr=lr; self.wd=wd; self.dropout=dropout; self.seeds=seeds
    def fit(self, X, y):
        X=np.asarray(X,np.float32); y=np.asarray(y,np.float32).reshape(-1,1)
        self.ymu_=float(y.mean()); self.ysd_=float(y.std()+1e-6)
        yn=(y-self.ymu_)/self.ysd_
        self.models_=[]
        for s in range(self.seeds):
            torch.manual_seed(s)
            m=nn.Sequential(nn.Linear(X.shape[1],self.hidden), nn.ReLU(), nn.Dropout(self.dropout),
                            nn.Linear(self.hidden,self.hidden//2), nn.ReLU(), nn.Linear(self.hidden//2,1))
            opt=torch.optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.wd)
            xt=torch.tensor(X); yt=torch.tensor(yn)
            m.train()
            for _ in range(self.epochs):
                opt.zero_grad(); loss=nn.functional.mse_loss(m(xt),yt); loss.backward(); opt.step()
            m.eval(); self.models_.append(m)
        return self
    def predict(self, X):
        X=torch.tensor(np.asarray(X,np.float32))
        with torch.no_grad():
            p=np.mean([m(X).numpy() for m in self.models_],axis=0).ravel()
        return p*self.ysd_+self.ymu_


_orig_create = MM.create_model
def create_model_patched(model_name, model_params=None):
    spec = model_name if isinstance(model_name, ModelSpec) else None
    name = (spec.model_name if spec else model_name).lower()
    if name == "our_mlp":
        params = dict(spec.model_params) if spec else dict(model_params or {})
        est = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()), ("model", TorchMLP(**params))])
        return SklearnNowcastModel(model_name="our_mlp", estimator=est)
    return _orig_create(model_name, model_params)
MM.create_model = create_model_patched
PL.create_model = create_model_patched   # pipeline이 import한 바인딩도 교체

_orig_grid = HT.get_default_parameter_grid
def grid_patched(model_name):
    if str(model_name).lower() == "our_mlp":
        return {}   # 튜닝 없이 base params 1회
    return _orig_grid(model_name)
HT.get_default_parameter_grid = grid_patched

# ---- ctx ----
repo=os.getcwd(); os.environ["GDP_NOWCAST_USE_S3"]="0"; os.environ["GDP_NOWCAST_LOCAL_PATH"]=repo
from pathlib import Path
specfile=str(Path(repo)/"data/meta/Spec_kim_34var_transformchange_named.xlsx")
ctx=AIRunContext(specfile=specfile, input_data=str(Path(repo)/"data/vintages"), repo_root=Path(repo),
                 forecasting_week="2026-01-23", sample_yrs=12, dfm_suffix="11")
fc=TabularFeatureConfig(lookback_months=1)
tuning=TuningConfig(min_train_size=8, max_candidates=10)
news=NewsDecompositionConfig(latest_only=True, max_pairs_per_model=1)
dfm_root=Path(repo)/"output/model/DFM/11"
grel=str(Path(repo)/"data/GDP_releases.pkl")
QSUB=os.environ.get("QSUB")
quarters=[str(p) for p in pd.period_range("2018Q1","2025Q4",freq="Q")]
if QSUB: quarters=QSUB.split(",")

SEEDS=int(os.environ.get("MLP_SEEDS","1")); EPOCHS=int(os.environ.get("MLP_EPOCHS","150"))
# our_mlp만 학습 (다른 모델은 이미 정확 재현함 → 스킵해서 속도↑)
tree_specs=[ModelSpec("our_mlp", {"epochs":EPOCHS,"hidden":32,"seeds":SEEDS}, model_id="our_mlp")]
print(f"[run] our_mlp만 | quarters={len(quarters)} seeds={SEEDS} epochs={EPOCHS}", flush=True)
res=run_tree_ensemble_backtest(ctx, feature_config=fc, linear_specs=[],
     tree_specs=tree_specs, backtest_config=BacktestConfig(tuning_mode="quarter_batch",
     quarter_batch_selection_mode="fvintage_window", min_train_rows=8, verbose=True),
     tuning_config=tuning, dfm_snapshot_root=dfm_root, gdp_release_file=grel, quarters=quarters, news_config=None)

res.predictions.to_csv("output/csv/_phase_b_fair_predictions.csv", index=False)
mods=sorted(res.predictions.model_name.unique())
print("예측된 모형:", [m for m in mods if "our_mlp" in m or "mlp" in m])
# 채점: our_mlp 단독 + ensemble_dfm_our_mlp (있으면)
grid,_=H.load_grid()
sc=H.score(res.predictions)
print("\n=== 공정 재실험 결과 (our_mlp 관련, flash w[-19,-1] avg RMSE) ===")
print(sc[sc.index.str.contains("our_mlp")].to_string())
print("\n[참고] dfm 0.865 / xgboost 0.937 / ensemble_dfm_xgboost 0.765")
sc.to_csv("output/csv/_phase_b_fair_scores.csv")
print("PHASE_B_FAIR_DONE")
