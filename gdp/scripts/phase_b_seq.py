"""Phase B — 공정 조건 시퀀스 AttnLSTM. max_lag=12 랙 피처를 (13개월×변수) 시퀀스로 reshape.
그들 backtest에 our_seq로 주입 → XGBoost와 동일 walk-forward/140행. lag-only 피처.
"""
import os, sys, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); sys.path.insert(0,"."); sys.path.insert(0,"/Users/user_1/vibe/bistro-lstm")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from lstm_model import AttentionLSTMForecaster
import gdp_nowcasting.ml_models as MM
import gdp_nowcasting.hyperparameter_tuning as HT
import gdp_nowcasting.pipeline_ai_ml as PL
from gdp_nowcasting.ml_models import SklearnNowcastModel, ModelSpec, get_linear_kernel_model_specs
from gdp_nowcasting.pipeline_ai_ml import AIRunContext, BacktestConfig, run_tree_ensemble_backtest
from gdp_nowcasting.ml_preprocessing import TabularFeatureConfig
from gdp_nowcasting.hyperparameter_tuning import TuningConfig
from pathlib import Path
import phase_b_harness as H

LAGRE = re.compile(r"^(.+)__lag(\d+)$")

class TorchSeq(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=150, seeds=1, hidden=32, dropout=0.3):
        self.epochs=epochs; self.seeds=seeds; self.hidden=hidden; self.dropout=dropout
    def _reshape(self, Xdf):
        vars_, lags_ = self.vars_, self.lags_
        n=len(Xdf); arr=np.zeros((n,len(lags_),len(vars_)),np.float32)
        for vi,v in enumerate(vars_):
            for li,lg in enumerate(lags_):
                col=f"{v}__lag{lg:02d}"
                if col in Xdf.columns: arr[:,li,vi]=Xdf[col].to_numpy(np.float32)
        return np.nan_to_num(arr)
    def fit(self, X, y):
        Xdf=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        parsed=[(m.group(1),int(m.group(2))) for c in Xdf.columns for m in [LAGRE.match(c)] if m]
        self.vars_=sorted(set(v for v,_ in parsed)); self.lags_=sorted(set(l for _,l in parsed))
        A=self._reshape(Xdf)  # (n, L, V)
        self.mu_=A.reshape(-1,A.shape[-1]).mean(0); self.sd_=A.reshape(-1,A.shape[-1]).std(0)+1e-6
        A=(A-self.mu_)/self.sd_
        y=np.asarray(y,np.float32); self.ym_=float(y.mean()); self.ys_=float(y.std()+1e-6)
        yn=((y-self.ym_)/self.ys_).reshape(-1,1)
        self.models_=[]
        for s in range(self.seeds):
            torch.manual_seed(s)
            m=AttentionLSTMForecaster(n_vars=len(self.vars_), d_model=16, hidden_dim=self.hidden,
                                      n_layers=1, n_heads=2, pred_len=1, dropout=self.dropout)
            opt=torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
            xt=torch.tensor(A); yt=torch.tensor(yn); m.train()
            for _ in range(self.epochs):
                opt.zero_grad(); loss=nn.functional.mse_loss(m(xt)["mu"],yt); loss.backward(); opt.step()
            m.eval(); self.models_.append(m)
        return self
    def predict(self, X):
        Xdf=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        A=(self._reshape(Xdf)-self.mu_)/self.sd_
        xt=torch.tensor(A)
        with torch.no_grad():
            p=np.mean([m(xt)["mu"].numpy() for m in self.models_],axis=0).ravel()
        return p*self.ys_+self.ym_

_orig=MM.create_model
def cm(model_name, model_params=None):
    spec=model_name if isinstance(model_name,ModelSpec) else None
    name=(spec.model_name if spec else model_name).lower()
    if name=="our_seq":
        params=dict(spec.model_params) if spec else dict(model_params or {})
        return SklearnNowcastModel(model_name="our_seq", estimator=TorchSeq(**params))
    return _orig(model_name, model_params)
MM.create_model=cm; PL.create_model=cm
_og=HT.get_default_parameter_grid
HT.get_default_parameter_grid=lambda n: {} if str(n).lower()=="our_seq" else _og(n)

repo=os.getcwd(); os.environ["GDP_NOWCAST_USE_S3"]="0"; os.environ["GDP_NOWCAST_LOCAL_PATH"]=repo
ctx=AIRunContext(specfile=str(Path(repo)/"data/meta/Spec_kim_34var_transformchange_named.xlsx"),
    input_data=str(Path(repo)/"data/vintages"), repo_root=Path(repo),
    forecasting_week="2026-01-23", sample_yrs=12, dfm_suffix="11")
fc=TabularFeatureConfig(lookback_months=13, add_missing_mask=False, add_release_lag=False,
                        rolling_windows=(), delta_lags=())
tuning=TuningConfig(min_train_size=8, max_candidates=10)
QSUB=os.environ.get("QSUB"); SEEDS=int(os.environ.get("SEEDS","1")); EP=int(os.environ.get("EPOCHS","150"))
quarters=QSUB.split(",") if QSUB else [str(p) for p in pd.period_range("2018Q1","2025Q4",freq="Q")]
specs=[ModelSpec("our_seq",{"epochs":EP,"seeds":SEEDS},model_id="our_seq")]
print(f"[run] our_seq(AttnLSTM, lag12) quarters={len(quarters)} seeds={SEEDS} ep={EP}",flush=True)
res=run_tree_ensemble_backtest(ctx, feature_config=fc, linear_specs=[], tree_specs=specs,
    backtest_config=BacktestConfig(tuning_mode="quarter_batch", quarter_batch_selection_mode="fvintage_window",
    min_train_rows=8, verbose=True), tuning_config=tuning,
    dfm_snapshot_root=Path(repo)/"output/model/DFM/11", gdp_release_file=str(Path(repo)/"data/GDP_releases.pkl"),
    quarters=quarters, news_config=None)
res.predictions.to_csv("output/csv/_phase_b_seq_predictions.csv",index=False)
sc=H.score(res.predictions)
print("\n=== our_seq (AttnLSTM 시퀀스, 공정) flash w[-19,-1] avg RMSE ===")
print(sc[sc.index.str.contains("our_seq")].to_string())
print("[참고] dfm 0.865 / our_mlp 1.023 / DFM+XGBoost 0.765")
print("PHASE_B_SEQ_DONE")
