"""Phase B: NCDENow-style 근사 (DFM 대용 PCA 요인 경로 + Neural CDE, 오일러 이산화)를
그들 backtest에 공정 주입. Lim·Choi et al. (CIKM 2024, arXiv:2409.08732)의 구조를
소형화한 프로토타입: 요인 추출(PCA k=4) → 잠재 CDE dz = f_θ(z)·dX → 회귀.

phase_b_seq.py(our_seq)와 동일한 lag12 시퀀스 하네스 — AttnLSTM과 직접 비교 가능.
"""
import os, sys, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA

import gdp_nowcasting.ml_models as MM
import gdp_nowcasting.hyperparameter_tuning as HT
import gdp_nowcasting.pipeline_ai_ml as PL
from gdp_nowcasting.ml_models import SklearnNowcastModel, ModelSpec
from gdp_nowcasting.pipeline_ai_ml import AIRunContext, BacktestConfig, run_tree_ensemble_backtest
from gdp_nowcasting.ml_preprocessing import TabularFeatureConfig
from gdp_nowcasting.hyperparameter_tuning import TuningConfig
import phase_b_harness as H

LAGRE = re.compile(r"^(.+)__lag(\d+)$")

class CDEFunc(nn.Module):
    """f_θ: z(h) → h×(k+1) 행렬 (경로 채널 k개 + 시간 채널 1개)."""
    def __init__(self, h, k):
        super().__init__()
        self.h, self.k = h, k
        self.net = nn.Sequential(nn.Linear(h, 32), nn.Tanh(), nn.Linear(32, h * (k + 1)))
    def forward(self, z):
        return self.net(z).view(-1, self.h, self.k + 1).tanh()

class NCDELite(BaseEstimator, RegressorMixin):
    """PCA 요인 경로 위의 discrete-euler Neural CDE: z_{t+1} = z_t + f_θ(z_t)·ΔX_t"""
    def __init__(self, k=4, hidden=16, epochs=200, seeds=1, lr=1e-3, wd=1e-3):
        self.k=k; self.hidden=hidden; self.epochs=epochs; self.seeds=seeds; self.lr=lr; self.wd=wd
    def _reshape(self, Xdf):
        n=len(Xdf); arr=np.zeros((n,len(self.lags_),len(self.vars_)),np.float32)
        for vi,v in enumerate(self.vars_):
            for li,lg in enumerate(self.lags_):
                c=f"{v}__lag{lg:02d}"
                if c in Xdf.columns: arr[:,li,vi]=Xdf[c].to_numpy(np.float32)
        return np.nan_to_num(arr)
    def _factors(self, A):
        """(n,L,V) → PCA 요인 경로 (n,L,k) + 시간 채널 → (n,L,k+1)"""
        n,L,V=A.shape
        F=self.pca_.transform(((A-self.mu_)/self.sd_).reshape(-1,V)).reshape(n,L,self.k)
        t=np.broadcast_to(np.linspace(0,1,L,dtype=np.float32)[None,:,None],(n,L,1))
        return np.concatenate([F.astype(np.float32),t],axis=2)
    def fit(self, X, y):
        Xdf=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        parsed=[(m.group(1),int(m.group(2))) for c in Xdf.columns for m in [LAGRE.match(c)] if m]
        self.vars_=sorted(set(v for v,_ in parsed)); self.lags_=sorted(set(l for _,l in parsed))
        A=self._reshape(Xdf); n,L,V=A.shape
        flat=A.reshape(-1,V); self.mu_=flat.mean(0); self.sd_=flat.std(0)+1e-6
        self.pca_=PCA(n_components=self.k, random_state=0).fit(((A-self.mu_)/self.sd_).reshape(-1,V))
        P=torch.tensor(self._factors(A))                       # (n,L,k+1)
        y=np.asarray(y,np.float32); self.ym_=float(y.mean()); self.ys_=float(y.std()+1e-6)
        yt=torch.tensor(((y-self.ym_)/self.ys_).reshape(-1,1))
        self.models_=[]
        for s in range(self.seeds):
            torch.manual_seed(s)
            f=CDEFunc(self.hidden,self.k); z0=nn.Linear(self.k+1,self.hidden); ro=nn.Linear(self.hidden,1)
            params=list(f.parameters())+list(z0.parameters())+list(ro.parameters())
            opt=torch.optim.Adam(params,lr=self.lr,weight_decay=self.wd)
            dX=P[:,1:,:]-P[:,:-1,:]                            # (n,L-1,k+1)
            for _ in range(self.epochs):
                opt.zero_grad()
                z=z0(P[:,0,:])
                for t in range(dX.shape[1]):
                    z=z+torch.einsum("bhc,bc->bh", f(z), dX[:,t,:])
                loss=nn.functional.mse_loss(ro(z),yt); loss.backward(); opt.step()
            self.models_.append((f,z0,ro))
        return self
    def predict(self, X):
        Xdf=X if isinstance(X,pd.DataFrame) else pd.DataFrame(X)
        P=torch.tensor(self._factors(self._reshape(Xdf)))
        dX=P[:,1:,:]-P[:,:-1,:]
        outs=[]
        with torch.no_grad():
            for f,z0,ro in self.models_:
                z=z0(P[:,0,:])
                for t in range(dX.shape[1]):
                    z=z+torch.einsum("bhc,bc->bh", f(z), dX[:,t,:])
                outs.append(ro(z).numpy().ravel())
        return np.mean(outs,axis=0)*self.ys_+self.ym_

_orig=MM.create_model
def cm(model_name, model_params=None):
    spec=model_name if isinstance(model_name,ModelSpec) else None
    name=(spec.model_name if spec else model_name).lower()
    if name=="our_ncde":
        params=dict(spec.model_params) if spec else dict(model_params or {})
        return SklearnNowcastModel(model_name="our_ncde", estimator=NCDELite(**params))
    return _orig(model_name, model_params)
MM.create_model=cm; PL.create_model=cm
_og=HT.get_default_parameter_grid
HT.get_default_parameter_grid=lambda n: {} if str(n).lower()=="our_ncde" else _og(n)

repo=os.getcwd(); os.environ["GDP_NOWCAST_USE_S3"]="0"; os.environ["GDP_NOWCAST_LOCAL_PATH"]=repo
ctx=AIRunContext(specfile=str(Path(repo)/"data/meta/Spec_kim_34var_transformchange_named.xlsx"),
    input_data=str(Path(repo)/"data/vintages"), repo_root=Path(repo),
    forecasting_week="2026-01-23", sample_yrs=12, dfm_suffix="11")
fc=TabularFeatureConfig(lookback_months=13, add_missing_mask=False, add_release_lag=False,
                        rolling_windows=(), delta_lags=())
tuning=TuningConfig(min_train_size=8, max_candidates=10)
QSUB=os.environ.get("QSUB"); SEEDS=int(os.environ.get("SEEDS","1")); EP=int(os.environ.get("EPOCHS","200"))
quarters=QSUB.split(",") if QSUB else [str(p) for p in pd.period_range("2018Q1","2025Q4",freq="Q")]
print(f"[run] our_ncde (PCA4+euler-CDE, lag12) quarters={len(quarters)} seeds={SEEDS} ep={EP}",flush=True)
res=run_tree_ensemble_backtest(ctx, feature_config=fc, linear_specs=[],
    tree_specs=[ModelSpec("our_ncde",{"epochs":EP,"seeds":SEEDS},model_id="our_ncde")],
    backtest_config=BacktestConfig(tuning_mode="quarter_batch", quarter_batch_selection_mode="fvintage_window",
    min_train_rows=8, verbose=True), tuning_config=tuning,
    dfm_snapshot_root=Path(repo)/"output/model/DFM/11", gdp_release_file=str(Path(repo)/"data/GDP_releases.pkl"),
    quarters=quarters, news_config=None)
res.predictions.to_csv("output/csv/_phase_b_ncde_predictions.csv",index=False)
sc=H.score(res.predictions)
print("\n=== our_ncde (NCDENow-style 근사, 공정) flash w[-19,-1] avg RMSE ===")
print(sc[sc.index.str.contains("ncde")].to_string())
print("[참고] our_seq(AttnLSTM) 1.277 / DFM+our_seq 0.975 / DFM+XGBoost 0.765 / DFM 0.865")
print("PHASE_B_NCDE_DONE")
