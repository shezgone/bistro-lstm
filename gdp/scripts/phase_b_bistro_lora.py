"""실험 B: BISTRO LoRA 적응 — C2-LoRA(v1)와 동일 레시피, walk-forward 평가 (2026-08-26).

C2 대조 원칙: 학습 데이터(빈티지 경로 v1: 분기당 완성 경로 1개, flash 앵커, est_flag 채널),
사전 등록 설정(lr 1e-4, 500 steps, batch 16), 연 1회 fresh 재적응(2021~2025)을 전부 동일하게.
다른 것은 모델뿐 — BISTRO(Moirai-base 91M, BIS WP 1337 공개 체크포인트).

구현: uni2ts에 내장 LoRA 없음 → 수동 LoRALinear 래퍼로 attention q/k/v/out_proj(12층)에
rank 8 LoRA 주입(peft는 이 venv의 torch 2.4.1과 transformers 충돌로 배제),
MoiraiForecast._convert로 패킹 재사용, PackedNLLLoss로 자체 학습 루프.
patch_size는 train/predict 모두 8로 고정 (auto는 학습과 불일치 위험) — 공정 비교용
zero-shot patch=8 재채점은 phase_b_moirai_c2p.py PATCH=8로 별도 산출.

env: YEARS / NUM_STEPS / LR / SEED / TAG / RANK
결과 (2026-08-26, 평가창 2021~2025 20분기, seed 7):
  BISTRO-LoRA 0.5490 — zs(p8) 0.5957 대비 -7.8%, zs(auto) 0.6072 대비 -9.6%
  **C2f-LoRA(0.571, 3-seed)를 역전 — LoRA 후 두 체크포인트 순위가 뒤집힘**
  흡수율 +10.1%(zs-p8) → +1.4%: 역행이 거의 사라짐 (C2-LoRA는 흡수 불변과 대조)
  연도별: 2021 -3% / 2022 -14% / 2023 -5% / 2024 -7% / **2025(준청정) 0.546 — XGB(0.567)를 앞섬**
    → C2-LoRA의 회고 편중과 달리 개선이 전 연도 고름 + 깨끗한 구간에서 최강
  슬롯: 조기=BLo 단독 0.5093 / (CLo+BLo)/2 0.5071 vs XGB 단독 0.5178
  단서: patch 고정(8) 효과가 일부 포함(zs auto→p8만으로 -1.9%), 오차상관 0.914로 슬롯 이득 얕음,
    seed 1개(확장 필요), DM 미검정.
해석(잠정): any-variate 계열은 '고칠 것'(흡수 부재)이 명확해 LoRA의 개선 여지가 컸고,
  C2는 이미 흡수하는 상태라 개선 여지가 작았다 — "적응 이득은 출발점의 결함 크기에 비례".

seed 확장 (2026-08-27, s11 0.5597 / s23 0.6021):
  3-seed 평균 0.5703 ± 0.0281 — C2f-LoRA(0.5711 ± 0.0016)와 평균 동률, 분산 17배.
  → 8/26의 "역전" 주장 철회: seed 7이 행운의 추첨이었음. 흡수율도 seed별 +1.4~+5.7% 산포.
  단, 3-seed 예측 평균(seed 앙상블): BISTRO-LoRA 0.5599(흡수 +3.2%, 2025 0.556) —
  분산을 평균으로 흡수하면 C2f-LoRA 앙상블(0.5641)보다 근소 우위 + 2025 XGB(0.567) 상회 유지.
교훈: 수동 LoRA(kaiming 초기화 + 소표본)는 seed 분산이 크다 — 단일 seed 결과 절대 금지,
  실무 형태는 "3-seed 예측 평균"이 정직한 대표값.

실행: /Users/user/vibe/bistro-lstm/.venv-moirai/bin/python phase_b_bistro_lora.py
"""
import os, sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.loss.packed import PackedNLLLoss
import torch.nn as nn
import phase_b_harness as H
import fast_signals as FS

BISTRO_CKPT = "/Users/user/vibe/bistro-bis/bistro-finetuned"
COVARS = ["I_m", "I_s", "M_s", "S_es", "S_cb", "S_mo", "B_bx", "B_bi", "R_s", "M_fi"]
FASTC = ["kospi_mret", "krw_mret", "esi_raw", "esi_mom"]
PLEN = 6
CTX = 128                       # 8의 배수 고정 (패치 정렬)
PATCH = 8
NUM_STEPS = int(os.environ.get("NUM_STEPS", "500"))
LR = float(os.environ.get("LR", "1e-4"))
SEED = int(os.environ.get("SEED", "7"))
RANK = int(os.environ.get("RANK", "8"))
BATCH = 16
TAG = os.environ.get("TAG", "our_bistro_lora")
YEARS = [int(y) for y in os.environ.get("YEARS", "2021,2022,2023,2024,2025").split(",")]

grid, _ = H.load_grid()
g = grid.copy(); g["vintage"] = pd.to_datetime(g["vintage"]).dt.strftime("%Y-%m-%d")
FLASH = g[["tq", "flash"]].drop_duplicates().set_index("tq")["flash"].to_dict()
ALL_Q = sorted(g.tq.unique(), key=lambda x: pd.Period(x, "Q"))
loss_fn = PackedNLLLoss()

def load_panel(tq, vintage):
    files = sorted(glob.glob(f"output/model/DFM/11/{tq}/*.csv"))
    cands = [f for f in files if os.path.basename(f)[:-4] <= vintage]
    if not cands: return None
    df = pd.read_csv(cands[-1])
    dcol = df.columns[0]; df[dcol] = pd.to_datetime(df[dcol])
    df = df.rename(columns={dcol: "date"}).set_index("date")
    return df[["N_gdp"] + [c for c in COVARS if c in df.columns]].astype(np.float32)

def with_fast_and_mask(panel, vintage):
    fs = FS.monthly_covariates(panel.index, vintage)
    panel = panel.copy()
    for c in FASTC:
        panel[c] = fs[c].values.astype(np.float32)
    vm = pd.Timestamp(vintage) + pd.offsets.MonthEnd(0)
    est_cut = vm - pd.offsets.MonthEnd(1)
    panel["est_flag"] = (panel.index > est_cut).astype(np.float32)
    return panel

def qend_of(tq):
    return pd.Period(tq, "Q").end_time.normalize().replace(day=1) + pd.offsets.MonthEnd(0)

def anchor_flashes(panel, upto_tq):
    panel = panel.copy()
    for q in ALL_Q:
        if pd.Period(q, "Q") > pd.Period(upto_tq, "Q"): break
        qe = qend_of(q)
        if qe in panel.index and q in FLASH and pd.notna(FLASH[q]):
            panel.loc[qe, "N_gdp"] = np.float32(FLASH[q])
    return panel

def build_training_paths(first_vintage_of_year):
    """v1 레시피: 분기당 완성 경로 1개 (C2와 동일)."""
    paths = []
    fv = pd.Timestamp(first_vintage_of_year)
    for q in ALL_Q:
        if qend_of(q) + pd.Timedelta(weeks=6) >= fv: continue
        sub = g[g.tq == q].sort_values("week_idx")
        if sub.empty: continue
        v_last = sub.vintage.iloc[-1]
        p = load_panel(q, v_last)
        if p is None: continue
        p = with_fast_and_mask(p, v_last)
        p = p[p.index <= qend_of(q)]
        p = anchor_flashes(p, q)
        if len(p) < CTX // 2 + PLEN: continue
        paths.append(p)
    return paths

COVCOLS = None
def path_to_tensors(p):
    """경로 → (past_target, past_feat, future_target). past는 CTX로 좌측 패딩/절단."""
    global COVCOLS
    if COVCOLS is None:
        COVCOLS = [c for c in p.columns if c != "N_gdp"]
    tgt = p["N_gdp"].to_numpy(np.float32)
    cov = p[COVCOLS].to_numpy(np.float32)
    past_t, fut_t = tgt[:-PLEN], tgt[-PLEN:]
    past_c = cov[:-PLEN]
    T = len(past_t)
    if T >= CTX:
        past_t, past_c = past_t[-CTX:], past_c[-CTX:]
        pad = 0
    else:
        pad = CTX - T
        past_t = np.concatenate([np.zeros(pad, np.float32), past_t])
        past_c = np.concatenate([np.zeros((pad, past_c.shape[1]), np.float32), past_c])
    return past_t, past_c, fut_t, pad

def make_forecaster(module):
    return MoiraiForecast(module=module, prediction_length=PLEN, context_length=CTX,
                          patch_size=PATCH, num_samples=100, target_dim=1,
                          feat_dynamic_real_dim=0,
                          past_feat_dynamic_real_dim=len(COVCOLS))

class LoRALinear(nn.Module):
    """수동 LoRA: y = Wx + (alpha/r)·B(Ax). 원 가중치 동결."""
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for prm in self.base.parameters(): prm.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

def inject_lora(module, r, alpha, targets=("q_proj", "k_proj", "v_proj", "out_proj")):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r, alpha))
        else:
            inject_lora(child, r, alpha, targets)
    return module

def train_lora(paths, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    module = MoiraiModule.from_pretrained(BISTRO_CKPT)
    for prm in module.parameters(): prm.requires_grad = False
    module = inject_lora(module, RANK, RANK * 2)
    n_tr = sum(prm.numel() for prm in module.parameters() if prm.requires_grad)
    fc = make_forecaster(module)
    opt = torch.optim.AdamW([prm for prm in module.parameters() if prm.requires_grad], lr=LR)
    tensors = [path_to_tensors(p) for p in paths]
    module.train()
    for step in range(NUM_STEPS):
        idx = np.random.choice(len(tensors), size=min(BATCH, len(tensors)), replace=len(tensors) < BATCH)
        pt = torch.tensor(np.stack([tensors[i][0] for i in idx]))[..., None]        # (B,CTX,1)
        pc = torch.tensor(np.stack([tensors[i][1] for i in idx]))                    # (B,CTX,K)
        ft = torch.tensor(np.stack([tensors[i][2] for i in idx]))[..., None]        # (B,PLEN,1)
        pads = np.stack([tensors[i][3] for i in idx])
        is_pad = torch.zeros(pt.shape[:2], dtype=torch.bool)
        for bi, pd_ in enumerate(pads):
            if pd_ > 0: is_pad[bi, :pd_] = True
        obs_p = ~is_pad[..., None]
        target, obs, sid, tid, vid, pmask = fc._convert(
            PATCH, past_target=pt, past_observed_target=obs_p, past_is_pad=is_pad,
            future_target=ft, future_observed_target=torch.ones_like(ft, dtype=torch.bool),
            future_is_pad=torch.zeros(ft.shape[:2], dtype=torch.bool),
            past_feat_dynamic_real=pc,
            past_observed_feat_dynamic_real=torch.ones_like(pc, dtype=torch.bool))
        distr = module(target, obs, sid, tid, vid, pmask,
                       torch.full_like(sid, PATCH))
        loss = loss_fn(distr, target, pmask, obs, sid, vid)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{NUM_STEPS} loss {float(loss):.4f}", flush=True)
    module.eval()
    print(f"  [적응 완료] LoRA 파라미터 {n_tr/1e3:.0f}K / 전체 91.4M", flush=True)
    return module

def moirai_predict(fc_pred, hist, h):
    from gluonts.dataset.common import ListDataset
    hist = hist.tail(200)
    covcols = [c for c in hist.columns if c != "N_gdp"]
    pred = fc_pred.create_predictor(batch_size=1, device="cpu")
    ds = ListDataset([{"target": hist["N_gdp"].values.astype(np.float32),
                       "start": pd.Period(hist.index[0], freq="M"),
                       "past_feat_dynamic_real": hist[covcols].values.T.astype(np.float32)}],
                     freq="M", one_dim_target=True)
    fcst = list(pred.predict(ds))[0]
    return float(np.median(fcst.samples[:, h - 1]))

rows = []
for year in YEARS:
    sub0 = g[g.tq == f"{year}Q1"].sort_values("week_idx")
    fv = sub0.vintage.iloc[0]
    paths = build_training_paths(fv)
    _ = path_to_tensors(paths[0])           # COVCOLS 초기화
    print(f"\n=== {year}: BISTRO LoRA 적응 (학습 분기 {len(paths)}개, 기준 {fv}) ===", flush=True)
    module = train_lora(paths, SEED)
    for tq in [q for q in ALL_Q if q.startswith(str(year))]:
        sub = g[g.tq == tq].drop_duplicates(["vintage", "week_idx"]).sort_values("week_idx")
        qe = qend_of(tq)
        for r in sub.itertuples(index=False):
            yp = np.nan
            panel = load_panel(tq, r.vintage)
            if panel is not None:
                panel = with_fast_and_mask(panel, r.vintage)
                edge = pd.Timestamp(r.vintage) + pd.offsets.MonthEnd(0)
                hist = panel[panel.index <= edge]
                hist = anchor_flashes(hist, tq)
                h = int((qe.to_period("M") - hist.index[-1].to_period("M")).n) if len(hist) else 99
                if h < 1:
                    yp = float(panel.loc[qe, "N_gdp"]) if qe in panel.index else np.nan
                elif h <= PLEN and len(hist) >= 60:
                    try:
                        fc_pred = MoiraiForecast(module=module, prediction_length=h, context_length=len(hist.tail(200)),
                                                 patch_size=PATCH, num_samples=100, target_dim=1,
                                                 feat_dynamic_real_dim=0,
                                                 past_feat_dynamic_real_dim=len([c for c in hist.columns if c != "N_gdp"]))
                        yp = moirai_predict(fc_pred, hist, h)
                    except Exception as e:
                        print(f"  predict err {tq} {r.vintage}: {type(e).__name__}: {e}", flush=True)
            rows.append({"tq": tq, "vintage": r.vintage, "week_idx": r.week_idx,
                         "flash": r.flash, "model_name": TAG, "y_pred": yp})
        print(f"[{tq}] done", flush=True)
    del module

pred = pd.DataFrame(rows)
pred.to_csv(f"output/csv/_phase_b_{TAG}_predictions.csv", index=False)
print(f"\n=== {TAG} flash w[-19,-1] avg RMSE (평가 연도: {YEARS}) ===")
print(H.score(pred).to_string())
print("PHASE_B_BISTRO_LORA_DONE")
