"""Train LSTM with the report's optimal config (matches CPI_Forecasting_Benchmark_v1)."""
import os
import sys
import lstm_core
import train_and_evaluate as te

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

lstm_core.SEQ_LEN = 36
te.SEQ_LEN = 36
te.MACRO_CSV = os.path.join(DATA_DIR, "macro_panel_optimal18.csv")

_orig_LSTMConfig = lstm_core.LSTMConfig

def OptimalLSTMConfig(variates, **kw):
    kw.setdefault("seq_len", 36)
    kw.setdefault("hidden_dim", 128)
    kw.setdefault("n_layers", 2)
    kw.setdefault("d_model", 64)
    kw.setdefault("n_heads", 4)
    kw.setdefault("dropout", 0.2)
    return _orig_LSTMConfig(variates=variates, **kw)

lstm_core.LSTMConfig = OptimalLSTMConfig
te.LSTMConfig = OptimalLSTMConfig

if __name__ == "__main__":
    print("=" * 60)
    print("Training with REPORT-OPTIMAL config:")
    print(f"  CSV: {te.MACRO_CSV}")
    print(f"  SEQ_LEN: {te.SEQ_LEN}")
    print(f"  hidden=128, n_layers=2, d_model=64, n_heads=4")
    print("=" * 60, flush=True)

    sys.argv = ["train_optimal.py", "--epochs", "300", "--patience", "20"]
    te.main()
