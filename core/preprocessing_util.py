"""
Preprocessing Utility for BISTRO-LSTM
=====================================
월별 매크로 패널 데이터를 LSTM 학습용으로 전처리.
- Z-score 정규화 (학습 세트 기준)
- Train/Test 분리
- 슬라이딩 윈도우 시퀀스 생성
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def load_macro_panel(
    csv_path: str,
    target_col: str = "CPI_KR_YoY",
    selected_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    macro_panel.csv 로딩.

    Parameters
    ----------
    csv_path : CSV 파일 경로
    target_col : 타겟 변수명
    selected_vars : 선택할 변수 리스트 (None이면 전체)

    Returns
    -------
    pd.DataFrame (PeriodIndex, columns = [target_col, ...covariates])
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.PeriodIndex(df.index, freq="M")

    if selected_vars is not None:
        # target은 항상 포함
        cols = [target_col] + [v for v in selected_vars if v != target_col]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    else:
        # target을 첫 번째 열로 이동
        cols = [target_col] + [c for c in df.columns if c != target_col]
        df = df[cols]

    # 결측값 처리: forward-fill → backward-fill
    df = df.ffill().bfill()

    # inf 제거
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df


def split_train_test(
    df: pd.DataFrame,
    train_end: str = "2022-12",
    test_start: str = "2023-01",
    test_end: str = "2023-12",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/Test 분리."""
    train = df[df.index <= pd.Period(train_end, freq="M")]
    test = df[
        (df.index >= pd.Period(test_start, freq="M")) &
        (df.index <= pd.Period(test_end, freq="M"))
    ]
    return train, test


class ZScoreNormalizer:
    """
    Per-variable z-score 정규화.
    학습 세트 기준으로 mean/std 계산, 테스트 세트에 동일 적용.
    """

    def __init__(self):
        self.mean_: Optional[pd.Series] = None
        self.std_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "ZScoreNormalizer":
        self.mean_ = df.mean()
        self.std_ = df.std().replace(0, 1)  # 0으로 나누기 방지
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean_) / self.std_

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df * self.std_ + self.mean_

    def inverse_transform_target(
        self,
        values: np.ndarray,
        target_col: str = "CPI_KR_YoY",
    ) -> np.ndarray:
        """타겟 변수만 역변환."""
        return values * self.std_[target_col] + self.mean_[target_col]


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    target_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    슬라이딩 윈도우로 학습용 시퀀스 생성.

    Parameters
    ----------
    data : (T, n_vars) 정규화된 데이터
    seq_len : 입력 시퀀스 길이
    pred_len : 예측 길이
    target_idx : 타겟 변수 인덱스

    Returns
    -------
    X : (n_samples, seq_len, n_vars) 입력 시퀀스
    y : (n_samples, pred_len) 타겟 시퀀스
    """
    T = len(data)
    n_samples = T - seq_len - pred_len + 1

    if n_samples <= 0:
        raise ValueError(
            f"데이터 길이({T})가 seq_len({seq_len}) + pred_len({pred_len})보다 작습니다."
        )

    X = np.zeros((n_samples, seq_len, data.shape[1]))
    y = np.zeros((n_samples, pred_len))

    for i in range(n_samples):
        X[i] = data[i:i + seq_len]
        y[i] = data[i + seq_len:i + seq_len + pred_len, target_idx]

    return X, y


def prepare_walk_forward_splits(
    df: pd.DataFrame,
    normalizer_class: type = ZScoreNormalizer,
    seq_len: int = 120,
    pred_len: int = 12,
    target_idx: int = 0,
    val_years: List[int] = None,
) -> List[Dict]:
    """
    Walk-forward CV용 데이터 분할.

    Parameters
    ----------
    df : 전체 학습 데이터 (PeriodIndex)
    normalizer_class : 정규화 클래스
    seq_len : 입력 시퀀스 길이
    pred_len : 예측 길이
    target_idx : 타겟 변수 인덱스
    val_years : 검증 연도 리스트 (default [2018, 2019, 2020, 2021, 2022])

    Returns
    -------
    List of dict, each with keys:
        fold, train_X, train_y, val_X, val_y, normalizer, train_end, val_year
    """
    if val_years is None:
        val_years = [2018, 2019, 2020, 2021, 2022]

    splits = []

    for fold_idx, val_year in enumerate(val_years):
        train_end = f"{val_year - 1}-12"
        val_start = f"{val_year}-01"
        val_end = f"{val_year}-12"

        train_df = df[df.index <= pd.Period(train_end, freq="M")]
        val_df = df[
            (df.index >= pd.Period(val_start, freq="M")) &
            (df.index <= pd.Period(val_end, freq="M"))
        ]

        if len(train_df) < seq_len + pred_len:
            continue

        # 정규화 (학습 세트 기준)
        norm = normalizer_class()
        train_normed = norm.fit_transform(train_df).values
        val_normed = norm.transform(val_df).values

        # 학습 시퀀스 생성
        train_X, train_y = create_sequences(
            train_normed, seq_len, pred_len, target_idx
        )

        # 검증: 학습 데이터 끝 seq_len개월 + val 데이터로 시퀀스 생성
        combined = np.vstack([train_normed[-seq_len:], val_normed])
        if len(combined) >= seq_len + pred_len:
            val_X, val_y = create_sequences(
                combined, seq_len, pred_len, target_idx
            )
        else:
            continue

        splits.append({
            "fold": fold_idx,
            "train_X": train_X,
            "train_y": train_y,
            "val_X": val_X,
            "val_y": val_y,
            "normalizer": norm,
            "train_end": train_end,
            "val_year": val_year,
        })

    return splits
