from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess(
    test_size: float = 0.2,
    *,
    return_metadata: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | Dict[str, Any]:
    """Preprocess the merged dataset into train/test splits for classification.

    The target is the 10-day direction label created in `price_reaction.py`.
    """
    path = "data/processed/ml_merged.parquet"
    print(f"📂 Loading dataset from {path}")
    df = pd.read_parquet(path)

    if "label" not in df.columns:
        raise ValueError("Merged dataset must contain a 'label' column (10-day direction).")

    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["event_date"] = pd.to_datetime(df["event_date"])

    target_col = "label"
    leakage_cols = [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "overnight_gap",
        "return_1d_norm",
        "return_3d_norm",
        "return_5d_norm",
        "return_10d_norm",
        "abs_return_1d",
        "momentum_flag",
        "vol_norm",
        "actual",
        "estimate",
        "surprise",
        "surprise_percent",
        "surprise_pct",
        "eps_direction",
        "eps_change_qoq",
        "eps_change_yoy",
        "eps_diff",
        "revenue_diff",
        "eps_surprise_pct",
        "revenue_surprise_pct",
    ]

    exclude_cols = {"symbol", "event_date", target_col, *leakage_cols}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("\n🧾 Excluding columns:", sorted(exclude_cols))
    print("✅ Including features:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")

    df_sorted = df.sort_values("event_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    split_idx = max(1, min(len(df_sorted) - 1, split_idx))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    split_date = test_df["event_date"].min()
    print(f"\n⏱️ Time-based split with cutoff at {split_date.date() if pd.notna(split_date) else split_date}")
    print(f" - Train window: {train_df['event_date'].min().date()} → {train_df['event_date'].max().date()} ({len(train_df)} rows)")
    print(f" - Test window:  {test_df['event_date'].min().date()} → {test_df['event_date'].max().date()} ({len(test_df)} rows)")

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"\n🔢 Numeric columns: {len(numeric_cols)}")
    print(f"🔤 Categorical columns: {categorical_cols}")

    scaler = None
    numeric_impute_values = None
    if numeric_cols:
        numeric_impute_values = X_train[numeric_cols].median()
        X_train_numeric = X_train[numeric_cols].fillna(numeric_impute_values)
        X_test_numeric = X_test[numeric_cols].fillna(numeric_impute_values)
        scaler = StandardScaler()
        X_num_train = pd.DataFrame(
            scaler.fit_transform(X_train_numeric),
            columns=numeric_cols,
            index=X_train.index,
        )
        X_num_test = pd.DataFrame(
            scaler.transform(X_test_numeric),
            columns=numeric_cols,
            index=X_test.index,
        )
    else:
        X_num_train = pd.DataFrame(index=X_train.index)
        X_num_test = pd.DataFrame(index=X_test.index)

    encoder = None
    categorical_fill_value = None
    if categorical_cols:
        categorical_fill_value = "missing"
        X_train_categorical = X_train[categorical_cols].fillna(categorical_fill_value)
        X_test_categorical = X_test[categorical_cols].fillna(categorical_fill_value)
        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoder.fit(X_train_categorical)
        cat_feature_names = encoder.get_feature_names_out(categorical_cols)

        X_cat_train = pd.DataFrame(
            encoder.transform(X_train_categorical),
            columns=cat_feature_names,
            index=X_train.index,
        )
        X_cat_test = pd.DataFrame(
            encoder.transform(X_test_categorical),
            columns=cat_feature_names,
            index=X_test.index,
        )
    else:
        X_cat_train = pd.DataFrame(index=X_train.index)
        X_cat_test = pd.DataFrame(index=X_test.index)

    X_train_processed = pd.concat([X_num_train, X_cat_train], axis=1)
    X_test_processed = pd.concat([X_num_test, X_cat_test], axis=1)

    print("\n📊 Label distribution (train):", y_train.value_counts().to_dict())
    print("📊 Label distribution (test):", y_test.value_counts().to_dict())
    print(f"\n✅ Training feature matrix shape: {X_train_processed.shape}")
    print(f"✅ Test feature matrix shape: {X_test_processed.shape}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_processed.to_parquet(out_dir / "X_train.parquet")
    X_test_processed.to_parquet(out_dir / "X_test.parquet")
    y_train.to_frame(name=target_col).to_parquet(out_dir / "y_train.parquet")
    y_test.to_frame(name=target_col).to_parquet(out_dir / "y_test.parquet")

    artifacts_path = out_dir / "preprocess_artifacts.joblib"
    joblib.dump(
        {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "scaler": scaler,
            "encoder": encoder,
            "numeric_impute_values": numeric_impute_values,
            "categorical_fill_value": categorical_fill_value,
            "feature_columns": feature_cols,
        },
        artifacts_path,
    )

    print(
        f"💾 Saved preprocessed data to:\n"
        f" - {out_dir / 'X_train.parquet'}\n"
        f" - {out_dir / 'X_test.parquet'}\n"
        f" - {out_dir / 'y_train.parquet'}\n"
        f" - {out_dir / 'y_test.parquet'}\n"
        f" - {artifacts_path}"
    )

    outputs: Dict[str, Any] = {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
    }

    if return_metadata:
        outputs.update(
            {
                "train_df": train_df.copy(),
                "test_df": test_df.copy(),
                "train_indices": train_df.index.to_list(),
                "test_indices": test_df.index.to_list(),
                "feature_columns": feature_cols,
            }
        )
        return outputs

    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    preprocess()