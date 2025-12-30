import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import joblib


@dataclass
class Config:
    threshold: float = 0.45
    random_state: int = 42


def load_and_prepare_xlsx(xlsx_path: str, sheet_name: str | int = 0) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    required = {"InvoiceNo", "StockCode", "Quantity", "UnitPrice", "CustomerID", "InvoiceDate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar var: {missing}. Mevcut kolonlar: {list(df.columns)}")

    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str)

    # Negatif/iade yok: sadece pozitif
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["LineAmount"] = df["Quantity"] * df["UnitPrice"]

    gcols = ["InvoiceNo", "CustomerID"]
    agg = df.groupby(gcols).agg(
        num_lines=("StockCode", "count"),
        unique_products=("StockCode", "nunique"),
        total_qty=("Quantity", "sum"),
        total_amount=("LineAmount", "sum"),
        avg_unit_price=("UnitPrice", "mean"),
        order_time=("InvoiceDate", "min"),
    ).reset_index()

    agg["order_hour"] = agg["order_time"].dt.hour.astype(int)
    agg["is_night"] = ((agg["order_hour"] >= 0) & (agg["order_hour"] <= 5)).astype(int)

    cust_stats = agg.groupby("CustomerID").agg(
        cust_mean_amount=("total_amount", "mean"),
        cust_std_amount=("total_amount", "std"),
        cust_median_amount=("total_amount", "median"),
        cust_order_count=("InvoiceNo", "nunique"),
    ).reset_index()

    agg = agg.merge(cust_stats, on="CustomerID", how="left")
    agg["cust_std_amount"] = agg["cust_std_amount"].fillna(0.0)

    eps = 1e-6
    agg["cust_amount_z"] = (agg["total_amount"] - agg["cust_mean_amount"]) / (agg["cust_std_amount"] + eps)

    agg["log_total_amount"] = np.log1p(agg["total_amount"])
    agg["log_total_qty"] = np.log1p(agg["total_qty"])

    return agg


def make_risk_label(df_orders: pd.DataFrame) -> pd.DataFrame:
    d = df_orders.copy()

    amt_p95 = d["total_amount"].quantile(0.95)
    qty_p95 = d["total_qty"].quantile(0.95)
    uniq_p90 = d["unique_products"].quantile(0.90)

    rule_high_amount = (d["total_amount"] >= amt_p95)
    rule_high_qty = (d["total_qty"] >= qty_p95)
    rule_many_unique = (d["unique_products"] >= uniq_p90)
    rule_night = (d["is_night"] == 1)
    rule_customer_anom = (d["cust_amount_z"] >= 2.0)

    d["risk_score"] = (
        rule_high_amount.astype(int) * 2
        + rule_customer_anom.astype(int) * 2
        + rule_high_qty.astype(int) * 1
        + rule_many_unique.astype(int) * 1
        + rule_night.astype(int) * 1
    )

    d["risk_label"] = (d["risk_score"] >= 4).astype(int)
    return d


def build_models():
    # RF yok!
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=400, random_state=42),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, default="data/Online Retail.xlsx")
    parser.add_argument("--sheet", type=str, default="0")  # 0 veya "Online Retail"
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--best_name", type=str, default="best_model.joblib")
    parser.add_argument("--select", type=str, default="recall", choices=["recall", "f1"],
                        help="Best modeli neye göre seçeceğiz? recall: riskliyi kaçırma, f1: denge")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = Config()

    sheet_name: str | int = int(args.sheet) if args.sheet.isdigit() else args.sheet

    orders = load_and_prepare_xlsx(args.xlsx, sheet_name=sheet_name)
    orders = make_risk_label(orders)

    feature_cols = [
        "num_lines", "unique_products", "total_qty", "total_amount", "avg_unit_price",
        "order_hour", "is_night",
        "cust_mean_amount", "cust_median_amount", "cust_order_count", "cust_amount_z",
        "log_total_amount", "log_total_qty",
    ]

    df = orders.dropna(subset=feature_cols + ["risk_label"]).copy()
    X = df[feature_cols]
    y = df["risk_label"].astype(int)

    print(f"[INFO] Orders: {len(df)} | Risk rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=cfg.random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop",
    )

    models = build_models()

    best_pipe = None
    best_name = None
    best_score = -1.0

    for name, clf in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (proba >= cfg.threshold).astype(int)

        print("\n" + "=" * 60)
        print(f"[MODEL] {name}")
        print(confusion_matrix(y_test, y_pred))
        rep = classification_report(y_test, y_pred, digits=4, output_dict=True)
        print(classification_report(y_test, y_pred, digits=4))

        try:
            auc = roc_auc_score(y_test, proba)
            print(f"AUC: {auc:.4f}")
        except Exception:
            pass

        recall_1 = rep["1"]["recall"] if "1" in rep else 0.0
        f1_1 = rep["1"]["f1-score"] if "1" in rep else 0.0

        if args.select == "recall":
            score = recall_1
            print(f"Select metric: recall(1) = {score:.4f}")
        else:
            score = f1_1
            print(f"Select metric: f1(1) = {score:.4f}")

        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_name = name

    if best_pipe is None:
        raise RuntimeError("Model eğitimi başarısız oldu.")

    out_path = os.path.join(args.outdir, args.best_name)
    joblib.dump(
        {
            "pipeline": best_pipe,
            "feature_cols": feature_cols,
            "threshold": cfg.threshold,
            "best_model_name": best_name,
            "select_metric": args.select,
            "best_score": best_score,
        },
        out_path,
    )

    print("\n" + "=" * 60)
    print(f"[SAVED] Best: {best_name} | {args.select}: {best_score:.4f}")
    print(f"[SAVED] -> {out_path}")


if __name__ == "__main__":
    main()
