import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import tensorflow as tf
from tensorflow import keras
from tensorflow import layers

DATA_PATH = os.path.join("data", "Online Retail.xlsx")
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
MODEL_DIR = "models"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_prepare_invoice_level_df(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    # Temel temizlik
    df = df.dropna(subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "Country"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["Country"] = df["Country"].astype(str)

    # Etiket: InvoiceNo 'C' ile başlıyorsa iptal/iade (1), değilse (0)
    df["label"] = df["InvoiceNo"].str.upper().str.startswith("C").astype(int)

    # Amount
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    # Tarih
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Invoice seviyesine özet
    invoice = (
        df.groupby(["InvoiceNo", "Country"], as_index=False)
          .agg(
              num_items=("StockCode", "count"),
              unique_products=("StockCode", "nunique"),
              total_qty=("Quantity", "sum"),
              total_amount=("Amount", "sum"),
              avg_price=("UnitPrice", "mean"),
              first_time=("InvoiceDate", "min"),
              label=("label", "max"),
          )
    )

    invoice["hour"] = invoice["first_time"].dt.hour
    invoice["dayofweek"] = invoice["first_time"].dt.dayofweek
    invoice["month"] = invoice["first_time"].dt.month
    invoice = invoice.drop(columns=["first_time"])

    invoice = invoice.replace([np.inf, -np.inf], np.nan).dropna()
    return invoice


def build_mlp(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    print(">>> TRAIN SCRIPT STARTED")

    invoice = load_and_prepare_invoice_level_df(DATA_PATH)

    y = invoice["label"].astype(int).values
    X = invoice.drop(columns=["label", "InvoiceNo"])  # InvoiceNo modelde kullanılmayacak

    numeric_features = ["num_items", "unique_products", "total_qty", "total_amount", "avg_price", "hour", "dayofweek", "month"]
    categorical_features = ["Country"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # sparse -> dense
    X_train_dense = X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t
    X_test_dense = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t

    model = build_mlp(X_train_dense.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        X_train_dense, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )

    y_prob = model.predict(X_test_dense).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    print("\nROC-AUC:", auc)
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    model.save(os.path.join(MODEL_DIR, "mlp_return_model.keras"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))

    # Grafikler
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(FIG_DIR, "loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history.history["auc"], label="train_auc")
    plt.plot(history.history["val_auc"], label="val_auc")
    plt.legend()
    plt.title("AUC")
    plt.savefig(os.path.join(FIG_DIR, "auc.png"), dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(FIG_DIR, "roc_curve.png"), dpi=200)
    plt.close()

    print("\n✅ Model kaydedildi: models/")
    print("✅ Grafikler kaydedildi: outputs/figures/")


if __name__ == "__main__":
    main()
