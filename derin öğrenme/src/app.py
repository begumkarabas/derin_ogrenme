import joblib
import numpy as np
import pandas as pd
import gradio as gr

BUNDLE_PATH = "models/best_model.joblib"

bundle = joblib.load(BUNDLE_PATH)
pipe = bundle["pipeline"]
FEATURES = bundle["feature_cols"]
THRESHOLD = float(bundle["threshold"])
MODEL_NAME = bundle.get("best_model_name", "model")


def risk_tahmin(
    satir_sayisi,
    farkli_urun_sayisi,
    toplam_adet,
    toplam_tutar,
    ort_fiyat,
    siparis_saati,
    musteri_ort_harcama,
    musteri_medyan_harcama,
    musteri_siparis_sayisi,
    musteri_sapma_z,
):

    if toplam_adet <= 0 or toplam_tutar <= 0 or ort_fiyat <= 0:
        return {"Hatalı giriş": 1.0}, 0.0, "Toplam adet / tutar / ortalama fiyat pozitif olmalı."

    if not (0 <= siparis_saati <= 23):
        return {"Hatalı giriş": 1.0}, 0.0, "Sipariş saati 0-23 arası olmalı."

    gece_mi = 1 if 0 <= int(siparis_saati) <= 5 else 0

    log_toplam_tutar = float(np.log1p(toplam_tutar))
    log_toplam_adet = float(np.log1p(toplam_adet))

    row = {
        "num_lines": float(satir_sayisi),
        "unique_products": float(farkli_urun_sayisi),
        "total_qty": float(toplam_adet),
        "total_amount": float(toplam_tutar),
        "avg_unit_price": float(ort_fiyat),
        "order_hour": int(siparis_saati),
        "is_night": int(gece_mi),
        "cust_mean_amount": float(musteri_ort_harcama),
        "cust_median_amount": float(musteri_medyan_harcama),
        "cust_order_count": float(musteri_siparis_sayisi),
        "cust_amount_z": float(musteri_sapma_z),
        "log_total_amount": log_toplam_tutar,
        "log_total_qty": log_toplam_adet,
    }

    X = pd.DataFrame([row])[FEATURES]
    proba = float(pipe.predict_proba(X)[:, 1][0])
    riskli = int(proba >= THRESHOLD)

    nedenler = []
    if toplam_tutar >= max(1.0, musteri_ort_harcama) * 2.0:
        nedenler.append("Sipariş tutarı, müşterinin ortalamasının belirgin üstünde.")
    if musteri_sapma_z >= 2.0:
        nedenler.append("Müşteri geçmişine göre sapma yüksek (z skoru).")
    if farkli_urun_sayisi >= 10:
        nedenler.append("Farklı ürün sayısı yüksek.")
    if toplam_adet >= 20:
        nedenler.append("Toplam adet yüksek.")
    if gece_mi == 1:
        nedenler.append("Gece saatinde sipariş.")

    if not nedenler:
        nedenler.append("Belirgin tek bir kural baskın değil; model kombinasyona göre karar verdi.")

    sonuc = "RİSKLİ" if riskli else "NORMAL"
    aciklama = (
        f"Model: {MODEL_NAME} | Eşik (threshold): {THRESHOLD}\n"
        f"Sonuç: {sonuc}\n"
        f"Nedenler: " + " | ".join(nedenler)
    )

    label_dict = {"RİSKLİ": proba, "NORMAL": 1.0 - proba}
    return label_dict, proba, aciklama


demo = gr.Interface(
    fn=risk_tahmin,
    inputs=[
        gr.Number(label="Sipariş satır sayısı", value=5),
        gr.Number(label="Farklı ürün sayısı", value=4),
        gr.Number(label="Toplam adet", value=8),
        gr.Number(label="Toplam tutar", value=250.0),
        gr.Number(label="Ortalama ürün fiyatı", value=30.0),
        gr.Slider(0, 23, step=1, label="Sipariş saati (0-23)", value=14),

        gr.Number(label="Müşterinin ortalama harcaması", value=120.0),
        gr.Number(label="Müşterinin medyan harcaması", value=90.0),
        gr.Number(label="Müşterinin sipariş sayısı", value=12),

        gr.Number(label="Müşteri sapma skoru (z)", value=1.2),
    ],
    outputs=[
        gr.Label(label="Sınıf", num_top_classes=2),
        gr.Number(label="Risk olasılığı (P[risk=1])"),
        gr.Textbox(label="Açıklama", lines=4),
    ],
    title="E-Ticaret Sipariş Risk Tahmini",
)

if __name__ == "__main__":
    demo.launch(share=True)
