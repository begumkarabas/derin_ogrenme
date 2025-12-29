import joblib
import pandas as pd
import gradio as gr
import tensorflow as tf

MODEL_PATH = "models/mlp_return_model.keras"
PREP_PATH = "models/preprocessor.joblib"

model = tf.keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREP_PATH)

def predict(num_items, unique_products, total_qty, total_amount, avg_price, hour, dayofweek, month, country):
    X = pd.DataFrame([{
        "num_items": num_items,
        "unique_products": unique_products,
        "total_qty": total_qty,
        "total_amount": total_amount,
        "avg_price": avg_price,
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "Country": country,
    }])

    Xt = preprocessor.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt

    prob = float(model.predict(Xt).ravel()[0])
    label = "İade/İptal olasılığı yüksek" if prob >= 0.5 else "Normal"
    return label, float(f"{prob:.6f}")

demo = gr.Interface(
    fn=predict,
    inputs=[
                gr.Number(label="Siparişteki Ürün Satırı Sayısı", value=5),
        gr.Number(label="Benzersiz Ürün Sayısı", value=5),
        gr.Number(label="Toplam Ürün Adedi", value=10),
        gr.Number(label="Toplam Sipariş Tutarı (£)", value=200.0),
        gr.Number(label="Ortalama Ürün Fiyatı (£)", value=20.0),
        gr.Slider(0, 23, value=12, step=1, label="Sipariş Saati (0–23)"),
        gr.Slider(0, 6, value=2, step=1, label="Haftanın Günü (0=Pzt, 6=Paz)"),
        gr.Slider(1, 12, value=6, step=1, label="Sipariş Ayı (1=Ocak, 12=Aralık)"),
        gr.Textbox(label="Ülke", value="United Kingdom"),
    ],
    outputs=[gr.Textbox(label="Tahmin"), gr.Number(label="İptal Edilme Olasılığı")],
    title="Online Retail İade/İptal Tahmini",
)

if __name__ == "__main__":
    demo.launch(share=True)

