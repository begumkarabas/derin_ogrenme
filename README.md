# Online Alışveriş Siparişlerinde İade/İptal Tahmini

Bu projede, online alışveriş siparişlerinin iade veya iptal edilme durumunu tahmin etmek amacıyla
**sıfırdan eğitilmiş bir derin öğrenme modeli (MLP)** geliştirilmiştir.

Model, gerçek bir e-ticaret veri seti üzerinde eğitilmiş ve Keras kullanılarak oluşturulmuştur.
Ayrıca modelin test edilebilmesi için Gradio tabanlı basit bir kullanıcı arayüzü geliştirilmiştir.

## Kullanılan Teknolojiler
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  
- Gradio  

## Proje Yapısı
project/
├── data/
├── src/
│ ├── train.py
│ └── app.py
├── models/
├── outputs/
└── requirements.txt


## Çalıştırma
Gerekli kütüphaneleri yüklemek için:
pip install -r requirements.txt

## Modeli eğitmek için
python src/train.py

## Gradio arayüzünü çalıştırmak için:
python src/app.py


