E-Ticaret Sipariş Risk Tahmini

Bu proje, e-ticaret platformlarında verilen siparişlerin riskli veya normal olarak sınıflandırılmasını amaçlayan bir makine öğrenmesi uygulamasıdır.
Siparişe ait tutar, adet, ürün çeşitliliği, zaman bilgileri ve müşteri geçmişi gibi özellikler kullanılarak bir siparişin riskli olma olasılığı tahmin edilmektedir.

Proje kapsamında, gerçek bir e-ticaret veri seti üzerinde sıfırdan model eğitimi, model değerlendirmesi ve interaktif bir demo arayüzü geliştirilmiştir.



Proje Özeti

Problem Türü: İkili sınıflandırma (Normal / Riskli)
Veri Seti: Online Retail (UCI Machine Learning Repository)
Kullanılan Modeller:
  Logistic Regression
  Multi Layer Perceptron (MLP)
Öncelikli Metrik: Riskli sınıf için Recall
Arayüz: Gradio tabanlı web uygulaması


Veri Seti

Projede kullanılan veri seti, Birleşik Krallık merkezli bir online perakende firmasının 2010–2011 yılları arasındaki satış işlemlerini içermektedir.

Veri setinde yer alan başlıca alanlar: InvoiceNo, StockCode, Quantity, UnitPrice, InvoiceDate, CustomerID

Not: Çalışmada yalnızca pozitif satış işlemleri kullanılmış, iade ve iptal satırları veri setinden çıkarılmıştır.

Ham veri satır bazlı olduğu için, veri sipariş (invoice) seviyesine indirgenmiş ve özet özellikler oluşturulmuştur.


Kullanılan Yöntemler

Logistic Regression
Açıklanabilirliği yüksek
Riskli siparişleri kaçırmamak için recall odaklı
Operasyonel kullanım için uygun

Multi Layer Perceptron (MLP)
Daha karmaşık örüntüleri öğrenebilme
Yanlış alarm sayısını azaltma potansiyeli

Modeller scikit-learn kullanılarak, herhangi bir hazır model veya önceden eğitilmiş yapı olmadan sıfırdan eğitilmiştir.


Model Değerlendirme

Model performansı aşağıdaki metriklerle değerlendirilmiştir:
Accuracy
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC

Deneysel sonuçlara göre:
Logistic Regression modeli, riskli siparişleri yakalama konusunda yüksek recall değerine sahiptir.
ROC-AUC değeri ≈ 0.99 olarak elde edilmiştir.

Bu durum, modelin riskli ve normal siparişleri ayırt etmede başarılı olduğunu göstermektedir.

Kurulum ve Çalıştırma
Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

Modeli eğitin
python src/train.py

Web arayüzünü başlatın
python src/app.py
