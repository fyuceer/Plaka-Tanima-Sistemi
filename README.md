# 🚗 Plaka Tanıma Sistemi

YOLO ve EasyOCR kullanılarak geliştirilen, kamera veya video üzerinden **Türk plakalarını tespit edip okuyabilen gerçek zamanlı bilgisayarlı görü uygulamasıdır.**

Sistem, görüntü içerisindeki plakaları YOLO modeli ile tespit eder, tespit edilen plaka bölgesini görüntü işleme yöntemleriyle işler ve EasyOCR kullanarak plaka karakterlerini okur.

Okunan plakalar, OCR güven değerleri ile birlikte belirli bir süre içerisinde tekrar kaydedilmelerini önleyen bir mekanizma kullanılarak CSV dosyasına kaydedilir.

Projenin amacı, görüntü işleme ve yapay zekâ yöntemlerini kullanarak araç plakalarının otomatik olarak tespit edilmesi ve okunmasını sağlamaktır.

Sistem özellikle gerçek zamanlı kamera görüntülerinde çalışabilecek şekilde tasarlanmıştır.

##  Özellikler

-  Gerçek zamanlı plaka tespiti
-  Kamera ve video dosyası desteği
-  YOLO tabanlı plaka tespiti
-  EasyOCR ile plaka karakterlerinin okunması
-  Plaka görüntüsüne ön işleme uygulanması
-  YOLO tespit güven değerinin gösterilmesi
-  OCR güven değerinin hesaplanması
- 🇹🇷 Türk plaka formatı kontrolü
-  Okunan plakaların CSV dosyasına kaydedilmesi
-  Aynı plakanın tekrar tekrar kaydedilmesini önleyen cooldown mekanizması
-  Komut satırı üzerinden model, kamera, güven eşiği ve kayıt ayarlarının değiştirilebilmesi

##  Sistem Akışı

Projenin temel çalışma akışı aşağıdaki gibidir:

**Kamera / Video → YOLO → Plaka Tespiti → Plaka Bölgesinin Kesilmesi → Görüntü Ön İşleme → EasyOCR → Plaka Format Kontrolü → CSV Kayıt**

### 1. Görüntü Alımı

Sistem kamera veya video dosyasından görüntü alır.

### 2. Plaka Tespiti

YOLO modeli görüntü içerisindeki plaka bölgelerini tespit eder.

Her tespit için bir güven değeri hesaplanır.

### 3. Plaka Bölgesinin Ayrılması

Tespit edilen bounding box koordinatları kullanılarak plaka görüntüden ayrılır.

### 4. Görüntü Ön İşleme

Plaka görüntüsü gri tonlamaya dönüştürülür ve Otsu eşikleme yöntemi uygulanır.

Bu işlem OCR'ın karakterleri daha kolay ayırt etmesine yardımcı olur.

### 5. OCR ile Karakter Tanıma

EasyOCR kullanılarak plaka içerisindeki karakterler okunur.

Okunan karakterler temizlenerek yalnızca harf ve rakamlardan oluşan bir metin elde edilir.

### 6. Plaka Format Kontrolü

OCR sonucunun Türk plaka formatına uygun olup olmadığı kontrol edilir.

Geçerli formatta olmayan sonuçlar kayıt altına alınmaz.

### 7. CSV Kayıt

Başarılı şekilde okunan plakalar:

- Tarih ve saat
- Plaka
- OCR güven değeri

bilgileriyle birlikte `plates.csv` dosyasına kaydedilir.

Aynı plakanın kısa süre içerisinde tekrar kaydedilmesini önlemek için cooldown mekanizması kullanılmaktadır.

## Kullanılan Teknolojiler

- **Python**
- **YOLO / Ultralytics**
- **OpenCV**
- **EasyOCR**
- **NumPy**
- **CSV**
- **Regular Expressions (Regex)**

##  Proje Yapısı

```text
Plaka-Tanima-Sistemi/
│
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```
## Model Dosyası

Sistem YOLO modeli olarak `best.pt` dosyasını kullanmaktadır.

Model dosyasını proje klasörüne yerleştirdikten sonra klasör yapısı aşağıdaki gibi olmalıdır:

```text
Plaka-Tanima-Sistemi/
│
├── best.pt
├── main.py
├── requirements.txt
└── README.md
```
## Kullanım

### Kamera ile çalıştırma

Varsayılan kamera kaynağını kullanmak için:

```bash
python main.py
```

### Farklı bir kamera kullanma
```bash
python main.py --source 1
```

### Video dosyası ile çalıştırma
```bash
python main.py --source video.mp4
```

### YOLO güven eşiğini değiştirme
```bash
python main.py --conf 0.5
```

### CPU kullanarak çalıştırma

### GPU kullanılmasını istemiyorsanız:
```bash
python main.py --no-gpu
```

### Çıkış
```bash
Program çalışırken görüntü penceresinde q tuşuna basarak uygulamadan çıkabilirsiniz.
```
## Kayıt Formatı

Okunan plakalar `plates.csv` dosyasına aşağıdaki formatta kaydedilir:

```text
timestamp,plate,confidence
2026-10-05 16:30:25,34ABC123,0.91
2026-10-05 16:31:12,06XYZ45,0.87
```
## Komut Satırı Parametreleri

Program aşağıdaki parametreleri desteklemektedir:

| Parametre | Açıklama | Varsayılan |
| --- | --- | --- |
| `--model` | YOLO model dosyasının yolu | `best.pt` |
| `--source` | Kamera numarası veya video yolu | `0` |
| `--conf` | YOLO güven eşiği | `0.4` |
| `--output` | CSV kayıt dosyasının yolu | `plates.csv` |
| `--cooldown` | Aynı plakanın tekrar kayıt süresi | `30 saniye` |
| `--no-gpu` | EasyOCR'ı CPU üzerinde çalıştırır | `Kapalı` |

##  Geliştirme Alanları

Projenin ilerleyen aşamalarında aşağıdaki geliştirmeler yapılabilir:

- Birden fazla plakanın aynı anda takip edilmesi
- OCR doğruluğunu artırmak için farklı görüntü ön işleme yöntemlerinin denenmesi
- Plaka karakterlerinin daha gelişmiş doğrulama yöntemleriyle kontrol edilmesi
- Araç takip sistemi eklenmesi
- Tespit edilen araçların giriş/çıkış zamanlarının tutulması
- Daha kapsamlı raporlama ve istatistiklerin eklenmesi
- Farklı kamera ve video kaynaklarıyla performans testlerinin yapılması

##  Geliştirici
**Feyza Sultan Yüceer**
