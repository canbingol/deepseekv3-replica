# TurnaMoE: Modular LLM with Mixture-of-Experts and SentencePiece

Bu proje, Türkçe metinler üzerinde eğitilebilecek modüler bir LLM (Large Language Model) mimarisi sunar. Model, **DeekSeekV3** mimarisine sahiptir ve **SentencePiece** ile özel bir tokenizer kullanır.

## Özellikler

- Transformer tabanlı mimari
- MoE (Mixture-of-Experts) entegrasyonu
- Rotary Positional Embedding desteği
- LoRA projeksiyonları
- Tokenizer: SentencePiece (BPE)
- Eğitim ve validasyon ayrımı
- MoE gating analizleri ve parametre logları

## Kurulum

```bash
pip install torch sentencepiece matplotlib
```

## 1. Tokenizer Eğitimi

Tokenizer'ı eğitmek için `train_tokenizer.py` scripti çalıştırılır. Bu adım, `8k_data.txt` dosyasındaki veriyi kullanarak `tokenizer.model` dosyasını üretir:

```bash
python train_tokenizer.py
```

## 2. Veri Hazırlama

Eğitim verisi `data.py` tarafından `train_loader` ve `val_loader` olarak hazırlanır. Bu veri, tokenizer ile tokenleştirildikten sonra `max_seq_len` uzunluklarında sliding window kullanılarak bölünür.

## 3. Model Eğitimi

Model eğitimi için `train_model.py` dosyası çalıştırılır:

```bash
python train_model.py
```

Eğitim sırasında:

- Eğitim ve validasyon kayıpları `logs/train_loss.txt` ve `logs/val_loss.txt` dosyalarına kaydedilir
- 10 adımda bir validasyon yapılır
- 100 adımda bir örnek metin üretimi yapılır
- Her epoch sonunda `checkpoint_epoch_*.pt` dosyası oluşturulur
- Eğitim tamamlandığında `loss_curve_side_by_side.png` olarak görselleştirme yapılır

## Detaylı İnceleme

Bu projeyle ilgili teknik detayları aşağıdaki Medium yazımda paylaştım:

[DeepSeekV3 Farkı Ne?](https://medium.com/@canbing0l/deepseekv3-farkı-ne-0c9575ad1239?source=user_profile_page---------0-------------619695bebe0d----------------------)
