import torch

#763. epochtaki modelin mimarisini ve ağırlıklarını ayrı bir dosyaya kaydetme   

# raw string ile tanımladığımız model yolu (başına r eklendi):
model_yolu = r'C:\Users\TR\Desktop\myz dönem projesi\NARX-Transformer-SoH-main\models\trained_model_0.017068_763.pt'

# Modeli doğrudan yükleyin ve değerlendirme moduna alın
# (PyTorch'un son sürümlerinde tam modeli yüklerken weights_only=False eklemek gerekir)
model = torch.load(model_yolu, weights_only=False)
model.eval()

print("Model başarıyla yüklendi!")

# Model mimarisini ayrı bir dosyaya kaydet
mimari_dosyasi = "egitilmis_mimari_ozeti.txt"
with open(mimari_dosyasi, "w", encoding="utf-8") as f:
    f.write("--- EĞİTİLMİŞ MODELİN MİMARİSİ (KATEGORİZE EDİLMİŞ) ---\n\n")
    f.write("=======================================================\n")
    f.write("1. BACKBONE (Özellik Çıkarımı: Evrişim Katmanları)\n")
    f.write("=======================================================\n")
    f.write(f"  (conv_layer): {model.conv_layer}\n")
    f.write(f"  (conv_layer2): {model.conv_layer2}\n\n")
    
    f.write("=======================================================\n")
    f.write("2. NECK (Zaman/Sıra İlişkilendirmesi: Transformer Katmanları)\n")
    f.write("=======================================================\n")
    f.write(f"  (encoder_layer): {model.encoder_layer}\n")
    f.write(f"  (decoder_layer): {model.decoder_layer}\n\n")
    
    f.write("=======================================================\n")
    f.write("3. HEAD (Sonuç/Tahmin Üretimi: Doğrusal Katmanlar)\n")
    f.write("=======================================================\n")
    f.write(f"  (cap_linear_layer): {model.cap_linear_layer}\n")
    f.write(f"  (final_linear_layer): {model.final_linear_layer}\n")

print(f"Model mimarisi '{mimari_dosyasi}' dosyasına kaydedildi.")

def classify_layer(name):
    if 'conv_layer' in name:
        return '1. BACKBONE (Özellik Çıkarımı: Evrişim Katmanları)'
    elif 'cap_linear_layer' in name or 'encoder_layer' in name or 'decoder_layer' in name:    
        return '2. NECK (Zaman/Sıra İlişkilendirmesi: Transformer Katmanları)'
    elif 'linear_layer' in name:
        return '3. HEAD (Sonuç/Tahmin Üretimi: Doğrusal Katman)'
    return '4. DİĞER'


    
 
    
dosya_adi = "egitilmis_model_ozeti.txt"
with open(dosya_adi, "w", encoding="utf-8") as f:
    f.write("--- EĞİTİLMİŞ MODELİN AĞIRLIKLARI (ÖZET) ---\n")
    
    # Parametreleri kategorize et
    categorized_params = {}
    for name, param in model.named_parameters():
        cat = classify_layer(name)
        if cat not in categorized_params:
            categorized_params[cat] = []
        categorized_params[cat].append((name, param))

    # Kategorileri yazdır
    for cat in sorted(categorized_params.keys()):
        f.write(f"\n=======================================================\n")
        f.write(f"{cat}\n")
        f.write(f"=======================================================\n")
        for name, param in categorized_params[cat]:
            flat_weights = param.data.flatten()
            mean_val = flat_weights.mean().item()
            ilk_5 = flat_weights[:5].tolist()
            
            f.write(f"\nKatman: {name} | Boyut: {param.data.shape}\n")
            f.write(f"Ortalama: {mean_val:.6f}\n")
            f.write(f"İlk 5 Değer: {[round(x, 6) for x in ilk_5]}\n")

    total_params_trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write(f"\n-------------------------------------------------------\n")
    f.write(f"Eğitilmiş Modelin Toplam Parametre Sayısı: {total_params_trained:,}\n")

print(f"Eğitilmiş modelin ağırlık detayları '{dosya_adi}' dosyasına kaydedildi.")

