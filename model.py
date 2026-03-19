import torch
import torch.nn as nn

class NARX_Transformer(nn.Module):
    def __init__(self, feature_dim1,feature_dim2, num_attention, num_cycles, num_preds):
        super(NARX_Transformer, self).__init__()
        self.num_cycles = num_cycles
        self.num_preds = num_preds
        self.cap_linear_layer = nn.Linear(self.num_cycles-1, feature_dim2)
        self.final_linear_layer = nn.Linear(feature_dim2, 1)

        # self.conv_layer = nn.Conv1d(3, 512, kernel_size=16, stride=8)
        self.conv_layer = nn.Conv2d(num_cycles, feature_dim1, kernel_size=3, stride=1,padding=1)
        self.conv_layer2 = nn.Conv2d(feature_dim1,feature_dim2,kernel_size=3)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim2, nhead=num_attention, batch_first=True)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim2, nhead=num_attention, batch_first=True)

    def forward(self, my_data, capacity):
        """
        NARX_Transformer'ın ileri geçişi (forward).

        Girdiler:
        - my_data: batchlenmiş giriş özellikleri (dataset'e göre şekli değişebilir)
        - capacity: geçmiş kapasite değerleri (decoder girişi olarak kullanılır)

        Dönen değer:
        - output_cap: bir sonraki kapasite tahmini; sigmoid ile (0,1) aralığına sınırlandırılır
        """
        # Konvolüsyonel gömme (embedding)
        embedded_data = self.conv_layer(my_data)
        embedded_data = self.conv_layer2(embedded_data).squeeze(-1)
        # (batch, seq_len, feature) düzenine getir
        embedded_data = embedded_data.permute(0, 2, 1)
        # Transformer kodlayıcı (encoder)
        encoded_data = self.encoder_layer(embedded_data)

        # Kapasite geçmişini decoder girdi boyutuna eşle
        tgt = self.cap_linear_layer(capacity)
        tgt = tgt.unsqueeze(1)
        # Decoder, encoded_data üzerinde attention uygular
        decoded_data = self.decoder_layer(tgt, encoded_data)
        decoded_data = decoded_data.squeeze(1)
        # Son doğrusal katmanla skalar çıktıya projeksiyon
        output_cap = self.final_linear_layer(decoded_data)
        # Autoregressive çıkarım sırasında tahminlerin kontrolsüz yükselmesini engellemek için (0,1) aralığına sıkıştır
        output_cap = torch.sigmoid(output_cap)
        return output_cap
    
    def pred_sequence(self, my_data, capacity):
        """
        Çok-adımlı (multi-step) tahmin üretir: her adımda `forward` çağrılır ve
        elde edilen tahminler kapasite geçmişine eklenir.

        Not: `forward` çıktıları sigmoid ile (0,1) aralığında olduğundan `pred_caps`
        bu aralığı koruyacaktır.
        """
        # Verilen kapasite geçmişi ile başla (son self.num_cycles-1 değerler)
        pred_caps = torch.stack([capacity[:, i] for i in range(self.num_cycles - 1)], axis=-1)
        for cycle in range(self.num_preds):
            # Bu adım için kayan pencere şeklinde my_data seç
            window = my_data[:, cycle:cycle + self.num_cycles]
            # En güncel (num_cycles-1) kapasite değerlerini decoder girdisi olarak al
            recent_caps = pred_caps[:, - (self.num_cycles - 1) :]
            pred = self.forward(window, recent_caps)
            # pred boyutu: (batch, 1) — son eksene ekle
            pred_caps = torch.cat([pred_caps, pred], axis=-1)
        return pred_caps


class GRU_CNN(nn.Module):
    def __init__(self):
        super(GRU_CNN, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=32, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 64, kernel_size=32, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 64, kernel_size=32, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 64, kernel_size=32, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 64, kernel_size=32, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 64, kernel_size=32, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256, 64)

        self.gru = nn.GRU(3, 256, batch_first=True)
        self.dense2 = nn.Linear(256, 64)

        self.concat = nn.Linear(128, 1)

    def forward(self, input_stream):
        x1 = self.conv_block(input_stream)
        x1 = self.flatten(x1)
        x1 = self.dense1(x1)

        _, x2 = self.gru(input_stream.permute(0, 2, 1))
        x2 = x2.squeeze(0)
        x2 = self.dense2(x2)

        combined = torch.cat((x1, x2), dim=1)
        output = self.concat(combined)

        return output
    




# import yaml
# 
# # YAML configuration dosyasını yükle
# with open('config.yaml', 'r') as file:
#     cfg = yaml.safe_load(file)
# 
# # Modelinizi config parametreleriyle başlatın
# custom_model = NARX_Transformer(
#     feature_dim1=cfg['FEATURE_DIM1'],
#     feature_dim2=cfg['FEATURE_DIM2'],
#     num_attention=cfg['NUM_ATTENTION'],
#     num_cycles=cfg['NUM_CYCLES'],
#     num_preds=cfg['NUM_PREDS']
# )
# 
# with open("model_ozeti.txt", "w", encoding="utf-8") as f:
#     f.write("Modelinizin Katmanları ve Ağırlıkları:\n\n")
#     # Modelin tüm öğrenilebilir parametrelerini (ağırlıklar ve bias'lar) yazdırma
#     for name, param in custom_model.named_parameters():
#         if param.requires_grad:
#             f.write(f"Katman Adı: {name}, Ağırlık Boyutu: {param.data.shape}\n")
#             # İsterseniz ağırlıkların ilk 5 değerini de yazdırabilirsiniz:
#             # f.write(f"İlk 5 ağırlık değeri: {param.data.flatten()[:5]}\n\n")
# 
#     f.write("\nModelin toplam parametre sayısı:\n")
#     total_params = sum(p.numel() for p in custom_model.parameters() if p.requires_grad)
#     f.write(f"{total_params:,}\n")
# 
# print("Model bilgileri 'model_ozeti.txt' dosyasına başarıyla kaydedildi.")
import torch

# HATA 1: Windows yollarında ters slash (\) kullanırken Python'un hata vermemesi için
# ya yolun başına r harfi koymalısınız (raw string), ya da / kullanmalısınız.
# HATA 2: Yukarıdaki hatalı NARX_Transformer.IMAGENET1K_V1 kodunu hala silmemişsiniz,
# bu yüzden o hatalı yeri silerek temizledim.

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
    f.write("--- EĞİTİLMİŞ MODELİN MİMARİSİ ---\n\n")
    f.write(str(model))

print(f"Model mimarisi '{mimari_dosyasi}' dosyasına kaydedildi.")
# Eğitilmiş modelin ağırlık isimlerini, ortalamasını ve ilk 5 değerini dosyaya yazdır
dosya_adi = "egitilmis_model_ozeti.txt"
with open(dosya_adi, "w", encoding="utf-8") as f:
    f.write("--- EĞİTİLMİŞ MODELİN AĞIRLIKLARI (ÖZET) ---\n")
    for name, param in model.named_parameters():
        flat_weights = param.data.flatten()
        mean_val = flat_weights.mean().item()
        ilk_5 = flat_weights[:5].tolist()
        
        f.write(f"\nKatman: {name} | Boyut: {param.data.shape}\n")
        f.write(f"Ortalama: {mean_val:.6f}\n")
        f.write(f"İlk 5 Değer: {[round(x, 6) for x in ilk_5]}\n")

    total_params_trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write(f"\nEğitilmiş Modelin Toplam Parametre Sayısı: {total_params_trained:,}\n")

print(f"Eğitilmiş modelin ağırlık detayları '{dosya_adi}' dosyasına kaydedildi.")
