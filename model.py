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