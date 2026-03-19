import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm, trange
import yaml
import os
import seaborn as sns
sns.set_theme()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import NARX_Transformer
from dataset import load_NASA

# YAML configuration dosyasını yükle.
with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

# confi dosyasından hiperparametreleri okuyun ve değişkenlere ata. 
NUM_CYCLES = cfg['NUM_CYCLES']
NUM_PREDS = cfg['NUM_PREDS']
FEATURE_DIM1 = cfg['FEATURE_DIM1']
FEATURE_DIM2 = cfg['FEATURE_DIM2']
NUM_ATTENTION = cfg['NUM_ATTENTION']
EPOCHS = cfg['EPOCHS']
LEARNING_RATE = cfg['LEARNING_RATE']
BATCH_SIZE = cfg['BATCH_SIZE']

# GPU erişilebilir mi kontrol et değilse cpu kullan.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# veriyi yükle. load_NASA fonksiyonu şurda tanımlanmıştır: NASA verilerini belirtilen klasörden yükler, belirli sayıda döngü ve tahmin için verileri hazırlar, verileri eğitim ve test setlerine böler ve isteğe bağlı olarak verileri ölçeklendirir.
train_dataset, test_dataset = load_NASA(folder='NASA_DATA', num_cycles=NUM_CYCLES+NUM_PREDS-1, split_ratio=0.5, scale_data=True)

# Train/test split
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True) 
#Shuffle kullanılmasının nedeni, her epoch'ta verilerin farklı sırayla model tarafından görülmesini sağlamak ve 
#böylece modelin genelleme yeteneğini artırmaktır. Eğer shuffle=False olarak ayarlanırsa, model her epoch'ta verileri aynı sırayla görecektir, bu da overfitting'e yol açabilir. 


# NN model
model = NARX_Transformer(feature_dim1=FEATURE_DIM1, 
                         feature_dim2=FEATURE_DIM2, 
                         num_attention=NUM_ATTENTION, 
                         num_cycles=NUM_CYCLES, 
                         num_preds=NUM_PREDS).to(device)

# Maaliyet fonksiyonu ve Adam optimizer kullanarak modelin parametrelerini optimize edin. L1Loss, tahmin edilen çıktılar ile gerçek çıktılar arasındaki mutlak farkın ortalamasını hesaplar. 
#Adam optimizer ise, öğrenme oranını adaptif olarak ayarlayarak modelin parametrelerini günceller ve genellikle hızlı ve etkili bir şekilde konverge eder.
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Eğitim döngüsü ve model kaydetme. Her epoch'ta modelin performansını değerlendirin ve eğer test kaybı önceki en iyi kayıptan daha düşükse, modeli kaydedin.
best_epoch = 0
best_loss = float('inf')
Loss_log = []
os.makedirs('models', exist_ok=True)


model.train()
t_range = trange(EPOCHS)
for epoch in t_range:
    
    train_losses = []
    for inputs, outputs in train_dataloader: # train_dataloader, train setinden verileri yükler.
        inputs = inputs.float().to(device)
        outputs = outputs.float().to(device)

        predicted_outputs = model.pred_sequence(inputs, outputs) #model.pred_sequence fonksiyonu şurda tanımlanmıştır: Çok-adımlı (multi-step) tahmin üretir: her adımda `forward` çağrılır ve elde edilen tahminler kapasite geçmişine eklenir.
        #Modelin ileri besleme (forward pass) adımıdır. inputs ve geçmiş outputs değerlerini alarak bir tahmin dizisi üretir.
        
        optimizer.zero_grad() #Bir önceki adımdan kalan gradyanları (türevleri) sıfırlar. PyTorch gradyanları biriktirdiği için bu adım hayati önem taşır; aksi takdirde model yanlış yöne sapar.
        loss = criterion(predicted_outputs[:,NUM_CYCLES-1:], outputs[:,NUM_CYCLES-1:]) #Hata payını hesaplar. L1Loss, tahmin edilen çıktılar ile gerçek çıktılar arasındaki mutlak farkın ortalamasını hesaplar.
        loss.backward() #Hata payının gradyanlarını hesaplar. PyTorch'un otomatik gradyan hesaplamasını kullanarak, loss fonksiyonunun türevini hesaplar ve modelin parametrelerini güncellemek için kullanılır.
        optimizer.step() #Modelin parametrelerini günceller. optimizer.step() fonksiyonu, gradyan inişi (gradient descent) algoritmasını uygular ve modelin parametrelerini günceller. Bu adım, modelin performansını artırmak için kritik bir adımdır.
        train_losses.append(loss.item()) #Hata payını kaydeder. loss.item() fonksiyonu, loss değerini bir Python float olarak döndürür ve bu değeri train_losses listesine ekler. Bu liste, her epoch'ta tüm batch'lerin kaybının ortalamasını tutar.

    test_losses = []
    for inputs, outputs in test_dataloader: # test_dataloader, test setinden verileri yükler.
        inputs = inputs.float().to(device)
        outputs = outputs.float().to(device)

        with torch.no_grad(): #PyTorch'un gradyan hesaplamasını devre dışı bırakır. Bu, test sırasında performansı artırır ve gereksiz bellek kullanımını önler.
            predicted_outputs = model.pred_sequence(inputs, outputs) 
            #Modelin ileri besleme (forward pass) adımıdır. inputs ve geçmiş outputs değerlerini alarak bir tahmin dizisi üretir.
            
            test_loss = criterion(predicted_outputs[:,NUM_CYCLES-1:], outputs[:,NUM_CYCLES-1:]) #Hata payını hesaplar. L1Loss, tahmin edilen çıktılar ile gerçek çıktılar arasındaki mutlak farkın ortalamasını hesaplar.
            test_losses.append(test_loss.item()) #Hata payını kaydeder. loss.item() fonksiyonu, loss değerini bir Python float olarak döndürür ve bu değeri test_losses listesine ekler. Bu liste, her epoch'ta tüm batch'lerin kaybının ortalamasını tutar.
    Loss_log.append([np.mean(train_losses),np.mean(test_losses)]) #Her epoch'ta train ve test kayıplarını kaydeder. Loss_log listesine her epoch'ta train ve test kayıplarının ortalamasını ekler. Bu liste, her epoch'ta train ve test kayıplarının ortalamasını tutar.

    #Epoch ilerlemesini göstermek için tqdm kullanarak eğitim ve test kayıplarını güncelleyin. tqdm, uzun süren işlemler sırasında ilerleme çubuğu göstererek kullanıcıya bilgi verir.
    t_range.set_description(f"train loss: {np.mean(train_losses)}, test loss: {np.mean(test_losses)}")
    t_range.refresh()

    # Eğer test kaybı önceki en iyi kayıptan daha düşükse, modeli kaydedin. Bu, modelin performansını izlemek ve en iyi modeli saklamak için önemlidir.
    if np.mean(test_losses) < best_loss:
        best_epoch = epoch
        best_loss = np.mean(test_losses)
        torch.save(model, f'models/trained_model_{best_loss:.6f}_{best_epoch}.pt')

Loss_log = np.array(Loss_log)
np.save(f'models/training_log_{NUM_CYCLES}_{NUM_PREDS}.npy', Loss_log)

plt.figure(figsize=(8,5))
plt.plot(Loss_log[:best_epoch, 0])
plt.plot(Loss_log[:best_epoch, 1])
plt.legend(["Train Loss","Test Loss"])
plt.grid("on")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()