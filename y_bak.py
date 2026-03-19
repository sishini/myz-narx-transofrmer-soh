import numpy as np
import matplotlib.pyplot as plt

try:
    # y verisini yükle (Hedef SoH değerleri)
    data_y = np.load('NASA_DATA/data_y.npy')
    print("Hedef veri yüklendi! Boyut:", data_y.shape)

    # y verisini çizdir
    plt.figure(figsize=(10, 5))
    plt.plot(data_y, color='red', linewidth=2, label='Gerçek SoH (Kapasite)')
    plt.title('NASA Veri Seti - Batarya Kapasite Kaybı (Hedef Veri)')
    plt.xlabel('Döngü Sayısı')
    plt.ylabel('Kapasite (Ah)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
except Exception as e:
    print("Hata:", e)