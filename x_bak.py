import numpy as np
import matplotlib.pyplot as plt

# Dosyayı yükle 
data = np.load('NASA_DATA/data_x.npy')

# Verinin boyutunu terminale yazdır 
print("Veri Boyutu:", data.shape) 

# Veriyi görselleştir 
plt.plot(data[0, :, 0]) 
plt.title("NASA Veri Seti - Ham Voltaj/Akım Grafiği")
plt.show()