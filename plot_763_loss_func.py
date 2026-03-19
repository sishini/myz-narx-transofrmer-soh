import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#763. epochiçin loss fonksiyonu grafiği çizdirme

log = np.load('models/training_log_7_7.npy')
train_loss, test_loss = log[:,0], log[:,1]

# 763. epoch için bu noktayı işaretle
target_epoch = 763
if target_epoch < len(train_loss):
    t_val = train_loss[target_epoch]
    v_val = test_loss[target_epoch]
else:
    raise ValueError(f"Epoch {target_epoch} out of range (max {len(train_loss)-1})")

plt.figure(figsize=(8,5))
plt.plot(train_loss, label='Train Loss')
plt.plot(test_loss, label='Test Loss')
plt.scatter([target_epoch],[t_val], c='red', label=f'Train @ {target_epoch}', zorder=5)
plt.scatter([target_epoch],[v_val], c='blue', label=f'Test @ {target_epoch}', zorder=5)
plt.axvline(target_epoch, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NARX_Transformer Training/Test Loss with epoch 763 marker')
plt.legend()
plt.grid(True)
plt.savefig('models/loss_curve_763.png')
print('Loss curve for epoch 763 saved to models/loss_curve_763.png')
print('Epoch 763: train', t_val, 'test', v_val)
