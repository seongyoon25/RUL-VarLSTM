import numpy as np
import matplotlib.pyplot as plt
import json
from utils import CallModel

file_path = 'data_example_1.csv' # or 'data_example_2.csv'
input_window = 100
input_size = 1

model = CallModel(input_window, input_size)
pred_mc_mean, pred_mc_upper, pred_mc_lower, x_dat = model.evaluate(file_path, cap_max=1.1, alpha=5, tmax=100, titer=1)

a1 = plt.plot(np.arange(input_window), x_dat / x_dat[0]*100, '+', linewidth=2, color='blue', markersize=4)
a2 = plt.plot(np.arange(input_window, input_window+len(pred_mc_mean)),
              pred_mc_mean / x_dat[0]*100, linewidth=2, color='red')
a3 = plt.fill_between(np.arange(input_window, input_window+len(pred_mc_mean)),
                      pred_mc_lower / x_dat[0]*100, pred_mc_upper / x_dat[0]*100,
                      facecolor=np.array([255, 255, 73]) / 255, edgecolor=np.array([255, 199, 38]) / 255)
a100 = plt.axhline(y=100, linestyle='--', color='grey')
a90 = plt.axhline(y=90, linestyle='--', color='grey')
a80 = plt.axhline(y=80, linestyle='--', color='grey')
a70 = plt.axhline(y=70, linestyle='--', color='grey')
a60 = plt.axhline(y=60, linestyle='--', color='grey')
plt.xlabel('Cycles', fontsize=14)
plt.ylabel('Retention (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend([a1[0], (a3, a2[0])], ['Observation', 'Forecast with CI'], fontsize=12)
plt.ylim([55, 105])
plt.xlim([0, 800])

plt.tight_layout()
plt.savefig('Prediction.tiff', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
