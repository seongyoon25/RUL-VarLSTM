import numpy as np
import pandas as pd
import torch

from scipy.signal import medfilt
from models import VarLSTM


class CallModel:
    def __init__(self, sequence_len=100, input_size=1):
        self.sequence_len = sequence_len
        self.input_size = input_size

        path_base = 'CALCE_VarLSTM_256state_1dim.pt'
        self.path = path_base

        self.model = VarLSTM(self.sequence_len, self.input_size)
        self.model.load_state_dict(torch.load(self.path, map_location='cpu'))
        self.model.eval()

    def evaluate(self, file_path, cap_max=1.1, alpha=5, tmax=100, titer=1):

        capacity = pd.read_csv(file_path, header=None).values.flatten()
        smoothed = medfilt(capacity, 5)
        noise = smoothed - capacity
        capacity_filtered = np.where(np.abs(noise) > 2.0 * np.std(noise), smoothed, capacity)

        x_dat = capacity_filtered.reshape(1, self.sequence_len, self.input_size) / cap_max
        x_dat_tensor = torch.FloatTensor(x_dat)

        pred_mc_bundle = []
        for _ in range(titer):
            self.model.set_dropout(tmax)
            pred_mc = []
            pred_train = self.model(torch.cat(tmax * [x_dat_tensor], 0))
            pred_mc.append(pred_train.detach().numpy())
            for i in range(20):
                pred_train = self.model(pred_train)
                pred_mc.append(pred_train.detach().numpy())
                if (np.percentile(pred_train.detach().numpy()[:, :, 0], 95, axis=0) < 0.5*capacity_filtered[0]).all():
                    break

            pred_mc = np.concatenate(pred_mc, 1)
            pred_mc_bundle.append(pred_mc)

        pred_mc_bundle = np.concatenate(pred_mc_bundle, 0) * cap_max
        pred_mc_mean = np.percentile(pred_mc_bundle[:, :, 0], 50, axis=0)
        pred_mc_upper = np.percentile(pred_mc_bundle[:, :, 0], 100 - alpha / 2, axis=0)
        pred_mc_lower = np.percentile(pred_mc_bundle[:, :, 0], alpha / 2, axis=0)

        return pred_mc_mean, pred_mc_upper, pred_mc_lower, capacity_filtered.flatten()
