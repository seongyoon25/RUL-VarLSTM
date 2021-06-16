import torch
import torch.nn as nn


class VarLSTM(nn.Module):

    def __init__(self, sequence_len=100, input_size=1, state_size=256, dropout_prob=.50):
        super(VarLSTM, self).__init__()

        self.sequence_len = sequence_len
        self.input_size = input_size
        self.state_size = state_size
        self.dropout_prob = dropout_prob

        self.rnn1_f = nn.LSTMCell(2 * self.input_size, self.state_size)
        self.rnn1_b = nn.LSTMCell(2 * self.input_size, self.state_size)
        self.rnn2_f = nn.LSTMCell(2 * self.state_size, self.state_size)
        self.rnn2_b = nn.LSTMCell(2 * self.state_size, self.state_size)

        self.out = nn.Linear(2 * self.state_size, self.input_size)  # estimate SOH

    def forward(self, x0):
        # x: (batch, input_sequence_len, input_size)
        # h: (batch, state_size), (batch, state_size)
        x = x0 - x0.mean(1, keepdim=True)
        x = torch.cat([x, torch.cat(self.sequence_len * [x0.mean(1, keepdim=True)], 1)], 2)

        batch = x.shape[0]
        h = torch.zeros(batch, self.state_size)
        c = torch.zeros(batch, self.state_size)
        h1_f, h1_b, h2_f, h2_b = h, h, h, h
        c1_f, c1_b, c2_f, c2_b = c, c, c, c

        pred_array_f = []
        for t in range(self.sequence_len):
            (h1_f, c1_f) = self.rnn1_f(x[:, t, :], (h1_f * self.mask1, c1_f))
            pred_array_f.append(h1_f)
        pred_array_f = torch.stack(pred_array_f, 1)

        pred_array_b = []
        for t in range(self.sequence_len):
            (h1_b, c1_b) = self.rnn1_b(x[:, -t - 1, :], (h1_b * self.mask1, c1_b))
            pred_array_b.append(h1_b)
        pred_array_b = torch.stack(pred_array_b, 1)

        pred_array = torch.cat([pred_array_f, pred_array_b], 2)

        pred_array_f = []
        for t in range(self.sequence_len):
            (h2_f, c2_f) = self.rnn2_f(pred_array[:, t, :] * self.mask2, (h2_f * self.mask3, c2_f))
            pred_array_f.append(h2_f)
        pred_array_f = torch.stack(pred_array_f, 1)

        pred_array_b = []
        for t in range(self.sequence_len):
            (h2_b, c2_b) = self.rnn2_b(pred_array[:, -t - 1, :] * self.mask2, (h2_b * self.mask3, c2_b))
            pred_array_b.append(h2_b)
        pred_array_b = torch.stack(pred_array_b, 1)

        pred_array = torch.cat([pred_array_f, pred_array_b], 2)

        pred = self.out(pred_array * self.mask4)

        return pred + x0.mean(1, keepdim=True)

    def set_dropout(self, batch):
        mask_dist = torch.distributions.bernoulli.Bernoulli(1 - self.dropout_prob)
        mask1 = mask_dist.sample([batch, self.state_size])
        mask2 = mask_dist.sample([batch, 2 * self.state_size])
        mask3 = mask_dist.sample([batch, self.state_size])
        mask4 = mask_dist.sample([batch, 1, 2 * self.state_size])
        self.mask1 = mask1
        self.mask2 = mask2
        self.mask3 = mask3
        self.mask4 = mask4
