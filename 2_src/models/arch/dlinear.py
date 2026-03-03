import torch
import torch.nn as nn

from layers.revin import RevIN


class MovingAverage(nn.Module):
    """
    移動平均ブロック (Causal Padding対応)
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1)
        x_padded = torch.cat([front, x], dim=1)

        # [Batch, Seq, Chan] -> [Batch, Chan, Seq] for AvgPool
        x_padded = x_padded.permute(0, 2, 1)
        out = self.avg(x_padded)
        return out.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """
    系列分解ブロック (Series Decomposition)
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, individual=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = input_dim
        self.revin = RevIN(input_dim)
        self.decomposition = SeriesDecomp(kernel_size=25)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(input_dim)])
            self.Linear_Trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(input_dim)])
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

        self.projection = nn.Linear(input_dim * pred_len, pred_len)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channel]
        x = self.revin(x, "norm")
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), self.channels, self.pred_len], device=x.device)
            trend_output = torch.zeros([trend_init.size(0), self.channels, self.pred_len], device=x.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        x = x.reshape(x.shape[0], -1)
        x = self.projection(x)

        return x
