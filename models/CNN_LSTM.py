import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers

        kernel_size = 3 if self.seq_len >= 3 else 1
        padding = kernel_size // 2
        dropout = configs.dropout if self.num_layers > 1 else 0.0

        self.temporal_conv = nn.Conv1d(
            in_channels=self.enc_in,
            out_channels=self.hidden_size,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.activation = nn.ReLU()
        self.encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(self.hidden_size, self.pred_len * self.enc_in)

    def forecast(self, x_enc):
        means = x_enc.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - means) / stdev

        conv_in = x_norm.permute(0, 2, 1)
        conv_out = self.activation(self.temporal_conv(conv_in)).permute(0, 2, 1)
        enc_out, _ = self.encoder(conv_out)
        last_hidden = enc_out[:, -1, :]
        dec_out = self.projection(last_hidden).view(-1, self.pred_len, self.enc_in)

        return dec_out * stdev.expand(-1, self.pred_len, -1) + means.expand(-1, self.pred_len, -1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        return None
