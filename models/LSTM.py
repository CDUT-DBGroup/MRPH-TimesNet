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

        dropout = configs.dropout if self.num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.enc_in,
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

        enc_out, _ = self.encoder(x_norm)
        last_hidden = enc_out[:, -1, :]
        dec_out = self.projection(last_hidden).view(-1, self.pred_len, self.enc_in)

        return dec_out * stdev.expand(-1, self.pred_len, -1) + means.expand(-1, self.pred_len, -1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc)
        return None
