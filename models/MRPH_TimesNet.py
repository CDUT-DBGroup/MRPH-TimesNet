import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # Parameter-efficient multi-scale convolution design from TimesNet.
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

class MSRLA(nn.Module):
    """
    Multi-Scale Rainfall Lag Attention (MSRLA)
    Captures dependencies across different time scales (3, 7, 15, 30 days).
    """
    def __init__(self, d_model, scales=[3, 7, 15, 30]):
        super(MSRLA, self).__init__()
        self.scales = scales
        self.d_model = d_model
        
        # Convolutions for different scales
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=s, padding=s//2, groups=d_model)
            for s in scales
        ])
        
        # Attention Projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in scales])
        self.value_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in scales])
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Scale weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def forward(self, x):
        # x: [Batch, Time, Channels]
        B, T, C = x.shape
        
        # Transpose for Conv1d: [B, C, T]
        x_t = x.permute(0, 2, 1)
        
        # Query from original input
        Q = self.query_proj(x) # [B, T, C]
        
        scale_outputs = []
        
        for i, conv in enumerate(self.convs):
            # Extract multi-scale features
            feat_s = conv(x_t).permute(0, 2, 1) # [B, T, C]
            
            # Key and Value
            K = self.key_projs[i](feat_s)
            V = self.value_projs[i](feat_s)
            
            # Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
            attn = F.softmax(scores, dim=-1)
            
            out_s = torch.matmul(attn, V)
            scale_outputs.append(out_s)
            
        # Fusion
        final_out = torch.zeros_like(x)
        weights = F.softmax(self.scale_weights, dim=0)
        
        for i, out_s in enumerate(scale_outputs):
            final_out += weights[i] * out_s
            
        return self.out_proj(final_out)

class PCHTE(nn.Module):
    """
    Physically Constrained Hydrological Temporal Embedding (PCHTE)
    Embeds physical constraints: Infiltration and Water Balance.
    Equation: I(t) = alpha * P(t-tau) * exp(-beta * tau)
    """
    def __init__(self, d_model, rainfall_idx=0, water_level_idx=5, max_lag=30):
        super(PCHTE, self).__init__()
        self.d_model = d_model
        self.rainfall_idx = rainfall_idx
        self.water_level_idx = water_level_idx
        self.max_lag = max_lag
        
        # Learnable Physical Parameters
        # alpha: Infiltration coefficient (0-1)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0)) # sigmoid(0) = 0.5
        # beta: Decay coefficient (>0)
        self.beta_raw = nn.Parameter(torch.tensor(-2.0)) # softplus(-2) is small positive
        # gamma: Water inflow response coefficient (0-1)
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        
        self.embedding = nn.Linear(1, d_model)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)
    
    @property
    def beta(self):
        return F.softplus(self.beta_raw)
    
    @property
    def gamma(self):
        return torch.sigmoid(self.gamma_raw)

    def forward(self, x_enc):
        # x_enc: [Batch, Time, Features]
        # We need raw features here, assuming x_enc is the raw input or contains it?
        # Typically x_enc in Model.forward is already normalized. 
        # But for physical consistency, relative changes matter more.
        
        B, T, F_dim = x_enc.shape
        
        # Extract Rainfall (P) and Water Level (H)
        # Handle case where indices might be out of bounds
        r_idx = min(self.rainfall_idx, F_dim-1)
        w_idx = min(self.water_level_idx, F_dim-1)
        
        P = x_enc[:, :, r_idx].unsqueeze(1) # [B, 1, T] for Conv1d
        H = x_enc[:, :, w_idx] # [B, T]
        
        # 1. Calculate Infiltration I(t)
        # Kernel: alpha * exp(-beta * tau)
        tau = torch.arange(self.max_lag, device=x_enc.device).float()
        kernel = self.alpha * torch.exp(-self.beta * tau)
        kernel = kernel.view(1, 1, -1) # [Out_C, In_C, Kernel_Size]
        kernel = torch.flip(kernel, dims=[-1]) # Flip for convolution to match lag definition
        
        # Padding for causal convolution
        padding = self.max_lag - 1
        I = F.conv1d(P, kernel, padding=padding) # [B, 1, T + pad]
        I = I[:, :, :T] # Truncate to original length [B, 1, T]
        I = I.squeeze(1) # [B, T]
        
        # 2. Calculate Water Level Change delta_H
        # delta_H(t) = H(t) - H(t-1)
        delta_H = torch.zeros_like(H)
        delta_H[:, 1:] = H[:, 1:] - H[:, :-1]
        
        # 3. Water Balance Constraint
        # Q_phys = gamma * I + (1-gamma) * delta_H
        # Note: This is a simplified representation of the physical process embedded into the model
        Q_phys = self.gamma * I + (1 - self.gamma) * delta_H
        
        # Embed to d_model
        phys_emb = self.embedding(Q_phys.unsqueeze(-1)) # [B, T, D]
        
        return phys_emb

class Model(nn.Module):
    """
    MRPH-TimesNet: Multi-scale Rainfall Lag Attention & Physically Constrained Hydrological Temporal Embedding
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        # New Modules
        # We assume column 0 is Rainfall and column 5 is Water Level based on dataset inspection
        # Adjust these indices if the dataset structure changes
        self.pchte = PCHTE(configs.d_model, rainfall_idx=0, water_level_idx=5)
        self.msrla = MSRLA(configs.d_model, scales=[3, 7, 15, 30])
        # Start with small fusion weights so the extra branches help only when useful.
        self.phys_gate_raw = nn.Parameter(torch.tensor(-2.0))
        self.msrla_gate_raw = nn.Parameter(torch.tensor(-2.0))
        self.phys_norm = nn.LayerNorm(configs.d_model)
        self.msrla_norm = nn.LayerNorm(configs.d_model)
        
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
                
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None

    @property
    def phys_gate(self):
        return torch.sigmoid(self.phys_gate_raw)

    @property
    def msrla_gate(self):
        return torch.sigmoid(self.msrla_gate_raw)

    def _fuse_physical_modules(self, enc_out, raw_x):
        phys_emb = self.phys_norm(self.pchte(raw_x))
        enc_out = enc_out + self.phys_gate * phys_emb

        msrla_out = self.msrla(enc_out)
        enc_out = enc_out + self.msrla_gate * self.msrla_norm(msrla_out)
        return enc_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raw_x = x_enc
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # 1. Data Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self._fuse_physical_modules(enc_out, raw_x)
        
        # Align temporal dimension for prediction
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
            
        # 4. TimesNet Blocks
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            
        # Project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        raw_x = x_enc
        # ... (Similar changes can be applied here if needed, but skipping for brevity as focus is forecasting)
        # For consistency, just calling parent implementation or keeping as is without PCHTE/MSRLA for now
        # unless user requests imputation.
        
        # Normalization
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._fuse_physical_modules(enc_out, raw_x)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        raw_x = x_enc
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._fuse_physical_modules(enc_out, raw_x)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        raw_x = x_enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self._fuse_physical_modules(enc_out, raw_x)
        
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output
