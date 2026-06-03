import argparse
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from sklearn.metrics import r2_score
from torch import optim

from data_provider.data_factory import data_provider
from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding
from models import TimesNet as OfficialTimesNet
from utils.metrics import NSE, metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual


DATASET_PRESETS = {
    "RF610": {
        "dataset_label": "RF610",
        "model_id": "water_MRPH_TimesNet_610",
        "data_path": "water_timeseries_610.csv",
        "seed": 67,
    },
    "RF670": {
        "dataset_label": "RF670",
        "model_id": "water_MRPH_TimesNet_670",
        "data_path": "water_timeseries_670.csv",
        "seed": 488,
    },
}


ABLATION_CONFIGS = {
    "baseline": {
        "label": "Baseline",
        "use_pchte": False,
        "use_msrla": False,
        "use_fusion_gate": False,
        "des": "Ablation_Baseline",
    },
    "pchte": {
        "label": "(a)",
        "use_pchte": True,
        "use_msrla": False,
        "use_fusion_gate": False,
        "des": "Ablation_A",
    },
    "msrla": {
        "label": "(b)",
        "use_pchte": False,
        "use_msrla": True,
        "use_fusion_gate": False,
        "des": "Ablation_B",
    },
    "full_no_gate": {
        "label": "(c)",
        "use_pchte": True,
        "use_msrla": True,
        "use_fusion_gate": False,
        "des": "Ablation_C",
    },
    "full": {
        "label": "(d)",
        "use_pchte": True,
        "use_msrla": True,
        "use_fusion_gate": True,
        "des": "Ablation_D",
    },
}


def fft_for_period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        bsz, time_steps, num_channels = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]], device=x.device
                )
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(bsz, length // period, period, num_channels).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(bsz, -1, num_channels)
            res.append(out[:, : (self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, time_steps, num_channels, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class MSRLA(nn.Module):
    def __init__(self, d_model, scales=None):
        super().__init__()
        self.scales = scales or [3, 7, 15, 30]
        self.d_model = d_model
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=s, padding=s // 2, groups=d_model) for s in self.scales]
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in self.scales])
        self.value_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in self.scales])
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)) / len(self.scales))

    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        query = self.query_proj(x)
        scale_outputs = []
        for i, conv in enumerate(self.convs):
            feat_s = conv(x_t).permute(0, 2, 1)
            key = self.key_projs[i](feat_s)
            value = self.value_projs[i](feat_s)
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
            attn = F.softmax(scores, dim=-1)
            scale_outputs.append(torch.matmul(attn, value))

        final_out = torch.zeros_like(x)
        weights = F.softmax(self.scale_weights, dim=0)
        for i, out_s in enumerate(scale_outputs):
            final_out += weights[i] * out_s
        return self.out_proj(final_out)


class PCHTE(nn.Module):
    def __init__(self, d_model, rainfall_idx=0, water_level_idx=5, max_lag=30):
        super().__init__()
        self.rainfall_idx = rainfall_idx
        self.water_level_idx = water_level_idx
        self.max_lag = max_lag
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        self.beta_raw = nn.Parameter(torch.tensor(-2.0))
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
        _, seq_len, feat_dim = x_enc.shape
        rainfall_idx = min(self.rainfall_idx, feat_dim - 1)
        water_level_idx = min(self.water_level_idx, feat_dim - 1)

        rainfall = x_enc[:, :, rainfall_idx].unsqueeze(1)
        water_level = x_enc[:, :, water_level_idx]

        tau = torch.arange(self.max_lag, device=x_enc.device).float()
        kernel = self.alpha * torch.exp(-self.beta * tau)
        kernel = torch.flip(kernel.view(1, 1, -1), dims=[-1])
        infiltration = F.conv1d(rainfall, kernel, padding=self.max_lag - 1)[:, :, :seq_len].squeeze(1)

        delta_h = torch.zeros_like(water_level)
        delta_h[:, 1:] = water_level[:, 1:] - water_level[:, :-1]
        q_phys = self.gamma * infiltration + (1 - self.gamma) * delta_h
        return self.embedding(q_phys.unsqueeze(-1))


class AblationMRPHTimesNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.use_pchte = configs.use_pchte
        self.use_msrla = configs.use_msrla
        self.use_fusion_gate = configs.use_fusion_gate

        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.pchte = PCHTE(configs.d_model, rainfall_idx=0, water_level_idx=5)
        self.msrla = MSRLA(configs.d_model, scales=[3, 7, 15, 30])
        self.phys_gate_raw = nn.Parameter(torch.tensor(-2.0))
        self.msrla_gate_raw = nn.Parameter(torch.tensor(-2.0))
        self.phys_norm = nn.LayerNorm(configs.d_model)
        self.msrla_norm = nn.LayerNorm(configs.d_model)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name in {"imputation", "anomaly_detection"}:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    @property
    def phys_gate(self):
        return torch.sigmoid(self.phys_gate_raw)

    @property
    def msrla_gate(self):
        return torch.sigmoid(self.msrla_gate_raw)

    def _fuse_physical_modules(self, enc_out, raw_x):
        if self.use_pchte:
            phys_emb = self.phys_norm(self.pchte(raw_x))
            enc_out = enc_out + (self.phys_gate * phys_emb if self.use_fusion_gate else phys_emb)

        if self.use_msrla:
            msrla_out = self.msrla_norm(self.msrla(enc_out))
            enc_out = enc_out + (self.msrla_gate * msrla_out if self.use_fusion_gate else msrla_out)
        return enc_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raw_x = x_enc
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self._fuse_physical_modules(enc_out, raw_x)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        dec_out = self.projection(enc_out)
        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        raise NotImplementedError("Ablation runner currently supports forecasting only.")


def print_dataset_time_spans(root_path, data_path, seq_len):
    csv_path = os.path.join(root_path, data_path)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    total_rows = len(df)
    num_train = int(total_rows * 0.8)
    num_test = int(total_rows * 0.1)
    num_vali = total_rows - num_train - num_test
    border1s = [0, num_train - seq_len, total_rows - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, total_rows]

    print("\n=== Dataset Time Spans (8:1:1) ===")
    for split_name, b1, b2 in zip(["train", "val", "test"], border1s, border2s):
        split_dates = df.iloc[b1:b2]["date"]
        print(
            f"{split_name}: rows={len(split_dates)}, "
            f"start={split_dates.iloc[0].strftime('%Y-%m-%d')}, "
            f"end={split_dates.iloc[-1].strftime('%Y-%m-%d')}"
        )
    print("==================================\n")


def set_random_seed(seed, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


class ExpAblation:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        if self.args.ablation_mode == "baseline":
            model = OfficialTimesNet.Model(self.args).float()
        else:
            model = AblationMRPHTimesNet(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                total_loss.append(criterion(pred, true))
                preds.append(pred.numpy())
                trues.append(true.numpy())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        self.model.train()
        return float(total_loss), float(NSE(preds, trues))

    def train(self, setting):
        _, train_loader = self._get_data("train")
        _, vali_loader = self._get_data("val")
        _, test_loader = self._get_data("test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        best_vali_loss = float("inf")
        best_epoch = 0
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        target = batch_y[:, -self.args.pred_len :, f_dim:]
                        loss = criterion(outputs, target)
                    train_loss.append(loss.item())
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    target = batch_y[:, -self.args.pred_len :, f_dim:]
                    loss = criterion(outputs, target)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_mse, vali_nse = self.vali(vali_loader, criterion)
            test_mse, test_nse = self.vali(test_loader, criterion)
            if vali_mse < best_vali_loss:
                best_vali_loss = float(vali_mse)
                best_epoch = epoch + 1

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali MSE: {3:.7f} "
                "Test MSE: {4:.7f} | Vali NSE: {5:.7f} Test NSE: {6:.7f} | "
                "Best Vali Loss So Far: {7:.7f} (Epoch {8})".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_mse,
                    test_mse,
                    vali_nse,
                    test_nse,
                    best_vali_loss,
                    best_epoch,
                )
            )
            early_stopping(vali_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, load_checkpoint=False):
        test_data, test_loader = self._get_data("test")
        if load_checkpoint:
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints", setting, "checkpoint.pth")))

        preds = []
        trues = []
        folder_path = Path(self.args.ablation_root) / self.args.dataset_label / setting
        folder_path.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pred_vis = np.concatenate((input_data[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pred_vis, str(folder_path / f"{i}.pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        nse = float(NSE(preds, trues))
        r2 = float(r2_score(trues.flatten(), preds.flatten()))

        np.save(folder_path / "metrics.npy", np.array([mae, mse, rmse, mape, mspe], dtype=float))
        np.save(folder_path / "pred.npy", preds)
        np.save(folder_path / "true.npy", trues)

        raw_df = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        num_train = int(len(raw_df) * 0.8)
        num_test = int(len(raw_df) * 0.1)
        target_mean = float(raw_df.iloc[:num_train][self.args.target].mean())
        target_scale = float(raw_df.iloc[:num_train][self.args.target].std(ddof=0))
        if abs(target_scale) < 1e-12:
            target_scale = 1.0

        pred_inverse = preds.reshape(-1) * target_scale + target_mean
        true_inverse = trues.reshape(-1) * target_scale + target_mean
        test_dates = raw_df.iloc[len(raw_df) - num_test :]["date"].reset_index(drop=True)
        prediction_table = pd.DataFrame(
            {
                "date": test_dates.dt.strftime("%Y-%m-%d"),
                "true_value": true_inverse,
                "pred_value": pred_inverse,
            }
        )
        prediction_table.to_csv(folder_path / "test_predictions_inverse.csv", index=False)

        metrics_payload = {
            "dataset": self.args.dataset_label,
            "setting": setting,
            "seed": int(self.args.seed),
            "ablation_mode": self.args.ablation_mode,
            "ablation_label": self.args.ablation_label,
            "modules": {
                "TimesNet": True,
                "PCHTE": bool(self.args.use_pchte),
                "MSRLA": bool(self.args.use_msrla),
                "FusionGate": bool(self.args.use_fusion_gate),
            },
            "metrics": {
                "nse": nse,
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "r2": r2,
            },
        }
        with open(folder_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

        print(
            f"[{self.args.dataset_label}][{self.args.ablation_label}] "
            f"NSE={nse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}, R2={r2:.4f}"
        )
        return metrics_payload

    def print_physical_params(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if not hasattr(model, "pchte"):
            print("\n=== Physical Parameters ===")
            print("Baseline uses official TimesNet and has no PCHTE/Fusion parameters.")
            print("=================================\n")
            return
        print("\n=== PCHTE Physical Parameters ===")
        print(f"Alpha (Infiltration Coeff): {model.pchte.alpha.item():.4f}")
        print(f"Beta (Decay Coeff): {model.pchte.beta.item():.4f}")
        print(f"Gamma (Response Coeff): {model.pchte.gamma.item():.4f}")
        print("=================================\n")


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_common_args(parser):
    parser.add_argument("--task_name", type=str, default="long_term_forecast", help="task name")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model", type=str, default="MRPH_TimesNet", help="model name")
    parser.add_argument("--data", type=str, default="custom", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="root path of the data file")
    parser.add_argument("--features", type=str, default="MS", help="forecasting task, options:[M, S, MS]")
    parser.add_argument("--target", type=str, default="water_inflow", help="target feature in S or MS task")
    parser.add_argument("--freq", type=str, default="d", help="freq for time features encoding")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")
    parser.add_argument("--seq_len", type=int, default=24, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=1, help="start token length")
    parser.add_argument("--pred_len", type=int, default=1, help="prediction sequence length")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="subset for M4")
    parser.add_argument("--inverse", action="store_true", default=False, help="inverse output data")
    parser.add_argument("--expand", type=int, default=2, help="expansion factor for Mamba")
    parser.add_argument("--d_conv", type=int, default=4, help="conv kernel size for Mamba")
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=1, help="output size")
    parser.add_argument("--d_model", type=int, default=32, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=32, help="dimension of fcn")
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--factor", type=int, default=3, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        default=True,
        help="whether to use distilling in encoder, using this argument means not using distilling",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in encoder")
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=150, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="optimizer learning rate")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", default=False, help="use automatic mixed precision training")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False, help="use multiple gpus")
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
    parser.add_argument("--augmentation_ratio", type=int, default=0, help="How many times to augment")
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        default=True,
        help="disable deterministic behavior if you want faster but less reproducible runs",
    )
    parser.add_argument("--jitter", action="store_true", default=False, help="jitter augmentation")
    parser.add_argument("--ablation_root", type=str, default="./results/dl_result/ablation", help="ablation output root")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["RF610", "RF670"],
        choices=list(DATASET_PRESETS.keys()),
        help="datasets to run",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(ABLATION_CONFIGS.keys()),
        choices=list(ABLATION_CONFIGS.keys()),
        help="ablation modes to run",
    )
    return parser


def build_runtime_args(base_args, dataset_name, ablation_mode):
    args = deepcopy(base_args)
    dataset_preset = DATASET_PRESETS[dataset_name]
    ablation_preset = ABLATION_CONFIGS[ablation_mode]
    args.dataset_label = dataset_name
    args.model_id = dataset_preset["model_id"]
    args.data_path = dataset_preset["data_path"]
    args.seed = dataset_preset["seed"]
    args.use_pchte = ablation_preset["use_pchte"]
    args.use_msrla = ablation_preset["use_msrla"]
    args.use_fusion_gate = ablation_preset["use_fusion_gate"]
    args.ablation_mode = ablation_mode
    args.ablation_label = ablation_preset["label"]
    args.des = ablation_preset["des"]
    return args


def build_setting(args, run_index):
    return "{}_{}_sl{}_pl{}_{}_{}".format(
        args.model_id,
        args.model,
        args.seq_len,
        args.pred_len,
        args.des,
        run_index,
    )


def save_dataset_summary(dataset_name, rows, output_root):
    summary_dir = Path(output_root) / dataset_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_path = summary_dir / f"ablation_{dataset_name.lower()}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    ranking_df = summary_df.sort_values(by="nse", ascending=False).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
    ranking_path = summary_dir / f"ablation_{dataset_name.lower()}_ranking_by_nse.csv"
    ranking_df.to_csv(ranking_path, index=False)
    return summary_path, ranking_path


def main():
    parser = add_common_args(argparse.ArgumentParser(description="MRPH-TimesNet ablation experiment runner"))
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(device_id) for device_id in device_ids]
        args.gpu = args.device_ids[0]

    print("Ablation args:")
    print(args)

    all_results = {}
    for dataset_name in args.datasets:
        dataset_rows = []
        for ablation_mode in args.modes:
            run_args = build_runtime_args(args, dataset_name, ablation_mode)
            set_random_seed(run_args.seed, deterministic=run_args.deterministic)
            print(
                "\n========== Running {0} | {1} ==========".format(
                    dataset_name,
                    run_args.ablation_label,
                )
            )
            print_dataset_time_spans(run_args.root_path, run_args.data_path, run_args.seq_len)

            for ii in range(run_args.itr):
                setting = build_setting(run_args, ii)
                exp = ExpAblation(run_args)
                if run_args.is_training:
                    print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    exp.train(setting)
                    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                    result = exp.test(setting)
                else:
                    print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                    result = exp.test(setting, load_checkpoint=True)

                exp.print_physical_params()
                dataset_rows.append(
                    {
                        "dataset": dataset_name,
                        "ablation": run_args.ablation_mode,
                        "label": run_args.ablation_label,
                        "A_TimesNet": 1,
                        "B_PCHTE": int(run_args.use_pchte),
                        "C_MSRLA": int(run_args.use_msrla),
                        "D_FusionGate": int(run_args.use_fusion_gate),
                        "seed": run_args.seed,
                        "nse": result["metrics"]["nse"],
                        "rmse": result["metrics"]["rmse"],
                        "mae": result["metrics"]["mae"],
                        "mape": result["metrics"]["mape"],
                        "r2": result["metrics"]["r2"],
                        "setting": result["setting"],
                        "model_dir": str(Path(run_args.ablation_root) / dataset_name / setting),
                    }
                )
                clear_cuda_cache()

        summary_path, ranking_path = save_dataset_summary(dataset_name, dataset_rows, args.ablation_root)
        all_results[dataset_name] = {
            "summary": str(summary_path),
            "ranking": str(ranking_path),
            "rows": dataset_rows,
        }
        print(f"\nSaved {dataset_name} summary to: {summary_path}")
        print(f"Saved {dataset_name} ranking to: {ranking_path}")

    merged_rows = [row for item in all_results.values() for row in item["rows"]]
    merged_path = Path(args.ablation_root) / "ablation_all_summary.csv"
    pd.DataFrame(merged_rows).to_csv(merged_path, index=False)
    print(f"\nSaved merged summary to: {merged_path}")


if __name__ == "__main__":
    main()
