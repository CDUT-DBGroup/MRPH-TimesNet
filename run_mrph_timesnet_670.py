import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from models import MRPH_TimesNet
from utils.metrics import metric, NSE
from utils.tools import EarlyStopping, adjust_learning_rate, visual


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


def set_random_seed(seed: int, deterministic: bool = True) -> None:
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


class Exp_MRPH_TimesNet:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu
            ) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        model = MRPH_TimesNet.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                total_loss.append(criterion(pred, true))
                preds.append(pred.numpy())
                trues.append(true.numpy())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mse = float(total_loss)
        nse = NSE(preds, trues)
        self.model.train()
        return mse, nse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        best_vali_loss = float("inf")
        best_epoch = 0

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

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

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())

                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
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
            vali_mse, vali_nse = self.vali(vali_data, vali_loader, criterion)
            test_mse, test_nse = self.vali(test_data, test_loader, criterion)
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        folder_path = "./results/dl_result/test_results/" + setting + "/"
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_data.shape
                        input_data = test_data.inverse_transform(input_data.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pred_vis = np.concatenate((input_data[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pred_vis, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("test shape:", preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        nse = NSE(preds, trues)
        print(f"mse:{mse}, rmse:{rmse}, mae:{mae}, nse:{nse}")

        from sklearn.metrics import r2_score

        r2 = r2_score(trues.flatten(), preds.flatten())
        print(f"R2:{r2}")

        results_log_path = "./results/dl_result/result_long_term_forecast.txt"
        os.makedirs(os.path.dirname(results_log_path), exist_ok=True)
        with open(results_log_path, "a", encoding="utf-8") as f:
            f.write(setting + "  \n")
            f.write(f"mse:{mse}, rmse:{rmse}, mae:{mae}, nse:{nse}, r2:{r2}")
            f.write("\n\n")

        np.save(os.path.join(folder_path, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)

        raw_df = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        num_train = int(len(raw_df) * 0.8)
        num_test = int(len(raw_df) * 0.1)
        target_mean = float(raw_df.iloc[:num_train][self.args.target].mean())
        target_scale = float(raw_df.iloc[:num_train][self.args.target].std(ddof=0))
        if abs(target_scale) < 1e-12:
            target_scale = 1.0

        true_inverse = trues.reshape(-1) * target_scale + target_mean
        pred_inverse = preds.reshape(-1) * target_scale + target_mean
        test_dates = raw_df.iloc[len(raw_df) - num_test:]["date"].reset_index(drop=True)

        prediction_table = pd.DataFrame(
            {
                "date": test_dates.dt.strftime("%Y-%m-%d"),
                "true_value": true_inverse,
                "pred_value": pred_inverse,
            }
        )
        prediction_table.to_csv(os.path.join(folder_path, "test_predictions_inverse.csv"), index=False)
        prediction_table.to_csv(
            os.path.join("./results/dl_result", f"{setting}_test_predictions_inverse.csv"),
            index=False,
        )

    def print_physical_params(self):
        print("\n=== PCHTE Physical Parameters ===")
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        print(f"Alpha (Infiltration Coeff): {model.pchte.alpha.item():.4f}")
        print(f"Beta (Decay Coeff): {model.pchte.beta.item():.4f}")
        print(f"Gamma (Response Coeff): {model.pchte.gamma.item():.4f}")
        print("=================================\n")


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="MRPH-TimesNet Experiment")

    parser.add_argument("--task_name", type=str, default="long_term_forecast", help="task name")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="water_MRPH_TimesNet_670", help="model id")
    parser.add_argument("--model", type=str, default="MRPH_TimesNet", help="model name")

    parser.add_argument("--data", type=str, default="custom", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="water_timeseries_670.csv", help="data file")
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
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")

    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=150, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="Comparison_Exp", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", default=False, help="use automatic mixed precision training")

    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False, help="use multiple gpus")
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")

    parser.add_argument("--augmentation_ratio", type=int, default=0, help="How many times to augment")
    parser.add_argument("--seed", type=int, default=488, help="random seed")
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        default=True,
        help="disable deterministic behavior if you want faster but less reproducible runs",
    )
    parser.add_argument("--jitter", action="store_true", default=False, help="jitter augmentation")

    args = parser.parse_args()
    set_random_seed(args.seed, deterministic=args.deterministic)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(device_id) for device_id in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)
    print(f"Reproducibility seed fixed at: {args.seed}")
    print_dataset_time_spans(args.root_path, args.data_path, args.seq_len)

    Exp = Exp_MRPH_TimesNet

    if args.is_training:
        for ii in range(args.itr):
            setting = "{}_{}_sl{}_pl{}_{}_{}".format(
                args.model_id,
                args.model,
                args.seq_len,
                args.pred_len,
                args.des,
                ii,
            )
            exp = Exp(args)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)

            print(f">>>>>>>Physical Parameters : {setting}<<<<<<<<<<<<<<<<<<<<<")
            exp.print_physical_params()
            clear_cuda_cache()
    else:
        ii = 0
        setting = "{}_{}_sl{}_pl{}_{}_{}".format(
            args.model_id,
            args.model,
            args.seq_len,
            args.pred_len,
            args.des,
            ii,
        )
        exp = Exp(args)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        exp.print_physical_params()
        clear_cuda_cache()


if __name__ == "__main__":
    main()
