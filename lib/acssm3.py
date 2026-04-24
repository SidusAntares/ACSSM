import os
import time
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime

from model.adaptation import ClassCenterAligner, TFMPTF_Projector, AttentionGate
from model.moco import AdaMoCo3D
# from model import Decoder
from model.TFMPTF import TFMPTF
import torch.nn.functional as F

now = datetime.now()
run_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

import lib.sde2 as sde
from lib.losses import MSE_, GNLL_, BNLL_, CNLL_, F1_
from lib.data_utils import adjust_obs_for_extrapolation
from lib.utils import get_time
from einops import rearrange

def match(domain):
    if domain == 'france/30TXT/2017':
        return 'FR1'
    elif domain == 'france/31TCJ/2017':
        return 'FR2'
    elif domain == 'denmark/32VNH/2017':
        return 'DK1'
    else:
        return 'AT1'

def lyap_loss(src_lyap, trg_lyap, min_bins=5, max_bins=20, min_range=0.1):
    src_lyap = src_lyap.float()
    trg_lyap = trg_lyap.float()

    # 动态范围计算（使用detach避免梯度传播）
    with torch.no_grad():
        src_max = src_lyap.abs().max().item()
        trg_max = trg_lyap.abs().max().item()
        dynamic_range = max(src_max, trg_max, min_range)

        # 自适应bins计算（基于数据量）
        total_elements = min(src_lyap.numel(), trg_lyap.numel())
        adaptive_bins = max(min_bins, min(max_bins, total_elements // 1000))
        num_bins = max(adaptive_bins, min_bins)  # 确保至少有最小bins数

    # 直方图计算
    src_lyap_dist = torch.histc(
        src_lyap.abs().flatten().detach(),  # detach避免不必要的梯度计算
        bins=num_bins,
        min=0,
        max=dynamic_range
    )
    trg_lyap_dist = torch.histc(
        trg_lyap.abs().flatten().detach(),
        bins=num_bins,
        min=0,
        max=dynamic_range
    )

    # 数值稳定性：添加小常数防止除零
    eps = 1e-8
    src_lyap_dist = src_lyap_dist + eps
    trg_lyap_dist = trg_lyap_dist + eps

    # 归一化为概率分布
    src_lyap_dist = src_lyap_dist / src_lyap_dist.sum()
    trg_lyap_dist = trg_lyap_dist / trg_lyap_dist.sum()

    # 计算累积分布函数（CDF）
    src_cum = torch.cumsum(src_lyap_dist, dim=0)
    trg_cum = torch.cumsum(trg_lyap_dist, dim=0)

    # 1D Wasserstein距离（Earth Mover's Distance）
    wasserstein_dist = torch.sum(torch.abs(src_cum - trg_cum))

    # 缩放因子：将离散距离映射到连续空间
    bin_width = dynamic_range / num_bins
    dyn_loss = wasserstein_dist * bin_width

    # 添加小的正则化项确保数值稳定性
    dyn_loss = dyn_loss + 1e-6 * (src_lyap.abs().mean() + trg_lyap.abs().mean())

    return dyn_loss


class Decoder(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, args, in_dim):
        super().__init__()

        self.dataset = args.dataset
        self.task = args.task
        self.ld = args.state_dim
        self.in_dim = in_dim
        self.od = args.out_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, self.od)  # od = num_classes
        )

    def forward(self, input):

        if self.dataset == 'pendulum' and self.task == 'interpolation':
            n, b, t, d = input.size()
            input = rearrange(input, 'n b t d -> (n b t) d')
            out = self.decoder(input)
            out = rearrange(out, '(n b t) x y z -> n b t x y z', n=n, b=b, t=t)
        else:
            out = self.decoder(input)
        return out

class ACSSM():
    def __init__(self, args):
        super(ACSSM, self).__init__()
        self.n_epoch = args.epochs
        self.task = args.task
        self.cut_time = args.cut_time
        self.device = args.device
        self.dataset = args.dataset

        self.dynamics = sde.LinearSDE(args)
        self.dynamics = self.dynamics.to(self.device)
        self.decoder = Decoder(args, args.state_dim).to(self.device)

        self.source_name = match(args.source)
        self.target_name = match(args.target)

        self.log_history = []
        self.noise_scale = args.ns
        self.seed = args.seed

        # losses
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="mean")

        # adaptation
        self.lambda_ctr = args.lambda_ctr
        self.lambda_align = args.lambda_align
        # self.momentum_classifier = Decoder(args).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.dynamics.parameters()) +
            list(self.decoder.parameters()),
            lr=args.lr, weight_decay=args.wd)
        self.adaptation_optimizer = torch.optim.AdamW(
            list(self.decoder.parameters()),
            lr=args.lr,
            weight_decay=args.wd
        )

    def add_noise(self, src_obs, trg_obs):
        assert trg_obs.shape == src_obs.shape
        trg_energy = trg_obs.pow(2).mean().sqrt()  # 全局 RMS
        src_energy = src_obs.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # 单样本 RMS
        energy_ratio = trg_energy / (src_energy + 1e-8)
        noise_pattern = trg_obs.mean(dim=0, keepdim=True) - src_obs.mean(dim=(1, 2), keepdim=True)
        return src_obs + noise_pattern * self.noise_scale * energy_ratio

    def train_and_eval_adaptation(self, src_train_loader, src_test_loader, trg_train_loader, trg_test_loader):
        assert self.dataset == 'timematch' and self.task == 'classification', "the function created for timematch classification"
        self.pertrain_start_time = time.time()
        print("========== Stage 1: Pretraining ==========")
        for epoch in range(self.n_epoch):
            epoch_ll = 0
            epoch_mse = 0
            epoch_loss = 0
            num_data = 0

            for _, (src_data, trg_data) in enumerate(zip(src_train_loader, trg_train_loader)):
                src_obs = src_data['inp_obs'].to(self.device)
                truth = src_data['evd_obs'].to(self.device)
                src_times = src_data['inp_tid'].to(self.device)
                src_labels = src_data['aux_obs'].to(self.device)
                src_valid = src_data['obs_valid'].to(self.device)
                mask_truth = src_data['mask_truth'].to(self.device)
                mask_obs = src_data['mask_obs'].to(self.device)

                self.optimizer.zero_grad()

                hidden_states, L_alpha = self.dynamics(src_obs, src_times, src_valid, mask_obs, n_samples=3,
                                                       epoch=epoch)
                mean = self.decoder(hidden_states)

                # Example loss
                batch_len = truth.size(0)

                train_nll, train_mse = CNLL_(src_labels, mean)
                loss = train_nll + L_alpha

                loss.backward()

                nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                self.optimizer.step()

                epoch_mse += train_mse.item()
                epoch_ll += train_nll.item()
                epoch_loss += loss.item()
                num_data += batch_len

            print('--------------[ {} || {} ]--------------'.format(epoch, self.n_epoch))
            print(
                'Pretrain [Time elapsed] : {0}:{1:02d}:{2:05.2f}'.format(*get_time(time.time() - self.pertrain_start_time)))
            with torch.no_grad():
                test_mse, test_nll, impute_mse, impute_nll, test_f1 = self.eval_pretrain(src_test_loader)

            log_dict = {
                "train_nll": epoch_ll / num_data,
                "train_loss": epoch_loss / num_data,
                "eval_nll": test_nll,
            }
            print("Pretrain [Train  ] NLL : {:.6f} || ACC : {:.6f}".format(epoch_ll / num_data, epoch_mse / num_data))
            print("Pretrain [Eval  ] NLL : {:.6f} || ACC : {:.6f} || Macro F1 : {:.6f}".format(test_nll, test_mse, test_f1))
            log_dict.update({
                "train_acc": epoch_mse / num_data,
                "eval_acc": test_mse,
                "eval_macro_f1": test_f1
            })

            log_dict["epoch"] = epoch
            self.log_history.append(log_dict)

            if (epoch + 1) % 50 == 0:
                if not os.path.exists(f'./checkpoints/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/'):
                    os.makedirs(f'./checkpoints/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/')
                torch.save({
                    'epoch': epoch,
                    'dynamics_state_dict': self.dynamics.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },
                    f'./checkpoints/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/pretrain_{epoch + 1}.pt')
        log_df = pd.DataFrame(self.log_history)
        csv_path = f"./adapttaion_logs/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/pretrain_{epoch + 1}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        log_df.to_csv(csv_path, index=False)
        print(f"Training log saved to {csv_path}")

    def adaptation(self, src_train_loader, trg_train_loader, trg_test_loader, load_source_checkpoint=None):
        assert self.dataset == 'timematch' and self.task == 'classification', "the function created for timematch classification"
        self.ad_start_time = time.time()

        print("========== Stage 2: Domain Adaptation Fine-tuning ==========")
        if load_source_checkpoint:
            checkpoint = torch.load(load_source_checkpoint)
            self.dynamics.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded source model from {load_source_checkpoint}")
        for name, param in self.dynamics.named_parameters():
            if 'coeff_net' in name or 'D' in name:  # 冻结动态核心
                param.requires_grad = False
            else:
                param.requires_grad = True  # 保持编码器可训练

        for epoch in range(self.n_epoch):
            epoch_ll = 0
            epoch_mse = 0
            epoch_loss = 0
            epoch_dyn_loss = 0
            num_data = 0

            for _, (src_data, trg_data) in enumerate(zip(src_train_loader, trg_train_loader)):
                src_obs = src_data['inp_obs'].to(self.device)
                src_times = src_data['inp_tid'].to(self.device)
                src_labels = src_data['aux_obs'].to(self.device)
                src_valid = src_data['obs_valid'].to(self.device)
                mask_obs = src_data['mask_obs'].to(self.device)

                trg_obs = trg_data['inp_obs'].to(self.device)
                trg_times = trg_data['inp_tid'].to(self.device)
                trg_valid = trg_data['obs_valid'].to(self.device)

                self.adaptation_optimizer.zero_grad()

                src_obs = self.add_noise(src_obs, trg_obs)

                with torch.no_grad():
                    src_feat, L_alpha = self.dynamics(
                        src_obs, src_times, src_valid, mask_obs,
                        n_samples=3, epoch=None
                    )
                with torch.no_grad():
                    trg_feat, _ = self.dynamics(
                        trg_obs, trg_times, trg_valid, None,
                        n_samples=3, epoch=None
                    )
                with torch.no_grad():
                    src_lyap = self.dynamics.alphas[:, 1:] - self.dynamics.alphas[:, :-1]  # 源域
                    trg_lyap = self.dynamics.alphas[:, 1:] - self.dynamics.alphas[:, :-1]  # 目标域
                dyn_loss = lyap_loss(src_lyap, trg_lyap, min_bins=8, max_bins=25)
                with torch.no_grad():
                    dyn_weight = 0.5 * (1 - torch.exp(-torch.tensor(epoch / 10.0, device=self.device)))

                logits_src = self.decoder(src_feat)
                logits_trg = self.decoder(trg_feat)
                train_nll, _ = CNLL_(src_labels, logits_src)
                loss_cls = train_nll  # + L_alpha
                loss = loss_cls + dyn_weight * dyn_loss

                with torch.no_grad():
                    lyap_magnitude = src_lyap.abs().mean()
                    grad_scale = torch.exp(-lyap_magnitude)  # 混沌区域梯度衰减

                for param in self.decoder.parameters():
                    if param.grad is not None:
                        param.grad *= grad_scale

                loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)
                self.adaptation_optimizer.step()

                epoch_ll += loss_cls.item()
                epoch_loss += loss.item()
                epoch_dyn_loss += dyn_loss.item()
                num_data += src_obs.size(0)

            print('--------------[ {} || {} ]--------------'.format(epoch, self.n_epoch))
            print('Adaptation [Time elapsed] : {0}:{1:02d}:{2:05.2f}'.format(*get_time(time.time() - self.ad_start_time)))
            with torch.no_grad():
                test_acc, test_f1 = self.eval_adaptation(trg_test_loader)

            log_dict = {
                "train_nll": epoch_ll / num_data,
                "train_dyn_loss": epoch_dyn_loss / num_data,
                "train_loss": epoch_loss / num_data,
            }
            print("Adaptation [Train  ] TOTAL : {:.6f} || DYN : {:.6f}".format(epoch_loss / num_data,
                                                                                         epoch_dyn_loss / num_data))
            print("Adaptation [Eval  ] ACC : {:.6f} || Macro F1 : {:.6f}".format(test_acc, test_f1))

            log_dict.update({
                "train_acc": epoch_mse / num_data,
                "eval_acc": test_acc,
                "eval_macro_f1": test_f1
            })

            log_dict["epoch"] = epoch
            self.log_history.append(log_dict)

            if (epoch + 1) % 50 == 0:
                if not os.path.exists(f'./checkpoints/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/'):
                    os.makedirs(f'./checkpoints/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/')
                torch.save({
                    'epoch': epoch,
                    'dynamics_state_dict': self.dynamics.state_dict(),  # 冻结的主干
                    'adaptation_optimizer_state_dict': self.adaptation_optimizer.state_dict(),
                },
                    f'./checkpoints/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/adaptation_{epoch + 1}.pt')
        log_df = pd.DataFrame(self.log_history)
        csv_path = f"./adapttaion_logs/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}/{run_id}_{self.seed}_{self.noise_scale}/v3/adaptation_{epoch + 1}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        log_df.to_csv(csv_path, index=False)
        print(f"Training log saved to {csv_path}")

    def eval_pretrain(self, dl):
        assert self.dataset == 'timematch' and self.task == 'classification'
        epoch_mse = 0
        epoch_ll = 0
        epoch_f1 = 0

        epoch_impute_mse = 0
        epoch_impute_ll = 0

        num_data = 0
        num_batches = 0
        for _, data in enumerate(dl):
            obs = data['inp_obs'].to(self.device)
            truth = data['evd_obs'].to(self.device)
            obs_times = data['inp_tid'].to(self.device)
            labels = data['aux_obs'].to(self.device)
            obs_valid = data['obs_valid'].to(self.device)
            mask_obs = data['mask_obs'].to(self.device)

            hidden_states, L_alpha = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=32, epoch=None)
            mean = self.decoder(hidden_states)

            batch_len = truth.size(0)
            eval_nll, eval_mse = CNLL_(labels, mean)
            macro_f1 = F1_(labels, mean, average='macro', zero_division=0)
            epoch_f1 += macro_f1
            num_batches += 1

            epoch_mse += eval_mse.item()
            epoch_ll += eval_nll.item()

            num_data += batch_len
        avg_macro_f1 = epoch_f1 / num_batches
        return epoch_mse / num_data, epoch_ll / num_data, epoch_impute_mse / num_data, epoch_impute_ll / num_data, avg_macro_f1

    def eval_adaptation(self, dl):
        """域适应专用评估（使用对齐模块的分类器）"""
        epoch_mse = 0
        epoch_ll = 0
        epoch_f1 = 0

        num_data = 0
        num_batches = 0

        self.decoder.eval()

        with torch.no_grad():
            for _, data in enumerate(dl):
                obs = data['inp_obs'].to(self.device)
                obs_times = data['inp_tid'].to(self.device)
                labels = data['aux_obs'].to(self.device)
                obs_valid = data['obs_valid'].to(self.device)
                mask_obs = data['mask_obs'].to(self.device)

                # ACSSM提取隐状态
                hidden_states, _ = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=32, epoch=None)

                logits = self.decoder(hidden_states)

                num_data += obs.size(0)
                eval_nll, eval_mse = CNLL_(labels, logits)
                macro_f1 = F1_(labels, logits, average='macro', zero_division=0)
                epoch_f1 += macro_f1
                num_batches += 1
                epoch_mse += eval_mse.item()
                epoch_ll += eval_nll.item()

        avg_acc = epoch_mse / num_data
        avg_f1 = epoch_f1 / num_batches
        return avg_acc, avg_f1


