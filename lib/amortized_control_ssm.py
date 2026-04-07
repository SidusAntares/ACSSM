import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime

from tfda.loss import CrossEntropyLabelSmooth

now = datetime.now()
run_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

import lib.sde as sde
from lib.losses import MSE_, GNLL_, BNLL_, CNLL_, F1_
from lib.data_utils import adjust_obs_for_extrapolation
from lib.utils import get_time

def match(domain):
    if domain == 'france/30TXT/2017':
        return 'FR1'
    elif domain == 'france/31TCJ/2017':
        return 'FR2'
    elif domain == 'denmark/32VNH/2017':
        return 'DK1'
    else:
        return 'AT1'



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
        self.optimizer = torch.optim.AdamW(self.dynamics.parameters(), lr=args.lr, weight_decay=args.wd)

        self.source_name = match(args.source)
        self.target_name = match(args.target)

        self.log_history = []
        self.noise_scale = args.ns
        self.seed = args.seed

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(args.num_classes, self.device, epsilon=0.1, )
        self.kl_loss = nn.KLDivLoss(reduction="mean")

    def add_nosie(self, src_obs, trg_obs):
        assert trg_obs.shape == src_obs.shape
        trg_energy = trg_obs.pow(2).mean().sqrt()  # 全局 RMS
        src_energy = src_obs.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # 单样本 RMS
        energy_ratio = trg_energy / (src_energy + 1e-8)
        noise_pattern = trg_obs.mean(dim=0, keepdim=True) - src_obs.mean(dim=(1, 2), keepdim=True)
        return src_obs + noise_pattern * self.noise_scale * energy_ratio

    def train_and_eval_adaptation(self, src_data_loader, trg_test_loder, trg_train_loader):
        assert self.dataset == 'timematch' and self.task == 'classification',"the function created for timematch classification"
        self.start_time = time.time()
        for epoch in range(self.n_epoch):
            epoch_ll = 0
            epoch_mse = 0
            epoch_loss = 0
            num_data = 0

            for _, (src_data, trg_data) in enumerate(zip(src_data_loader, trg_train_loader)):

                src_obs = src_data['inp_obs'].to(self.device)
                truth = src_data['evd_obs'].to(self.device)
                src_times = src_data['inp_tid'].to(self.device)
                src_labels = src_data['aux_obs'].to(self.device)
                src_valid = src_data['obs_valid'].to(self.device)
                mask_truth = src_data['mask_truth'].to(self.device)
                mask_obs = src_data['mask_obs'].to(self.device)

                trg_obs = trg_data['inp_obs'].to(self.device)
                trg_times = trg_data['inp_tid'].to(self.device)
                trg_valid = trg_data['obs_valid'].to(self.device)
                # 这里需要修改，使用数据原长，意味着加噪函数逻辑需要修改
                assert trg_obs.shape == src_obs.shape
                src_obs = self.add_nosie(src_obs, src_times)

                self.optimizer.zero_grad()
                # ACSSM api 输出隐状态
                out, L_alpha = self.dynamics(src_obs, src_times, src_valid, mask_obs, n_samples=3, epoch=epoch)
                mean, var = out

                # Example loss
                batch_len = truth.size(0)

                train_nll, train_mse = CNLL_(src_labels, mean)


                loss = train_nll + L_alpha

                loss.backward()

                nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1)
                self.optimizer.step()

                epoch_mse += train_mse.item()
                epoch_ll += train_nll.item()
                epoch_loss += loss.item()
                num_data += batch_len

            print('--------------[ {} || {} ]--------------'.format(epoch, self.n_epoch))
            print('[Time elapsed] : {0}:{1:02d}:{2:05.2f}'.format(*get_time(time.time() - self.start_time)))
            with torch.no_grad():
                test_mse, test_nll, impute_mse, impute_nll, test_f1 = self.eval_func(trg_test_loder)

            log_dict = {
                "train_nll": epoch_ll / num_data,
                "train_loss": epoch_loss / num_data,
                "eval_nll": test_nll,
            }
            print("[Train  ] NLL : {:.6f} || ACC : {:.6f}".format(epoch_ll / num_data, epoch_mse / num_data))
            print("[Eval  ] NLL : {:.6f} || ACC : {:.6f} || Macro F1 : {:.6f}".format(test_nll, test_mse, test_f1))
            log_dict.update({
                "train_acc": epoch_mse / num_data,
                "eval_acc": test_mse,
                "eval_macro_f1": test_f1
            })

            log_dict["epoch"] = epoch
            self.log_history.append(log_dict)

            if (epoch + 1) % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save({
                    'epooch': epoch,
                    'model_state_dict': self.dynamics.state_dict()},
                    f'./checkpoints/{self.dataset}_{self.task}_{epoch + 1}.pt')
        log_df = pd.DataFrame(self.log_history)
        csv_path = f"./new_logs/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}_{self.seed}_{self.noise_scale}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        log_df.to_csv(csv_path, index=False)
        print(f"Training log saved to {csv_path}")

    def train_and_eval(self, train_dl, eval_dl):

        self.start_time=time.time()
        for epoch in range(self.n_epoch):
            epoch_ll = 0
            epoch_mse = 0
            epoch_loss = 0
            num_data = 0
            for _, data in enumerate(train_dl):

                if self.dataset == 'pendulum':
                    if self.task == 'regression':
                        obs, truth, obs_times, obs_valid = [j.to(self.device).to(torch.float32) for j in data]
                        mask_obs = None
                        mask_truth = None
                    else:
                        obs, truth, obs_valid, obs_times, mask_truth = [j.to(self.device).to(torch.float32) for j in data]
                        mask_obs = None
                elif self.dataset == 'person_activity':
                    obs = data['inp_obs'].to(self.device)
                    truth = data['evd_obs'].to(self.device)
                    obs_times = data['inp_tid'].to(self.device)
                    labels = data['aux_obs'].to(self.device)
                    b, t, _ = obs.size()
                    obs_valid = torch.ones(b, t).to(obs.device)
                    mask_obs = None
                    mask_truth = None
                elif self.dataset == 'timematch':
                    obs = data['inp_obs'].to(self.device)
                    truth = data['evd_obs'].to(self.device)
                    obs_times = data['inp_tid'].to(self.device)
                    labels = data['aux_obs'].to(self.device)
                    obs_valid = data['obs_valid'].to(self.device)
                    mask_truth = data['mask_truth'].to(self.device)
                    mask_obs = data['mask_obs'].to(self.device)
                else:
                    obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [j.to(self.device).to(torch.float32) for j in data]
                    if self.task == 'extrapolation':
                        obs, obs_valid = adjust_obs_for_extrapolation(obs, obs_valid, obs_times, self.cut_time)

                self.optimizer.zero_grad()
                out, L_alpha = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=3, epoch=epoch)
                mean, var = out

                # Example loss
                batch_len = truth.size(0)

                if self.dataset == 'pendulum' and self.task == 'interpolation':
                    train_nll = BNLL_(truth, mean) * batch_len
                    train_mse = MSE_(truth.flatten(start_dim=2), mean.flatten(start_dim=3), mask = mask_truth.flatten(start_dim=2)) * batch_len
                elif self.task == 'classification':
                    train_nll, train_mse = CNLL_(labels, mean)
                else:
                    train_nll = GNLL_(truth, mean, var, mask = mask_truth) * batch_len
                    train_mse = MSE_(truth, mean, mask = mask_truth) * batch_len

                loss = train_nll + L_alpha

                loss.backward()
                if self.task == 'classification':
                    nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1)
                self.optimizer.step()

                epoch_mse += train_mse.item()
                epoch_ll += train_nll.item()
                epoch_loss += loss.item()
                num_data += batch_len

            print('--------------[ {} || {} ]--------------'.format(epoch, self.n_epoch))
            print('[Time elapsed] : {0}:{1:02d}:{2:05.2f}'.format(*get_time(time.time() - self.start_time)))
            with torch.no_grad():
                test_mse, test_nll, impute_mse, impute_nll,  test_f1 = self.eval_func(eval_dl)

            log_dict = {
                "train_nll": epoch_ll / num_data,
                "train_loss": epoch_loss / num_data,
                "eval_nll": test_nll,
            }
            if self.task == 'classification':
                print("[Train  ] NLL : {:.6f} || ACC : {:.6f}".format(epoch_ll/num_data, epoch_mse/num_data))
                print("[Eval  ] NLL : {:.6f} || ACC : {:.6f} || Macro F1 : {:.6f}".format(test_nll, test_mse, test_f1))
                # wandb.log({"train_acc" : (epoch_mse/num_data)}, step=epoch)
                # wandb.log({"eval_acc" : test_mse}, step=epoch)
                # wandb.log({"eval_macro_f1" : test_f1}, step=epoch)
                log_dict.update({
                    "train_acc": epoch_mse / num_data,
                    "eval_acc": test_mse,
                    "eval_macro_f1": test_f1
                })

            else:
                print("[Train  ] NLL : {:.6f} || MSE : {:.6f}".format(epoch_ll/num_data, epoch_mse/num_data))
                print("[Eval  ] NLL : {:.6f} || MSE : {:.6f}".format(test_nll, test_mse))
                # wandb.log({"train_mse" : (epoch_mse/num_data)}, step=epoch)
                # wandb.log({"eval_mse" : test_mse}, step=epoch)
                log_dict.update({
                    "train_mse": epoch_mse / num_data,
                    "eval_mse": test_mse
                })

            # wandb.log({"train_nll" : (epoch_ll/num_data)}, step=epoch)
            # wandb.log({"train_loss" : (epoch_loss/num_data)}, step=epoch)
            # wandb.log({"eval_nll" : test_nll}, step=epoch)

            if self.task == 'extrapolation' or self.task == 'interpolation':
                print("[Impute ] NLL : {:.6f} || MSE : {:.6f}".format(impute_nll, impute_mse))
                # wandb.log({"impute_nll" : impute_nll}, step=epoch)
                # wandb.log({"impute_mse" : impute_mse}, step=epoch)
                log_dict.update({
                    "impute_nll": impute_nll,
                    "impute_mse": impute_mse
                })

            log_dict["epoch"] = epoch
            self.log_history.append(log_dict)

            if (epoch+1) % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save({
                    'epooch' : epoch,
                    'model_state_dict' : self.dynamics.state_dict()},
                           f'./checkpoints/{self.dataset}_{self.task}_{epoch+1}.pt')
        log_df = pd.DataFrame(self.log_history)
        csv_path = f"./logs/{self.dataset}_{self.task}/{self.source_name}/{self.target_name}_{run_id}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        log_df.to_csv(csv_path, index=False)
        print(f"Training log saved to {csv_path}")

    def eval_func(self, dl):
        epoch_mse = 0
        epoch_ll = 0
        epoch_f1 = 0

        epoch_impute_mse = 0
        epoch_impute_ll = 0
        
        num_data = 0
        num_batches = 0
        for _, data in enumerate(dl):
        
            if self.dataset == 'pendulum':
                if self.task == 'regression':
                    obs, truth, obs_times, obs_valid = [j.to(self.device).to(torch.float32) for j in data]
                    mask_obs = None
                    mask_truth = None
                else:
                    obs, truth, obs_valid, obs_times, mask_truth = [j.to(self.device).to(torch.float32) for j in data]
                    mask_obs = None
            elif self.dataset == 'person_activity':
                obs = data['inp_obs'].to(self.device)
                truth = data['evd_obs'].to(self.device)
                obs_times = data['inp_tid'].to(self.device)
                labels = data['aux_obs'].to(self.device)
                b, t, _ = obs.size()
                obs_valid = torch.ones(b, t).to(obs.device)
                mask_obs = None
                mask_truth = None
            elif self.dataset == 'timematch':
                obs = data['inp_obs'].to(self.device)
                truth = data['evd_obs'].to(self.device)
                obs_times = data['inp_tid'].to(self.device)
                labels = data['aux_obs'].to(self.device)
                obs_valid = data['obs_valid'].to(self.device)
                mask_truth = data['mask_truth'].to(self.device)
                mask_obs = data['mask_obs'].to(self.device)
            else:
                obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [j.to(self.device).to(torch.float32) for j in data]
                if self.task == 'extrapolation':
                    obs, obs_valid = adjust_obs_for_extrapolation(obs, obs_valid, obs_times, self.cut_time)

            out, _ = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=32, epoch=None)
            mean, var = out
            
            batch_len = truth.size(0)
            if self.dataset == 'pendulum' and self.task == 'interpolation':
                eval_nll = BNLL_(truth, mean) * batch_len
                eval_mse = MSE_(truth.flatten(start_dim=2), mean.flatten(start_dim=3), mask = mask_truth.flatten(start_dim=2)) * batch_len
            elif self.task == 'classification':
                eval_nll, eval_mse = CNLL_(labels, mean)
                macro_f1 = F1_(labels, mean, average='macro', zero_division=0)
                epoch_f1 += macro_f1
                num_batches += 1
            else:
                eval_nll = GNLL_(truth, mean, var, mask=mask_truth) * batch_len
                eval_mse = MSE_(truth, mean, mask=mask_truth) * batch_len
                
            epoch_mse += eval_mse.item()
            epoch_ll += eval_nll.item()
            
            if self.task == 'extrapolation' or self.dataset == 'pendulum':

                if self.dataset == 'pendulum' and self.task == 'interpolation':
                    mask_impute = (1 - obs_valid)[..., None, None, None] * mask_truth
                    impute_nll = BNLL_(truth, mean) * batch_len
                    impute_mse = MSE_(truth.flatten(start_dim=2), mean.flatten(start_dim=3), mask = mask_impute.flatten(start_dim=2)) * batch_len
                elif self.dataset != 'pendulum' and self.task == 'extrapolation':
                    mask_impute = (1 - obs_valid)[..., None] * mask_truth
                    impute_mse = MSE_(truth, mean, mask=mask_impute) * batch_len
                    impute_nll = GNLL_(truth, mean, var, mask=mask_impute) * batch_len
                    
                    epoch_impute_mse += impute_mse.item()
                    epoch_impute_ll += impute_nll.item()

            num_data += batch_len
        avg_macro_f1 = epoch_f1 / num_batches if self.task == 'classification' else None
        return epoch_mse/num_data, epoch_ll/num_data, epoch_impute_mse/num_data, epoch_impute_ll/num_data, avg_macro_f1

    def generate_traj(self, data):
        obs, truth, obs_valid, obs_times = [j.to(self.device).to(torch.float32) for j in data]
        mask_obs = None
        out, KL = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=32)
        mean, var = out
        return truth, mean


