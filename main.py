import configs
import torch

import random
import os
import numpy as np
import argparse
from dataset import PixelSetData

from lib.data_utils import load_data
from lib.utils import set_seed, count_parameters
from lib.amortized_control_ssm import ACSSM
from timematch_utils import label_utils
from timematch_utils.train_utils import bool_flag

parser = argparse.ArgumentParser('ACSSM')
parser.add_argument('--info_type', type=str, default="full", help="Assimilation type to encode the information")
parser.add_argument('--cut-time', type=int, default=None, help='Timepoint at which extrapolation starts.')
parser.add_argument('-b', '--batch-size', type=int, default=None, help="Batch size for training and test set.")
parser.add_argument('--task', type=str, default=None, help="Target task.")
parser.add_argument('--sample-rate', type=float, default=None, help='Sample time points to increase irregularity of timestamps. For example, if sample_rate=0.5 half of the time points are discarded at random in the data preprocessing.')
parser.add_argument('--impute-rate', type=float, default=None, help='Remove time points for interpolation. For example, if impute_rate=0.3 the model is given 70% of the time points and tasked to reconstruct the entire series.')
parser.add_argument('--unobserved-rate', type=float, default=0.2, help='Percentage of features to remove per timestamp in order to increase sparseness across dimensions (applied only for USHCN)')
parser.add_argument('--data-random-seed', type=int, default=None, help="Random seed for subsampling timepoints and features.")
parser.add_argument('-rs', '--random-seed', type=int, default=42, help="Random seed for initializing model parameters.")
parser.add_argument('--num-workers', type=int, default=None, help="Number of workers to use in dataloader.")
parser.add_argument('--pin-memory', type=bool, default=True, help="If to pin memory in dataloader.")
parser.add_argument('--state-dim', type=int, default=None, help="Dimension of latent states")
parser.add_argument('--out-dim', type=int, default=None, help="Dimension of output")
parser.add_argument('--n_layer', type=int, default=None, help="Number of layer for transformer")
parser.add_argument('--drop_out', type=float, default=None, help="Dropout rate for transformer")
parser.add_argument('--lamda_1', type=float, default=None, help="Adjusting the lagrange term in ELBO")
parser.add_argument('--lamda_2', type=float, default=None, help="Adjusting the mayer term in ELBO")
parser.add_argument('--init_sigma', type=float, default=None, help="Adjusting initial covaraince of latent dynamics")
parser.add_argument('--ts', type=float, default=None, help="Time scaler")
parser.add_argument('--num-basis', type=int, default=None, help="Number of basis matrices to use in transition model for locally-linear transitions. L in paper")

parser.add_argument('--problem_name', type=str, default='timematch_classification', help="Problem to solve")
parser.add_argument('--dataset', type=str, default='timematch', help="Dataset to use. Available datasets are physionet, ushcn and pendulum.")
parser.add_argument('--data_root', type=str, default='/mnt/d/All_Documents/documents/ViT/dataset/timematch', help="Dataset to use. Available datasets are physionet, ushcn and pendulum.")
# parser.add_argument('--data_root', default='/data/user/DBL/timematch_data', type=str,
#                         help='Path to datasets root directory.')

# timamatch
parser.add_argument("--gpu", type=int, default=0, help="GPU device")

parser.add_argument('--batch_size', type=int, default=500, help='Batch size for training.')
parser.add_argument('--balance_source', type=bool_flag, default=True, help='Use class balanced batches for source.')

parser.add_argument('--source', default='france/31TCJ/2017', type=str, help='Source domain.')
parser.add_argument('--target', default='france/30TXT/2017', type=str)
parser.add_argument('--num_folds', default=1, type=int, help='Number of cross-validation folds.')

parser.add_argument('--with_shift_aug', default=True, action='store_true',
                    help='whether to apply random temporal shift augmentation')
parser.add_argument('--shift_aug_p', default=1.0, type=float,
                    help='probability to apply temporal shift augmentation')
parser.add_argument('--max_shift_aug', default=60, type=int,
                    help='highest shift to apply for temporal shift augmentation')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--classes')

# innovation
parser.add_argument('--ns', default=0.01, type=float)

problem_name = parser.parse_args().problem_name
default_config = {
    'pendulum_regression':     configs.get_pendulum_regression_configs,
    'physionet_interpolation': configs.get_physionet_interpolation_configs,
    'physionet_extrapolation': configs.get_physionet_extrapolation_configs,
    'ushcn_interpolation':     configs.get_ushcn_interpolation_configs,
    'ushcn_extrapolation':     configs.get_ushcn_extrapolation_configs,
    'person_activity_classification': configs.get_person_activity_classification_configs,
    'timematch_classification': configs.get_timematch_classification_configs,

}.get(problem_name)()
parser.set_defaults(**default_config)

args = parser.parse_args()

args.device = 'cuda:' + str(args.gpu)
# seed = args.random_seed
set_seed(args.seed)

# def match(domain):
#     match domain:
#         case 'france/30TXT/2017':
#             return 'FR1'
#         case 'france/31TCJ/2017':
#             return 'FR2'
#         case 'denmark/32VNH/2017':
#             return 'DK1'
#     return 'AT1'

def main(args):
    config = args
    source_classes = label_utils.get_classes(
        config.source.split('/')[0],
        combine_spring_and_winter=config.combine_spring_and_winter
    )
    source_data = PixelSetData(config.data_root, config.source, source_classes)
    labels, counts = np.unique(source_data.get_labels(), return_counts=True)
    source_classes = [source_classes[i] for i in labels[counts >= 200]]
    print(len(source_classes))
    args.classes = source_classes
    args.num_classes = len(source_classes)
    args.out_dim = len(source_classes)
    # ========= print options =========
    for o in vars(args):
        print("#", o, ":", getattr(args, o))
    run = ACSSM(args)
    # wandb.init(project="acssm", config=args, save_code=True, mode="online",
    #            name=f'{args.problem_name}')
    # wandb.init(project="acssm", config=args, save_code=True, mode="offline",
    #            name=f'{args.problem_name}')
    #wandb sync同步日志
    print(f"# param of model: {count_parameters(run.dynamics)}")
    train_dl, valid_dl, trg_train_loader = load_data(args)
    run.train_and_eval_adaptation(train_dl, valid_dl, trg_train_loader)
    
main(args)