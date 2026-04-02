import ml_collections


def get_timematch_classification_configs():
    config = ml_collections.ConfigDict()

    config.data_random_seed = 0
    config.T = 100

    config.dataset = 'timematch'  # ← 关键！
    config.task = 'classification'
    config.info_type = 'full'
    config.num_workers = 8
    config.pin_memory = True
    config.ts = 1.0  # TimeMatch 时间戳已是整数，无需缩放
    config.lamda_1 = 1e-6
    config.lamda_2 = 1e-8

    config.epochs = 50
    config.lr = 1e-4
    config.wd = 1e-3
    config.batch_size = 500
    config.num_basis = 256
    config.state_dim = 288
    config.n_layer = 1
    config.drop_out = 0.2
    config.init_sigma = 1.

    config.out_dim = 12  # ← 根据你的 TimeMatch 类别数调整！

    # TimeMatch 特有参数（会被 parser 覆盖）
    # config.data_root = '/data/user/DBL/timematch_data'
    config.data_root = '/mnt/d/All_Documents/documents/ViT/dataset/timematch'
    config.seq_length = 30
    config.num_pixels = 1
    config.val_ratio = 0.1
    config.test_ratio = 0.2
    config.combine_spring_and_winter = False

    # 论文创新
    config.noise_scale = 0.01

    return config