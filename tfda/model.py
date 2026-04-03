import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, num_neighbors):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(128):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]

        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)

    # Pseudolabels
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    # First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard

def refine_predictions(
    features,
    probs,
    banks, num_neighbors):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(features, feature_bank, probs_bank, num_neighbors)

    return pred_labels, probs, pred_labels_all, pred_labels_hard

@torch.no_grad()
def eval_and_label_dataset(epoch, FE, classifier, banks, test_dataloader, train_dataloader, num_neighbors):
    print("Evaluating Dataset!")

    FE.eval()
    classifier.eval()
    logits, indices, gt_labels = [], [], []
    features = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # for batch_idx, batch in enumerate(train_dataloader):
            test_inputs, test_targets, test_idxs, test_inputs_f = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), \
            batch[6].cuda()
            # inputs, targets, idxs = batch[0].cuda(), batch[2].cuda(), batch[3].cuda()
            feats, _, _, _, _ = FE(test_inputs, test_inputs_f)
            # print(f'batch_idx:{batch_idx}')
            # print(f'feats.shape:{feats.shape}')

            logits_cls = classifier(feats)

            features.append(feats)
            gt_labels.append(test_targets)
            logits.append(logits_cls)
            indices.append(test_idxs)

    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: 16384],  # hanya diacak urutan indeksnya menggunakan rand_idxs
        "probs": probs[rand_idxs][: 16384],
        "ptr": 0,
    }

    FE.train()
    classifier.train()
    return banks

@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])