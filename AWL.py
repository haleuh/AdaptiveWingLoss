import argparse
import math
import numpy as np
import torch
from core import models


def transform(point, center, scale, resolution, rotation=0, invert=False):
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if rotation != 0:
        rotation = -rotation
        r = np.eye(3)
        ang = rotation * math.pi / 180.0
        s = math.sin(ang)
        c = math.cos(ang)
        r[0][0] = c
        r[0][1] = -s
        r[1][0] = s
        r[1][1] = c

        t_ = np.eye(3)
        t_[0][2] = -resolution / 2.0
        t_[1][2] = -resolution / 2.0
        t_inv = torch.eye(3)
        t_inv[0][2] = resolution / 2.0
        t_inv[1][2] = resolution / 2.0
        t = reduce(np.matmul, [t_inv, r, t_, t])

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.matmul(t, _pt))[0:2]

    return new_point.astype(int)


def get_preds_fromhm(hm, center=None, scale=None, rot=None):
    max, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if 0 < pX < 63 and 0 < pY < 63:
                diff = torch.FloatTensor([hm_[pY, pX + 1] - hm_[pY, pX - 1],
                                          hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(preds[i, j], center, scale, hm.size(2), rot, True)

    return preds, preds_orig


def main():
    # Model
    model_ft = models.FAN(num_modules=4, num_landmarks=98)
    # Load pretrained weights
    checkpoint = torch.load(args.pretrained_weights)
    pretrained_weights = checkpoint['state_dict']
    model_weights = model_ft.state_dict()
    pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}
    model_weights.update(pretrained_weights)
    model_ft.load_state_dict(model_weights)
    # GPU
    model_ft = model_ft.to(device)
    model_ft.eval()
    # Detect landmarks
    landmarks = []
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs, boundary_channels = model_ft(inputs)
        for i in range(inputs.shape[0]):
            pred_heatmap = outputs[-1][:, :-1, :, :][i].detach().cpu()
            pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
            pred_landmarks = pred_landmarks.squeeze().numpy() * 4
            landmarks.append(pred_landmarks)
    return landmarks


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    # Checkpoint and pretrained weights
    parser.add_argument('--pretrained_weights', type=str, help='pretrained model')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main()
