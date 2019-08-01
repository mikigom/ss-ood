import torch
import numpy as np


def get_one_hot(targets, nb_classes):
    targets = np.array([targets])
    res = np.eye(nb_classes)[targets.reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])[0]


b_ls = [True, False]
s_h_ls = [-1, 0, 1]
s_w_ls = [-1, 0, 1]
k_ls = [0, 1, 2, 3]
transformation2label = {}

coding_len = len(b_ls) * len(s_h_ls) * len(s_w_ls) * len(k_ls)
code = 0
for i in b_ls:
    for j in s_h_ls:
        for k in s_w_ls:
            for l in k_ls:
                transformation2label[(i, j, k, l)] = code
                code += 1


def horizontal_flip(x, b):
    if b is True:
        return x[:, ::-1]
    else:
        return x


def translation(x, s_h, s_w):
    assert s_h in [-1, 0, 1]
    assert s_w in [-1, 0, 1]

    h, w, _ = x.shape
    pad_h = int(0.25 * h)
    pad_w = int(0.25 * w)

    x = np.pad(x, [(pad_h, pad_w), (pad_h, pad_w), (0, 0)], mode='reflect')

    if s_h == -1:
        h_start = 0
    elif s_h == 0:
        h_start = pad_h
    elif s_h == 1:
        h_start = 2 * pad_h

    if s_w == -1:
        w_start = 0
    elif s_w == 0:
        w_start = pad_w
    elif s_w == 1:
        w_start = 2 * pad_w

    x = x[h_start:h_start + h, w_start:w_start + w]

    return x


def rotation(x, k):
    return np.rot90(x, k=k, axes=(0, 1))


def apply_transformation(x, b, s_h, s_w, k):
    x = horizontal_flip(x, b)
    x = translation(x, s_h, s_w)
    x = rotation(x, k)
    return x


def sample_transformation():
    b = np.random.choice(b_ls)
    s_h = np.random.choice(s_h_ls)
    s_w = np.random.choice(s_w_ls)
    k = np.random.choice(k_ls)
    return b, s_h, s_w, k


def batch_apply_transformation_and_get_label(xs):
    xs = xs.cpu().numpy()
    xs = np.moveaxis(xs, 1, 3)
    batch_num = xs.shape[0]

    images = []
    labels = []
    for i in range(batch_num):
        x = xs[i]

        b, s_h, s_w, k = sample_transformation()
        x = apply_transformation(x, b, s_h, s_w, k)
        y = transformation2label[(b, s_h, s_w, k)]

        images.append(x)
        labels.append(y)

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)

    images = np.moveaxis(images, 3, 1)
    images = torch.from_numpy(images).cuda().float()
    labels = torch.from_numpy(labels).cuda().long()
    return images, labels


def apply_transformation_with_order(xs):
    xs = xs.cpu().numpy()
    xs = np.moveaxis(xs, 1, 3)
    batch_num = xs.shape[0]

    images = []
    for i in range(batch_num):
        for b in b_ls:
            for s_h in s_h_ls:
                for s_w in s_w_ls:
                    for k in k_ls:
                        images.append(apply_transformation(xs[i], b, s_h, s_w, k))

    images = np.stack(images, axis=0)

    images = np.moveaxis(images, 3, 1)
    images = torch.from_numpy(images).cuda().float()
    return images


def all_apply_transformation_and_get_label(xs):
    xs = xs.cpu().numpy()
    xs = np.moveaxis(xs, 1, 3)
    batch_num = xs.shape[0]

    images = []
    labels = []
    for i in range(batch_num):
        for b in b_ls:
            for s_h in s_h_ls:
                for s_w in s_w_ls:
                    for k in k_ls:
                        x = xs[i]
                        x = apply_transformation(x, b, s_h, s_w, k)
                        y = transformation2label[(b, s_h, s_w, k)]

                        images.append(x)
                        labels.append(y)

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)

    images = np.moveaxis(images, 3, 1)
    images = torch.from_numpy(images).cuda().float()
    labels = torch.from_numpy(labels).cuda().long()
    return images, labels
