import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from itertools import chain
# deprecated
def my_softmax(input, axis=1):
    # trans_input = input.transpose(axis, 0).contiguous()
    # soft_max_1d = F.softmax(trans_input)
    # return soft_max_1d.transpose(axis, 0)
    return torch.softmax(input, dim=axis)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return torch.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10, inference_period=40):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    if len(logits.size()) > 3:
        num_samples = logits.size(1) // inference_period
        gumbel_noise_slim = [
            [sample_gumbel([logits.size(0), logits.size(2), logits.size(3)], eps=eps).unsqueeze(1)] * inference_period
            for _ in range(num_samples)
        ]
        gumbel_noise_slim = list(chain(*gumbel_noise_slim))
        gumbel_noise = torch.cat(gumbel_noise_slim, dim=1)
    else:
        gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)

def gumbel_sigmoid(logits, tau=1, eps=1e-10):
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + gumbel_noise
    return torch.sigmoid(y/tau)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, inference_period=40):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps, inference_period=inference_period)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data(batch_size=1, data_dir='', suffix='',dynamic=False):
    loc_train = np.load(data_dir + 'loc_train' + suffix + '.npy')
    vel_train = np.load(data_dir +'vel_train' + suffix + '.npy')
    edges_train = np.load(data_dir +'edges_train' + suffix + '.npy')

    loc_valid = np.load(data_dir +'loc_valid' + suffix + '.npy')
    vel_valid = np.load(data_dir +'vel_valid' + suffix + '.npy')
    edges_valid = np.load(data_dir +'edges_valid' + suffix + '.npy')

    loc_test = np.load(data_dir +'loc_test' + suffix + '.npy')
    vel_test = np.load(data_dir +'vel_test' + suffix + '.npy')
    edges_test = np.load(data_dir +'edges_test' + suffix + '.npy')

    # if it is dynamic edges
    # print(edges_train.shape)
    if dynamic:
        assert(len(edges_train.shape) == 4)

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, list(edges_train.shape[:-2])+ [-1])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, list(edges_valid.shape[:-2]) + [-1])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, list(edges_test.shape[:-2]) + [-1])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    if dynamic:
        edges_train = edges_train[:,:, off_diag_idx]
        edges_valid = edges_valid[:,:, off_diag_idx]
        edges_test = edges_test[:,:, off_diag_idx]
    else:
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def load_kuramoto_data(batch_size=1, suffix=''):
    feat_train = np.load('data/feat_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/feat_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Normalize each feature dim. individually
    feat_max = feat_train.max(0).max(0).max(0)
    feat_min = feat_train.min(0).min(0).min(0)

    feat_max = np.expand_dims(np.expand_dims(np.expand_dims(feat_max, 0), 0), 0)
    feat_min = np.expand_dims(np.expand_dims(np.expand_dims(feat_min, 0), 0), 0)

    # Normalize to [-1, 1]
    feat_train = (feat_train - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_valid = (feat_valid - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_test = (feat_test - feat_min) * 2 / (feat_max - feat_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_kuramoto_data_old(batch_size=1, suffix=''):
    feat_train = np.load('data/old_kuramoto/feat_train' + suffix + '.npy')
    edges_train = np.load('data/old_kuramoto/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/old_kuramoto/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/old_kuramoto/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/old_kuramoto/feat_test' + suffix + '.npy')
    edges_test = np.load('data/old_kuramoto/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_motion_data(batch_size=1, suffix=''):
    feat_train = np.load('data/motion_train' + suffix + '.npy')
    feat_valid = np.load('data/motion_valid' + suffix + '.npy')
    feat_test = np.load('data/motion_test' + suffix + '.npy')
    adj = np.load('data/motion_adj' + suffix + '.npy')

    # NOTE: Already normalized

    # [num_samples, num_nodes, num_timesteps, num_dims]
    num_nodes = feat_train.shape[1]

    edges_train = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_train.shape[0], axis=0)
    edges_valid = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_valid.shape[0], axis=0)
    edges_test = np.repeat(np.expand_dims(adj.flatten(), 0),
                           feat_test.shape[0], axis=0)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(np.array(edges_train, dtype=np.int64))
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(np.array(edges_valid, dtype=np.int64))
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(np.array(edges_test, dtype=np.int64))

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
        [num_nodes, num_nodes])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds*(torch.log(preds + eps) + np.log(num_edge_types))

    # print(preds.size())
    return kl_div.sum() / (num_atoms * preds.size(0))
    # return kl_div.mean()

def flip_sample(weights):
    p = torch.sigmoid(weights).clamp(0.1,0.9)
    b = torch.bernoulli(p.detach())
    sample = b * torch.max(p,1.-p)
    return sample

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def edge_pr(preds, target, label=1):
    _, preds = preds.max(-1)
    label = torch.tensor(label).to(preds.device)
    pos = torch.tensor(1.).to(preds.device)
    neg = torch.tensor(0.).to(preds.device)
    _, pred = preds.max(-1)
    base_p = torch.where(target == label, pos, neg).sum()
    base_r = torch.where(preds == label, pos, neg).sum()
    tp = torch.where((preds == target) & (preds == label), pos, neg).sum()
    cal = lambda x, y: (x/y).item()
    precision = cal(tp, base_p)
    recall = cal(tp, base_r)
    return precision, recall

def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    # print(preds)
    # print(preds[0,:,0])
    # print(target[0,:,0])
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.view(-1).shape[0])

def nll_gauss_logvar(mean, logvar, x):
    pi = torch.FloatTensor([np.pi])
    if mean.is_cuda:
        pi = pi.cuda()
    nll_element = (x - mean).pow(2) / torch.exp(logvar) + logvar + torch.log(2*pi)
    return torch.sum(nll_element) / (x.size(0) * x.size(1))
    # return torch.mean(nll_element)
    
# input shape is seq_len * 2(x,y)
def draw_trajectory(seq1, seq2, filename='test.png'):
    # plt.gca().set_color_cycle(['red', 'blue'])
    # print(seq1.shape)
    fig = plt.figure()
    ax = plt.subplot(111)
    num_atom = seq1.shape[0]
    for i in range(num_atom):
        if seq1 is not None:
            ax.plot(seq1[i,:,0], seq1[i,:,1], label='atom_{}'.format(i))
        if seq2 is not None:
            ax.plot(seq2[i,:,0], seq2[i,:,1], label='predicion_{}'.format(i))
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

    plt.savefig(filename)
    plt.close()

# input shape is seq_len * 2(x,y)

def draw_trajectory_2(seq1, seq2, filename='test.png'):
    # plt.gca().set_color_cycle(['red', 'blue'])
    # print(seq1.shape)
    fig = plt.figure()
    ax = plt.subplot(1,2,1)
    num_atom = seq1.shape[0]
    for i in range(num_atom):
        ax.plot(seq1[i, :, 0], seq1[i, :, 1])
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5)
    ax.set_title('Ground-truth')
    ax = plt.subplot(1,2,2)
    for i in range(num_atom):
        ax.plot(seq2[i, :, 0], seq2[i, :, 1])
        
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5)
    ax.set_title('Predictions')
    plt.savefig(filename)
    plt.close()

def plot_samples(t1, t2, e1, e2):
    # t1: gt trajectory 1 has shape [length, num_atom, 2]
    # t2: prediction trajactory 2 has shape [length, num_atom, 2]
    # e1, edge 1 has shape [length, num_atom  * (num_atom-1)]
    # e2, edge 2 has shape [length, num_atom  * (num_atom-1)]
    length =  t1.shape[0]
    fig = plt.figure()
    ax = plt.subplot(2, length)
    for i in range(length):
        # plot gt
        ax[0,i].scatter(t1[i,:,0],t1[i,:,1])
        # plot prediction
        ax[1,i].scatter(t2[i,:,0],t2[i,:,1])

