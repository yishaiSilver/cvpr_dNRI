from dnri.utils.flags import build_flags
import dnri.models.model_builder as model_builder
from dnri.datasets.spring_data import SpringData
import dnri.datasets.dyari_utils as dyari_utils
import dnri.training.train as train
import dnri.training.train_utils as train_utils
import dnri.training.evaluate as evaluate
import dnri.utils.misc as misc

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np


def plot_sample(model, dataset, num_samples, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1)
    use_gt_edges = params.get('use_gt_edges')
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    batch_count = 0
    all_errors = []
    burn_in_steps = 8
    forward_pred_steps = 40
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        gt_edges = batch.get('edges', None)
        with torch.no_grad():
            model_inputs = inputs[:, :burn_in_steps]
            gt_predictions = inputs[:, burn_in_steps:burn_in_steps+forward_pred_steps]
            if gpu:
                model_inputs = model_inputs.cuda(non_blocking=True)
                if gt_edges is not None and use_gt_edges:
                    gt_edges = gt_edges.cuda(non_blocking=True)
            if not use_gt_edges:
                gt_edges=None
            model_preds = model.predict_future(model_inputs, forward_pred_steps).cpu()
            #total_se += F.mse_loss(model_preds, gt_predictions).item()
            print("MSE: ", torch.nn.functional.mse_loss(model_preds, gt_predictions).item())
            batch_count += 1
        fig, ax = plt.subplots()
        unnormalized_preds = dataset.unnormalize(model_preds)
        unnormalized_gt = dataset.unnormalize(inputs)
        def update(frame):
            ax.clear()
            ax.plot(unnormalized_gt[0, frame, 0, 0], unnormalized_gt[0, frame, 0, 1], 'bo')
            ax.plot(unnormalized_gt[0, frame, 1, 0], unnormalized_gt[0, frame, 1, 1], 'ro')
            ax.plot(unnormalized_gt[0, frame, 2, 0], unnormalized_gt[0, frame, 2, 1], 'go')
            ax.plot(unnormalized_gt[0, frame, 3, 0], unnormalized_gt[0, frame, 1, 1], 'mo')
            ax.plot(unnormalized_gt[0, frame, 4, 0], unnormalized_gt[0, frame, 2, 1], 'yo')
            if frame >= burn_in_steps:
                tmp_fr = frame - burn_in_steps
                ax.plot(unnormalized_preds[0, tmp_fr, 0, 0], unnormalized_preds[0, tmp_fr, 0, 1], 'bo', alpha=0.5)
                ax.plot(unnormalized_preds[0, tmp_fr, 1, 0], unnormalized_preds[0, tmp_fr, 1, 1], 'ro', alpha=0.5)
                ax.plot(unnormalized_preds[0, tmp_fr, 2, 0], unnormalized_preds[0, tmp_fr, 2, 1], 'go', alpha=0.5)
                ax.plot(unnormalized_preds[0, tmp_fr, 3, 0], unnormalized_preds[0, tmp_fr, 3, 1], 'mo', alpha=0.5)
                ax.plot(unnormalized_preds[0, tmp_fr, 4, 0], unnormalized_preds[0, tmp_fr, 4, 1], 'yo', alpha=0.5)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
        ani = animation.FuncAnimation(fig, update, interval=100, frames=burn_in_steps+forward_pred_steps)
        path = os.path.join(params['working_dir'], 'pred_trajectory_%d.mp4'%batch_ind)
        ani.save(path, codec='mpeg4')
        if batch_count >= num_samples:
            break

def eval_edges(model, data_loader, params):
    gpu = params.get('gpu', False)
    batch_size = params.get('batch_size', 1000)
    eval_metric = params.get('eval_metric')
    num_edge_types = params['num_edge_types']
    skip_first = params['skip_first']
    full_edge_count = 0.
    model.eval()
    correct_edges = 0.
    edge_count = 0.
    correct_0_edges = 0.
    edge_0_count = 0.
    correct_1_edges = 0.
    edge_1_count = 0.

    correct = num_predicted = num_gt = 0
    all_edges = []
    for batch_ind, batch in enumerate(data_loader):
        inputs = batch['inputs']
        gt_edges = batch['edges'].long()
        with torch.no_grad():
            if gpu:
                inputs = inputs[:40].cuda(non_blocking=True)
                gt_edges = gt_edges.cuda(non_blocking=True)

            print(inputs.shape)
            _, _, _, edges, _ = model.calculate_loss(inputs, is_train=False, return_logits=True)
            edges = edges.argmax(dim=-1)
            all_edges.append(edges.cpu())
            if len(edges.shape) == 3 and len(gt_edges.shape) == 2:
                gt_edges = gt_edges.unsqueeze(1).expand(gt_edges.size(0), edges.size(1), gt_edges.size(1))
            elif len(gt_edges.shape) == 3 and len(edges.shape) == 2:
                edges = edges.unsqueeze(1).expand(edges.size(0), gt_edges.size(1), edges.size(1))
            if edges.size(1) == gt_edges.size(1) - 1:
                gt_edges = gt_edges[:, :-1]
            edge_count += edges.numel()
            full_edge_count += gt_edges.numel()
            correct_edges += ((edges == gt_edges)).sum().item()
            edge_0_count += (gt_edges == 0).sum().item()
            edge_1_count += (gt_edges == 1).sum().item()
            correct_0_edges += ((edges == gt_edges)*(gt_edges == 0)).sum().item()
            correct_1_edges += ((edges == gt_edges)*(gt_edges == 1)).sum().item()
            correct += (edges*gt_edges).sum().item()
            num_predicted += edges.sum().item()
            num_gt += gt_edges.sum().item()
    prec = correct / (num_predicted + 1e-8)
    rec = correct / (num_gt + 1e-8)
    f1 = 2*prec*rec / (prec+rec+1e-6)
    all_edges = torch.cat(all_edges)
    return f1, correct_edges / (full_edge_count + 1e-8), correct_0_edges / (edge_0_count + 1e-8), correct_1_edges / (edge_1_count + 1e-8), all_edges

if __name__ == '__main__':
    parser = build_flags()
    parser.add_argument('--data_path')
    parser.add_argument('--error_out_name', default='prediction_errors.npy')
    parser.add_argument('--error_suffix')
    args = parser.parse_args()
    params = vars(args)

    misc.seed(args.seed)

    params['num_vars'] = params['num_agents'] = 5
    params['input_noise_type'] = 'none'
    params['input_size'] = 4
    params['input_time_steps'] = 40
    params['nll_loss_type'] = 'gaussian'
    params['prior_variance'] = 5e-5
    name = 'springs5'
    batch_size = 10
    # train_data, val_data, test_data, loc_max, loc_min, vel_max, vel_min = dyari_utils.load_data(batch_size, args.data_path, name)
    train_data = SpringData(name, args.data_path, 'train', params, num_in_path=False, transpose_data=False, max_len=50)
    val_data = SpringData(name, args.data_path, 'valid', params, num_in_path=False, transpose_data=False, max_len=50)

    model = model_builder.build_model(params)
    if args.mode == 'train':
        # with train_utils.build_writers(args.working_dir) as (train_writer, val_writer):
            # train.train(model, train_data, val_data, params, train_writer, val_writer)
        train.train(model, train_data, val_data, params, None, None)
    elif args.mode == 'eval':
        test_data = SpringData(name, args.data_path, 'test', params, num_in_path=False, transpose_data=False)
        test_cumulative_mse = evaluate.eval_forward_prediction(model, test_data, 40, 9, params)
        path = os.path.join(args.working_dir, args.error_out_name)
        np.save(path, test_cumulative_mse.cpu().numpy())
        test_mse_1 = test_cumulative_mse[0].item()
        test_mse_5 = test_cumulative_mse[4].item()
        test_mse_9 = test_cumulative_mse[8].item()
        print("\t1 STEP:  ",test_mse_1)
        print("\t5 STEP: ", test_mse_5)
        print("\t9 STEP: ",test_mse_9)
        # plot_sample(model, test_data, 30, params)
        eval_edges(model, val_data, params)