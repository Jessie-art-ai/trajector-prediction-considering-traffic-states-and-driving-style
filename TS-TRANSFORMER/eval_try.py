import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp

mlp.use('Agg')
from scipy.spatial import ConvexHull
import argparse
from data.nuscenes_pred_split import get_nuscenes_pred_split
from data.ethucy_split import get_ethucy_split
from utils.utils import print_log, AverageMeter, isfile, isfolder, find_unique_common_from_lists, load_list_from_folder, load_txt_file

""" Metrics """

def compute_ADE(pred_arr, gt_arr):
    ade = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist.mean(axis=-1)  # samples
        ade += dist.min(axis=0)  # (1, )
    ade /= len(pred_arr)
    return ade

def compute_FDE(pred_arr, gt_arr):
    fde = 0.0
    for pred, gt in zip(pred_arr, gt_arr):
        diff = pred - np.expand_dims(gt, axis=0)  # samples x frames x 2
        dist = np.linalg.norm(diff, axis=-1)  # samples x frames
        dist = dist[..., -1]  # samples
        fde += dist.min(axis=0)  # (1, )
    fde /= len(pred_arr)
    return fde

def align_gt(pred, gt):
    frame_from_data = pred[0, :, 0].astype('int64').tolist()
    frame_from_gt = gt[:, 0].astype('int64').tolist()
    common_frames, index_list1, index_list2 = find_unique_common_from_lists(frame_from_gt, frame_from_data)
    assert len(common_frames) == len(frame_from_data)

    hist_ind = np.arange(min(index_list1))  # added
    hist_new = gt[hist_ind, 2:]  # added
    gt_new = gt[index_list1, 2:]
    pred_new = pred[:, index_list2, 2:]
    return pred_new, gt_new, hist_new  # added

colors = ['red', 'green', 'purple', 'darkgoldenrod', 'darkorange', 'peru', 'slategrey', 'hotpink', 'yellow',
          'cyan', 'teal', 'rosybrown', 'yellowgreen', 'chocolate', 'saddlebrown', 'crimson', 'dimgray', 'gainsboro',
          'tan', 'lightsteelblue']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='rounD')
    parser.add_argument('--results_dir', default="results/rounD_agentformer/results/epoch_0005/test/samples")
    parser.add_argument('--data', default='test')
    parser.add_argument('--log_file', default=None)
    args = parser.parse_args()

    dataset = args.dataset.lower()
    results_dir = args.results_dir

    if dataset == 'nuscenes_pred':  # nuscenes
        data_root = f'datasets/nuscenes_pred'
        gt_dir = f'{data_root}/label/{args.data}'
        seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)
        seq_eval = globals()[f'seq_{args.data}']

    elif dataset == 'eot':  # added for EOT dataset
        gt_dir = 'datasets/EOT_split_preprocessed/test'
        data_root = 'datasets/EOT_split_preprocessed'

        seq_train, seq_test, seq_val = [], [], []
        for file in os.listdir(str(data_root) + '/train/'):
            seq_train.append(str(data_root) + '/train/' + file)

        for file in os.listdir(str(data_root) + '/test/'):
            seq_test.append(str(data_root) + '/test/' + file)

        for file in os.listdir(str(data_root) + '/val/'):
            seq_val.append(str(data_root) + '/val/' + file)

        seq_train = sorted(seq_train)
        seq_test = sorted(seq_test)
        seq_val = sorted(seq_val)
        seq_eval = globals()[f'seq_{args.data}']
        print("Loaded EOT data ....")
    elif dataset == 'round':  # added for rounD dataset
        gt_dir = 'rounD/merge_rounD/test'
        data_root = 'rounD/merge_rounD'
        print("rounD_in")
        seq_train, seq_test, seq_val = [], [], []
        for file in os.listdir(str(data_root) + '/train/'):
            seq_train.append(str(data_root) + '/train/' + file)

        for file in os.listdir(str(data_root) + '/test/'):
            seq_test.append(str(data_root) + '/test/' + file)

        for file in os.listdir(str(data_root) + '/val/'):
            seq_val.append(str(data_root) + '/val/' + file)

        seq_train = sorted(seq_train)
        seq_test = sorted(seq_test)
        seq_val = sorted(seq_val)
        seq_eval = globals()[f'seq_{args.data}']
        print("Loaded rounD data ....")
    else:  # ETH/UCY
        gt_dir = f'datasets/eth_ucy/{args.dataset}'
        seq_train, seq_val, seq_test = get_ethucy_split(args.dataset)
        seq_eval = globals()[f'seq_{args.data}']

    if args.log_file is None:
        results_dir_2 = 'results/rounD_agentformer/log'
        log_file = os.path.join(results_dir_2, 'log_eval.txt')
    else:
        log_file = args.log_file
    log_file = open(log_file, 'a+')
    print_log('loading results from %s' % results_dir, log_file)
    print_log('loading GT from %s' % gt_dir, log_file)

    stats_func = {
        'ADE': compute_ADE,
        'FDE': compute_FDE
    }

    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    seq_list, num_seq = load_list_from_folder(gt_dir)
    print_log('\n\nnumber of sequences to evaluate is %d' % len(seq_eval), log_file)
    count = 0
    for seq_name in seq_eval:
        seq_name = seq_name.split('/')[-1][:-4]
        gt_data, _ = load_txt_file(os.path.join(gt_dir, seq_name + '.txt'))
        gt_raw = []
        for line_data in gt_data:
            line_data = np.array([line_data.split(',')])[:, [0, 1, 13, 15]][0].astype('float32')
            if line_data[1] == -1: continue
            gt_raw.append(line_data)
        gt_raw = np.stack(gt_raw)

        data_filelist, _ = load_list_from_folder(os.path.join(results_dir, seq_name))

        for data_file in data_filelist:
            if isfile(data_file):
                print(" ** Most Likely **")
                all_traj = np.loadtxt(data_file, delimiter=',', dtype='float32')
                all_traj = np.expand_dims(all_traj, axis=0)
            elif isfolder(data_file):
                print("** Best of 20 samples **")
                sample_list, _ = load_list_from_folder(data_file)
                sample_all = []
                for sample in sample_list:
                    sample = np.loadtxt(sample, delimiter=' ', dtype='float32')
                    sample_all.append(sample)
                all_traj = np.stack(sample_all, axis=0)
            else:
                assert False, 'error'

            id_list = np.unique(all_traj[:, :, 1])
            frame_list = np.unique(all_traj[:, :, 0])
            agent_traj = []
            gt_traj = []
            hist_traj = []
            for idx in id_list:
                gt_idx = gt_raw[gt_raw[:, 1] == idx]
                ind = np.unique(np.where(all_traj[:, :, 1] == idx)[1].tolist())
                pred_idx = all_traj[:, ind, :]
                pred_idx, gt_idx, hist_idx = align_gt(pred_idx, gt_idx)
                hist_traj.append(hist_idx)
                agent_traj.append(pred_idx)
                gt_traj.append(gt_idx)

            fig, ax = plt.subplots()
            for idx in range(len(agent_traj)):
                history = hist_traj[idx]
                future = gt_traj[idx]
                predictions = agent_traj[idx]

                pred_x_mean = np.mean(predictions[:, :, 0], axis=0)
                pred_y_mean = np.mean(predictions[:, :, 1], axis=0)

                ax.plot(history[:, 0], history[:, 1], 'x', color=colors[idx], alpha=1, label=f'History {idx}')
                ax.plot(pred_x_mean, pred_y_mean, 'o', color=colors[idx], alpha=1, label=f'Predicted Trajectory {idx}')
                ax.plot(future[:, 0], future[:, 1], '*', color='blue', alpha=1, label=f'Ground Truth {idx}')

                pred_points = predictions.reshape(-1, 2)
                try:
                    hull = ConvexHull(pred_points)
                    x_hull = np.append(pred_points[hull.vertices, 0], pred_points[hull.vertices, 0][0])
                    y_hull = np.append(pred_points[hull.vertices, 1], pred_points[hull.vertices, 1][0])
                    ax.fill(x_hull, y_hull, alpha=0.1, c=colors[idx])
                except RuntimeError:
                    print("计算凸包错误")

            ax.legend()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Trajectories for Sequence {seq_name}')

            plt.savefig(f'E:/yjy/code/Improving-Multi-agent-Trajectory-Prediction-using-Traffic-States-on-Interactive-Driving-Scenarios-master/TS-TRANSFORMER/plots/convex_hull/plot_{count}.png', bbox_inches='tight')
            plt.close()
            print(f"plotted graph no {count}")
            count += 1

            for stats_name, meter in stats_meter.items():
                func = stats_func[stats_name]
                value = func(agent_traj, gt_traj)
                meter.update(value, n=len(agent_traj))

            stats_str = ' '.join([f'{x}: {y.val:.4f} ({y.avg:.4f})' for x, y in stats_meter.items()])
            print_log(f'evaluating seq {seq_name:s}, forecasting frame {int(frame_list[0]):06d} to {int(frame_list[-1]):06d} {stats_str}', log_file)

    print_log('-' * 30 + ' STATS ' + '-' * 30, log_file)
    for name, meter in stats_meter.items():
        print_log(f'{meter.count} {name}: {meter.avg:.4f}', log_file)
    print_log('-' * 67, log_file)
    log_file.close()
