import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched
import numpy as np

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        '''
        数据啊，记住这里需要改
        accum_iter 含义？ 
        '''
        #
        # xyz_samples = samples[:,:,:1152].to(device, non_blocking=True)
        # rgb_samples = samples[:,:,1152:].to(device, non_blocking=True)

        data = [data.to(args.device, non_blocking=True) for data in samples]
        tea_feat = data[0][:,:2].to(device)
        stu_feat = data[-1].to(device)


        with torch.cuda.amp.autocast():
            loss = model(tea_feat.float(), stu_feat.float())



        loss_value = loss.item()

        if not math.isfinite(loss_value):
            np.save("tea_feat.npy",tea_feat.float().cpu().numpy())
            np.save("stu_feat.npy", stu_feat.float().cpu().numpy())

            print("Loss is {}, stopping training".format(loss_value))
            return
        if accum_iter != 0:
            loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}