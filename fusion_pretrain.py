import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import timm
import timm.optim.optim_factory as optim_factory
import utils.misc as misc  ###
from utils.misc import NativeScalerWithGradNormCount as NativeScaler  ###
from engine_fusion_pretrain import train_one_epoch  ###
import img_dataset
import torch
from models.STG_NF.fusion import fusion


from models.STG_NF.model_pose import STG_NF
from models.pretrain_NF import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from img_dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params
from models.STG_NF.Student import Student,Feat_Student
from models.train_RGBEnc import Stu_Trainer
import matplotlib.pyplot as plt
from img_dataset import  PoseSegDataset
from models.stage4.model import Fusion




def main():
    parser = init_parser()
    args = parser.parse_args()
    print(args)


    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    log_writer = SummaryWriter(log_dir=args.log_dir)
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    # dataset_train = dataset.PreTrainTensorDataset(args.data_path)  #### 数据
    #
    # print(dataset_train)

    '''
    数据集的处理
    '''
    root = os.getcwd()
    if args.dataset == "ShanghaiTech":
        feat_path_root = os.path.join(root, r"data/ShanghaiTech/feat")
        if args.tea_pretrained is None:
            tea_pretrained = os.path.join(root,"checkpoints/ShanghaiTech_85_9.tar")
        else :
            tea_pretrained = os.path.join(args.tea_pretrained)

    else :
        feat_path_root = os.path.join(root, r"data/UBnormal/features/ubnormal_feature_i3d")
        if args.tea_pretrained is None:
            tea_pretrained = os.path.join(root,"checkpoints/UBnormal_unsupervised_71_8.tar")
        else :
            tea_pretrained = os.path.join(args.tea_pretrained)


    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)



    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, feat_path_root = feat_path_root,only_test=(pretrained is not None),only_train = False ) ## 这里产生数据集 #



    '''
    batch_size 处理   这部分最后要解决
    '''

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    '''
    模型的加载与优化器，      损失函数（没见过）怎么工作的
    '''
    model_args = init_model_params(args, dataset)
    print(model_args)
    model = fusion(**model_args)## 模型加载
    model.to(args.device)
    print("OK2")
    ### 分布式训练
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], find_unused_parameters=True)
    model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    print("OK3")

    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
    print("OK4")
    '''
    这里往后的没有看，记得要补充
    
    
    '''
    '''
    模型的训练与保存
    '''
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, loader["train"],
            optimizer, args.device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )  ### 对比学习 ，train_one_epoch 包括了train过程完整的一次包括梯度更新等，可以视为train.py   model里定义了对比学习损失函数
        print("OK5")
        if args.output_dir and (epoch % 1 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    main()
