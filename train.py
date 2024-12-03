import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# from models.STG_NF.Unet import Unet
from models.STG_NF.model_pose import STG_NF
from models.pretrain_NF import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params
from models.unamsk_encoder.RGBEnc import RGB_Encoder
from models.train_RGBEnc import RGBEnc_Trainer
import matplotlib.pyplot as plt
import time
from dataset import  PoseSegDataset
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def plot_and_save(x,y,auc,log_path,save_path):
    plt.plot(x,y)
    now_time = time.time()
    with open(log_path, "a") as f:
        f.write( str(auc * 100) + "%\n")
    plt.savefig(save_path+"%s"% now_time+".jpg", dpi=300)

def main():
    parser = init_parser()
    args = parser.parse_args()
    root = os.getcwd()


    if args.dataset == "ShanghaiTech":
        feat_path_root = os.path.join(root, r"../STG-NF-main/data/ShanghaiTech/feat")
        if args.tea_pretrained is None:
            tea_pretrained = os.path.join(root,"checkpoints/ShanghaiTech_85_9.tar")
        else:
            tea_pretrained = os.path.join(args.tea_pretrained)

    else:
        feat_path_root = os.path.join(root, r"../STG-NF-main/data/UBnormal/features/ubnormal_feature_i3d")
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
    # dataset :[(3,24,18),trans_index,segs_score_np(24,),labels(1,)]
    '''

    1. Load data from JSON files.
    2. Process each JSON file on a per-character basis in a loop: gen_dataset
       - First, obtain all frames related to a character.
       - Next, pack a certain number of consecutive frames into a group based on seg_len (24) and overlap.
    3. For each call to the dataset's __getitem__, data is organized in segments (segs): [(3, 24, 18), trans_index, segs_score_np(24,), labels(1,)].
    4. Both the training and testing processes operate on a per-segment basis.
    5. When calculating AUC, classify each segment to the corresponding frame in scene_clip_.

    '''
    model_args = init_model_params(args, dataset)
    print(model_args)
    tea_model = STG_NF(**model_args)

    num_of_params = calc_num_of_params(tea_model)
    trainer = Trainer(args, tea_model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs)) ## 训练器
    if args.stage == 1:
        print("Stage One : Train pretrained Model")
        if pretrained:
            trainer.load_checkpoint(pretrained)
        else:
            writer = SummaryWriter()
            trainer.train(log_writer=writer)


        normality_scores = trainer.test()
        auc, scores ,cal_abnormal_nums= score_dataset(normality_scores, dataset["test"].metadata, args=args)
        # Logging and recording results
        print("\n-------------------------------------------------------")
        print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
        with open("log/log.txt", "a") as f:
            f.write(str(args.epochs)+"\t" + str(args.K) + "\t" + str(args.L) + "\t"+ str(args.seg_len) + "\t"+ str(args.seg_stride) + "\t" + str(args.model_lr) + "\t"+ str(args.model_lr_decay) + "\t"+ str(auc * 100) + "%\n")

    ###################### Stage 3 : Train Feat_Student Model ######################
    if args.stage == 2  :
        tea_pretrained = tea_pretrained
        print("Stage 3 : Train Feat_Student Model")
        checkpoint = torch.load(tea_pretrained)
        tea_model.load_state_dict(checkpoint['state_dict'], strict=False)
        tea_model.set_actnorm_init()

        model_args = init_model_params(args, dataset)
        feat_Encoder_model = RGB_Encoder(1024)

        feat_Encoder_trainer = RGBEnc_Trainer(args, feat_Encoder_model, tea_model,loader['train'], loader['test'],
                          optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                          scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs)) ## 训练器
        if pretrained:  ## 加载预训练模型 /ssd/agqing/STG-NF-main/data/exp_dir/ShanghaiTech/Apr20_1139/Apr20_1143__checkpoint.pth.tar
            feat_stu_trainer.feat_load_checkpoint(pretrained)
            normality_scores = feat_Encoder_trainer.feat_test()  ## 训练器的测试过程
            auc, scores, cal_abnormal_nums = score_dataset(normality_scores, dataset["test"].metadata, args=args)

            print("\n-------------------------------------------------------")
            print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
            print("-------------------------------------------------------\n\n")
            plot_and_save(cal_abnormal_nums[:, 0], cal_abnormal_nums[:, 2], auc, log_path="log/log2.txt",
                          save_path="log/test_stage2_abnormal_rate_%s.png")
        else:
            writer = SummaryWriter()
            pretrained = feat_Encoder_trainer.feat_train(log_writer=writer)  ## 开始训练

    return pretrained #



def test(pretrained,batch_size):
    parser = init_parser()
    args = parser.parse_args()
    args.batch_size = batch_size
    args.checkpoint = "checkpoints/feat_checkpoint/" +pretrained


    root = os.getcwd()


    if args.dataset == "ShanghaiTech":
        feat_path_root = os.path.join(root, r"data/ShanghaiTech/feat")
        if args.tea_pretrained is None:
            tea_pretrained = os.path.join(root, "checkpoints/pretrained_ShanghaiTech.tar")
        else:
            tea_pretrained = os.path.join(args.tea_pretrained)

    else:
        feat_path_root = os.path.join(root, r"data/UBnormal/features/ubnormal_feature_i3d")
        if args.tea_pretrained is None:
            tea_pretrained = os.path.join(root, "checkpoints/pretrained_UBnormal.tar")
        else:
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
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, feat_path_root=feat_path_root,
                                             only_test=(pretrained is not None), only_train=False)  ## 这里产生数据集 #

    model_args = init_model_params(args, dataset)
    print(model_args)
    tea_model = STG_NF(**model_args)  ## 模型加载

    num_of_params = calc_num_of_params(tea_model)
    trainer = Trainer(args, tea_model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))  ## 训练器

    ###################### Stage 2 : Train Feat_Student Model ######################
    if args.stage == 2:
        tea_pretrained = tea_pretrained

        print("Stage 2 : Train Feat_Student Model")
        checkpoint = torch.load(tea_pretrained)
        tea_model.load_state_dict(checkpoint['state_dict'], strict=False)
        tea_model.set_actnorm_init()
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print(pretrained)

        model_args = init_model_params(args, dataset)
        feat_Encoder_model = RGB_Encoder(1024)

        feat_Encoder_trainer = RGBEnc_Trainer(args, feat_Encoder_model, tea_model, loader['train'], loader['test'],
                                       optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                                       scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr,
                                                                  epochs=args.epochs))

        feat_stu_trainer.feat_load_checkpoint(pretrained)
        normality_scores = feat_stu_trainer.feat_test()
        auc, scores ,cal_abnormal_nums= score_dataset(normality_scores, dataset["test"].metadata, args=args)

        print("\n-------------------------------------------------------")
        print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
        print("-------------------------------------------------------\n\n")


if __name__ == '__main__':
    pretrained = main()
    print(pretrained)
    test(pretrained, 32)
