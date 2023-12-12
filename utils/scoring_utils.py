import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import shanghaitech_hr_skip


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#
def score_dataset(score, metadata, args=None,log_times = 100): ####  metadata === load['test']

    gt_arr, scores_arr = get_dataset_scores(score, metadata, args=args) # gt_arr 真实, scores_arr   预测

    # Len = len(gt_arr)
    scores_arr = smooth_scores(scores_arr)
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc = score_auc(scores_np, gt_np)


    with open(r"result/%s.txt"%(str(args.dataset)),"w") as f:
        for i , _ in enumerate(gt_np):
            f.write(str(i)+","+str(gt_np[i])+","+str(scores_np[i])+"\n")

    list = []
    for i in range(gt_np.shape[0]):
        list.append([gt_np[i],scores_np[i], gt_np[i]-scores_np[i]])
    np_sort_lits = np.array(list)
    if args.stage == 1:
        np_sort_lits = np_sort_lits[np_sort_lits[:, 1].argsort()]
    else:
        np_sort_lits = np_sort_lits[np_sort_lits[:, 2].argsort()]


    cal_abnormal_nums = []
    j =0
    while  j + log_times < gt_np.shape[0]:
        num_1 = np.sum(np_sort_lits[j:j+log_times],axis=0)[0]
        j = j + log_times
        num_0 = log_times - num_1
        rate = num_0/log_times
        cal_abnormal_nums.append([j,j/gt_np.shape[0],rate])
        with open("sort.txt","a") as f:
            f.write(str(j) + "," + str(num_1) + "," + str(num_0)+ "," + str(rate) + "\n")
    num_1 = np.sum(np_sort_lits[j:gt_np.shape[0]], axis=0)[0]
    len = gt_np.shape[0] - j
    j = gt_np.shape[0]
    num_0 = len - num_1
    rate = num_0 / len
    cal_abnormal_nums.append([j,j / gt_np.shape[0], rate])
    cal_abnormal_nums = np.array(cal_abnormal_nums)
    #np.savez("shanghai_test_labels", gt_np)
    return auc, scores_np,cal_abnormal_nums


def get_dataset_scores(scores, metadata, args=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)
    # print(metadata_np.shape)
    np.savez("shanghai_test_meta",metadata_np)
    print("2次ok")

    if args.dataset == 'UBnormal':
        pose_segs_root = '../STG-NF-main/data/UBnormal/pose/test'
        clip_list = os.listdir(pose_segs_root) ##测试集的文件名字，原来测试集是自己做的，也就是说可以作者很可能会伪造结果？
        clip_list = sorted(
            fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
        per_frame_scores_root = 'data/UBnormal/gt/'
    elif args.dataset in "ShanghaiTech":
        per_frame_scores_root = '../STG-NF-main/data/ShanghaiTech/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)  ###
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    elif args.dataset == "Avenue":
        per_frame_scores_root = 'data/Avenue/gt'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy')) ### 这里可以直接更换成那个大的

    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args) ### 得到clip的分数，将人物进行归并了
        '''
        ## clip_gt 真实标签，具体到帧，clip_score 预测异常得分？是的
        ## z这个clip_score 是这样的，
        对于每一个视频clip，找到所有的人物ID
        循环遍历人物ID：
            找到所有对应的的人物帧ID（也就是__getitem__ 一次返回的数据） 
            初始化每一个人物帧的得分矩阵为无穷大，矩阵大小是视频帧数大小。
            对于每一个人物帧，实际上就是一个seg片段，将seg其中间位置赋分数为本段的分数 （这样做是否合理呢？？？？？？？）
        最后该一个视频中所有的得分矩阵，得分矩阵的每一个位置（代表一个帧）的分数为 上述所有得分矩阵对应位置的最小值
                
        '''
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt) # shape :107,每个clip的具体帧级别标签列表
            dataset_scores_arr.append(clip_score)


    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index] ## 第clip段的第t帧
            index += 1
    return dataset_gt_arr, dataset_scores_arr ## 返回真实与预测


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc = roc_auc_score(gt, scores_np)
    return auc


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):
    '''
    metadata_np: segs_meta（metadata):[[int(scene_id), clip_id, int(single_pose_meta[0] 人物ID), int(start_key seg的起始帧id)]......] 一共有人物idx个数*人物seg个片段*clip个数
    '''
    if args.dataset == 'UBnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        clip_id = type + "_" + clip_id
    elif args.dataset in 'ShanghaiTech':
        #print(clip.replace("label", "001").split('.')[0].split('_'))
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]

        if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
            return None, None
    else :
        scene_id, clip_id = 0 , int(clip[:2])



    clip_metadata_inds2 = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id)) # 因爲metadata_np 为 （数据集总长度16278，4），所以返回值是（第一维所在的位置，第二维所在的位置），我们只需要第一维,取的是编号。
    clip_metadata_inds=clip_metadata_inds2[0] ## (前面函数嵌套了107个)，因此这里提取的信息是，每一个视频clip所含有的人物帧的4种数据 ； 人物帧id就是dataset每一次iter一个的数据 107*{每个clip包含的人物帧id}
    clip_metadata = metadata[clip_metadata_inds] ##
    #print(metadata_np[90],metadata_np[91])#[  1 177   1  90] [  1 177   1  91]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata]) ##找到该clip的所有人物帧数据 (x,4) 的所有人物ID（debug的时候之所以这里为0 是因为数据集没有全部加载过去）
    clip_res_fn = os.path.join(per_frame_scores_root, clip) # 路径，每个集合里有人物ID集合
    clip_gt = np.load(clip_res_fn) ## 这是数据集：每个clip都是有帧数个01组成   对应clip的帧数个
    # print(clip_res_fn,clip_gt)
    if args.dataset != "UBnormal": # 在UBnormal中1是正常0 是异常，而ShanghaiTech 和 Avenue 正好相反
        clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf ### 对于每一个clip，维度为帧数大小
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}
    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0] ### 这个函数返回的是一个列表，列表中的数据是一个视频中人物帧编号的集合，即每一次__getitem__ 的数据的编号
        # print(person_metadata_inds,np.array(scores).shape)
        pid_scores =np.array(scores)[person_metadata_inds]
        #print(person_metadata_inds,pid_scores) # x个list，list元素个数是16278，分数都是在-3~-5之间的，维度与前面保持一致

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores ### 这个地方为什么要这个样子
    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))

    clip_score = np.amin(clip_ppl_score_arr, axis=0)# 行 :一维数组a中的最小值，二维数组需通过axis指定行或列，
    # print(clip_ppl_score_arr.shape,clip_score.shape)

    return clip_gt, clip_score ## 真实标签
