import os
import re

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
# from numpy import asarray

plt.style.use('seaborn-ticks')


def get_ab_labels(global_data_np_ab, segs_meta_ab, segs_root=None):
    pose_segs_root = segs_root
    clip_list = os.listdir(pose_segs_root)
    clip_list = sorted(
        fn.replace("alphapose_tracked_person.json", "annotations") for fn in clip_list if fn.endswith('.json'))
    per_frame_scores_root = 'data/UBnormal/videos/'
    labels = np.ones_like(global_data_np_ab)
    for clip in tqdm(clip_list):
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_annotations.*', clip)[0]
        if type == "normal":
            continue
        clip_id = type + "_" + clip_id
        clip_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                      (segs_meta_ab[:, 0] == scene_id))[0]
        clip_metadata = segs_meta_ab[clip_metadata_inds]
        clip_res_fn = os.path.join(per_frame_scores_root, "Scene{}".format(scene_id), clip)
        filelist = sorted(os.listdir(clip_res_fn))
        clip_gt_lst = [np.array(Image.open(os.path.join(clip_res_fn, fname)).convert('L')) for fname in filelist]
        # FIX shape bug
        clip_shapes = set([clip_gt.shape for clip_gt in clip_gt_lst])
        min_width = min([clip_shape[0] for clip_shape in clip_shapes])
        min_height = min([clip_shape[1] for clip_shape in clip_shapes])
        clip_labels = np.array([clip_gt[:min_width, :min_height] for clip_gt in clip_gt_lst])
        gt_file = os.path.join("data/UBnormal/gt", clip.replace("annotations", "tracks.txt"))
        clip_gt = np.zeros_like(clip_labels)
        with open(gt_file) as f:
            abnormality = f.readlines()
            for ab in abnormality:
                i, start, end = ab.strip("\n").split(",")
                for t in range(int(float(start)), int(float(end))):
                    clip_gt[t][clip_labels[t] == int(float(i))] = 1
        for t in range(clip_gt.shape[0]):
            if (clip_gt[t] != 0).any():  # Has abnormal event
                ab_metadata_inds = np.where(clip_metadata[:, 3].astype(int) == t)[0]
                # seg = clip_segs[ab_metadata_inds][:, :2, 0]
                clip_fig_idxs = set([arr[2] for arr in segs_meta_ab[ab_metadata_inds]])
                for person_id in clip_fig_idxs:
                    person_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                                    (segs_meta_ab[:, 0] == scene_id) &
                                                    (segs_meta_ab[:, 2] == person_id) &
                                                    (segs_meta_ab[:, 3].astype(int) == t))[0]
                    data = np.floor(global_data_np_ab[person_metadata_inds].T).astype(int)
                    if data.shape[-1] != 0:
                        if clip_gt[t][
                            np.clip(data[:, 0, 1], 0, clip_gt.shape[1] - 1),
                            np.clip(data[:, 0, 0], 0, clip_gt.shape[2] - 1)
                        ].sum() > data.shape[0] / 2:
                            # This pose is abnormal
                            labels[person_metadata_inds] = -1
    return labels[:, 0, 0, 0]


def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False,
                         global_pose_data=[], dataset="ShanghaiTech"):
    """
    注意这个函数在两个地方调用了
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array生成一个分段序列数组，每个对象是一个分段和一个对应的元数据数组
    这个函数的作用是将输入的姿态估计数据按照给定的参数（如seg_stride，seg_len等）进行分割，并返回分割后的数据和元数据数组。GPT
    clip_dict: 将json 加载出来，是一个字典形式
    """
    pose_segs_data = []
    score_segs_data = []
    pose_segs_meta = []
    #first_imgs_path_list_allpeople = [] ###
    person_keys = {}
    all_idx_img_index = []
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)): # 这是遍历某一scene_id的某clip_id的数据：也就是一段视频中出现几个人物就循环几次
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = single_pose_dict2np(clip_dict, idx)## 所有人物？（第二维） 的数据 [[0.9x,0.3,...],[..],..]; [index, first_frame] ;  的keys ;   的得分
        #print("sing_pose_meta",sing_pose_meta)
        '''
        #extract_img(clip_dict, idx) 现在就应该提取每一段一个照片即可，上面那个函数是把一个任务
        #sing_pose_keys: dict_keys(['0000', '0001', '0002', '0003',......'0113']
        # sing_pose_np ：[(17,3)....]  list 长度为帧数个
        # sing_pose_meta： [人物id，出现一个clip的第一帧的帧编号(比如‘0113’)]
        # sing_pose_keys:  single_person_dict_keys 一个人物id 在一个clip的 的所有帧编号 
        # sing_scores_np ： 得分
        '''

        if dataset == "UBnormal":
            key = ('{:02d}_{}_{:02d}'.format(int(scene_id), clip_id, int(idx)))
        else:
            key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))

        person_keys[key] = sing_pose_keys
        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np,img_index = split_pose_to_segments(sing_pose_np, # 这是将单个pose 分成没seg_len 为一组 连续帧动作
                                                                                            sing_pose_meta,
                                                                                            sing_pose_keys,
                                                                                            start_ofst, seg_stride,
                                                                                            seg_len,
                                                                                            scene_id=scene_id,
                                                                                            clip_id=clip_id,
                                                                                            single_score_np=sing_scores_np,
                                                                                            dataset=dataset,

                                                                                            )
        '''
        curr_pose_segs_np:是 [(seg_len,17,3).....]  list 为overlap将clip分成的seg个数 ，为一个视频中，一个人的动作 分成平均12帧一段，overlap为6 ;
        curr_pose_segs_meta ：[ [int(scene_id), clip_id, int(single_pose_meta[0] 人物ID), int(start_key 帧id)] ......],list 为overlap将clip分成的seg个数 表示一个人在一个视频中所有的seg片段，其中包含每个seg的起始帧
        curr_pose_score_np： 得分
        '''
        all_idx_img_index =  all_idx_img_index+ img_index ## 连接

        #print("?",idx,curr_pose_segs_np[0].shape)
        pose_segs_data.append(curr_pose_segs_np) #pose_segs_data 是[[(seg_len,17,3).....]......[(seg_len,17,3).....]] , 是该视频中所有人的数据；按照人来排序的，再次是时间帧:
        score_segs_data.append(curr_pose_score_np)
        #first_imgs_path_list_allpeople = first_imgs_path_list if len(first_imgs_path_list_allpeople)==0 else first_imgs_path_list_allpeople +  first_imgs_path_list  ###
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        pose_segs_meta += curr_pose_segs_meta
    if len(pose_segs_data) == 0:
        pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 3) #可能是将数据分成了几段，每一段都有17，3长度（大概率是这样）
        score_segs_data_np = np.empty(0).reshape(0, seg_len) # 一组中seg_len 个分数
    else:
        pose_segs_data_np = np.concatenate(pose_segs_data, axis=0) ## 合并前两维度 ，一个视频，不分人的界限，只是按照人的编号顺序在同一维度将以seg_len长度的连续帧段依次排开（x,seg_len,17,3）
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)
    global_pose_data_np = np.concatenate(global_pose_data, axis=0)# 在axis= 0 拼接
    del pose_segs_data

    # del global_pose_data

    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys, global_pose_data_np, global_pose_data, score_segs_data_np,all_idx_img_index ###
    else:
        return pose_segs_data_np, pose_segs_meta, global_pose_data_np, global_pose_data, score_segs_data_np,all_idx_img_index#,first_imgs_path_list_allpeople ###
    '''
    pose_segs_data_np:[shape（x,seg_len,17,3)...] list为clip中人物数量，一共有seg个片段*人物idx个数
    pose_segs_meta: [[int(scene_id), clip_id, int(single_pose_meta[0] 人物ID), int(start_key 帧id)]......] 一共有人物idx个数*人物seg个片段
    score_segs_data_np：
    global_pose_data_np:[(17,3)]以帧数为单位的，lsit长度为人物所在帧数*人物个数
    global_pose_data:保留前两维度： （人物个数，人物所在帧数，17,3）


    到这里位置，说明了函数是以seg划分的，那么训练和最后auc 以及test是怎么计算的呢？ 

    '''
    # 分段后的（骨骼）数据？ 怎么分段？一个文件的一个di中有(['0000', '0001', '0002', '0003', ....'0113'] 这么多个，将他们按照以6?个一组分段（也可以叫做分组）
    # 那么这么理解， 分段后（第二维） 的数据 [[0.9x,0.3,......],[......],......];
    #pose_segs_meta 分段后[index, first_frame] ;
    # 所有帧？（第二维） 的keys ;
    # global_pose_data：所有帧？（第二维） 的数据 [[0.9x,0.3,......],[......],......]
    # score_segs_data_np:分段后？（第二维） 的得分

# def extract_img(person_dict, idx):
#     single_person = person_dict[str(idx)]

def single_pose_dict2np(person_dict, idx,seg_len=12):

    single_person = person_dict[str(idx)]
    sing_pose_np = []
    sing_scores_np = []
    imgs_names = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys()) #dict_keys(['0000', '0001', '0002', '0003',......'0113']
    ### print("single_person_dict_keys",idx,single_person_dict_keys)
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3) # (51,) --> (17,3)
        sing_pose_np.append(curr_pose_np)
        sing_scores_np.append(single_person[key]['scores'])
    sing_pose_np = np.stack(sing_pose_np, axis=0) ## 增加一个维度的链接 从axis=0可以区分不同的原先数据
    sing_scores_np = np.stack(sing_scores_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys, sing_scores_np
## 所有帧？（第二维） 的数据 [[0.9x,0.3,......],[......],......]; [index, first_frame] ; 所有帧？（第二维） 的keys ;  所有帧？（第二维） 的得分


def is_single_person_dict_continuous(sing_person_dict):
    """
    Checks if an input clip is continuous or if there are frames missing
    :return:
    """
    start_key = min(sing_person_dict.keys())
    person_dict_items = len(sing_person_dict.keys())
    sorted_seg_keys = sorted(sing_person_dict.keys(), key=lambda x: int(x))
    return is_seg_continuous(sorted_seg_keys, start_key, person_dict_items)


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12, # seg_dist=6, seg_len=12 这两个变量就实现了重复overlap 采集
                           scene_id='', clip_id='', single_score_np=None, dataset="ShanghaiTech" , img_path=None,idx=None):
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_score_np = np.empty([0, seg_len])
    pose_segs_meta = []
    img_index= []
    #first_imgs_path_list = [] ### 表示这一段视频中的第一帧画面信息
    num_segs = np.ceil((clip_t - seg_len) / seg_dist +1 ).astype(np.int) #视频重采样，overlap个数，是不是少了+1 ？？？？ 这个是向上取整返回值能
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len) and len(single_pose_np) > start_ind + seg_len:#####只要seg_len 长度，不足seg_len的不要
            index = single_pose_keys_sorted[start_ind + seg_len//2]
            img_index.append(index)


            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            curr_score = single_score_np[start_ind:start_ind + seg_len].reshape(1, seg_len)
            # else:#
            #     np_seg1 = single_pose_np[start_ind:len(single_pose_np)]
            #     np_seg2 = single_pose_np[len(single_pose_np)-1:len(single_pose_np)]
            #     np_score1 = single_score_np[start_ind:len(single_pose_np)]
            #     np_score2 = single_score_np[len(single_pose_np)-1:len(single_pose_np)]
            #     for i in range(start_ind+seg_len-len(single_pose_np)):
            #         np_seg1 = np.concatenate((np_seg1,np_seg2),axis=0)
            #         np_score1 = np.concatenate((np_score1, np_score2), axis=0)
            #     curr_segment = np_seg1.reshape(1, seg_len, kp_count, kp_dim)
            #     curr_score = np_score1.reshape(1, seg_len)


            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_score_np = np.append(pose_score_np, curr_score, axis=0)


            if dataset == "UBnormal":
                pose_segs_meta.append([int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)])
            else:
                pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
    #print(single_pose_np.shape,num_segs, pose_segs_np.shape)
    return pose_segs_np, pose_segs_meta, pose_score_np,img_index#,first_imgs_path_list
## pose_segs_np : 为一个视频中，一个人的动作 分成平均12帧一段，overlap为6

# def img_to_np(img_path):
#     img = Image.open(img_path)
#     np_img = np.asarray(img)
#
#     return np_img  # 要resize 处理啊




