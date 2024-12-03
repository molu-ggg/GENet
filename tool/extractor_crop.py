import os

import json
path = r"/data/imt3090/agqing/STG-NF-main/data/ShanghaiTech/pose/train/01_069_alphapose_tracked_person.json"
# with open(path,"r") as fn:
#     data = json.load(fn)
#     print(data.keys())
#     print(data['1'].keys())
import numpy as np
# clip_t = 24
# seg_len = 12
# seg_dist = 6
# start_ofst = 0
# num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int)  # 视频重采样，overlap个数，是不是少了+1 ？？？？
# print("num:",num_segs)
# for seg_ind in range(num_segs):
#     start_ind = start_ofst + seg_ind * seg_dist
#     print(start_ind)
import cv2
import json
def cut_person_img(img_path,json_xy,alpha = 0.1):
    image = cv2.imread(img_path)


    json_xy = json_xy.reshape(-1,3)
    #print(json_xy.shape)
    h,w,c  = image.shape
    json_xy = np.array(json_xy)
    small_x = np.min(json_xy[:,0])
    big_x = np.max(json_xy[:,0])
    small_y = np.min(json_xy[:,1])
    big_y = np.max(json_xy[:,1])
    W = big_x- small_x
    H = big_y - small_y
    small_x = max(int(small_x -alpha* W) ,0)
    big_x = min(int(big_x + alpha* W),w)


    small_y = max(int(small_y -alpha* H) ,0)
    big_y = min(int(big_y + alpha* H),h)

    if small_x >  big_x:
        big_x += alpha* H
    if small_y > big_y:
        big_y += alpha *W

    crop_img = image[small_y:big_y,small_x:big_x,:]


    return crop_img

def json_to_img(json_path,img_path_root,save_path_root):


    with open(json_path, 'r', encoding='utf-8') as fp:
        json_dict = json.load(fp)
    person_dict = json_dict.keys()
    # name_list = json_path.split('.')[0].split('\\')[-1]
    name = json_path.split('.')[0].split('/')[-1].replace("_alphapose_tracked_person","")
    print(name)
    scene = "Scene" + str(name.split("_")[2])


    # scene_id = name_list[0]
    # clip_id = name_list[1]
    save_path_file =os.path.join(save_path_root,name)
    if not os.path.exists(save_path_file):
        os.mkdir(save_path_file)

    img_path_file = os.path.join(os.path.join(img_path_root,scene),name)
    # if not os.path.exists(img_path_file):
    #     with open("UBnormal_log/UBnormal_extrator_crop.txt","a") as f :
    #         f.write(img_path_file+ "is not exists"+"\n")

    #
    # if not os.path.exists(save_path_file):
    #     os.mkdir(save_path_file)

    # s_clip_id = clip_id.zfill(4)

    for person in person_dict:
        person_id = int(person)
        s_person = person.zfill(4)
        person_path = os.path.join(save_path_file,s_person)
        if not os.path.exists(person_path):
            os.mkdir(person_path)
        for frame in json_dict[person].keys():
            #print(frame)
            s_frame = frame.zfill(4)
            frame_id = int(frame)
            #print(json_dict[person][frame])
            json_xy = np.array(json_dict[person][frame]['keypoints'])
            # name = scene_id + '_' + s_clip_id + '_' + s_person + '_' + s_frame + '.jpg'
            frame_name = name + "_" + s_person + '_' + s_frame + '.jpg'
            img_path = img_path_file+'/'+ str(int(frame))+'.jpg' #
            #print(img_path)
            if not os.path.exists(img_path):
                with open('UBnormal_log/error.txt', 'a', encoding="utf-8") as f:
                    f.write(img_path + "\t is not exist,IMgae is not exist in images\n")
                continue
            crop_img = cut_person_img(img_path, json_xy, alpha=0.2)

            crop_img_path = os.path.join(person_path,frame_name)
            # if not os.path.exists()
            print(crop_img_path)

            cv2.imwrite(crop_img_path,crop_img)

def main_deal(json_path):
    ## train
    # json_to_img(r"D:\OneDrive\A-expriments\STG-NF-main\data\ShanghaiTech\pose\train\08_001_alphapose_tracked_person.json",
    #             r"D:\.mydatasets\ShanghaiTech\training\training\train_frame",
    #             r"D:\OneDrive\A-expriments\STG-NF-main\data\ShanghaiTech\crop\train")
    path  = os.path.join(json_path,'test')
    for json in os.listdir(path):
        if json.endswith("person.json"):
            json_to_img(os.path.join(path,json), r"/ssd/agqing/data/UBnormal/images", r"/ssd/agqing/data/UBnormal/crop/test")
    # ## test:
    # for json in os.listdir(os.path.join(json_path,'test')):
    #     if json.endswith("person.json"):
    #         json_to_img(os.path.join(os.path.join(json_path,'test'),json), r"D:\.mydatasets\ShanghaiTech\testing\test_frame", r"D:\OneDrive\A-expriments\STG-NF-main\data\ShanghaiTech\crop\test")

main_deal(r"/ssd/agqing/STG-NF-main/data/UBnormal/pose")






