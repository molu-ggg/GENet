import os

import numpy as np

def deal_to_20(path,dst):
    print("why")
    # with open(r"/mydata/AST/data/done.txt") as f :
    #     list = f.readlines()
    # print(list)



    for file in os.listdir(path):###

        # if file + "\n" in list:
        #     print("ok")
        #     continue
        file_path = os.path.join(path,file)
        np_img = np.load(file_path)
        id = int(file[-8:-4])
        print(file)
        print(np_img.shape)

        for i in range(20):#
            start = i*128
            if start >= np_img.shape[0]:
                continue
            end = min((i+1)*128,np_img.shape[0])
            name =str( id + i ).zfill(4)
            new_img = np_img[start:end]
            dst_path = os.path.join(dst,file[:-8]+name)
            np.save(dst_path,new_img)

if __name__ == "__main__":
    deal_to_20(r"/mydata/AST/data/backgroud/backgroud_train", r"/mydata/AST/data/backgroud/1")