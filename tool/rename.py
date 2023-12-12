import os

def rename(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path,folder)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path,file)
            re_img_int = int(file.split(".")[0])
            re_img = str(re_img_int)+".jpg"
            re_img_path = os.path.join(folder_path,re_img)
            os.rename(img_path,re_img_path)
rename(r"D:\.mydatasets\ShanghaiTech\testing\test_frame")

