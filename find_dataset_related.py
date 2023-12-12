import os

root = os.getcwd()

ShanghaiTech = "data/ShanghaiTech/images"
UBnormal = "data/UBnormal/images"

dataset_root = os.path.join(os.path.join(root,UBnormal),"train")
ans =0
write_list = []
for file in sorted(os.listdir(dataset_root)):
    file_path= os.path.join(dataset_root,file)

    count = len(os.listdir(file_path))
    write_list.append([file,ans,ans+count])
    ans = ans + count

with open("UBnormal_train.txt","w") as f:
    for i in range(len(write_list)):
        f.write(str(write_list[i][0])+","+str(write_list[i][1])+","+str(write_list[i][2])+"\n")





