

# [A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection](https://arxiv.org/abs/2312.04119)

paper link :https://arxiv.org/abs/2312.04119

The essence of this algorithm is to explore a novel guidance-exploration network architecture, which resembles the teacher-student architecture pattern. However, what sets it apart is that the teacher network is no longer a network with a massive number of parameters. Instead, it can be quite small. Additionally, the input modalities are diverse, and the encoder architectures vary accordingly. Detect anomalies based on the differences in modality outputs.

![image-20231212115445824](https://raw.githubusercontent.com/molu-ggg/image230306/master/image/imgimage-20231212115445824.png)

The implementation code for the paper.

Note: This repository corresponds to the previous version of the algorithm, similar to multilevel-1. The key difference is that RGBENc employs a traditional convolutional neural network. The achieved AUC values on the ShanghaiTech and UBnormal datasets are 86.7% and 73.2%, respectively.

Certainly, please provide the specific steps of the algorithm, and I'll do my best to assist or provide information based on the details you provide.

# Install

 CUDA 11/10 Pytorch==1.10

```
pip install -r requirements.txt
```



# Stage1 - Pretrain 

Drawing inspiration from the STG-NF paper, you can download pre-trained models for extracting skeletal information using the alphahpose algorithm from this source.

```python
python train.py --stage 1 
```



# Stage2 - Train 

We require pre-extracted features, and the download files are provided here:

Note: We have observed that even without pre-extracting features, the algorithm can still achieve satisfactory results

Train:

```
python train.py --stage 2 --epochs 10 --dataset ShanghaiTech 
```

# Test 

We provide pre-trained models for both Stage 1 and Stage 2, which you can directly download and load:

```
python train.py --stage 2 --epochs 15 --checkpoint checkpoints/... --dataset ShanghaiTech
python train.py --stage 2 --epochs 15 --checkpoint checkpoints/... --dataset UBnormal --seg_len  16 
```



# Cite

If you find this paper useful, please cite this work as follows:

```
@article{yang2023multilevel,
  title = {A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection},
  author = {Guoqing Yang et al.},
  journal={arXiv preprint arXiv:2312.04119},
  year = {2023},
}
```

