

# \[ACMMM24\] A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection

paper link :https://dl.acm.org/doi/abs/10.1145/3664647.3680592

The essence of this algorithm is to explore a novel guidance-exploration network architecture, which resembles the teacher-student architecture pattern. However, what sets it apart is that the teacher network is no longer a network with a massive number of parameters. Instead, it can be quite small. Additionally, the input modalities are diverse, and the encoder architectures vary accordingly. Detect anomalies based on the differences in modality outputs.



The implementation code for the paper.

- 2023-12: the code of our paper "A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection" reached its upper limit. The performance on the SHT dataset can reach 86.9%, and on the UBnormal dataset it can reach 73.5%.

- 2024-03: relevant explanations were supplemented.

- 2024-08: with the model iteration of Simple-GENet, considering that the inference speed of the algorithm was too slow, we further optimized the version of the algorithm. We removed the two-stage mask method and only retained and optimized the model in the first stage. The training and inference speeds have been improved.

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
@inproceedings{10.1145/3664647.3680592,
author = {Yang, Guoqing and Luo, Zhiming and Gao, Jianzhe and Lai, Yingxin and Yang, Kun and He, Yifan and Li, Shaozi},
title = {A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection},
year = {2024},
isbn = {9798400706868},
url = {https://doi.org/10.1145/3664647.3680592},
doi = {10.1145/3664647.3680592},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {5865–5873},
numpages = {9},
}
```

