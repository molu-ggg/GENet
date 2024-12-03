from models.STG_NF.model_pose import FlowNet
from models.STG_NF.Student import Feat_Student
import torch
import torch.nn as nn
import  numpy as np
import math

'''
1. 提取特征  extractor_model
2.对比学习 fusion
'''
class fusion(nn.Module):
    def __init__(self,
            pose_shape,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            learn_top,
            device,
            R=0,
            edge_importance=False,
            temporal_kernel_size=None,
            strategy='uniform',
            max_hops=8,):
        super(fusion,self).__init__()
        # self.tea = FlowNet(
        #     pose_shape=pose_shape,
        #     hidden_channels=hidden_channels,
        #     K=K,
        #     L=L,
        #     actnorm_scale=actnorm_scale,
        #     flow_permutation=flow_permutation,
        #     flow_coupling=flow_coupling,
        #     LU_decomposed=LU_decomposed,
        #     edge_importance=edge_importance,
        #     temporal_kernel_size=temporal_kernel_size,
        #     strategy=strategy,
        #     max_hops=max_hops,
        #     device=device,)
        # self.stu = Feat_Student(1024,n_blocks=4)
        self.extractor_model = extractor_model(
            pose_shape,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            learn_top,
            device,
            R=0,
            edge_importance=False,
            temporal_kernel_size=None,
            strategy='uniform',
            max_hops=8,
        ).to(device)
        self.stu_head = nn.Linear(24,32).to(device)  ###
        self.tea_head = nn.Linear(24,32).to(device)  ###
        self.T = 1

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        value = nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
        if not math.isfinite(value.item()):
            np.save("logits.npy",logits.cpu().detach().numpy() )
            np.save("labels.npy",labels.cpu().detach().numpy() )


        return value
    def feature_fusion(self, x_tea, x_stu):

        feat_tea,feat_stu = self.extractor_model(x_tea,x_stu)

        feature = torch.cat([feat_tea, feat_stu], dim=2)

        return feat_stu,feature

    def forward(self,x_tea,x_stu):
        feat_tea, feat_stu = self.extractor_model(x_tea, x_stu)
        # B,C,T,V = feat_tea.shape
        # feat_tea = feat_tea.view(B,C*T*V)
        # B, C, T, V = feat_stu.shape
        # feat_stu = feat_stu.view(B,C*T*V)


        q = self.tea_head(feat_tea.view(-1, feat_tea.shape[2])) ##
        k = self.stu_head(feat_stu.view(-1, feat_stu.shape[2])) ##

        tea_feat = x_tea.view(-1, x_tea.shape[2])
        rgb_feat = x_stu.view(-1, x_stu.shape[2])

        # q = self.tea_head(feat_tea) ##
        # k = self.stu_head(feat_stu) ##


        patch_no_zeros_indices = torch.nonzero(torch.all(tea_feat != 0, dim=1)) #### 这句话什么意思？不懂啊？
        loss = self.contrastive_loss(q[patch_no_zeros_indices, :].squeeze(), k[patch_no_zeros_indices, :].squeeze())

        return loss

#
class extractor_model(nn.Module):
    def __init__(self,
                 pose_shape,
                 hidden_channels,
                 K,
                 L,
                 actnorm_scale,
                 flow_permutation,
                 flow_coupling,
                 LU_decomposed,
                 learn_top,
                 device,
                 R=0,
                 edge_importance=False,
                 temporal_kernel_size=None,
                 strategy='uniform',
                 max_hops=8,
                 ):
        super(extractor_model, self).__init__()
        self.tea = FlowNet(
            pose_shape=pose_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            edge_importance=edge_importance,
            temporal_kernel_size=temporal_kernel_size,
            strategy=strategy,
            max_hops=max_hops,
            device=device,)
        self.stu = Feat_Student(1024,n_blocks=4)


    def forward(self, x_tea, x_stu):


        x_tea,_ = self.tea(x_tea)
        x_stu = self.stu(x_stu)
        return x_tea,x_stu






