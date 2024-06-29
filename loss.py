import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfSupervisedLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, proj_1, proj_2):
        """
        Args:
            proj_1: 第一个视图的投影向量 (batch_size, projection_dim)
            proj_2: 第二个视图的投影向量 (batch_size, projection_dim)
        Returns:
            loss: 自监督对比学习损失
        """
        batch_size = proj_1.shape[0]

        # 计算两个视图之间的相似度矩阵
        proj_1_norm = F.normalize(proj_1, dim=1)
        proj_2_norm = F.normalize(proj_2, dim=1)
        similarity_matrix = torch.matmul(proj_1_norm, proj_2_norm.T) / self.temperature

        # 创建标签张量
        labels = torch.arange(batch_size).to(proj_1.device)

        # 计算损失
        loss = self.criterion(similarity_matrix, labels)

        return loss

class SupervisedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss