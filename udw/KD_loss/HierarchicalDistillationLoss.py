import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDistillationLoss(nn.Module):
    """
    综合蒸馏损失：
    - 结构梯度损失（SOGDL，使用 mask, agree, conflict 引导学生特征梯度）
    - 特征对齐损失（使用 agree, conflict 加权对齐特征）
    - 预测误差分布对齐（对齐教师-学生与GT的误差分布）
    """
    def __init__(self, teacher_channels, student_channels, num_classes,
                 lambda_grad=1.0, lambda_feat=1.0, lambda_dist=1.0):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_feat = lambda_feat
        self.lambda_dist = lambda_dist

        # 特征投影层：将学生特征投影到教师特征维度（每个尺度独立）
        self.feat_proj = nn.ModuleList([
            nn.Conv2d(sc, tc, kernel_size=1)
            for sc, tc in zip(student_channels, teacher_channels)
        ])

        # 梯度蒸馏损失
        self.grad_loss = StructureOrientedGradientDistillation()

        # 特征对齐损失使用 L1 或 L2
        self.feat_criterion = nn.MSELoss(reduction='none')  # 逐像素，后续加权

        # 预测误差分布对齐使用 L1 距离
        self.dist_criterion = nn.L1Loss()

    def forward(self, teacher_out, student_out, gt,
                teacher_feats, student_feats,
                teacher_mask_prompt, teacher_agree, teacher_conflict):
        """
        参数:
            teacher_out: (B, C, H, W) 教师预测
            student_out: (B, C, H, W) 学生预测
            gt: (B, H, W) 或 (B, 1, H, W) 真实标签，像素值范围为 0..C-1
            teacher_feats: list of 4 tensors 教师多尺度特征，形状 (B, C_i, H_i, W_i)
            student_feats: list of 4 tensors 学生多尺度特征，形状 (B, C'_i, H'_i, W'_i)
            teacher_mask_prompt: (B, 1, H_high, W_high) 教师掩码提示（最高分辨率）
            teacher_agree: list of 4 tensors 每层的 agree map，形状 (B,1,H_i,W_i)
            teacher_conflict: list of 4 tensors 每层的 conflict map，形状 (B,1,H_i,W_i)
        返回:
            total_loss, dict of individual losses
        """
        device = teacher_out.device
        B, C, H, W = teacher_out.shape
        # 确保 gt 形状与预测一致
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)  # (B,1,H,W)
        gt = F.interpolate(gt.float(), size=(H,W), mode='nearest').long()

        # 1. 结构梯度损失（对每一层计算，取平均）
        grad_loss = 0.0
        for i, (t_feat, s_feat) in enumerate(zip(teacher_feats, student_feats)):
            # 学生特征先投影到教师通道数
            s_feat_proj = self.feat_proj[i](s_feat)
            # 插值到教师特征相同分辨率
            t_feat_res = t_feat
            s_feat_res = F.interpolate(s_feat_proj, size=t_feat_res.shape[-2:],
                                        mode='bilinear', align_corners=False)
            # 获取对应层的提示
            agree = teacher_agree[i]
            conflict = teacher_conflict[i]
            mask_prompt = F.interpolate(teacher_mask_prompt, size=t_feat_res.shape[-2:],
                                        mode='bilinear', align_corners=False)
            # 计算梯度损失
            loss_g, _ = self.grad_loss(s_feat_res, agree, conflict, mask_prompt)
            grad_loss += loss_g
        grad_loss /= len(teacher_feats)

        # 2. 特征对齐损失（使用 agree, conflict 加权）
        feat_loss = 0.0
        for i, (t_feat, s_feat) in enumerate(zip(teacher_feats, student_feats)):
            s_feat_proj = self.feat_proj[i](s_feat)
            t_feat_res = t_feat
            s_feat_res = F.interpolate(s_feat_proj, size=t_feat_res.shape[-2:],
                                        mode='bilinear', align_corners=False)
            agree = teacher_agree[i]
            conflict = teacher_conflict[i]
            # 权重 = agree * (1 - conflict)，约束在 [0,1]
            weight = torch.sigmoid(agree - conflict)  # 与 SOGDL 一致
            # 逐像素加权 L2
            diff = self.feat_criterion(s_feat_res, t_feat_res.detach())
            # 对空间和通道取平均（通道维度取平均，因为权重是空间单通道）
            weighted_diff = diff.mean(dim=1, keepdim=True) * weight
            feat_loss += weighted_diff.mean()
        feat_loss /= len(teacher_feats)

        # 3. 预测误差分布对齐
        # 计算教师预测与GT的逐像素欧氏距离（平方和沿通道）
        teacher_logits = teacher_out  # (B,C,H,W)
        student_logits = student_out
        # 将 GT 转为 one-hot 形式，用于计算每个像素的误差平方
        gt_onehot = F.one_hot(gt.squeeze(1), num_classes=teacher_out.shape[1]).permute(0,3,1,2).float()
        teacher_err = (teacher_logits - gt_onehot) ** 2   # 逐像素平方误差
        student_err = (student_logits - gt_onehot) ** 2
        # 对每个像素，取所有类别的平均（或直接求二范数），得到 (B,1,H,W)
        teacher_err_map = teacher_err.mean(dim=1, keepdim=True)
        student_err_map = student_err.mean(dim=1, keepdim=True)
        # 对齐两个误差分布（L1距离）
        dist_loss = self.dist_criterion(student_err_map, teacher_err_map.detach())

        total_loss = (self.lambda_grad * grad_loss +
                      self.lambda_feat * feat_loss +
                      self.lambda_dist * dist_loss)

        loss_dict = {
            'grad_loss': grad_loss.item(),
            'feat_loss': feat_loss.item(),
            'dist_loss': dist_loss.item(),
            'total_loss': total_loss.item()
        }
        return total_loss, loss_dict


class StructureOrientedGradientDistillation(nn.Module):
    """结构导向梯度蒸馏（复用原定义）"""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()

    def gradient(self, x):
        """计算空间梯度幅值"""
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = F.pad(dx, (0,1,0,0))
        dy = F.pad(dy, (0,0,0,1))
        grad = torch.sqrt(dx**2 + dy**2 + 1e-6)
        return grad

    def forward(self, student_feat, teacher_agree, teacher_conflict, mask_prompt):
        # 尺度对齐已在外部完成
        # 构造权重场 Wt = sigmoid(agree - conflict) * mask
        weight = torch.sigmoid(teacher_agree - teacher_conflict) * mask_prompt
        # 教师结构梯度
        gt_grad = self.gradient(weight)
        # 学生结构梯度（取通道平均）
        student_map = student_feat.mean(dim=1, keepdim=True)
        student_grad = self.gradient(student_map)
        loss = self.loss_fn(student_grad, gt_grad.detach())
        return loss, {'Wt': weight, 'Gt': gt_grad, 'Gs': student_grad}