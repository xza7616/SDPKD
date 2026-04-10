import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from MLPDecoder import DecoderHead
from thop import profile

class Adapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.blk = blk
        dim = blk.attn.qkv.in_features

        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

        # 可学习缩放因子，防止破坏预训练特征
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        res = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = res + self.scale * x
        return self.blk(x)
# =========================================================
# Utils
# =========================================================
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


# =========================================================
# Prompt Generator
# =========================================================
class PromptGenerator(nn.Module):
    def __init__(self, high_c, K_pos=5, K_neg=5):
        super().__init__()
        self.K_pos = K_pos
        self.K_neg = K_neg

        self.mask_head = nn.Sequential(
            nn.Conv2d(high_c, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )

    @torch.no_grad()
    def forward(self, rgb_feats, dep_feats):
        fr = rgb_feats[-1]
        fd = dep_feats[-1]
        B, _, H, W = fr.shape

        # ---------- score map ----------
        rgb_score = normalize(torch.norm(fr, dim=1, keepdim=True))

        dep_gray = fd.mean(1, keepdim=True)
        gx = dep_gray[:, :, :, 1:] - dep_gray[:, :, :, :-1]
        gy = dep_gray[:, :, 1:, :] - dep_gray[:, :, :-1, :]
        gx = F.pad(gx, (0,1,0,0))
        gy = F.pad(gy, (0,0,0,1))
        dep_grad = torch.sqrt(gx**2 + gy**2 + 1e-6)
        dep_score = 1 - normalize(dep_grad)

        score_map = normalize(0.6 * rgb_score + 0.4 * dep_score)

        # ---------- point prompt ----------
        point_coords, point_labels, point_weights = [], [], []

        for b in range(B):
            flat = score_map[b, 0].view(-1)
            pos_vals, pos_idx = torch.topk(flat, self.K_pos)
            neg_vals, neg_idx = torch.topk(-flat, self.K_neg)

            pos_y, pos_x = pos_idx // W, pos_idx % W
            neg_y, neg_x = neg_idx // W, neg_idx % W

            coords = torch.cat([
                torch.stack([pos_x, pos_y], 1),
                torch.stack([neg_x, neg_y], 1)
            ], 0)

            labels = torch.cat([
                torch.ones(self.K_pos, device=flat.device),
                torch.zeros(self.K_neg, device=flat.device)
            ])

            weights = torch.cat([
                pos_vals / (pos_vals.sum() + 1e-6),
                (-neg_vals) / ((-neg_vals).sum() + 1e-6)
            ])

            point_coords.append(coords)
            point_labels.append(labels)
            point_weights.append(weights)

        points = (
            torch.stack(point_coords).float(),
            torch.stack(point_labels)
        )
        point_weights = torch.stack(point_weights)

        # ---------- mask prompt ----------
        fg = normalize(torch.sigmoid(self.mask_head(fr)))
        blur = F.avg_pool2d(fg, 7, 1, 3)
        boundary = normalize(torch.abs(fg - blur))

        dep_var = F.avg_pool2d(
            (dep_gray - F.avg_pool2d(dep_gray, 5, 1, 2))**2,
            5, 1, 2
        )
        bg = normalize(dep_var)

        mask_prompt = normalize(fg * (1 - boundary) * (1 - bg))

        # ---------- agree / conflict ----------
        agree_list, conflict_list = [], []
        for i in range(4):
            fr_i = F.normalize(rgb_feats[i], dim=1)
            fd_i = F.normalize(dep_feats[i], dim=1)

            sim = normalize((fr_i * fd_i).sum(1, keepdim=True))
            dep_var_i = F.avg_pool2d(
                (dep_feats[i] - F.avg_pool2d(dep_feats[i], 5, 1, 2))**2,
                5, 1, 2
            ).mean(1, keepdim=True)

            conflict = normalize((1 - sim) * normalize(dep_var_i))
            agree_list.append(sim)
            conflict_list.append(conflict)

        return points, point_weights, mask_prompt, agree_list, conflict_list


# =========================================================
# Prompt Feature Fusion（已修正）
# =========================================================
class PromptFeatureFusion(nn.Module):
    def __init__(self, c, token_dim=256):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, c)

        self.spatial_proj = nn.Sequential(
            nn.Conv2d(token_dim, token_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim // 4, 1, 1)
        )

    def forward(self, feat, sparse_token, dense_token,
                point_weights, point_labels,
                agree, conflict):

        B, C_feat, H, W = feat.shape
        _, N_sparse, C_token = sparse_token.shape  # ✅ 不再覆盖 C_feat

        # ----- channel modulation -----
        w = point_weights * (2 * point_labels - 1)
        w = w.unsqueeze(-1)

        point_tokens = sparse_token[:, :w.shape[1], :]
        token = (point_tokens * w).sum(dim=1)   # [B, C_token]

        ch_bias = self.token_proj(token).view(B, C_feat, 1, 1)
        feat = feat + ch_bias

        # ----- spatial modulation -----
        dense = F.interpolate(dense_token, (H, W),
                              mode='bilinear', align_corners=False)
        spatial_attn = torch.sigmoid(self.spatial_proj(dense))
        feat = feat * (1 + spatial_attn)

        # ----- stability -----
        # agree = F.interpolate(agree, (H, W), mode='bilinear', align_corners=False)
        # conflict = F.interpolate(conflict, (H, W), mode='bilinear', align_corners=False)
        # feat = feat * (1 + agree) * (1 - conflict)

        return feat



class PromptConstrainedTransport(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.flow_proj =  nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # ---------- Agreement 可信度建模 ----------
        self.agree_refine = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

        # ---------- Conflict 抑制建模 ----------
        self.conflict_refine = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, fr, fd, agree, conflict):
        B, C, H, W = fr.shape

        # ---------- 尺度对齐 ----------
        agree = F.interpolate(agree, (H, W), mode='bilinear', align_corners=False)
        conflict = F.interpolate(conflict, (H, W), mode='bilinear', align_corners=False)

        # ---------- 学习型可信度建模 ----------
        A = torch.sigmoid(self.agree_refine(agree))        # (B,1,H,W)
        C = torch.sigmoid(self.conflict_refine(conflict))  # (B,1,H,W)

        # ---------- 软约束 Gate ----------
        G = A * (1.0 - C)                                 # (B,1,H,W)

        # ---------- 跨模态残差 ----------
        flow = self.flow_proj(fd - fr)                    # (B,C,H,W)

        # ---------- 约束传输 ----------
        fr = fr + G * flow

        return fr


# =========================================================
# Main Model
# =========================================================
class EnDecoderModel(nn.Module):
    def __init__(self,
                 checkpoint_path='/media/yuride/date/xza2/Sam_prepth2.0/sam2_hiera_large.pt',
                 num_classes=8):
        super().__init__()

        samR = build_sam2("sam2_hiera_l.yaml", checkpoint_path)
        samD = build_sam2("sam2_hiera_l.yaml", checkpoint_path)

        del samR.sam_mask_decoder
        del samR.memory_encoder
        del samR.memory_attention
        del samR.mask_downsample
        del samR.obj_ptr_proj
        del samR.obj_ptr_tpos_proj
        del samR.image_encoder.neck

        del samD.sam_mask_decoder
        del samD.memory_encoder
        del samD.memory_attention
        del samD.mask_downsample
        del samD.obj_ptr_proj
        del samD.obj_ptr_tpos_proj
        del samD.image_encoder.neck

        self.encoderR = samR.image_encoder.trunk
        self.encoderD = samD.image_encoder.trunk
        self.prompt_encoder = samR.sam_prompt_encoder

        # 冻结主干
        for p in self.encoderR.parameters():
            p.requires_grad = False

        for p in self.encoderD.parameters():
            p.requires_grad = False

        # 注入 Adapter
        self.encoderR.blocks = nn.ModuleList(
            [Adapter(b) for b in self.encoderR.blocks]
        )

        self.encoderD.blocks = nn.ModuleList(
            [Adapter(b) for b in self.encoderD.blocks]
        )

        # self.channels = [112, 224, 448, 896]
        self.channels = [144, 288, 576, 1152]

        self.prompt_gen = PromptGenerator(self.channels[-1])

        self.prompt_fusion = nn.ModuleList([
            PromptFeatureFusion(c) for c in self.channels
        ])

        self.transport = nn.ModuleList([
            PromptConstrainedTransport(c) for c in self.channels
        ])

        self.decoder = DecoderHead(
            in_channels=self.channels,
            num_classes=num_classes,
            embed_dim=512
        )

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, rgb, dep):
        rgb_feats = self.encoderR(rgb)
        dep_feats = self.encoderD(dep)

        points, point_weights, mask_prompt, agree_list, conflict_list = \
            self.prompt_gen(rgb_feats, dep_feats)

        sparse_token, dense_token = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_prompt
        )

        fused_feats = []
        for i in range(4):
            fr = self.prompt_fusion[i](
                rgb_feats[i], sparse_token, dense_token,
                point_weights, points[1],
                agree_list[i], conflict_list[i]
            )
            fd = self.prompt_fusion[i](
                dep_feats[i], sparse_token, dense_token,
                point_weights, points[1],
                agree_list[i], conflict_list[i]
            )

            fused_feats.append(
                self.transport[i](fr, fd, agree_list[i], conflict_list[i])
            )

        out = self.decoder(fused_feats)
        return self.upsample(out),fused_feats, mask_prompt, agree_list, conflict_list


# =========================================================
# TEST
# =========================================================
if __name__ == '__main__':
    rgb = torch.randn(1, 3, 480, 640).cuda()
    dep = torch.randn(1, 3, 480, 640).cuda()

    model = EnDecoderModel().cuda()
    out = model(rgb, dep)

    print("Output:", out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Params: {total_params / 1e6:.2f} M")
    print(f"Trainable Params: {trainable_params / 1e6:.2f} M")

    flops, params = profile(model, inputs=(rgb, dep))
    print('Flops', flops / 1e9, 'G')
