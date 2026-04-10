import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from udw.toolbox.backbone.mix_transformer import mit_b2
from MLPDecoder import DecoderHead


# =========================================================
# ============ Adaptive Diffusion Operator =================
# =========================================================
class DiffusionOperator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.laplace = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        nn.init.constant_(self.laplace.weight, 0)

        self.alpha = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lap = self.laplace(x)
        a = self.alpha(x)
        return a * lap


# =========================================================
# ============ Absorption Operator =========================
# =========================================================
class AbsorptionOperator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.beta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        b = self.beta(x)
        return -b * x


# =========================================================
# ============ Structure Source Operator ===================
# =========================================================
class SourceOperator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.highpass = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.inject = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        hp = self.highpass(x) - F.avg_pool2d(x, 3, stride=1, padding=1)
        return self.inject(hp)


# =========================================================
# ============ Operator Evolution Block ====================
# =========================================================
class OperatorBlock(nn.Module):
    """
    One explicit PDE evolution step:
    F_{t+1} = F_t + dt * (Diffusion + Absorption + Source)
    """
    def __init__(self, dim, dt=0.2):
        super().__init__()
        self.diff = DiffusionOperator(dim)
        self.absorb = AbsorptionOperator(dim)
        self.source = SourceOperator(dim)
        self.dt = dt

        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        update = self.diff(x) + self.absorb(x) + self.source(x)
        out = x + self.dt * update
        return self.norm(out)

class SemanticFlow(nn.Module):
    """
    Top-down 语义约束流
    Bottom-up 细节补偿流
    """
    def __init__(self, c_high, c_low):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_high, c_low, 2, stride=2)
        self.fuse = nn.Sequential(
            nn.Conv2d(c_low*2, c_low, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(c_low, c_low, 3, padding=1)
        )

    def forward(self, high, low):
        high_up = self.up(high)
        fused = torch.cat([high_up, low], dim=1)
        return self.fuse(fused)


# =========================================================
# ======================= OFD-Net ==========================
# =========================================================
class EncoderDecoder(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = mit_b2()
        # self.in_channels = [32, 64, 160, 256]
        self.in_channels = [64, 128, 320, 512]

        # operator blocks (2 steps per stage)
        self.ops = nn.ModuleList([
            nn.Sequential(OperatorBlock(64), OperatorBlock(64)),
            nn.Sequential(OperatorBlock(128), OperatorBlock(128)),
            nn.Sequential(OperatorBlock(320), OperatorBlock(320)),
            nn.Sequential(OperatorBlock(512), OperatorBlock(512)),
        ])

        self.flow32 = SemanticFlow(512, 320)
        self.flow21 = SemanticFlow(320, 128)
        self.flow10 = SemanticFlow(128, 64)

        self.decoder = DecoderHead(self.in_channels, num_classes)

    def forward(self, x):
        size = x.shape[2:]

        f1, f2, f3, f4 = self.backbone(x)



        f1 = self.ops[0](f1)
        f2 = self.ops[1](f2)
        f3 = self.ops[2](f3)
        f4 = self.ops[3](f4)

        f3 = self.flow32(f4, f3)
        f2 = self.flow21(f3, f2)
        f1 = self.flow10(f2, f1)

        out = self.decoder([f1, f2, f3, f4])
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        return out, f1, f2, f3, f4

    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.backbone.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.backbone.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")


# =========================================================
# ======================== Test ============================
# =========================================================
if __name__ == "__main__":
    model = EncoderDecoder(num_classes=8).cuda()
    x = torch.randn(1, 3, 480, 640).cuda()

    flops, params = profile(model, (x,))
    print("FLOPs:", flops / 1e9, "G")
    print("Params:", params / 1e6, "M")
    # b0 6.49M   b2 28.94  b1  17.89M

    y = model(x)
    print("Output:", y.shape)
