import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ss2d import SS2D_my
    from csms6s import DynamicCenterScan, DynamicCenterMerge
except ImportError:
    pass

# ==========================================
# Multi-Scale Feature Enhancer (MSFE)
# ==========================================
class MSFE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Branch 1: Local Context (3x3 equivalent)
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)), 
            nn.BatchNorm2d(in_channels), nn.GELU(), 
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1)), 
            nn.BatchNorm2d(in_channels), nn.GELU()
        )
        # Branch 2: Broader Context (5x5 equivalent)
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(0, 2)), 
            nn.BatchNorm2d(in_channels), nn.GELU(), 
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(2, 0)), 
            nn.BatchNorm2d(in_channels), nn.GELU()
        )
        # Branch 3: Pixel Context (1x1)
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=(0, 0)), 
            nn.BatchNorm2d(in_channels), nn.GELU()
        )
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0), 
            nn.BatchNorm2d(in_channels), nn.GELU()
        )
        
    def forward(self, x):
        return self.fusion(F.gelu(self.scale1(x) + self.scale2(x) + self.scale3(x)))

# ==========================================
# Mamba Spatial Block
# ==========================================
class MambaSpatial(nn.Module):
    def __init__(self, in_features, d_conv=3, expand=1, d_state=16):
        super().__init__()
        
        d_inner = int(expand * in_features)
        self.in_proj = nn.Linear(in_features, d_inner)
        self.in_proj_skip = nn.Conv1d(1, 1, kernel_size=5, padding=2)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, in_features)
        self.conv2d = nn.Conv2d(d_inner, d_inner, groups=d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2)
        
        # Dynamic SSM Branch
        self.mamba = SS2D_my(d_model=d_inner, d_state=d_state, ssm_ratio=1, d_conv=d_conv, scan_type="spa", k_group=2)
        
        # Local Geometric Path (LGP)
        self.local_path = nn.Sequential(
            nn.Conv2d(d_inner, d_inner, 3, 1, 1, groups=d_inner),
            nn.BatchNorm2d(d_inner),
            nn.SiLU()
        )

    def get_dynamic_index(self, x):
        B, C, H, W = x.shape
        center_feat = x[:, :, H//2, W//2] 
        x_flat = x.view(B, C, -1)
        x_norm = F.normalize(x_flat, dim=1)
        c_norm = F.normalize(center_feat, dim=1).unsqueeze(2)
        sim = torch.bmm(x_norm.transpose(1, 2), c_norm).squeeze(2)
        _, sort_idx = torch.sort(sim, descending=True, dim=1)
        return sort_idx

    def forward(self, x):
        x_in = x.permute(0, 2, 3, 1).contiguous()
        z = self.in_proj_skip(torch.mean(x_in, dim=[1, 2]).unsqueeze(1)).unsqueeze(1)
        x_feat = self.in_proj(x_in).permute(0, 3, 1, 2).contiguous()
        x_conv = self.act(self.conv2d(x_feat))
        
        # Dynamic Scan Index Generation
        with torch.no_grad(): 
            sort_idx = self.get_dynamic_index(x_conv)
            
        class ProxyScan:
            @staticmethod
            def apply(z): return DynamicCenterScan.apply(z, sort_idx)
            
        class ProxyMerge:
            @staticmethod
            def apply(z): return DynamicCenterMerge.apply(z, sort_idx)
            
        # Global SSM Modeling
        out_mamba = self.mamba(x_conv, CrossScan=ProxyScan, CrossMerge=ProxyMerge)
            
        # Feature Fusion (Global SSM + LGP)
        out = out_mamba + self.local_path(x_conv)
            
        out = out.permute(0, 2, 3, 1) * F.softmax(z, dim=1)
        return self.out_proj(out).permute(0, 3, 1, 2).contiguous()

# ==========================================
# Channel Layer Normalization
# ==========================================
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# ==========================================
# SD-Mamba Basic Block
# ==========================================
class SD_Block(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.norm = ChanLayerNorm(in_features) 
        self.spa = MambaSpatial(in_features)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.spa(x)
        return x + res

# ==========================================
# Spatial-Dynamic Mamba (SD-Mamba)
# ==========================================
class SD_Mamba(nn.Module):
    def __init__(self, in_features=200, num_classes=16, patch_size=11, hidden_dim=64):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_features, hidden_dim, 1), 
            nn.BatchNorm2d(hidden_dim), 
            nn.SiLU()
        )
            
        # Spatial-Dynamic Block
        self.block = SD_Block(hidden_dim)
        
        # Multi-Scale Feature Enhancer
        self.feature_enhancer = MSFE(hidden_dim)
            
        # Classifier Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.feature_enhancer(x) 
        return self.head(x)