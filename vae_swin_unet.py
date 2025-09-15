import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os
import math
import matplotlib.pyplot as plt

# ----------------------------
# Window util
# ----------------------------
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = windows.shape[0] // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# ----------------------------
# Swin components
# ----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)                               # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)               # (B, L, C) L=(H/4)*(W/4)
        x = self.norm(x)
        return x, (H, W)

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)                          # (B, (H/2)*(W/2), 2C)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias
        Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads))
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, (window_size, window_size), num_heads, qkv_bias, attn_drop, drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # shifted-window mask 미리 계산
        if shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # window attention
        x_windows = window_partition(shifted_x, self.window_size)                 # (nW*B, w, w, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)    # (nW*B, w*w, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path1(x)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        # drop_path는 float 또는 리스트를 허용
        if isinstance(drop_path, (list, tuple)):
            dp_list = list(drop_path)
            assert len(dp_list) == depth
        else:
            dp_list = [float(drop_path)] * depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, drop_path=dp_list[i], norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

# ----------------------------
# Swin-Small Encoder (features + VAE heads)
# ----------------------------
class SwinSmallEncoder(nn.Module):
    """
    Swin-Small 설정: embed_dim=96, depths=[2,2,18,2], num_heads=[3,6,12,24]
    outputs:
      - features: [C1(96,H/4), C2(192,H/8), C3(384,H/16), C4(768,H/32)]
      - mu, logvar from GAP(C4)
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2,2,18,2], num_heads=[3,6,12,24],
                 window_size=7, mlp_ratio=4., latent_dim=512, drop_path_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm)

        # stage resolutions
        H0 = img_size // patch_size                         # 56
        self.stages = nn.ModuleList()
        self.stage_res = []
        dims = [int(embed_dim * 2**i) for i in range(4)]

        in_res = (H0, H0)
        # drop path 스케줄 준비 (전체 depth에 대해 선형 증가)
        total_depth = sum(depths)
        if total_depth > 1:
            dp_rates = [drop_path_rate * (idx / (total_depth - 1)) for idx in range(total_depth)]
        else:
            dp_rates = [drop_path_rate]
        dp_ptr = 0

        for i in range(4):
            layer = BasicLayer(
                dim=dims[i], input_resolution=in_res, depth=depths[i],
                num_heads=num_heads[i], window_size=window_size,
                mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm,
                drop_path=dp_rates[dp_ptr: dp_ptr + depths[i]],
                downsample=PatchMerging if (i < 3) else None
            )
            self.stages.append(layer)
            self.stage_res.append(in_res)
            if i < 3:
                in_res = (in_res[0]//2, in_res[1]//2)
            dp_ptr += depths[i]

        # VAE heads from C4(=dims[3]=768)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj_to_mu_logvar = nn.Linear(dims[3], latent_dim * 2)

        # remember for decoder
        self.out_channels = dims
    def forward(self, x):
        # patch embed
        x, (H, W) = self.patch_embed(x)         # (B, initial_L, embed_dim)
        features = []
        cur = x

        for i, stage in enumerate(self.stages):
            # stage forward: cur -> (B, L_i, C_i)
            cur = stage(cur)
            B, L, C = cur.shape

            # 안전하게 H_i, W_i 계산 (PatchMerging 때문에 하드코딩 금지)
            h = int(math.sqrt(L))
            w = h
            assert h * w == L, f"Stage {i}: sequence length {L} is not a perfect square"

            # (B, L, C) -> (B, C, h, w)
            feat = cur.transpose(1, 2).reshape(B, C, h, w)
            features.append(feat)

            # 다음 스테이지 입력을 위해 다운샘플을 수동 적용
            if stage.downsample is not None:
                cur = stage.downsample(cur)
        # C4 for VAE heads (마지막 features[-1])
        C4 = features[-1]                        # (B, C4, h4, w4)  -> 보통 7x7
        pooled = self.global_pool(C4).flatten(1) # (B, C4)
        mu_logvar = self.proj_to_mu_logvar(pooled)  # (B, 2*latent)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        return features, mu, logvar
    
            
# U-Net Decoder
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.pre_up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.pre_skip = nn.Sequential(
            nn.Conv2d(skip_ch, out_ch, kernel_size=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        # x를 skip 크기에 맞게 업샘플링
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = self.pre_up(x)
        skip = self.pre_skip(skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, latent_dim=512, bottleneck_hw=7, out_ch=3):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.latent_dim = latent_dim
        self.bottleneck_hw = bottleneck_hw
        self.out_ch = out_ch

        # z를 더 가벼운 채널(decoder_channels[0])로 바로 투영하여 메모리 절감
        self.z_to_c4 = nn.Linear(latent_dim, decoder_channels[0]*bottleneck_hw*bottleneck_hw)

        # C4 refine (동일 채널 유지)
        self.c4_refine = ConvBlock(decoder_channels[0], decoder_channels[0])

        # UpBlocks
        self.up3 = UpBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1])
        self.up2 = UpBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2])
        self.up1 = UpBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])

        # 최종 conv
        self.final_conv = nn.Conv2d(decoder_channels[3], out_ch, kernel_size=1)
    def forward(self, z, features):
        C1, C2, C3, C4 = features
        B = z.size(0)

        # z -> C4 feature
        x = self.z_to_c4(z)
        x = x.view(B, self.decoder_channels[0], self.bottleneck_hw, self.bottleneck_hw)

        x = self.c4_refine(x)

        x = self.up3(x, C3)  # 7->14

        x = self.up2(x, C2)  # 14->28

        x = self.up1(x, C1)  # 28->56

        x = self.final_conv(x)

        
        return x
# ----------------------------
# VAE (Swin-S + U-Net)
# ----------------------------
class SwinUNetVAE(nn.Module):
    def __init__(self, img_size=224, latent_dim=512, drop_path_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.encoder = SwinSmallEncoder(img_size=img_size, latent_dim=latent_dim,
                                        embed_dim=96, depths=[2,2,18,2], num_heads=[3,6,12,24],
                                        window_size=7, mlp_ratio=4., drop_path_rate=drop_path_rate)
        # bottleneck spatial size = 224/(4*2*2*2)=7
        self.decoder = UNetDecoder(encoder_channels=[96,192,384,768],
                                   decoder_channels=[512,256,128,64],
                                   latent_dim=latent_dim, bottleneck_hw=7, out_ch=3)

    def encode(self, x):
        features, mu, logvar = self.encoder(x)
        return features, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, features):
        return self.decoder(z, features)

    def forward(self, x):
        features, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, features)
        if x_recon.shape[-2:] != x.shape[-2:]:
            x_recon = F.interpolate(x_recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x_recon, mu, logvar

    def get_loss(self, x, x_recon, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

# ----------------------------
# Dataset & transforms
# ----------------------------
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image

def get_transforms(img_size=224):
    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

# ----------------------------
# Train utils (원본 유지)
# ----------------------------
def train_vae(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda', patience=10, visualizer=None,
              beta_start=0.0, beta_end=1.0, beta_warmup_epochs=20, use_amp=True, max_grad_norm=1.0,
              save_dir='./saved_models'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_checkpoint = None

    def get_beta(epoch_idx: int) -> float:
        if beta_warmup_epochs <= 0:
            return beta_end
        t = min(max(epoch_idx, 0), beta_warmup_epochs)
        return beta_start + (beta_end - beta_start) * (t / beta_warmup_epochs)

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0
        num_train_batches = 0
        beta = get_beta(epoch)

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                recon_batch, mu, logvar = model(data)
                total_loss, recon_loss, kl_loss = model.get_loss(data, recon_batch, mu, logvar, beta=beta)
                # 평균 단위로 보고하려면 배치 크기/픽셀 수로 정규화 옵션 고려 가능
            scaler.scale(total_loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += total_loss.detach().item()
            num_train_batches += 1

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Beta: {beta:.3f}, '
                      f'Loss: {total_loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}')

        # validation
        model.eval()
        val_loss_sum = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                recon_batch, mu, logvar = model(data)
                total_loss, _, _ = model.get_loss(data, recon_batch, mu, logvar, beta=beta)
                val_loss_sum += total_loss.item()
                num_val_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(num_train_batches, 1)
        avg_val_loss = val_loss_sum / max(num_val_batches, 1)
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} (beta={beta:.3f})')

        if visualizer is not None:
            visualizer.update(epoch, avg_train_loss, avg_val_loss)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
            }
            print(f'  -> 새로운 최고 성능! Val Loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            print(f'  -> Val Loss 개선 없음 ({patience_counter}/{patience})')

        if patience_counter >= patience:
            print(f'\n{patience} epoch 동안 개선이 없어 학습을 중단합니다. 최고 성능: Val Loss {best_val_loss:.4f}')
            break

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'vae_model_epoch_{epoch+1}.pth')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}, save_path)

    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint['model'])
        best_model_path = os.path.join(save_dir, 'vae_swin_unet_best_model.pth')
        torch.save(best_checkpoint, best_model_path)
        print(f'최고 성능 모델 저장: {best_model_path}')
    return model

# ----------------------------
# ImageGenerator (간단 통합 버전)
# ----------------------------
# class ImageGenerator:
#     def __init__(self, model_path=None, img_size=224, latent_dim=512, device='cuda'):
#         self.img_size = img_size
#         self.latent_dim = latent_dim
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.model = SwinUNetVAE(img_size=img_size, latent_dim=latent_dim)
#         if model_path and os.path.exists(model_path):
#             state = torch.load(model_path, map_location=self.device)
#             try:
#                 self.model.load_state_dict(state, strict=True)
#                 print(f"모델 로드 완료: {model_path}")
#             except RuntimeError as e:
#                 print(f"경고: strict=True 로드 실패, strict=False로 재시도합니다.\n{e}")
#                 missing_loaded = self.model.load_state_dict(state, strict=False)
#                 print(f"모델 부분 로드 완료: {model_path}\n미로딩 키(참고): {missing_loaded.missing_keys}")
#         elif model_path:
#             print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
#         self.model.to(self.device)
#         self.model.eval()

#     def generate_images(self, num_images=4, nrow=2, save_path='vae_unet_generated_images.png', show=False):
#         with torch.no_grad():
#             z = torch.randn(num_images, self.latent_dim, device=self.device)
#             noise_imgs = torch.randn(num_images, 3, self.img_size, self.img_size, device=self.device)
#             features, _, _ = self.model.encode(noise_imgs)
#             generated = self.model.decode(z, features)
#             if generated.shape[-2:] != (self.img_size, self.img_size):
#                 generated = F.interpolate(generated, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
#             generated = torch.sigmoid(generated)
#             grid = make_grid(generated.cpu(), nrow=nrow, padding=2, normalize=False)
#             img = grid.permute(1, 2, 0).numpy()
#             plt.figure(figsize=(8, 8))
#             plt.imshow(img); plt.axis('off'); plt.tight_layout()
#             if save_path:
#                 plt.savefig(save_path, dpi=150, bbox_inches='tight')
#                 print(f"생성된 이미지 저장: {save_path}")
#             if show:
#                 plt.show()
#             plt.close()
#             return generated

# ----------------------------
# Main / FLOPs
# ----------------------------
if __name__ == "__main__":
    import importlib
    try:
        fvcore_nn = importlib.import_module("fvcore.nn")
        FlopCountAnalysis = fvcore_nn.FlopCountAnalysis
    except Exception:
        FlopCountAnalysis = None
    model = SwinUNetVAE(img_size=224, latent_dim=512)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델의 총 파라미터 수: {total_params}")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    if FlopCountAnalysis is not None:
        flops = FlopCountAnalysis(model, dummy_input)
        print(flops.total())  # FLOPs 수
    else:
        print("fvcore가 없어 FLOPs 계산을 건너뜁니다.")
