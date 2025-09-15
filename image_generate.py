import torch
import torchvision
import matplotlib.pyplot as plt
import os
from vae_swin_unet import SwinUNetVAE


class ImageGenerator:
    """SwinUNetVAE 모델을 사용한 이미지 생성 클래스"""
    
    def __init__(self, model_path='vae_unet_final_model.pth', img_size=224, latent_dim=512, device='cuda'):
        self.model_path = model_path
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드 (현재 사용 중인 SwinUNetVAE)
        self.model = SwinUNetVAE(
            img_size=img_size,
            latent_dim=latent_dim
        )
        
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
            missing = set(model_state.keys()) - set(filtered.keys())
            self.model.load_state_dict(filtered, strict=False)
            print(f"체크포인트 부분 로드: {len(filtered)}/{len(model_state)} 매칭 | 파일: {model_path}")
            if missing:
                print(f"참고: 불일치로 미로딩된 키 수: {len(missing)}")
        else:
            print(f"경고: 모델 파일을 찾을 수 없습니다: {model_path}")
            
        self.model.to(self.device)
        self.model.eval()
    
    def generate_images(self, num_images=64, nrow=8, save_path='vae_unet_generated_images.png', show=True):
        """랜덤 이미지 생성"""
        with torch.no_grad():
            # 랜덤 latent vector 생성
            z = torch.randn(num_images, self.latent_dim, device=self.device)
            
            # 노이즈 이미지를 인코더에 통과시켜 스킵 피처 생성
            noise_imgs = torch.randn(num_images, 3, self.img_size, self.img_size, device=self.device)
            features, _, _ = self.model.encode(noise_imgs)

            # 생성
            generated_images = self.model.decode(z, features)
            if generated_images.shape[-2:] != (self.img_size, self.img_size):
                generated_images = torch.nn.functional.interpolate(
                    generated_images, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False
                )
            # 값 범위 안정화
            generated_images = torch.sigmoid(generated_images)
            
            # 그리드로 배열
            grid_img = torchvision.utils.make_grid(
                generated_images.cpu(), 
                nrow=nrow, 
                padding=2, 
                normalize=False
            )
            grid_img = grid_img.permute(1, 2, 0).numpy()
            
            # 시각화 및 저장
            plt.figure(figsize=(12, 12))
            plt.imshow(grid_img)
            plt.axis('off')
            plt.title(f'Generated Images ({num_images} samples)')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"생성된 이미지 저장: {save_path}")
            
            if show:
                plt.show()
            
            plt.close()
            
            return generated_images
    
    def reconstruct_images(self, data_loader, num_samples=4, save_path='vae_unet_reconstruction.png', show=True):
        """실제 이미지 재구성"""
        with torch.no_grad():
            # 데이터에서 샘플 가져오기
            data_iter = iter(data_loader)
            real_images = next(data_iter)[:num_samples].to(self.device)
            
            # 재구성 (encode -> reparameterize -> decode)
            features, mu, logvar = self.model.encode(real_images)
            z = self.model.reparameterize(mu, logvar)
            reconstructed = self.model.decode(z, features)
            if reconstructed.shape[-2:] != (self.img_size, self.img_size):
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False
                )
            
            # 원본과 재구성 이미지 비교
            comparison = torch.cat([real_images.cpu(), reconstructed.cpu()], dim=0)
            
            grid_img = torchvision.utils.make_grid(
                comparison, 
                nrow=num_samples, 
                padding=2, 
                normalize=True
            )
            grid_img = grid_img.permute(1, 2, 0).numpy()
            
            plt.figure(figsize=(15, 6))
            plt.imshow(grid_img)
            plt.axis('off')
            plt.title('Upper: Original Images, Lower: Reconstructed Images')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"재구성 이미지 저장: {save_path}")
            
            if show:
                plt.show()
            
            plt.close()
            
            return real_images, reconstructed
    def reconstruct_single_image(self, data_loader, num_samples=2 , save_path='vae_reconstruction.png', show=True):
        """실제 이미지 재구성"""
        with torch.no_grad():
            # 데이터에서 샘플 가져오기
            data_iter = iter(data_loader)
            real_images = next(data_iter)[:num_samples].to(self.device)
            
            # 재구성 (encode -> reparameterize -> decode)
            features, mu, logvar = self.model.encode(real_images)
            z = self.model.reparameterize(mu, logvar)
            reconstructed = self.model.decode(z, features)
            if reconstructed.shape[-2:] != (self.img_size, self.img_size):
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False
                )
            
            # 원본과 재구성 이미지 비교
            comparison = torch.cat([real_images.cpu(), reconstructed.cpu()], dim=0)
            
            grid_img = torchvision.utils.make_grid(
                comparison, 
                nrow=num_samples, 
                padding=2, 
                normalize=True
            )
            grid_img = grid_img.permute(1, 2, 0).numpy()
            
            plt.figure(figsize=(15, 6))
            plt.imshow(grid_img)
            plt.axis('off')
            plt.title('LEFT: Original Images, RIGHT: Reconstructed Images')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"재구성 이미지 저장: {save_path}")
            
            if show:
                plt.show()
            
            plt.close()
            
            return real_images, reconstructed

