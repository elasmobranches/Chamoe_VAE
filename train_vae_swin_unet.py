#!/usr/bin/env python3
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
import sys
from vae_swin_unet import SwinUNetVAE, ImageDataset, get_transforms, train_vae
from image_generate import ImageGenerator
from val_visualizer import ValidationVisualizer


def main():
    # 설정
    img_size = 224
    batch_size = 8
    num_epochs = 1000
    learning_rate = 1e-4
    latent_dim = 512
    
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터 경로
    train_dir = 'v61/train'
    val_dir = 'v61/val'
    
    # 데이터셋 생성
    transform = get_transforms(img_size=img_size)
    
    train_dataset = ImageDataset(train_dir, transform=transform)
    val_dataset = ImageDataset(val_dir, transform=transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    
    # 모델 생성 (SwinUNetVAE의 현재 시그니처에 맞게 수정)
    model = SwinUNetVAE(
        img_size=img_size,
        latent_dim=latent_dim
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Validation 시각화 준비
    visualizer = ValidationVisualizer(save_dir='./validation_plots')
    
    # 모델 학습
    print("Starting training...")
    trained_model = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=learning_rate,
        device=device,
        patience=20,  # 20 epoch 동안 개선 없으면 중단
        visualizer=visualizer,  # visualizer 전달
        beta_start=0.0, beta_end=1.0, beta_warmup_epochs=20,
        use_amp=True, max_grad_norm=1.0,
        save_dir='./saved_models'
    )
    
    # 최종 모델 저장
    # 최종 모델 저장
    # best 체크포인트는 train_vae 내부에서 저장됨
    torch.save(trained_model.state_dict(), 'vae_unet_final_model.pth')
    print("Training completed! Final model saved as 'vae_unet_final_model.pth' (best checkpoint stored under saved_models)")
    
    # Validation 결과 시각화
    print("\n=== Validation 결과 시각화 ===")
    visualizer.print_summary()
    visualizer.save_all_plots('vae_unet_training')
    
    # 이미지 생성 및 재구성 테스트
    print("\n=== 이미지 생성 테스트 ===")
    generator = ImageGenerator(
        model_path='vae_unet_final_model.pth',
        img_size=img_size,
        latent_dim=latent_dim,
        device=device
    )
    
    # 1. 랜덤 이미지 생성
    print("1. 랜덤 이미지 생성 중...")
    generator.generate_images(
        num_images=4, 
        nrow=2, 
        save_path='vae_unet_generated_images.png',
        show=False  # 서버 환경에서는 show=False
    )
    
    # 2. 실제 이미지 재구성
    print("2. 실제 이미지 재구성 중...")
    generator.reconstruct_images(
        data_loader=val_loader,
        num_samples=4,
        save_path='vae_unet_reconstruction.png',
        show=False  # 서버 환경에서는 show=False
    )
    
    print("\n이미지 생성 완료!")
    print("- 생성된 이미지: vae_unet_generated_images.png")
    print("- 재구성 이미지: vae_unet_reconstruction.png")


if __name__ == "__main__":
    main()
