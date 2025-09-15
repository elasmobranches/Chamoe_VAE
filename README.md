# SwinUNet-VAE

Swin Transformer encoder + U-Net decoder로 구성된 간결한 Variational Autoencoder 구현체입니다. 이미지 재구성과 생성에 초점을 맞추어, Swin의 윈도우 어텐션과 U-Net의 스킵 커넥션을 결합했습니다.

## 주요 특징
- Swin-S 인코더: 윈도우 어텐션, 쉬프티드 윈도우, 드롭패스 스케줄
- U-Net 디코더: 가벼운 채널 구성과 스킵 커넥션
- 학습 유틸: KL warmup(beta schedule), AMP, gradient clipping, 체크포인트 저장
- 시각화: 학습/검증 손실 비교, 최소 검증 손실 지점 어노테이션

## 파일 구성
- `vae_swin_unet.py`: 모델(SwinSmallEncoder + UNetDecoder), 데이터셋/변환, 학습 루프 포함
- `val_visualizer.py`: 학습/검증 손실 시각화 유틸
- `image_generate.py`: 학습된 모델을 이용한 이미지 생성/재구성 헬퍼
- `train_vae_swin_unet.py`: 학습 엔트리포인트
- `generate_single_image.py`: 단일 이미지를 재구성해 저장하는 스크립트

이미지 산출물(.png 등)과 모델 가중치(.pth 등)는 버전 관리에서 제외합니다(`.gitignore`).

## 설치
```bash
pip install -r requirements.txt
```

## 데이터셋 준비
- 단순 폴더 구조(클래스 폴더 없이 파일 나열)를 가정
- `vXX/train`, `vXX/val` 경로 하위에 이미지(.jpg/.png/...) 

예시:
```text
v61/
  ├─ train/
  │   ├─ 000001_img.jpg
  │   └─ ...
  └─ val/
      ├─ 000101_img.jpg
      └─ ...
```

## 학습
`train_vae_swin_unet.py`의 경로/하이퍼파라미터
```bash
python train_vae_swin_unet.py
```
- 최고 성능 모델은 `saved_models/vae_swin_unet_best_model.pth`로 저장됩니다.
- 마지막 모델 상태는 `vae_unet_final_model.pth`로 저장됩니다.

## 추론/시각화
단일 이미지 재구성 샘플:
```bash
python generate_single_image.py \
  --data_dir v61/val \
  --model_path vae_unet_final_model.pth \
  --img_size 224 \
  --latent_dim 512 \
  --save_path sample.png
```

학습 중/후 검증 결과 시각화:
- `val_visualizer.py`를 통해 학습/검증 손실 비교 그래프와 러닝 커브를 저장
- `train_vae_swin_unet.py`에서 `ValidationVisualizer`가 자동으로 결과를 저장(`validation_plots/`).
