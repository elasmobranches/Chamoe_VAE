import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

class ValidationVisualizer:
    """VAE 학습 과정의 validation 결과를 시각화하는 클래스"""
    
    def __init__(self, save_dir='./plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 학습 기록을 저장할 리스트들
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.epochs = []
        
    def update(self, epoch, train_loss, val_loss, recon_loss=None, kl_loss=None):
        """각 epoch의 결과를 업데이트"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if recon_loss is not None:
            self.recon_losses.append(recon_loss)
        if kl_loss is not None:
            self.kl_losses.append(kl_loss)
    
    # def plot_losses(self, save_path=None, show=False):
    #     """Loss 곡선 시각화"""
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
    #     # 1. Train vs Val Loss
    #     axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss', color='blue', alpha=0.7)
    #     axes[0, 0].plot(self.epochs, self.val_losses, label='Val Loss', color='red', alpha=0.7)
    #     axes[0, 0].set_xlabel('Epoch')
    #     axes[0, 0].set_ylabel('Loss')
    #     axes[0, 0].set_title('Training vs Validation Loss')
    #     axes[0, 0].legend()
    #     axes[0, 0].grid(True, alpha=0.3)
        
    #     # 2. Validation Loss만 따로
    #     axes[0, 1].plot(self.epochs, self.val_losses, label='Val Loss', color='red', linewidth=2)
    #     axes[0, 1].set_xlabel('Epoch')
    #     axes[0, 1].set_ylabel('Validation Loss')
    #     axes[0, 1].set_title('Validation Loss Over Time')
    #     axes[0, 1].legend()
    #     axes[0, 1].grid(True, alpha=0.3)
        
    #     # 3. Reconstruction Loss (있는 경우)
    #     if self.recon_losses:
    #         axes[1, 0].plot(self.epochs, self.recon_losses, label='Reconstruction Loss', color='green', alpha=0.7)
    #         axes[1, 0].set_xlabel('Epoch')
    #         axes[1, 0].set_ylabel('Reconstruction Loss')
    #         axes[1, 0].set_title('Reconstruction Loss Over Time')
    #         axes[1, 0].legend()
    #         axes[1, 0].grid(True, alpha=0.3)
    #     else:
    #         axes[1, 0].text(0.5, 0.5, 'No Reconstruction Loss Data', 
    #                        horizontalalignment='center', verticalalignment='center', 
    #                        transform=axes[1, 0].transAxes)
        
    #     # 4. KL Loss (있는 경우)
    #     if self.kl_losses:
    #         axes[1, 1].plot(self.epochs, self.kl_losses, label='KL Loss', color='orange', alpha=0.7)
    #         axes[1, 1].set_xlabel('Epoch')
    #         axes[1, 1].set_ylabel('KL Divergence Loss')
    #         axes[1, 1].set_title('KL Divergence Loss Over Time')
    #         axes[1, 1].legend()
    #         axes[1, 1].grid(True, alpha=0.3)
    #     else:
    #         axes[1, 1].text(0.5, 0.5, 'No KL Loss Data', 
    #                        horizontalalignment='center', verticalalignment='center', 
    #                        transform=axes[1, 1].transAxes)
        
    #     plt.tight_layout()
        
    #     if save_path:
    #         plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #         print(f"Loss 곡선 저장: {save_path}")
        
    #     if show:
    #         plt.show()
        
    #     plt.close()
    
    def plot_loss_comparison(self, save_path=None, show=False):
        """Train vs Val Loss 비교 (단일 그래프)"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.epochs, self.train_losses, label='Training Loss', color='blue', alpha=0.8, linewidth=2)
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', color='red', alpha=0.8, linewidth=2)
        
        # 최소 validation loss 지점 표시
        if self.val_losses:
            min_val_idx = np.argmin(self.val_losses)
            min_val_loss = self.val_losses[min_val_idx]
            min_val_epoch = self.epochs[min_val_idx]
            
            plt.scatter([min_val_epoch], [min_val_loss], color='red', s=100, zorder=5)
            plt.annotate(f'Best Val: {min_val_loss:.4f}\nEpoch: {min_val_epoch}', 
                        xy=(min_val_epoch, min_val_loss), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss 비교 그래프 저장: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_learning_curve(self, save_path=None, show=False):
        """학습 곡선 (로그 스케일 포함)"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. 일반 스케일
        axes[0].plot(self.epochs, self.train_losses, label='Training Loss', alpha=0.7)
        axes[0].plot(self.epochs, self.val_losses, label='Validation Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Learning Curve (Linear Scale)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 로그 스케일
        axes[1].semilogy(self.epochs, self.train_losses, label='Training Loss', alpha=0.7)
        axes[1].semilogy(self.epochs, self.val_losses, label='Validation Loss', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (log scale)')
        axes[1].set_title('Learning Curve (Log Scale)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"학습 곡선 저장: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def print_summary(self):
        """학습 결과 요약 출력"""
        if not self.val_losses:
            print("학습 데이터가 없습니다.")
            return
        
        min_val_idx = np.argmin(self.val_losses)
        min_val_loss = self.val_losses[min_val_idx]
        min_val_epoch = self.epochs[min_val_idx]
        
        final_train_loss = self.train_losses[-1] if self.train_losses else "N/A"
        final_val_loss = self.val_losses[-1] if self.val_losses else "N/A"
        
        print("\n" + "="*50)
        print("학습 결과 요약")
        print("="*50)
        print(f"총 학습 Epoch: {len(self.epochs)}")
        print(f"최종 Training Loss: {final_train_loss:.4f}" if isinstance(final_train_loss, float) else f"최종 Training Loss: {final_train_loss}")
        print(f"최종 Validation Loss: {final_val_loss:.4f}" if isinstance(final_val_loss, float) else f"최종 Validation Loss: {final_val_loss}")
        print(f"최고 성능 Validation Loss: {min_val_loss:.4f} (Epoch {min_val_epoch})")
        
        if len(self.val_losses) > 1:
            improvement = self.val_losses[0] - min_val_loss
            improvement_pct = (improvement / self.val_losses[0]) * 100
            print(f"초기 대비 개선: {improvement:.4f} ({improvement_pct:.1f}%)")
        
        print("="*50)
    
    def save_all_plots(self, prefix='val_viz'):
        """모든 시각화를 한번에 저장"""
        #self.plot_losses(save_path=os.path.join(self.save_dir, f'{prefix}_losses.png'))
        self.plot_loss_comparison(save_path=os.path.join(self.save_dir, f'{prefix}_comparison.png'))
        self.plot_learning_curve(save_path=os.path.join(self.save_dir, f'{prefix}_learning_curve.png'))
        print(f"\n모든 시각화 저장 완료: {self.save_dir}")


if __name__ == "__main__":
    # 테스트용 코드
    visualizer = ValidationVisualizer()
    
    # 더미 데이터로 테스트
    for epoch in range(50):
        train_loss = 1.0 - epoch * 0.01 + np.random.normal(0, 0.02)
        val_loss = 1.0 - epoch * 0.008 + np.random.normal(0, 0.03)
        recon_loss = 0.8 - epoch * 0.008 + np.random.normal(0, 0.02)
        kl_loss = 0.2 - epoch * 0.002 + np.random.normal(0, 0.01)
        
        visualizer.update(epoch, train_loss, val_loss, recon_loss, kl_loss)
    
    visualizer.print_summary()
    visualizer.save_all_plots('test')
