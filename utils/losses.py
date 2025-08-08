import torch
import torch.nn.functional as F

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    计算 WGAN-GP 的梯度惩罚项。

    参数:
        discriminator (nn.Module): 判别器模型。
        real_samples (Tensor/Graph): 一批真实样本。
        fake_samples (Tensor/Graph): 一批生成样本。
        device (str/torch.device): 计算所在的设备。

    返回:
        Tensor: 标量梯度惩罚损失。
    
    使用场景:
        判别器每步训练时添加梯度惩罚项，防止判别器梯度爆炸或消失。
    """
    alpha = torch.rand(real_samples.shape[0], 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp

def calculate_reconstruction_loss(generated_X, real_X, mode='l1'):
    """
    计算生成器的重建损失，衡量生成的节点特征和真实健康样本的接近程度。

    参数:
        generated_X (Tensor): 生成节点特征，形状 (N, 10)。
        real_X (Tensor): 真实节点特征，形状 (N, 10)。
        mode (str): 损失类型，'l1' 或 'l2'，默认 'l1'。

    返回:
        Tensor: 标量重建损失。
    
    使用场景:
        生成器训练时联合对抗损失，优化生成样本的真实性。
    """
    if mode == 'l1':
        recon_loss = F.l1_loss(generated_X, real_X)
    else:
        recon_loss = F.mse_loss(generated_X, real_X)
    return recon_loss