# train.py
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn import L1Loss
import os

# 假设您的模型和工具函数在这些文件中
from models import Generator, Discriminator 
from utils import compute_gradient_penalty

# --- 1. 超参数和设置 ---
# 训练参数
batch_size = 8
num_epochs = 1000
lr_G = 0.0001
lr_D = 0.0001
ncritic = 5  # 判别器迭代次数

# 模型维度
latent_dim = 128
style_dim = 64
node_feature_dim = 10 # X_template的特征维度

# 损失权重
lambda_gp = 10
lambda_recon = 100
w_pos = 10.0
w_feat = 1.0

# 其他设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)


# --- 2. 数据加载 ---
# 假设您有一个包含 torch_geometric.data.Data 对象的列表
# my_dataset = [Data(...), Data(...), ...]
# 在这里，我们用一个虚拟数据集代替
# 您需要替换成真实的数据集加载逻辑
from torch_geometric.data import Data
# 示例: 创建一个虚拟数据集
# 您需要加载您自己的 "healthy_brain_graphs"
# my_dataset = torch.load("path/to/your/dataset.pt") 
N_nodes = 200 # 假设每个图有200个节点
M_edges = 800 # 假设每个图有800条边
num_graphs = 64
my_dataset = [Data(x=torch.randn(N_nodes, node_feature_dim), 
                   edge_index=torch.randint(0, N_nodes, (2, M_edges)),
                   pos=torch.randn(N_nodes, 3)) for _ in range(num_graphs)]

data_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

# 加载模板图 (Template Graph)
# 模板图应该只有一个，并且在所有生成过程中共享
# 模板图的节点数 N 和边数 M 应该与您的数据集匹配
X_template = torch.randn(N_nodes, node_feature_dim).to(device)
edge_index_template = torch.randint(0, N_nodes, (2, M_edges)).to(device)
# 在您的真实案例中，请加载固定的、有意义的模板
# X_template = torch.load("path/to/X_template.pt").to(device)
# edge_index_template = torch.load("path/to/edge_index_template.pt").to(device)


# --- 3. 模型实例化和优化器 ---
generator = Generator(latent_dim=latent_dim, style_dim=style_dim, node_feature_dim=node_feature_dim).to(device)
discriminator = Discriminator(node_feature_dim=node_feature_dim).to(device) # 根据您的D架构传入参数

# Adam优化器，betas适用于GAN训练
optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.9))

# 定义重建损失函数
l1_loss = L1Loss()


# --- 4. 训练循环 ---
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, real_graphs in enumerate(data_loader):
        real_graphs = real_graphs.to(device)
        
        # ============================================
        # (1) 训练判别器 (Train Discriminator)
        # ============================================
        for _ in range(ncritic):
            optimizer_D.zero_grad()

            # --- 计算真实样本的损失 ---
            # D的输入是一个图结构，PyG的DataLoader会自动批处理
            real_outputs = discriminator(real_graphs)
            loss_real = -torch.mean(real_outputs) # WGAN损失，希望D对真实样本打高分

            # --- 计算生成样本的损失 ---
            z = torch.randn(real_graphs.num_graphs, latent_dim, device=device)
            # 使用 .detach() 防止在训练D时更新G的梯度
            fake_graphs = generator(z, X_template, edge_index_template).detach()
            
            fake_outputs = discriminator(fake_graphs)
            loss_fake = torch.mean(fake_outputs) # WGAN损失，希望D对虚假样本打低分

            # --- 计算梯度惩罚 ---
            # PyG的Batch对象将所有图的x拼接在一起，我们需要确保对齐
            gradient_penalty = compute_gradient_penalty(
                discriminator, 
                real_graphs.x, 
                fake_graphs.x, 
                real_graphs.edge_index, # 使用真实图的结构
                getattr(real_graphs, 'edge_attr', None),
                device, 
                lambda_gp
            )

            # --- 计算总损失并更新 ---
            loss_D = loss_real + loss_fake + gradient_penalty
            loss_D.backward()
            optimizer_D.step()

        # ============================================
        # (2) 训练生成器 (Train Generator)
        # ============================================
        optimizer_G.zero_grad()
        
        # --- 生成新的假样本 ---
        z = torch.randn(real_graphs.num_graphs, latent_dim, device=device)
        fake_graphs_for_G = generator(z, X_template, edge_index_template)

        # --- 计算对抗损失 ---
        # 我们希望生成器能骗过判别器，即让D打出高分
        outputs_for_G = discriminator(fake_graphs_for_G)
        loss_adv = -torch.mean(outputs_for_G)
        
        # --- 计算重建损失 ---
        # 从数据集中采样另一批真实样本作为重建目标
        # 注意: 理论上最好用同一个batch，但为了代码简洁，重采样也可接受
        # 更好的做法是在循环开始时就采样好两批数据
        target_graphs = next(iter(data_loader)).to(device)

        # 提取坐标和特征进行比较
        # 假设生成器输出的x前3维是坐标Δp，后10维是特征Δf
        # 并且您的Generator实现中已经将 Δ 加到了template上
        fake_pos = fake_graphs_for_G.pos # 假设 .pos 属性存在
        fake_feat = fake_graphs_for_G.x   # 假设 .x 包含了节点特征

        target_pos = target_graphs.pos
        target_feat = target_graphs.x

        loss_rec_pos = l1_loss(fake_pos, target_pos)
        loss_rec_feat = l1_loss(fake_feat, target_feat)
        loss_rec = w_pos * loss_rec_pos + w_feat * loss_rec_feat
        
        # --- 计算总损失并更新 ---
        loss_G = loss_adv + lambda_recon * loss_rec
        loss_G.backward()
        optimizer_G.step()

    # --- 5. 打印日志和保存模型 ---
    if (epoch + 1) % 10 == 0:
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Loss_D: {loss_D.item():.4f} "
            f"Loss_G: {loss_G.item():.4f} "
            f"  (Adv: {loss_adv.item():.4f}, Recon: {loss_rec.item():.4f})"
        )

    if (epoch + 1) % 100 == 0:
        torch.save(generator.state_dict(), os.path.join(checkpoints_dir, f"generator_epoch_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(checkpoints_dir, f"discriminator_epoch_{epoch+1}.pth"))

print("Training finished.")