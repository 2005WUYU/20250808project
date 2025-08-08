import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history_dict, x_axis_data, save_path):
    """
    绘制训练过程中多个标量指标的变化曲线，并保存为图片。

    参数:
        history_dict (dict): 指标名称到数值列表的映射。
        x_axis_data (list/np.ndarray): 横坐标（如 epoch 数或 step 数）。
        save_path (str): 图像保存路径。

    返回:
        无（直接保存图片）。
    
    使用场景:
        GAN训练收敛动态监控，各种损失/分数可视化。
    """
    num_plots = len(history_dict)
    n_cols = 2
    n_rows = (num_plots + 1) // 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axs = axs.flatten() if num_plots > 1 else [axs]

    for i, (key, values) in enumerate(history_dict.items()):
        axs[i].plot(x_axis_data, values)
        axs[i].set_title(key, fontsize=14)
        axs[i].set_xlabel("Step/Epoch", fontsize=12)
        axs[i].set_ylabel("Value", fontsize=12)
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_distributions_comparison(dist_real, dist_fake, statistic_name, save_path, bins=60):
    """
    对比绘制真实样本与生成样本在某一统计量上的分布（直方图+KDE），并保存图片。

    参数:
        dist_real (np.ndarray): 真实样本分布，一维。
        dist_fake (np.ndarray): 生成样本分布，一维。
        statistic_name (str): 当前统计量名称。
        save_path (str): 图像保存路径。
        bins (int): 直方图分箱数，默认60。

    返回:
        无（直接保存图片）。
    
    使用场景:
        边长、特征、谱等分布对比，评价生成器多样性与真实性。
    """
    import seaborn as sns
    plt.figure(figsize=(6, 4))
    sns.histplot(dist_real, bins=bins, color='blue', alpha=0.6, stat='density', label='Real', kde=True)
    sns.histplot(dist_fake, bins=bins, color='orange', alpha=0.6, stat='density', label='Generated', kde=True)
    plt.title(f"Distribution Comparison for {statistic_name}", fontsize=14)
    plt.xlabel(statistic_name)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_brain_mesh(mesh_data, save_path, scalar_map=None, off_screen=True, window_size=(800, 600)):
    """
    可视化大脑皮层网格，支持多个网格并排显示，可选标量上色，支持无窗口渲染。

    参数:
        mesh_data (dict或list): 单个或多个网格，需包含:
            - 'vertices' (np.ndarray, N x 3): 顶点坐标。
            - 'faces' (np.ndarray, F x 3): 面定义（三角形）。
            - 'title' (str): 网格标题。
        save_path (str): 图片保存路径。
        scalar_map (dict, 可选): 标量上色，键为title，值为N x 1数组。
        off_screen (bool): 是否启用无窗口渲染，默认True。
        window_size (tuple): 窗口大小，默认800x600。

    返回:
        无（直接保存图片）。
    
    使用场景:
        训练后定性评估生成与真实大脑表面形态。
    """
    import pyvista as pv

    if isinstance(mesh_data, dict):
        mesh_data = [mesh_data]
    n_meshes = len(mesh_data)
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size, shape=(1, n_meshes))
    for i, mesh in enumerate(mesh_data):
        vertices = mesh['vertices']
        faces = mesh['faces']
        title = mesh.get('title', f"Mesh {i+1}")
        # pyvista需要faces格式为[n_faces, 4]，第一列是3（三角面），后面三个是索引
        pv_faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        surf = pv.PolyData(vertices, pv_faces)
        plotter.subplot(0, i)
        if scalar_map and title in scalar_map:
            scalars = scalar_map[title]
            surf["scalars"] = scalars
            plotter.add_mesh(surf, scalars="scalars", cmap="coolwarm", show_scalar_bar=True)
        else:
            plotter.add_mesh(surf, color="lightgray")
        plotter.add_text(title, font_size=12)
    plotter.camera_position = 'xy'
    plotter.show(screenshot=save_path)
    plotter.close()