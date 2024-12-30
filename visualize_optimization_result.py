import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_optimization_result(T_camera_table, camera_transforms, paper_transforms):
    """
    可视化优化结果。
    :param T_camera_table: 优化后的变换矩阵。
    :param camera_transforms: 相机坐标系中的标签变换。
    :param paper_transforms: 纸张坐标系中的标签变换。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制实际标签位置
    actual_positions = []
    predicted_positions = []
    for tag_id in camera_transforms:
        if tag_id in paper_transforms:
            # 相机坐标中的实际标签位置
            T_camera_tag = camera_transforms[tag_id]
            actual_positions.append(T_camera_tag[:3, 3])  # 提取平移部分

            # 预测标签位置
            T_paper_tag = paper_transforms[tag_id]
            T_predicted = T_camera_table @ np.linalg.inv(T_paper_tag)
            predicted_positions.append(T_predicted[:3, 3])  # 提取平移部分

    actual_positions = np.array(actual_positions)
    predicted_positions = np.array(predicted_positions)

    # 绘制实际位置
    ax.scatter(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2],
               c='r', label='Actual Positions')

    # 绘制预测位置
    ax.scatter(predicted_positions[:, 0], predicted_positions[:, 1], predicted_positions[:, 2],
               c='b', label='Predicted Positions')

    ax.set_title("Optimization Result Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# 调用可视化函数
# visualize_optimization_result(optimized_transform, camera_transforms, paper_transforms)


def calculate_error_statistics(T_camera_table, camera_transforms, paper_transforms):
    """
    计算误差的统计信息。
    :param T_camera_table: 优化后的变换矩阵。
    :param camera_transforms: 相机坐标系中的标签变换。
    :param paper_transforms: 纸张坐标系中的标签变换。
    :return: 误差列表和统计信息。
    """
    errors = []
    for tag_id in camera_transforms:
        if tag_id in paper_transforms:
            T_camera_tag = camera_transforms[tag_id]
            T_paper_tag = paper_transforms[tag_id]
            T_predicted = T_camera_table @ np.linalg.inv(T_paper_tag)

            # 计算 Frobenius 范数误差
            error = np.linalg.norm(T_camera_tag - T_predicted, ord='fro')
            errors.append(error)

    # 统计信息
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    error_max = np.max(errors)
    error_min = np.min(errors)

    print("Error Statistics:")
    print(f"Mean Error: {error_mean}")
    print(f"Standard Deviation: {error_std}")
    print(f"Max Error: {error_max}")
    print(f"Min Error: {error_min}")

    return errors

# 计算误差统计
# errors = calculate_error_statistics(optimized_transform, camera_transforms, paper_transforms)


def plot_error_distribution(errors):
    """
    绘制误差分布的直方图。
    :param errors: 每个标签的误差列表。
    """
    plt.figure()
    plt.hist(errors, bins=10, color='blue', alpha=0.7)
    plt.title("Error Distribution")
    plt.xlabel("Error Magnitude")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 绘制误差分布
# plot_error_distribution(errors)