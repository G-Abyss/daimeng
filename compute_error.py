import numpy as np
from scipy.optimize import minimize
from visualize_optimization_result import visualize_optimization_result, calculate_error_statistics, plot_error_distribution



def compute_error(T_camera_table, camera_transforms, paper_transforms):
    """
    计算误差：总误差为每个标签的误差之和。
    :param T_camera_table: 当前优化变量（4x4变换矩阵）。
    :param camera_transforms: 相机坐标系中的标签变换。
    :param paper_transforms: 纸张坐标系中的标签变换。
    :return: 总误差。
    """
    error = 0.0
    for tag_id in camera_transforms:
        if tag_id in paper_transforms:
            T_camera_tag = camera_transforms[tag_id]
            T_paper_tag = paper_transforms[tag_id]

            # 计算预测的位姿
            T_predicted = T_camera_table @ np.linalg.inv(T_paper_tag)

            # 误差为预测位姿与实际位姿的 Frobenius 范数
            diff = T_camera_tag - T_predicted
            error += np.linalg.norm(diff, ord='fro')
    return error


def optimize_camera_table_transform(camera_transforms, paper_transforms, initial_guess=None):
    """
    优化相机到桌面的变换矩阵。
    :param camera_transforms: 相机坐标系中的标签变换。
    :param paper_transforms: 纸张坐标系中的标签变换。
    :param initial_guess: 初始变换矩阵。
    :return: 优化后的变换矩阵（4x4）。
    """
    # 初始化
    if initial_guess is None:
        initial_guess = np.eye(4)  # 默认初始猜测为单位矩阵

    # 将初始矩阵展开为一维数组进行优化
    initial_guess_flat = initial_guess.flatten()

    # 优化目标函数
    def objective(flat_T_camera_table):
        # 将一维数组还原为 4x4 矩阵
        T_camera_table = flat_T_camera_table.reshape((4, 4))
        return compute_error(T_camera_table, camera_transforms, paper_transforms)

    # 使用 scipy.optimize 进行优化
    result = minimize(objective, initial_guess_flat, method='BFGS')

    # 将结果还原为 4x4 矩阵
    optimized_T_camera_table = result.x.reshape((4, 4))
    return optimized_T_camera_table

def evaluate_optimization(T_camera_table, camera_transforms, paper_transforms):
    """
    评估优化结果，计算每个标签的误差。
    :param T_camera_table: 优化后的变换矩阵。
    :param camera_transforms: 相机坐标系中的标签变换。
    :param paper_transforms: 纸张坐标系中的标签变换。
    :return: 每个标签的误差列表。
    """
    errors = {}
    for tag_id in camera_transforms:
        if tag_id in paper_transforms:
            T_camera_tag = camera_transforms[tag_id]
            T_paper_tag = paper_transforms[tag_id]
            T_predicted = T_camera_table @ np.linalg.inv(T_paper_tag)
            errors[tag_id] = np.linalg.norm(T_camera_tag - T_predicted, ord='fro')
    return errors


if __name__ == "__main__":
    # 初始化示例数据
    camera_transforms = {
        1: np.array([
            [1, 0, 0, 0.3],
            [0, 1, 0, 0.4],
            [0, 0, 1, 0.5],
            [0, 0, 0, 1]
        ])
    }

    paper_transforms = {
        1: np.array([
            [1, 0, 0, 0.1],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0.3],
            [0, 0, 0, 1]
        ])
    }

    # 初始猜测
    initial_guess = np.eye(4)

    # 优化
    optimized_transform = optimize_camera_table_transform(camera_transforms, paper_transforms, initial_guess)

    # 输出优化结果
    print("Optimized Transformation Matrix:")
    print(optimized_transform)


    # 评估优化结果
    errors = evaluate_optimization(optimized_transform, camera_transforms, paper_transforms)
    print("Per-tag Errors:")
    for tag_id, error in errors.items():
        print(f"Tag {tag_id}: Error = {error}")

    # 调用可视化函数
    visualize_optimization_result(optimized_transform, camera_transforms, paper_transforms)

    # 计算误差统计
    errors = calculate_error_statistics(optimized_transform, camera_transforms, paper_transforms)

    # 绘制误差分布
    plot_error_distribution(errors)