import cv2
import numpy as np
import sophus as sp
import apriltag
import compute_error as ce
# from ceres import Solver, Problem, CostFunction, LossFunction

# 代价函数，用于计算重投影误差
# class ReprojectionError(CostFunction):
#     def __init__(self, observed_x, observed_y):
#         super(ReprojectionError, self).__init__(2, 16)  # 残差维度为2，状态变量维度为16（4x4变换矩阵）
#         self.observed_x = observed_x
#         self.observed_y = observed_y

#     def Evaluate(self, parameters, residuals):
#         # 从参数中恢复变换矩阵
#         T = np.reshape(parameters, (4, 4))
        
#         # 假设世界坐标系中的点是 (x, y, z, 1)
#         world_point = np.array([self.tag_positions_world[i][0], self.tag_positions_world[i][1], self.tag_positions_world[i][2], 1])
        
#         # 将世界坐标系中的点变换到相机坐标系
#         camera_point = T.dot(world_point)
        
#         # 将相机坐标系中的点投影到像素坐标系
#         pixel_point = self.camera_matrix @ camera_point[:3]
#         pixel_point /= pixel_point[2]
        
#         # 计算残差
#         residuals[0] = pixel_point[0] - self.observed_x
#         residuals[1] = pixel_point[1] - self.observed_y
        
class CameraPaperCalibrator:
    def __init__(self):
        # Initialize your calibrator
        self.tags = []
        self.tag_paper_transforms = {}
        self.tag_camera_transforms = {}
        self.camera_matrix = []

    
    def add_tag_paper_transform(self, T,tag_id):
        """Add a tag's position in paper coordinates."""
        self.tag_paper_transforms[tag_id]=T

    def add_tag_camera_transform(self, T,tag_id):
        """Add a detected tag's position in camera coordinate"""
        self.tag_camera_transforms[tag_id]=T

    def calibrate(self, initial_guess=None):
        """Perform calibration and return the camera-paper transformation"""
        if initial_guess is None:
            initial_guess = np.eye(4)  # 4x4 identity matrix

        # 优化
        result = ce.optimize_camera_table_transform(self.tag_camera_transforms,self.tag_paper_transforms, initial_guess)
        
        # 调用可视化函数
        ce.visualize_optimization_result(result, self.tag_camera_transforms,self.tag_paper_transforms)

        # 计算误差统计
        errors = ce.calculate_error_statistics(result, self.tag_camera_transforms,self.tag_paper_transforms)

        # 绘制误差分布
        ce.plot_error_distribution(errors)
        # Define the problem
        # problem = Problem()

        # # 添加残差块
        # for i, tag in enumerate(self.tags):
        #     if i < len(self.tag_paper_transforms):
        #         T_table = self.tag_paper_transforms[i]
        #         world_point = T_table[:3, 3]  # 提取世界坐标系中的点

        #         # 定义代价函数
        #         error = ReprojectionError(tag.center[0], tag.center[1], world_point, self.camera_matrix)
        #         problem.AddResidualBlock(error, None, initial_guess)

        # # 配置求解器选项
        # options = SolverOptions()
        # options.linear_solver_type = 'DENSE_QR'
        # options.minimizer_progress_to_stdout = True

        # # 求解
        # solver = Solver(options)
        # result = solver.Solve(problem, initial_guess)

        return result  # Return the optimized transformation matrix
    
def main():
    calibrator = CameraPaperCalibrator()
    
    # 加载图片
    image = cv2.imread('target_photo.jpg')
    
    # 转为灰度图片
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # 创建AprilTag检测器
    detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9'))
     
    # 检测AprilTags
    tags = detector.detect(gray) 
    calibrator.tags = tags
    # 显示检测结果
    print("%d apriltags have been detected."%len(tags))
    
    # for tag in tags:
    #     cv2.circle(image, tuple(tag.corners[0].astype(int)), 4,(255,0,0), 2) # left-top
    #     cv2.circle(image, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-top
    #     cv2.circle(image, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-bottom
    #     cv2.circle(image, tuple(tag.corners[3].astype(int)), 4,(255,0,0), 2) # left-bottom
   
    # 设置Tag实际位置，假设图片中4个Tag中心点在世界坐标系的位置为(0,0,0)，(200,0,0)，(200,200,0)，(0,200,0)（从左下开始逆时针,单位mm）
    tag_positions_world = np.array([(200, 0, 0),
                                    (0, 0, 0),
                                    (200, 200, 0),
                                    (0, 200, 0)])
 
    cam_params = np.array([612.345825, 613.81781, 318.473251, 237.981806]) #假设的fx, fy, cx, cy
    # 相机内参
    calibrator.camera_matrix = np.array([[cam_params[0], 0, cam_params[2]],
                          [0, cam_params[1], cam_params[3]],
                          [0, 0, 1]])
    tag_len = 48    #假设的tag边长（单位：mm）
    # obj2world_T = np.array([0, 0, 0])    #假设的左下角Tag中心在世界坐标系中的坐标值
    
    tag_positions_camera = []
    tag_positions_table = []
    for i,tag in enumerate(tags):
        # 获得相机坐标系下Tag矩阵
        M, e1, e2 = detector.detection_pose(tag, cam_params)
        M[:3,3:] *= tag_len
        calibrator.add_tag_camera_transform(M,tag.tag_id)
        # 获得世界坐标系下Tag矩阵
        obj2world = np.identity(4)
        obj2world[:3,3] = tag_positions_world[i]
        calibrator.add_tag_paper_transform(obj2world,tag.tag_id)        
        
        for i in range(4):
            cv2.circle(image, tuple(tag.corners[i].astype(int)), 4, (255, 0, 0), 2)
        cv2.circle(image, tuple(tag.center.astype(int)), 4, (2, 180, 200), 4)
                
    cv2.imshow("target_photo",image)
    # cv2.waitKey()

    # Perform calibration
    result = calibrator.calibrate()

    print("Optimized Transformation Matrix:")
    print(result)
    
    

if __name__ == "__main__":
    main()
