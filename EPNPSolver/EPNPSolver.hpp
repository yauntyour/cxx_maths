#ifndef __EPNP_SOLVER_H__
#define __EPNP_SOLVER_H__

#include "Numcpp/Numcpp.hpp"
#include <vector>
#include <algorithm>

namespace epnp {

template<typename T>
class EPNPSolver {
private:
    // 相机内参矩阵
    np::Numcpp<T> K;
    
    // 控制点在世界坐标系中的坐标
    np::Numcpp<T> control_points_world;
    
    // 控制点在相机坐标系中的坐标
    np::Numcpp<T> control_points_camera;
    
    // 特征点对应的权重（齐次重心坐标）
    np::Numcpp<T> barycentric_coords;

public:
    EPNPSolver(const np::Numcpp<T>& camera_matrix) : K(camera_matrix) {
        if (K.row != 3 || K.col != 3) {
            throw std::invalid_argument("Camera matrix must be 3x3");
        }
    }

    // 主求解函数
    bool solve(const std::vector<std::vector<T>>& points3D, 
               const std::vector<std::vector<T>>& points2D,
               np::Numcpp<T>& R, np::Numcpp<T>& t) {
        
        if (points3D.size() != points2D.size() || points3D.size() < 4) {
            throw std::invalid_argument("Need at least 4 point correspondences");
        }

        size_t n = points3D.size();
        
        // 1. 选择控制点
        selectControlPoints(points3D);
        
        // 2. 计算齐次重心坐标
        computeBarycentricCoords(points3D);
        
        // 3. 构建线性系统
        np::Numcpp<T> M = buildLinearSystem(points2D);
        
        // 4. 求解线性系统
        if (!solveLinearSystem(M)) {
            return false;
        }
        
        // 5. 高斯牛顿优化
        gaussNewtonOptimization(points2D);
        
        // 6. 计算最终的旋转和平移
        return computePose(R, t);
    }

private:
    // 选择控制点（使用PCA方法）
    void selectControlPoints(const std::vector<std::vector<T>>& points3D) {
        size_t n = points3D.size();
        
        // 计算重心
        std::vector<T> centroid(3, 0);
        for (const auto& pt : points3D) {
            for (int i = 0; i < 3; ++i) {
                centroid[i] += pt[i];
            }
        }
        for (int i = 0; i < 3; ++i) {
            centroid[i] /= n;
        }
        
        // 构建协方差矩阵
        np::Numcpp<T> cov(3, 3, 0);
        for (const auto& pt : points3D) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    cov[i][j] += (pt[i] - centroid[i]) * (pt[j] - centroid[j]);
                }
            }
        }
        
        // 特征值分解
        auto eig_result = cov.eig();
        auto eigenvectors = eig_result[1]; // 特征向量矩阵
        
        // 控制点：重心 + 特征向量方向
        control_points_world = np::Numcpp<T>(4, 3);
        
        // 第一个控制点：重心
        for (int i = 0; i < 3; ++i) {
            control_points_world[0][i] = centroid[i];
        }
        
        // 其他三个控制点：重心 + 特征向量
        for (int j = 0; j < 3; ++j) {
            T lambda = sqrt(eig_result[0][0][j]); // 特征值的平方根
            for (int i = 0; i < 3; ++i) {
                control_points_world[j+1][i] = centroid[i] + lambda * eigenvectors[i][j];
            }
        }
    }

    // 计算齐次重心坐标
    void computeBarycentricCoords(const std::vector<std::vector<T>>& points3D) {
        size_t n = points3D.size();
        barycentric_coords = np::Numcpp<T>(n, 4);
        
        // 构建线性系统：每个点可以表示为控制点的线性组合
        for (size_t k = 0; k < n; ++k) {
            np::Numcpp<T> A(3, 4);
            np::Numcpp<T> b(3, 1);
            
            for (int i = 0; i < 3; ++i) {
                b[i][0] = points3D[k][i] - control_points_world[0][i];
                for (int j = 0; j < 3; ++j) {
                    A[i][j+1] = control_points_world[j+1][i] - control_points_world[0][i];
                }
            }
            
            // 求解最小二乘问题
            np::Numcpp<T> alpha = A.pseudoinverse() * b;
            
            // 设置重心坐标（第一个坐标由其他三个决定）
            barycentric_coords[k][0] = 1.0;
            for (int j = 0; j < 3; ++j) {
                barycentric_coords[k][j+1] = alpha[j][0];
                barycentric_coords[k][0] -= alpha[j][0];
            }
        }
    }

    // 构建线性系统 M * x = 0
    np::Numcpp<T> buildLinearSystem(const std::vector<std::vector<T>>& points2D) {
        size_t n = points2D.size();
        np::Numcpp<T> M(2 * n, 12, 0);
        
        T fx = K[0][0], fy = K[1][1];
        T cx = K[0][2], cy = K[1][2];
        
        for (size_t i = 0; i < n; ++i) {
            T u = points2D[i][0], v = points2D[i][1];
            
            for (int j = 0; j < 4; ++j) {
                T alpha = barycentric_coords[i][j];
                
                // 第一行方程
                M[2*i][3*j] = alpha * fx;
                M[2*i][3*j+2] = alpha * (cx - u);
                
                // 第二行方程  
                M[2*i+1][3*j+1] = alpha * fy;
                M[2*i+1][3*j+2] = alpha * (cy - v);
            }
        }
        
        return M;
    }

    // 求解线性系统
    bool solveLinearSystem(const np::Numcpp<T>& M) {
        // 使用SVD求解 M^T * M 的最小特征值对应的特征向量
        np::Numcpp<T> MtM = M.transpose() * M;
        
        auto eig_result = MtM.eig();
        auto eigenvalues = eig_result[0];
        auto eigenvectors = eig_result[1];
        
        // 找到最小特征值对应的特征向量
        size_t min_idx = 0;
        T min_eigenvalue = eigenvalues[0][0];
        for (size_t i = 1; i < 12; ++i) {
            if (eigenvalues[0][i] < min_eigenvalue) {
                min_eigenvalue = eigenvalues[0][i];
                min_idx = i;
            }
        }
        
        // 提取控制点在相机坐标系中的坐标
        control_points_camera = np::Numcpp<T>(4, 3);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                control_points_camera[i][j] = eigenvectors[j][min_idx + i * 3];
            }
        }
        
        return true;
    }

    // 高斯牛顿优化
    void gaussNewtonOptimization(const std::vector<std::vector<T>>& points2D) {
        const int max_iterations = 10;
        const T convergence_threshold = 1e-6;
        
        size_t n = points2D.size();
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // 计算雅可比矩阵和残差
            np::Numcpp<T> J(2 * n, 6, 0);
            np::Numcpp<T> r(2 * n, 1, 0);
            
            computeJacobianAndResidual(points2D, J, r);
            
            // 高斯牛顿步：delta = -(J^T J)^-1 J^T r
            np::Numcpp<T> JtJ = J.transpose() * J;
            np::Numcpp<T> Jtr = J.transpose() * r;
            np::Numcpp<T> delta = JtJ.pseudoinverse() * Jtr * (-1.0);
            
            // 更新控制点坐标
            updateControlPoints(delta);
            
            // 检查收敛
            if (delta.sum() < convergence_threshold) {
                break;
            }
        }
    }

    // 计算雅可比矩阵和残差
    void computeJacobianAndResidual(const std::vector<std::vector<T>>& points2D, 
                                   np::Numcpp<T>& J, np::Numcpp<T>& r) {
        size_t n = points2D.size();
        T fx = K[0][0], fy = K[1][1];
        T cx = K[0][2], cy = K[1][2];
        
        for (size_t i = 0; i < n; ++i) {
            // 计算重投影点
            std::vector<T> proj_pt = projectPoint(i);
            T u_proj = proj_pt[0], v_proj = proj_pt[1];
            T u_obs = points2D[i][0], v_obs = points2D[i][1];
            
            // 残差
            r[2*i][0] = u_obs - u_proj;
            r[2*i+1][0] = v_obs - v_proj;
            
            // 雅可比矩阵（数值微分近似）
            const T eps = 1e-6;
            for (int j = 0; j < 6; ++j) {
                // 扰动第j个参数
                np::Numcpp<T> control_points_plus = control_points_camera;
                perturbControlPoints(control_points_plus, j, eps);
                
                std::vector<T> proj_plus = projectPoint(i, control_points_plus);
                T du_dp = (proj_plus[0] - u_proj) / eps;
                T dv_dp = (proj_plus[1] - v_proj) / eps;
                
                J[2*i][j] = du_dp;
                J[2*i+1][j] = dv_dp;
            }
        }
    }

    // 投影点到图像平面
    std::vector<T> projectPoint(size_t point_idx, const np::Numcpp<T>& control_points = np::Numcpp<T>()) {
        const np::Numcpp<T>& cp = control_points.row == 0 ? control_points_camera : control_points;
        
        // 计算3D点在相机坐标系中的坐标
        std::vector<T> pt3D(3, 0);
        for (int j = 0; j < 4; ++j) {
            T alpha = barycentric_coords[point_idx][j];
            for (int k = 0; k < 3; ++k) {
                pt3D[k] += alpha * cp[j][k];
            }
        }
        
        // 投影到图像平面
        T x = pt3D[0], y = pt3D[1], z = pt3D[2];
        T u = (K[0][0] * x + K[0][2] * z) / z;
        T v = (K[1][1] * y + K[1][2] * z) / z;
        
        return {u, v};
    }

    // 扰动控制点（用于数值微分）
    void perturbControlPoints(np::Numcpp<T>& control_points, int param_idx, T delta) {
        // 参数顺序：前3个是旋转，后3个是平移
        if (param_idx < 3) {
            // 旋转扰动（简化处理）
            for (int i = 0; i < 4; ++i) {
                control_points[i][param_idx] += delta;
            }
        } else {
            // 平移扰动
            int trans_idx = param_idx - 3;
            for (int i = 0; i < 4; ++i) {
                control_points[i][trans_idx] += delta;
            }
        }
    }

    // 更新控制点坐标
    void updateControlPoints(const np::Numcpp<T>& delta) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                control_points_camera[i][j] += delta[j][0];
            }
        }
    }

    // 计算最终的旋转和平移
    bool computePose(np::Numcpp<T>& R, np::Numcpp<T>& t) {
        // 使用Umeyama算法计算相似变换
        // 计算两个点集的重心
        std::vector<T> centroid_world(3, 0), centroid_camera(3, 0);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                centroid_world[j] += control_points_world[i][j];
                centroid_camera[j] += control_points_camera[i][j];
            }
        }
        for (int j = 0; j < 3; ++j) {
            centroid_world[j] /= 4;
            centroid_camera[j] /= 4;
        }
        
        // 计算去中心化的点集
        np::Numcpp<T> X(3, 4), Y(3, 4);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                X[j][i] = control_points_world[i][j] - centroid_world[j];
                Y[j][i] = control_points_camera[i][j] - centroid_camera[j];
            }
        }
        
        // 计算协方差矩阵
        np::Numcpp<T> S = X * Y.transpose();
        
        // SVD分解
        auto svd_result = S.svd();
        auto U = svd_result[0];
        auto V = svd_result[2];
        
        // 计算旋转矩阵
        R = V * U.transpose();
        
        // 确保行列式为正（处理反射情况）
        if (R.determinant() < 0) {
            for (int i = 0; i < 3; ++i) {
                V[i][2] *= -1;
            }
            R = V * U.transpose();
        }
        
        // 计算平移向量
        t = np::Numcpp<T>(3, 1);
        for (int i = 0; i < 3; ++i) {
            t[i][0] = centroid_camera[i];
            for (int j = 0; j < 3; ++j) {
                t[i][0] -= R[i][j] * centroid_world[j];
            }
        }
        
        return true;
    }
};

// 便捷函数
template<typename T>
bool solveEPNP(const std::vector<std::vector<T>>& points3D,
               const std::vector<std::vector<T>>& points2D,
               const np::Numcpp<T>& camera_matrix,
               np::Numcpp<T>& R, np::Numcpp<T>& t) {
    
    EPNPSolver<T> solver(camera_matrix);
    return solver.solve(points3D, points2D, R, t);
}

} // namespace epnp

#endif // __EPNP_SOLVER_H__