#ifndef __KALMAN_FILTER__H__
#define __KALMAN_FILTER__H__

#include "Numcpp/Numcpp.hpp"
#include <functional>

namespace kalman
{

    template <typename T>
    class KalmanFilter
    {
    private:
        // 状态向量 (n x 1)
        np::Numcpp<T> x;

        // 状态协方差矩阵 (n x n)
        np::Numcpp<T> P;

        // 状态转移矩阵 (n x n)
        np::Numcpp<T> F;

        // 过程噪声协方差矩阵 (n x n)
        np::Numcpp<T> Q;

        // 观测矩阵 (m x n)
        np::Numcpp<T> H;

        // 观测噪声协方差矩阵 (m x m)
        np::Numcpp<T> R;

        // 控制输入矩阵 (n x p)
        np::Numcpp<T> B;

        // 单位矩阵 (n x n)
        np::Numcpp<T> I;

        // 状态维度
        size_t n;

        // 观测维度
        size_t m;

        // 控制输入维度
        size_t p;

    public:
        // 构造函数 - 基本版本
        KalmanFilter(size_t state_dim, size_t measurement_dim)
            : x(state_dim, 1, (T)0),
              P(state_dim, state_dim, (T)0),
              F(state_dim, state_dim, (T)0),
              Q(state_dim, state_dim, (T)0),
              H(measurement_dim, state_dim, (T)0),
              R(measurement_dim, measurement_dim, (T)0),
              B(1, 1, (T)0), // No control input
              I(state_dim, state_dim, (T)0),
              n(state_dim), m(measurement_dim), p(0)
        {
            initializeMatrices();
        }

        // 构造函数 - 带控制输入版本
        KalmanFilter(size_t state_dim, size_t measurement_dim, size_t control_dim)
            : x(state_dim, 1, (T)0),
              P(state_dim, state_dim, (T)0),
              F(state_dim, state_dim, (T)0),
              Q(state_dim, state_dim, (T)0),
              H(measurement_dim, state_dim, (T)0),
              R(measurement_dim, measurement_dim, (T)0),
              B(state_dim, control_dim, (T)0),
              I(state_dim, state_dim, (T)0),
              n(state_dim), m(measurement_dim), p(control_dim)
        {
            initializeMatrices();
        }

        // 初始化所有矩阵
        void initializeMatrices()
        {
            // 初始化状态向量和协方差矩阵
            x = np::Numcpp<T>(n, 1, (T)0);
            P = np::Numcpp<T>(n, n, (T)0);

            // 初始化状态转移矩阵为单位矩阵
            F = np::Numcpp<T>(n, n, (T)0);
            for (size_t i = 0; i < n; i++)
            {
                F[i][i] = 1.0;
            }

            // 初始化过程噪声协方差矩阵
            Q = np::Numcpp<T>(n, n, (T)0);
            for (size_t i = 0; i < n; i++)
            {
                Q[i][i] = 1.0;
            }

            // 初始化观测矩阵
            H = np::Numcpp<T>(m, n, (T)0);
            for (size_t i = 0; i < m && i < n; i++)
            {
                H[i][i] = 1.0;
            }

            // 初始化观测噪声协方差矩阵
            R = np::Numcpp<T>(m, m, (T)0);
            for (size_t i = 0; i < m; i++)
            {
                R[i][i] = 1.0;
            }

            // 初始化控制输入矩阵（如果有控制输入）
            if (p > 0)
            {
                B = np::Numcpp<T>(n, p, (T)0);
                for (size_t i = 0; i < n && i < p; i++)
                {
                    B[i][i] = 1.0;
                }
            }

            // 初始化单位矩阵
            I = np::Numcpp<T>(n, n, (T)0);
            for (size_t i = 0; i < n; i++)
            {
                I[i][i] = 1.0;
            }
        }

        // 设置初始状态
        void setInitialState(const np::Numcpp<T> &initialState)
        {
            if (initialState.row != n || initialState.col != 1)
            {
                throw std::invalid_argument("Initial state dimension mismatch");
            }
            x = initialState;
        }

        // 设置初始协方差
        void setInitialCovariance(const np::Numcpp<T> &initialCovariance)
        {
            if (initialCovariance.row != n || initialCovariance.col != n)
            {
                throw std::invalid_argument("Initial covariance dimension mismatch");
            }
            P = initialCovariance;
        }

        // 设置状态转移矩阵
        void setTransitionMatrix(const np::Numcpp<T> &transitionMatrix)
        {
            if (transitionMatrix.row != n || transitionMatrix.col != n)
            {
                throw std::invalid_argument("Transition matrix dimension mismatch");
            }
            F = transitionMatrix;
        }

        // 设置过程噪声协方差
        void setProcessNoiseCovariance(const np::Numcpp<T> &processNoise)
        {
            if (processNoise.row != n || processNoise.col != n)
            {
                throw std::invalid_argument("Process noise covariance dimension mismatch");
            }
            Q = processNoise;
        }

        // 设置观测矩阵
        void setObservationMatrix(const np::Numcpp<T> &observationMatrix)
        {
            if (observationMatrix.row != m || observationMatrix.col != n)
            {
                throw std::invalid_argument("Observation matrix dimension mismatch");
            }
            H = observationMatrix;
        }

        // 设置观测噪声协方差
        void setObservationNoiseCovariance(const np::Numcpp<T> &observationNoise)
        {
            if (observationNoise.row != m || observationNoise.col != m)
            {
                throw std::invalid_argument("Observation noise covariance dimension mismatch");
            }
            R = observationNoise;
        }

        // 设置控制输入矩阵
        void setControlMatrix(const np::Numcpp<T> &controlMatrix)
        {
            if (p == 0)
            {
                throw std::invalid_argument("Control dimension was set to 0");
            }
            if (controlMatrix.row != n || controlMatrix.col != p)
            {
                throw std::invalid_argument("Control matrix dimension mismatch");
            }
            B = controlMatrix;
        }

        // 预测步骤（无控制输入）
        void predict()
        {
            // 状态预测: x = F * x
            x = F * x;

            // 协方差预测: P = F * P * F^T + Q
            P = (F * P * F.transpose()) + Q;
        }

        // 预测步骤（有控制输入）
        void predict(const np::Numcpp<T> &u)
        {
            if (p == 0)
            {
                throw std::invalid_argument("Control dimension was set to 0");
            }
            if (u.row != p || u.col != 1)
            {
                throw std::invalid_argument("Control input dimension mismatch");
            }

            // 状态预测: x = F * x + B * u
            x = (F * x) + (B * u);

            // 协方差预测: P = F * P * F^T + Q
            P = (F * P * F.transpose()) + Q;
        }

        // 更新步骤
        void update(np::Numcpp<T> &z)
        {
            if (z.row != m || z.col != 1)
            {
                throw std::invalid_argument("Measurement dimension mismatch");
            }

            // 计算卡尔曼增益: K = P * H^T * (H * P * H^T + R)^(-1)
            np::Numcpp<T> Ht = H.transpose();
            np::Numcpp<T> S = (H * P * Ht) + R;
            np::Numcpp<T> K = P * Ht * S.inverse();

            // 更新状态估计: x = x + K * (z - H * x)
            np::Numcpp<T> y = z - (H * x);
            x = x + (K * y);

            // 更新协方差估计: P = (I - K * H) * P
            P = (I - (K * H)) * P;
        }

        // 获取当前状态估计
        np::Numcpp<T> getState() const
        {
            return x;
        }

        // 获取当前协方差估计
        np::Numcpp<T> getCovariance() const
        {
            return P;
        }

        // 获取状态维度
        size_t getStateDimension() const
        {
            return n;
        }

        // 获取观测维度
        size_t getMeasurementDimension() const
        {
            return m;
        }

        // 获取控制输入维度
        size_t getControlDimension() const
        {
            return p;
        }
        void setState(const np::Numcpp<T> &new_state)
        {
            if (new_state.row != n || new_state.col != 1)
            {
                throw std::invalid_argument("State dimension mismatch");
            }
            x = new_state;
        }

        void setCovariance(const np::Numcpp<T> &new_cov)
        {
            if (new_cov.row != n || new_cov.col != n)
            {
                throw std::invalid_argument("Covariance dimension mismatch");
            }
            P = new_cov;
        }
    };

} // namespace kalman

#endif // __KALMAN_FILTER__H__