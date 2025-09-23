#ifndef __KF__H__
#define __KF__H__

#include "Numcpp/Numcpp.hpp"
#include <functional>
#include <cmath>

namespace kf
{
    // 基类：卡尔曼滤波器
    template <typename T>
    class kf
    {
    protected:
        // 状态向量 (n x 1)
        np::Numcpp<T> x;

        // 状态协方差矩阵 (n x n)
        np::Numcpp<T> P;

        // 过程噪声协方差矩阵 (n x n)
        np::Numcpp<T> Q;

        // 观测噪声协方差矩阵 (m x m)
        np::Numcpp<T> R;

        // 单位矩阵 (n x n)
        np::Numcpp<T> I;

        // 状态维度
        size_t n;

        // 观测维度
        size_t m;

    public:
        kf(size_t state_dim, size_t measurement_dim)
            : x(state_dim, 1, (T)0),
              P(state_dim, state_dim, (T)0),
              Q(state_dim, state_dim, (T)0),
              R(measurement_dim, measurement_dim, (T)0),
              I(state_dim, state_dim, (T)0),
              n(state_dim), m(measurement_dim)
        {
            // 初始化单位矩阵
            for (size_t i = 0; i < n; i++)
            {
                I[i][i] = 1.0;
            }
        }

        virtual ~kf() = default;

        // 设置初始状态
        virtual void setInitialState(const np::Numcpp<T> &initialState)
        {
            if (initialState.row != n || initialState.col != 1)
            {
                throw std::invalid_argument("Initial state dimension mismatch");
            }
            x = initialState;
        }

        // 设置初始协方差
        virtual void setInitialCovariance(const np::Numcpp<T> &initialCovariance)
        {
            if (initialCovariance.row != n || initialCovariance.col != n)
            {
                throw std::invalid_argument("Initial covariance dimension mismatch");
            }
            P = initialCovariance;
        }

        // 设置过程噪声协方差
        virtual void setProcessNoiseCovariance(const np::Numcpp<T> &processNoise)
        {
            if (processNoise.row != n || processNoise.col != n)
            {
                throw std::invalid_argument("Process noise covariance dimension mismatch");
            }
            Q = processNoise;
        }

        // 设置观测噪声协方差
        virtual void setObservationNoiseCovariance(const np::Numcpp<T> &observationNoise)
        {
            if (observationNoise.row != m || observationNoise.col != m)
            {
                throw std::invalid_argument("Observation noise covariance dimension mismatch");
            }
            R = observationNoise;
        }
        virtual void GaussiandistributionNoiseMatrix(np::Numcpp<T> &mat)
        {
        }

        // 预测步骤（纯虚函数）
        virtual void predict() = 0;
        virtual void predict(const np::Numcpp<T> &u) = 0;

        // 更新步骤（纯虚函数）
        virtual void update(const np::Numcpp<T> &z) = 0;

        // 获取当前状态估计
        virtual np::Numcpp<T> getState() const
        {
            return x;
        }

        // 获取当前协方差估计
        virtual np::Numcpp<T> getCovariance() const
        {
            return P;
        }

        // 获取状态维度
        virtual size_t getStateDimension() const
        {
            return n;
        }

        // 获取观测维度
        virtual size_t getMeasurementDimension() const
        {
            return m;
        }
    };

    // 线性卡尔曼滤波器
    template <typename T>
    class kfl : public kf<T>
    {
    protected:
        using kf<T>::x;
        using kf<T>::P;
        using kf<T>::Q;
        using kf<T>::R;
        using kf<T>::I;
        using kf<T>::n;
        using kf<T>::m;

        // 状态转移矩阵 (n x n)
        np::Numcpp<T> F;

        // 观测矩阵 (m x n)
        np::Numcpp<T> H;

        // 控制输入矩阵 (n x p)
        np::Numcpp<T> B;

        // 控制输入维度
        size_t p;

    public:
        kfl() = default;
        // 构造函数 - 基本版本
        kfl(size_t state_dim, size_t measurement_dim)
            : kf<T>(state_dim, measurement_dim),
              F(state_dim, state_dim, (T)0),
              H(measurement_dim, state_dim, (T)0),
              B(1, 1, (T)0), // No control input
              p(0)
        {
            initializeMatrices();
        }

        // 构造函数 - 带控制输入版本
        kfl(size_t state_dim, size_t measurement_dim, size_t control_dim)
            : kf<T>(state_dim, measurement_dim),
              F(state_dim, state_dim, (T)0),
              H(measurement_dim, state_dim, (T)0),
              B(state_dim, control_dim, (T)0),
              p(control_dim)
        {
            initializeMatrices();
        }

        // 初始化所有矩阵
        void initializeMatrices()
        {
            // 初始化状态转移矩阵为单位矩阵
            F = np::Numcpp<T>(n, n, (T)0);
            for (size_t i = 0; i < n; i++)
            {
                F[i][i] = 1.0;
            }

            // 初始化观测矩阵
            H = np::Numcpp<T>(m, n, (T)0);
            for (size_t i = 0; i < m && i < n; i++)
            {
                H[i][i] = 1.0;
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

        // 设置观测矩阵
        void setObservationMatrix(const np::Numcpp<T> &observationMatrix)
        {
            if (observationMatrix.row != m || observationMatrix.col != n)
            {
                throw std::invalid_argument("Observation matrix dimension mismatch");
            }
            H = observationMatrix;
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
        void predict() override
        {
            // 状态预测: x = F * x
            x = F * x;

            // 协方差预测: P = F * P * F^T + Q
            P = (F * P * F.transpose()) + Q;
        }

        // 预测步骤（有控制输入）
        void predict(const np::Numcpp<T> &u) override
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
        void update(const np::Numcpp<T> &z) override
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

        // 获取控制输入维度
        size_t getControlDimension() const
        {
            return p;
        }
    };
    // 扩展卡尔曼滤波器 (EKF) - 处理非线性系统
    template <typename T>
    class ekf : public kf<T>
    {
    protected:
        using kf<T>::x;
        using kf<T>::P;
        using kf<T>::Q;
        using kf<T>::R;
        using kf<T>::I;
        using kf<T>::n;
        using kf<T>::m;

        // 非线性状态转移函数: x_k = f(x_{k-1}, u_{k-1})
        std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> f;

        // 非线性观测函数: z_k = h(x_k)
        std::function<np::Numcpp<T>(const np::Numcpp<T> &)> h;

        // 状态转移雅可比矩阵函数: F = ∂f/∂x
        std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> calculateF;

        // 观测雅可比矩阵函数: H = ∂h/∂x
        std::function<np::Numcpp<T>(const np::Numcpp<T> &)> calculateH;

        // 控制输入维度
        size_t p;

    public:
        // 构造函数
        ekf() = default;

        ekf(size_t state_dim, size_t measurement_dim,
            std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> state_transition_func,
            std::function<np::Numcpp<T>(const np::Numcpp<T> &)> observation_func,
            std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> state_transition_jacobian_func,
            std::function<np::Numcpp<T>(const np::Numcpp<T> &)> observation_jacobian_func,
            size_t control_dim = 1)
            : kf<T>(state_dim, measurement_dim),
              f(state_transition_func),
              h(observation_func),
              calculateF(state_transition_jacobian_func),
              calculateH(observation_jacobian_func),
              p(control_dim)
        {
        }

        // 预测步骤（无控制输入）
        void predict() override
        {
            predict(np::Numcpp<T>(p, 1, (T)0));
        }

        // 预测步骤（有控制输入）
        void predict(const np::Numcpp<T> &u) override
        {
            if (u.row != p || u.col != 1)
            {
                throw std::invalid_argument("Control input dimension mismatch");
            }

            // 计算雅可比矩阵 F
            np::Numcpp<T> F = calculateF(x, u);

            // 状态预测: x = f(x, u)
            x = f(x, u);

            // 协方差预测: P = F * P * F^T + Q
            P = (F * P * F.transpose()) + Q;
        }

        // 更新步骤
        void update(const np::Numcpp<T> &z) override
        {
            if (z.row != m || z.col != 1)
            {
                throw std::invalid_argument("Measurement dimension mismatch");
            }

            // 计算雅可比矩阵 H
            np::Numcpp<T> H = calculateH(x);

            // 计算卡尔曼增益: K = P * H^T * (H * P * H^T + R)^(-1)
            np::Numcpp<T> Ht = H.transpose();
            np::Numcpp<T> S = (H * P * Ht) + R;
            np::Numcpp<T> K = P * Ht * S.inverse();

            // 更新状态估计: x = x + K * (z - h(x))
            np::Numcpp<T> y = z - h(x);
            x = x + (K * y);

            // 更新协方差估计: P = (I - K * H) * P
            P = (I - (K * H)) * P;
        }

        // 获取控制输入维度
        size_t getControlDimension() const
        {
            return p;
        }
    };

    // 无迹卡尔曼滤波器 (UKF) - 处理非线性系统
    template <typename T>
    class ukf : public kf<T>
    {
    protected:
        using kf<T>::x;
        using kf<T>::P;
        using kf<T>::Q;
        using kf<T>::R;
        using kf<T>::I;
        using kf<T>::n;
        using kf<T>::m;

        // 非线性状态转移函数: x_k = f(x_{k-1}, u_{k-1})
        std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> f;

        // 非线性观测函数: z_k = h(x_k)
        std::function<np::Numcpp<T>(const np::Numcpp<T> &)> h;

        // 控制输入维度
        size_t p;

        // UKF参数
        T alpha, beta, kappa;
        size_t num_sigma_points;
        T lambda;

        // 计算Sigma点
        std::vector<np::Numcpp<T>> calculateSigmaPoints(const np::Numcpp<T> &mean, const np::Numcpp<T> &cov)
        {
            std::vector<np::Numcpp<T>> sigma_points;
            sigma_points.reserve(num_sigma_points);

            // 第一个Sigma点是均值
            sigma_points.push_back(mean);

            // 计算矩阵平方根
            np::Numcpp<T> S = cov;
            // 这里需要实现矩阵平方根计算，例如使用Cholesky分解
            // 简化实现：假设cov是对角矩阵
            np::Numcpp<T> sqrt_cov(n, n, (T)0);
            for (size_t i = 0; i < n; i++)
            {
                sqrt_cov[i][i] = std::sqrt(cov[i][i] * (n + lambda));
            }

            // 生成Sigma点
            for (size_t i = 0; i < n; i++)
            {
                np::Numcpp<T> point1 = mean + sqrt_cov[i];
                np::Numcpp<T> point2 = mean - sqrt_cov[i];
                sigma_points.push_back(point1);
                sigma_points.push_back(point2);
            }

            return sigma_points;
        }

        // 计算权重
        void calculateWeights(std::vector<T> &wm, std::vector<T> &wc)
        {
            wm.reserve(num_sigma_points);
            wc.reserve(num_sigma_points);

            T c = 0.5 / (n + lambda);
            wm.push_back(lambda / (n + lambda));
            wc.push_back(lambda / (n + lambda) + (1 - alpha * alpha + beta));

            for (size_t i = 1; i < num_sigma_points; i++)
            {
                wm.push_back(c);
                wc.push_back(c);
            }
        }

    public:
        ukf() = default;
        // 构造函数
        ukf(size_t state_dim, size_t measurement_dim,
            std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> state_transition_func,
            std::function<np::Numcpp<T>(const np::Numcpp<T> &)> observation_func,
            size_t control_dim = 0,
            T alpha = 1e-3, T beta = 2.0, T kappa = 0.0)
            : kf<T>(state_dim, measurement_dim),
              f(state_transition_func),
              h(observation_func),
              p(control_dim),
              alpha(alpha), beta(beta), kappa(kappa),
              num_sigma_points(2 * state_dim + 1)
        {
            lambda = alpha * alpha * (n + kappa) - n;
        }

        // 预测步骤（无控制输入）
        void predict() override
        {
            predict(np::Numcpp<T>(p, 1, (T)0));
        }

        // 预测步骤（有控制输入）
        void predict(const np::Numcpp<T> &u) override
        {
            if (u.row != p || u.col != 1)
            {
                throw std::invalid_argument("Control input dimension mismatch");
            }

            // 计算Sigma点
            std::vector<np::Numcpp<T>> sigma_points = calculateSigmaPoints(x, P);

            // 计算权重
            std::vector<T> wm, wc;
            calculateWeights(wm, wc);

            // 传播Sigma点通过状态转移函数
            std::vector<np::Numcpp<T>> propagated_sigma_points;
            propagated_sigma_points.reserve(num_sigma_points);
            for (const auto &point : sigma_points)
            {
                propagated_sigma_points.push_back(f(point, u));
            }

            // 计算预测状态均值
            np::Numcpp<T> x_pred(n, 1, (T)0);
            for (size_t i = 0; i < num_sigma_points; i++)
            {
                x_pred = x_pred + (propagated_sigma_points[i] * wm[i]);
            }

            // 计算预测状态协方差
            np::Numcpp<T> P_pred(n, n, (T)0);
            for (size_t i = 0; i < num_sigma_points; i++)
            {
                np::Numcpp<T> diff = propagated_sigma_points[i] - x_pred;
                P_pred = P_pred + (diff * diff.transpose() * wc[i]);
            }
            P_pred = P_pred + Q;

            // 更新状态和协方差
            x = x_pred;
            P = P_pred;
        }

        // 更新步骤
        void update(const np::Numcpp<T> &z) override
        {
            if (z.row != m || z.col != 1)
            {
                throw std::invalid_argument("Measurement dimension mismatch");
            }

            // 计算Sigma点
            std::vector<np::Numcpp<T>> sigma_points = calculateSigmaPoints(x, P);

            // 计算权重
            std::vector<T> wm, wc;
            calculateWeights(wm, wc);

            // 传播Sigma点通过观测函数
            std::vector<np::Numcpp<T>> measurement_sigma_points;
            measurement_sigma_points.reserve(num_sigma_points);
            for (const auto &point : sigma_points)
            {
                measurement_sigma_points.push_back(h(point));
            }

            // 计算预测观测均值
            np::Numcpp<T> z_pred(m, 1, (T)0);
            for (size_t i = 0; i < num_sigma_points; i++)
            {
                z_pred = z_pred + (measurement_sigma_points[i] * wm[i]);
            }

            // 计算观测协方差和互协方差
            np::Numcpp<T> P_zz(m, m, (T)0);
            np::Numcpp<T> P_xz(n, m, (T)0);
            for (size_t i = 0; i < num_sigma_points; i++)
            {
                np::Numcpp<T> z_diff = measurement_sigma_points[i] - z_pred;
                P_zz = P_zz + (z_diff * z_diff.transpose() * wc[i]);

                np::Numcpp<T> x_diff = sigma_points[i] - x;
                P_xz = P_xz + (x_diff * z_diff.transpose() * wc[i]);
            }
            P_zz = P_zz + R;

            // 计算卡尔曼增益
            np::Numcpp<T> K = P_xz * P_zz.inverse();

            // 更新状态估计
            np::Numcpp<T> y = z - z_pred;
            x = x + (K * y);

            // 更新协方差估计
            P = P - (K * P_zz * K.transpose());
        }

        // 获取控制输入维度
        size_t getControlDimension() const
        {
            return p;
        }
    };

    // 为向后兼容性定义别名
    template <typename T>
    using kfFilter = kfl<T>;

} // namespace kf

#endif // __KF__H__