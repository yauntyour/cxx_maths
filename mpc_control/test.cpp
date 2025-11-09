// simple_mpc_test.cpp
#include "MPC.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// 简单的质量-弹簧-阻尼系统
class SimpleSystem
{
public:
    // 状态: [位置, 速度]
    // 控制: [力]
    np::Numcpp<double> dynamics(const np::Numcpp<double> &state,
                                const np::Numcpp<double> &control)
    {
        double m = 1.0;  // 质量
        double k = 0.5;  // 弹簧系数
        double c = 0.1;  // 阻尼系数
        double dt = 0.1; // 采样时间

        double x = state[0][0];
        double v = state[1][0];
        double F = control[0][0];

        // 连续时间动力学: m*a + c*v + k*x = F
        // 离散化: x(k+1) = x(k) + v(k)*dt
        //         v(k+1) = v(k) + (F - k*x(k) - c*v(k))/m * dt

        double a = (F - k * x - c * v) / m;

        np::Numcpp<double> new_state(2, 1);
        new_state[0][0] = x + v * dt;
        new_state[1][0] = v + a * dt;

        return new_state;
    }
};

int main()
{
    std::cout << "=== Simple MPC Test ===" << std::endl;

    // 系统参数
    size_t state_dim = 2;   // [位置, 速度]
    size_t control_dim = 1; // [力]
    size_t output_dim = 2;  // 全状态输出
    size_t prediction_horizon = 10;
    size_t control_horizon = 5;

    // 创建MPC控制器
    mpc::MPC<double> controller(state_dim, control_dim, output_dim,
                                prediction_horizon, control_horizon);

    // 设置系统矩阵 (线性化的质量-弹簧-阻尼系统)
    np::Numcpp<double> A(2, 2);
    A << 1.0, 0.1,   // x(k+1) = x(k) + 0.1*v(k)
        -0.05, 0.99; // v(k+1) = v(k) - 0.05*x(k) + 0.99*v(k)

    np::Numcpp<double> B(2, 1);
    B << 0.005, // 位置受力的影响
        0.1;    // 速度受力的影响

    np::Numcpp<double> C(2, 2);
    C.set_identity();

    controller.set_system_matrices(A, B, C);

    // 设置权重矩阵
    np::Numcpp<double> Q(2, 2);
    Q.set_identity();
    Q[0][0] = 10.0; // 位置误差权重
    Q[1][1] = 1.0;  // 速度误差权重

    np::Numcpp<double> R(1, 1);
    R[0][0] = 0.1; // 控制力权重

    np::Numcpp<double> P(2, 2);
    P.set_identity();

    controller.set_weight_matrices(Q, R, P);

    // 设置约束
    np::Numcpp<double> umin(1, 1, -2.0); // 最小控制力
    np::Numcpp<double> umax(1, 1, 2.0);  // 最大控制力
    np::Numcpp<double> xmin(2, 1, -5.0); // 状态下限
    np::Numcpp<double> xmax(2, 1, 5.0);  // 状态上限

    controller.set_constraints(umin, umax, xmin, xmax);

    // 初始状态
    np::Numcpp<double> state(2, 1);
    state << 0.0, // 初始位置
        0.0;      // 初始速度

    // 参考位置 (阶跃信号)
    double target_position = 2.0;

    // 模拟参数
    int total_steps = 100;

    std::cout << "Starting MPC control..." << std::endl;
    std::cout << "Target position: " << target_position << std::endl;
    std::cout << std::endl;

    // 存储结果用于输出
    std::vector<double> positions;
    std::vector<double> velocities;
    std::vector<double> controls;
    std::vector<int> steps;

    for (int step = 0; step < total_steps; step++)
    {
        // 创建参考轨迹 (所有预测步长都指向目标)
        np::Numcpp<double> reference(2, prediction_horizon);
        for (size_t i = 0; i < prediction_horizon; i++)
        {
            reference[0][i] = target_position; // 目标位置
            reference[1][i] = 0.0;             // 目标速度为零
        }

        // MPC控制计算
        auto control = controller.step(state, reference);

        // 应用控制 (使用真实系统动力学)
        SimpleSystem system;
        state = system.dynamics(state, control);

        // 存储结果
        positions.push_back(state[0][0]);
        velocities.push_back(state[1][0]);
        controls.push_back(control[0][0]);
        steps.push_back(step);

        // 输出当前状态
        if (step % 10 == 0)
        {
            std::cout << "Step " << step
                      << ": Position = " << state[0][0]
                      << ", Velocity = " << state[1][0]
                      << ", Control = " << control[0][0] << std::endl;
        }
    }

    // 输出最终结果
    std::cout << std::endl;
    std::cout << "=== Final Results ===" << std::endl;
    std::cout << "Final position: " << state[0][0] << " (target: " << target_position << ")" << std::endl;
    std::cout << "Final velocity: " << state[1][0] << " (target: 0.0)" << std::endl;
    std::cout << "Steady-state error: " << std::abs(state[0][0] - target_position) << std::endl;

    // 简单的文本可视化
    std::cout << std::endl;
    std::cout << "=== Text Visualization ===" << std::endl;
    std::cout << "Position trajectory:" << std::endl;

    for (int i = 0; i < total_steps; i += 5)
    {
        double pos = positions[i];
        int bar_length = static_cast<int>((pos / target_position) * 20);
        bar_length = std::max(0, std::min(20, bar_length));

        std::cout << "Step " << i << ": [";
        for (int j = 0; j < bar_length; j++)
        {
            std::cout << "=";
        }
        for (int j = bar_length; j < 20; j++)
        {
            std::cout << " ";
        }
        std::cout << "] " << pos << std::endl;
    }

    return 0;
}