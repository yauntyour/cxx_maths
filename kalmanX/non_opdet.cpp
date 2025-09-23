#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "kf.hpp"

using namespace std;
using namespace cv;
using namespace kf;
using namespace np;

#define URL "rtsp://192.168.121.155:8080/h264.sdp"

// 非线性运动模型
void rand_move(Point2d &p)
{
    p.x = p.x + sin(p.y);
    p.y = p.x * p.x;
}

int main(int argc, char const *argv[])
{
    Mat bg(800, 800, CV_8UC3);
    Point2d true_position(10, 10);
    Point2d measured_position;

    // 初始化EKF
    Numcpp<double> X0(2, 1, 0.0); // 初始状态 [x, y]
    X0[0][0] = 10.0;
    X0[1][0] = 10.0;

    Numcpp<double> P0(2, 2, 0.0); // 初始协方差
    P0[0][0] = 1.0;               // x方差
    P0[1][1] = 1.0;               // y方差

    Numcpp<double> Q(2, 2, 0.0); // 过程噪声
    Q[0][0] = 0.1;
    Q[1][1] = 0.1;

    Numcpp<double> R(2, 2, 0.0); // 观测噪声
    R[0][0] = 1.0;
    R[1][1] = 1.0;

    // 定义状态转移函数
    auto state_transition = [](const Numcpp<double> &x, const Numcpp<double> &u)
    {
        Numcpp<double> new_x(2, 1, 0.0);
        new_x[0][0] = x[0][0] + sin(x[1][0]); // x + sin(y)
        new_x[1][0] = x[0][0] * x[0][0];      // x^2
        return new_x;
    };

    // 定义观测函数
    auto observation = [](const Numcpp<double> &x)
    {
        Numcpp<double> z(2, 1, 0.0);
        z[0][0] = x[0][0]; // 直接观测x
        z[1][0] = x[1][0]; // 直接观测y
        return z;
    };

    // 定义状态转移雅可比矩阵
    auto state_jacobian = [](const Numcpp<double> &x, const Numcpp<double> &u)
    {
        Numcpp<double> F(2, 2, 0.0);
        F[0][0] = 1.0;          // ∂f1/∂x = 1
        F[0][1] = cos(x[1][0]); // ∂f1/∂y = cos(y)
        F[1][0] = 2 * x[0][0];  // ∂f2/∂x = 2x
        F[1][1] = 0.0;          // ∂f2/∂y = 0
        return F;
    };

    // 定义观测雅可比矩阵
    auto observation_jacobian = [](const Numcpp<double> &x)
    {
        Numcpp<double> H(2, 2, 0.0);
        H[0][0] = 1.0; // ∂h1/∂x = 1
        H[0][1] = 0.0; // ∂h1/∂y = 0
        H[1][0] = 0.0; // ∂h2/∂x = 0
        H[1][1] = 1.0; // ∂h2/∂y = 1
        return H;
    };

    // 创建EKF
    ekf<double> kfer(2, 2, state_transition, observation,
                     state_jacobian, observation_jacobian);

    // 初始化EKF
    kfer.setInitialState(X0);
    kfer.setInitialCovariance(P0);
    kfer.setProcessNoiseCovariance(Q);
    kfer.setObservationNoiseCovariance(R);

    while (true)
    {
        // 清空画布
        bg = Scalar(0, 0, 0);

        // 真实运动
        rand_move(true_position);

        // 添加噪声的测量值
        measured_position.x = true_position.x + (rand() % 100 - 50) / 50.0;
        measured_position.y = true_position.y + (rand() % 100 - 50) / 50.0;

        // 创建测量向量
        Numcpp<double> z(2, 1, 0.0);
        z[0][0] = measured_position.x;
        z[1][0] = measured_position.y;

        // EKF预测
        kfer.predict();

        // EKF更新
        kfer.update(z);

        // 获取估计状态
        Numcpp<double> estimated_state = kfer.getState();
        Point2d estimated_position(estimated_state[0][0], estimated_state[1][0]);

        // 绘制
        circle(bg, Point2d(true_position.x + 400, true_position.y + 400), 5, Scalar(0, 255, 0), -1);           // 绿色：真实位置
        circle(bg, Point2d(measured_position.x + 400, measured_position.y + 400), 5, Scalar(0, 0, 255), -1);   // 红色：测量位置
        circle(bg, Point2d(estimated_position.x + 400, estimated_position.y + 400), 5, Scalar(255, 0, 0), -1); // 蓝色：估计位置

        // 显示坐标
        printf("True: (%.2f, %.2f), Measured: (%.2f, %.2f), Estimated: (%.2f, %.2f)\n",
               true_position.x, true_position.y,
               measured_position.x, measured_position.y,
               estimated_position.x, estimated_position.y);

        imshow("EKF Tracking", bg);
        if (waitKey(100) == 27) // 按ESC退出
        {
            break;
        }
    }
    return 0;
}