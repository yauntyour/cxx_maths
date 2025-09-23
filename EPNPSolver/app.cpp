// 示例使用代码
#include <iostream>
#include <cmath>
#include "EPNPSolver.hpp"

int main()
{
    // 相机内参矩阵
    np::Numcpp<double> K(3, 3);
    K << 800, 0, 320,
        0, 800, 240,
        0, 0, 1;

    // 3D-2D点对应
    std::vector<std::vector<double>> points3D = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}};
    std::vector<std::vector<double>> points2D = {
        {320, 240}, {400, 240}, {320, 160}, {400, 160}, {320, 320}};

    // 求解位姿
    np::Numcpp<double> R, t;
    if (epnp::solveEPNP(points3D, points2D, K, R, t))
    {
        std::cout << "Fail to get the R,t\n"
                  << std::endl;
    }
    std::cout << "Rotation Matrix:\n"
              << R << std::endl;
    std::cout << "Translation Vector:\n"
              << t << std::endl;
    std::cout << "Camera position in world" << R.transpose() * t * -1 << std::endl;

    double roll;
    double yaw;
    double pitch;
    if (std::abs(R[0][2]) > 0.9999)
    {
        roll = 0;
        if (R[0][2] < 0)
        {
            pitch = NP_PI / 2;
            yaw = atan2(R[1][0], R[2][0]);
        }
        else
        {
            pitch = -NP_PI / 2;
            yaw = atan2(-R[1][0], -R[2][0]);
        }
    }
    else
    {
        roll = atan2(-R[1][2], R[2][2]);
        pitch = asin(R[0][2]);
        yaw = atan2(-R[0][1], R[0][0]);
    }
    np::Numcpp<double> OL_mat = (np::Numcpp<double>(3, 1) << roll, pitch, yaw);

    OL_mat /= (2 * NP_PI);
    OL_mat *= 360;

    std::cout << "XYZ: " << OL_mat << std::endl;
    return 0;
}