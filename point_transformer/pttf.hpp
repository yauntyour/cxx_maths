#ifndef __PTTF__H__
#define __PTTF__H__
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "Numcpp/Numcpp.hpp"

#define Point(__point__, Name_mat, x, y, z) \
    double __point__[1][3] = {{x, y, z}};   \
    np::Numcpp<double> Name_mat(1, 3);      \
    MATtoNumcpp(__point__, Name_mat, 1, 3);

namespace pttf
{
    std::map<std::string, np::Numcpp<double>> xyz_centers;
    void add_xyz(std::string xyz_C, np::Numcpp<double> point = np::Numcpp<double>(1, 3, 0))
    {
        if (point.row != 1 || point.col != 3)
        {
            throw std::invalid_argument("World center point must 1x3 matrix");
        }
        else
        {
            xyz_centers.insert(std::pair<std::string, np::Numcpp<double>>(xyz_C, point));
        }
    }
    /*
    pitch()：俯仰，将物体绕X轴旋转（localRotationX）
    yaw()：航向，将物体绕Y轴旋转（localRotationY）
    roll()：横滚，将物体绕Z轴旋转（localRotationZ）
    */
    np::Numcpp<double> rotate(double pitch, double yaw, double roll)
    {
        double mRx[3][3] = {
            {1, 0, 0},
            {0, cos(pitch), -sin(pitch)},
            {0, sin(pitch), cos(pitch)}};
        np::Numcpp<double> Rx(3, 3);
        MATtoNumcpp(mRx, Rx, 3, 3);
        double mRy[3][3] = {
            {cos(yaw), 0, sin(yaw)},
            {0, 1, 0},
            {-sin(yaw), 0, cos(yaw)}};
        np::Numcpp<double> Ry(3, 3);
        MATtoNumcpp(mRy, Ry, 3, 3);
        double mRz[3][3] = {
            {1, 0, 0},
            {0, cos(pitch), -sin(pitch)},
            {0, sin(pitch), cos(pitch)}};
        np::Numcpp<double> Rz(3, 3);
        MATtoNumcpp(mRz, Rz, 3, 3);
        return Rx * Ry * Rz;
    }
    np::Numcpp<double> translation(std::string origin, std::string trans)
    {
        return xyz_centers[trans] - xyz_centers[origin];
    }
    // pitch: X, yaw: Y, roll: Z
    np::Numcpp<double> transform(np::Numcpp<double> points, std::string xyz_origin, std::string xyz_trans, double pitch, double yaw, double roll)
    {
        if (points.col != 3)
        {
            throw std::invalid_argument("points is a Nx3");
        }
        return points * rotate(pitch, yaw, roll) + translation(xyz_origin, xyz_trans);
    }
};
#endif //!__PTTF__H__