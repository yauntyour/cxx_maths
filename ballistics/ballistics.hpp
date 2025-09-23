#ifndef __BALLISTICS__H__
#define __BALLISTICS__H__

#include <cmath>
#include "Numcpp/Numcpp.hpp"

class ballistics
{
public:
    // 弹道系数
    double s = 0;
    // 阻力系数
    double k = 0;
    // 质量
    double m = 0;

    // 初末位置
    double x0 = 0;
    double y0 = 0;
    double z0 = 0;
    double x = 0;
    double y = 0;
    double z = 0;

    ballistics( // 弹道系数
        double x0,
        double y0,
        double z0,
        double x,
        double y,
        double z,
        double s = 0.1,
        double k = 0.1,
        double m = 1)
        : s(s),
          k(k), m(m), x0(x0), y0(y0), z0(z0), x(x), y(y), z(z)
    {
    }

    // 位移
    double dx = 0;
    double dy = 0;
    double dz = 0;

    // 飞行时间
    double tf;

    // 加速度
    double g = 9.8;

    // 发射角，发射动量
    double pitch;
    double yaw;
    double p;

    // 角度制
    double angle_pitch;
    double angle_yaw;
    // 计算飞行时间
    double ball_tf()
    {
        tf = -(m / k) * log(1 - s);
        return tf;
    }
    void ball_distance()
    {
        dx = x - x0;
        dy = y - y0;
        dz = z - z0;
    }
    void ball_calculate()
    {
        ball_distance();
        double r = sqrt(dx * dx + dy * dy);

        double p_xy = (k * r / s);
        double p_z = (k * dz / s) - ((g * m * m) / k) * ((s + log(1 - s)) / s);

        pitch = atan(p_z / p_xy);
        if (dx < 1e-6)
        {
            yaw = NP_PI / 2;
        }
        else
        {
            yaw = atan(dy / dx);
        }
        angle_pitch = pitch / (2 * NP_PI) * 360;
        angle_yaw = yaw / (2 * NP_PI) * 360;
    }
};
#endif //!__BALLISTICS__H__