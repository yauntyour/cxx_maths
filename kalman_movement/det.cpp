#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "kalman_filter.hpp"

using namespace std;
using namespace cv;

vector<Vec3f> cv_circle_find(Mat &src)
{
    Mat img;
    // CV det
    cvtColor(src, img, COLOR_BGR2GRAY);
    /*Mat kernel1 = (Mat_<float>(3, 3) << 0, -1, 0,
                   -1, 5, -1,
                   0, -1, 0); // 锐化卷积核
    filter2D(img, img, -1, kernel1);*/

    GaussianBlur(img, img, Size(5, 5), 0);

    adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 5, 2);
    vector<Vec3f> circles;
    HoughCircles(img, circles, HOUGH_GRADIENT, 1,
                 img.rows / 16,  // 更改此值以检测彼此距离不同的圆
                 100, 30, 20, 50 // 更改最后两个参数
                                 // (min_radius & max_radius) 以检测更大的圆
    );
    for (size_t i = 0; i < circles.size(); i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // 圆心
        circle(src, center, 1, Scalar(255, 0, 0), 3, LINE_AA);
        // 圆周
        int radius = c[2];
        circle(src, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
    }
    return circles;
}

Point2f hsv_circle_find(Mat &src)
{
    Mat hsv_img;
    cvtColor(src, hsv_img, COLOR_BGR2HSV);

    Scalar lower_red(0, 120, 70);
    Scalar upper_red(10, 255, 255);

    Mat mask;
    inRange(hsv_img, lower_red, upper_red, mask);

    Mat bg = Mat::zeros(src.size(), src.type());

    // replace to blue
    src.copyTo(bg, mask);
    bg.setTo(Scalar(255, 0, 0), mask);

    Mat res;
    cvtColor(bg, res, COLOR_BGR2GRAY);
    std::vector<std::vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(res, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Point center(0, 0);
    Point2f rect[4];
    for (auto &cnt : contours)
    {
        if (contourArea(cnt) > 200)
        {
            RotatedRect box = minAreaRect(cnt); // 计算每个轮廓最小外接矩形
            Rect boundRect = boundingRect(cnt);
            circle(src, Point(box.center.x, box.center.y), 5, Scalar(255, 0, 0), -1, 8); // 绘制最小外接矩形的中心点
            center = Point(box.center.x, box.center.y);
            rectangle(src, boundRect, Scalar(0, 0, 255), 2, 8);
            // rectangle(src, Point(boundRect.x, boundRect.y), Point(boundRect.x + boundRect.width, boundRect.y + boundRect.height), Scalar(0, 255, 0), 2, 8);
            break;
        }
    }
    return center;
}

int main(int argc, char const *argv[])
{
    VideoCapture capture("D:\\dev\\cxx_kalman\\ds\\kun.mp4");
    if (!capture.isOpened())
    {
        std::cerr << "Error: Could not open video." << std::endl;
        return -1;
    }

    int width = capture.get(CAP_PROP_FRAME_WIDTH);
    int height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int frameCount = capture.get(CAP_PROP_FRAME_COUNT);
    double fps = capture.get(CAP_PROP_FPS);

    printf("Width:%d,Height:%d,Total_FPS:%d,FPS:%f\n", width, height, frameCount, fps);

    //[x, y, vx, vy, ax, ay]
    kalman::KalmanFilter<double> kf(6, 2);

    np::Numcpp<double> F(6, 6, 0.0);
    double dt = 1.0 / fps;
    double g = -9.8;
    double a = 0.5;
    double F_mat[6][6] = {
        {1, 0, dt, 0, 0.5 * a * dt * dt, 0},
        {0, 1, 0, dt, 0, 0.5 * g * dt * dt},
        {0, 0, 1, 0, dt, 0},
        {0, 0, 0, 1, 0, dt},
        {0, 0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0, 1}};

    MATtoNumcpp(F_mat, F, F.row, F.col);
    kf.setTransitionMatrix(F);

    np::Numcpp<double> H(2, 6, 0.0);
    H[0][0] = 1;
    H[1][1] = 1;
    kf.setObservationMatrix(H);

    double Qmat[6][6] = {
        {0.05, 0, 0, 0, 0, 0},
        {0, 0.05, 0, 0, 0, 0},
        {0, 0, 0.05, 0, 0, 0},
        {0, 0, 0, 0.05, 0, 0},
        {0, 0, 0, 0, 0.01, 0},
        {0, 0, 0, 0, 0, 0.01}};
    np::Numcpp<double> Q(6, 6);
    MATtoNumcpp(Qmat, Q, Q.row, Q.col);
    Q *= 100;
    kf.setProcessNoiseCovariance(Q);

    np::Numcpp<double> R(2, 2, 0.0);
    R[0][0] = 0.01;
    R[1][1] = 0.01;
    kf.setObservationNoiseCovariance(R);

    Mat frame;
    bool first_detection = true;

    int lost_count = 0;
    const int max_lost = 30;

    while (true)
    {
        capture >> frame;
        if (frame.empty())
            break;

        Point2f detected_center = hsv_circle_find(frame);
        bool detected = (detected_center.x >= 0 && detected_center.y >= 0);

        if (detected)
        {
            lost_count = 0;
            if (first_detection)
            {
                np::Numcpp<double> newState(6, 1, 0.0);
                newState[0][0] = detected_center.x;
                newState[1][0] = detected_center.y;
                kf.setState(newState);
                first_detection = false;
            }
            else
            {
                np::Numcpp<double> z(2, 1, 0.0);
                z[0][0] = detected_center.x;
                z[1][0] = detected_center.y;
                kf.update(z);
            }
        }
        else
        {
            lost_count++;
            if (lost_count >= max_lost)
            {
                first_detection = true; // 重置滤波器
                lost_count = 0;
            }
        }

        kf.predict();
        np::Numcpp<double> state = kf.getState();
        double pred_x = state[0][0];
        double pred_y = state[1][0];
        circle(frame, Point(pred_x, pred_y), 50, Scalar(0, 255, 0), 3);

        imshow("Tracking", frame);
        if (waitKey(10) == 27)
            break; // 按ESC退出
    }

    return 0;
}