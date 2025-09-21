#include "qcnn.hpp"

// 定义更新权重的函数
// void updata(std::vector<np::Numcpp<double>> &results, np::Numcpp<double> &loss, size_t offset)
backward_func_make(double, updata)
{
    std::cout << "Updata[" << offset << "]:" << loss << results[offset];
}

int main(int argc, char const *argv[])
{
    std::vector<qcnn_layer<double>> list = {
        {w_1, [](Numcpp<double> &A, Numcpp<double> &B) -> Numcpp<double>
         {
             return A * B;
         },
         NULL},
        // 使用快捷宏创建lambda表达式
        {b_1, (active_lambda_make(double) {
             return A + B;
         }),
         updata}};
    qcnn<double> qc(list);
    std::cout << "arithmetic result: " << qc.arithmetic(input) << std::endl;
    auto loss = qc.loss(val);
    qc.updata(loss);
    auto s_loss = qc.loss_squ(val);
    std::cout << "s_loss: " << s_loss;
    return 0;
}
