#include "Numcpp.hpp"
#ifndef __QCNN__H__
#define __QCNN__H__
#include <vector>
#include <algorithm>
#include <functional>

namespace np
{
#define active_func_make(T, func_name) np::Numcpp<T> func_name(np::Numcpp<T> &A, np::Numcpp<T> &B)
#define active_lambda_make(T) [](np::Numcpp<T> & A, np::Numcpp<T> & B) -> np::Numcpp<T>
#define backward_func_make(T, func_name) void updata(std::vector<np::Numcpp<T>> &results, np::Numcpp<T> &loss, size_t offset)
#define backward_lambda_make(T) [](std::vector<np::Numcpp<T>> & results, np::Numcpp<T> & loss, size_t offset)

    /*
    active function: (np::Numcpp<T> &, np::Numcpp<T> &) -> np::Numcpp<T>
    */
    template <typename T>
    struct qcnn_layer
    {
        np::Numcpp<T> matrix;
        np::Numcpp<T> (*active)(np::Numcpp<T> &, np::Numcpp<T> &);
        void (*backward)(std::vector<np::Numcpp<T>> &results, np::Numcpp<T> &loss, size_t offset);
    };
    /*
    QCNN: Provide a neural network implemented with matrices and processors.
    Each layer contains a matrix and an activation function.
    The arithmetic() function will compute the result of the QCNN.
    The operator between two layers is defined as matrix multiplication followed by the activation function.
    */
    template <typename T>
    class qcnn
    {
    public:
        std::vector<qcnn_layer<T>> layers;
        std::vector<np::Numcpp<T>> results;
        qcnn(std::vector<qcnn_layer<T>> layers)
        {
            this->layers = layers;
        };
        ~qcnn()
        {
            layers.clear();
        };
        np::Numcpp<T> arithmetic(np::Numcpp<T> input)
        {

            for (size_t i = 0; i < layers.size(); i++)
            {
                if (i == 0)
                {
                    results.push_back(layers[i].active(input, layers[i].matrix));
                }
                else
                {
                    results.push_back(layers[i].active(results[i - 1], layers[i].matrix));
                }
            }
            return results[results.size() - 1];
        }
        np::Numcpp<T> loss_squ(np::Numcpp<T> validation)
        {
            return (results[results.size() - 1] - validation)<[](T x, T y) -> T
                                                              { return x * x; }>
                NULL;
        }
        np::Numcpp<T> loss(np::Numcpp<T> validation)
        {
            return (results[results.size() - 1]) - validation;
        }
        void updata(np::Numcpp<T> loss)
        {
            size_t i = 0;
            for (qcnn_layer<T> &layer : layers)
            {
                if (layer.backward != NULL)
                {
                    layer.backward(results, loss, i);
                    i += 1;
                }
            }
        }
    };
} // namespace qcnn

#endif //!__QCNN__H__