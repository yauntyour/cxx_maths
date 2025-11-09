#include <vector>

#include "Numcpp/Numcpp.hpp"

class MM_pred
{
private:
    float fps = 0;
    float dt = 0;

    np::Numcpp<float> X;
    np::Numcpp<float> P;
    np::Numcpp<float> Z;

public:
    bool initialized = false;
    MM_pred(size_t dimension, float dt = 0.1) : dt(dt)
    {
        X = np::Numcpp<float>(dimension, 1, 0.0f);
        P = np::Numcpp<float>(dimension, 1, 0.0f);
        Z = np::Numcpp<float>(dimension, 1, 0.0f);
    };
    void init(const std::vector<float> &init_state)
    {
        if (initialized == false)
        {
            if (init_state.size() != X.row)
                throw std::invalid_argument("Initial state size does not match state dimension");
            for (size_t i = 0; i < init_state.size(); i++)
            {
                X[i][0] = init_state[i];
            }
        }
    };
    void update(const std::vector<float> &meas, float fps)
    {
        if (meas.size() != X.row)
            throw std::invalid_argument("Measurement size does not match state dimension");
        for (size_t i = 0; i < meas.size(); i++)
        {
            Z[i][0] = meas[i];
        }
        this->fps = fps;
        if (fps > 0)
            dt = 1.0f / fps;
        np::Numcpp<float> dx = Z - X;
        this->P = dx / dt;
        this->X = Z;
    };
    void reset()
    {
        X = np::Numcpp<float>(X.row, 1, 0.0f);
        P = np::Numcpp<float>(P.row, 1, 0.0f);
        Z = np::Numcpp<float>(Z.row, 1, 0.0f);
    }
    np::Numcpp<float> predict(float t) const
    {
        return X + P * t;
    };
};
