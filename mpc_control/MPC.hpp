#ifndef __MPC__H__
#define __MPC__H__

#include "../Numcpp/Numcpp.hpp"
#include <functional>
#include <vector>
#include <memory>

namespace mpc
{
    template <typename T>
    class MPC
    {
    private:
        // System mat
        np::Numcpp<T> A; // state
        np::Numcpp<T> B; // control
        np::Numcpp<T> C; // output

        // weight mat
        np::Numcpp<T> Q; // state
        np::Numcpp<T> R; // control

        // end control mat
        np::Numcpp<T> P;
        np::Numcpp<T> K;

        // constraint
        np::Numcpp<T> umin, umax; // control
        np::Numcpp<T> xmin, xmax; // state

        // args
        size_t predict_Horizon;
        size_t control_Horizon;
        size_t state_dim;
        size_t control_dim;
        size_t output_dim;

        // the weights of weights
        T wy;  // output
        T wu;  // control
        T wdu; // the differentiation of control

        // system function
        std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> system_dynamics;
        std::function<np::Numcpp<T>(const np::Numcpp<T> &)> output_function;

        np::Numcpp<T> H;  // Hessian
        np::Numcpp<T> F;  // graded
        np::Numcpp<T> Ac; // non-equal constraint mat
        np::Numcpp<T> bc; // non-equal constraint vector

        // usually a system didn't has the equal constraint
        np::Numcpp<T> Ec; // equal constraint mat
        np::Numcpp<T> dc; // equal constraint vector

        np::Numcpp<T> Xk;
        np::Numcpp<T> rk;

    public:
        // default creative function
        MPC(size_t state_dim, size_t control_dim, size_t output_dim,
            size_t predict_horizon, size_t control_horizon)
            : state_dim(state_dim), control_dim(control_dim), output_dim(output_dim),
              predict_Horizon(predict_horizon), control_Horizon(control_horizon),
              wy(1.0), wu(0.1), wdu(0.01)
        {
            initialize_matrices();
            set_default_system();
        }

        // A,B,C infer creative function
        MPC(const np::Numcpp<T> &A_mat, const np::Numcpp<T> &B_mat, const np::Numcpp<T> &C_mat,
            size_t predict_horizon, size_t control_horizon)
            : A(A_mat), B(B_mat), C(C_mat),
              predict_Horizon(predict_horizon), control_Horizon(control_horizon),
              wy(1.0), wu(0.1), wdu(0.01)
        {
            state_dim = A.row;
            control_dim = B.col;
            output_dim = C.row;
            initialize_matrices();
        }

        // destroy function
        ~MPC() = default;

    private:
        void initialize_matrices()
        {
            // init the identity mat
            Q = np::Numcpp<T>(state_dim, state_dim);
            R = np::Numcpp<T>(control_dim, control_dim);
            P = np::Numcpp<T>(state_dim, state_dim);

            // init the constraint mat
            umin = np::Numcpp<T>(control_dim, 1, -1.0);
            umax = np::Numcpp<T>(control_dim, 1, 1.0);
            xmin = np::Numcpp<T>(state_dim, 1, -10.0);
            xmax = np::Numcpp<T>(state_dim, 1, 10.0);

            // init state
            Xk = np::Numcpp<T>(state_dim, 1, 0.0);
            rk = np::Numcpp<T>(output_dim, predict_Horizon, 0.0);
        }

        void set_default_system()
        {
            // set the default System mat
            A = np::Numcpp<T>(state_dim, state_dim);
            B = np::Numcpp<T>(state_dim, control_dim);
            C = np::Numcpp<T>(output_dim, state_dim);

            A.set_identity();
            B.set_identity();
            C.set_identity();

            // default function
            system_dynamics = [this](const np::Numcpp<T> &x, const np::Numcpp<T> &u)
            {
                return A * x + B * u;
            };
            output_function = [this](const np::Numcpp<T> &x)
            {
                return C * x;
            };
        }

    public:
        void set_system_matrices(const np::Numcpp<T> &A_mat, const np::Numcpp<T> &B_mat, const np::Numcpp<T> &C_mat)
        {
            A = A_mat;
            B = B_mat;
            C = C_mat;
        }

        void set_weight_matrices(const np::Numcpp<T> &Q_mat, const np::Numcpp<T> &R_mat, const np::Numcpp<T> &P_mat)
        {
            Q = Q_mat;
            R = R_mat;
            P = P_mat;
        }

        void set_weights(T output_weight, T control_weight, T control_rate_weight)
        {
            wy = output_weight;
            wu = control_weight;
            wdu = control_rate_weight;
        }

        void set_constraints(const np::Numcpp<T> &u_min, const np::Numcpp<T> &u_max,
                             const np::Numcpp<T> &x_min, const np::Numcpp<T> &x_max)
        {
            umin = u_min;
            umax = u_max;
            xmin = x_min;
            xmax = x_max;
        }

        void set_system_dynamics(std::function<np::Numcpp<T>(const np::Numcpp<T> &, const np::Numcpp<T> &)> dynamics)
        {
            system_dynamics = dynamics;
        }

        void set_output_function(std::function<np::Numcpp<T>(const np::Numcpp<T> &)> output_func)
        {
            output_function = output_func;
        }

        void set_current_state(const np::Numcpp<T> &state)
        {
            Xk = state;
        }

        void set_reference_trajectory(const np::Numcpp<T> &reference)
        {
            rk = reference;
        }

        void set_reference_point(const np::Numcpp<T> &reference_point)
        {
            for (size_t i = 0; i < predict_Horizon; i++)
            {
                for (size_t j = 0; j < output_dim; j++)
                {
                    rk[j][i] = reference_point[j][0];
                }
            }
        }

        // single build a optimization problem with matrix system
        void build_optimization_problem()
        {
            size_t N = predict_Horizon;
            size_t Nu = control_Horizon;
            size_t nx = state_dim;
            size_t nu = control_dim;
            size_t ny = output_dim;

            np::Numcpp<T> Phi(nx * N, nx);
            np::Numcpp<T> Gamma(nx * N, nu * control_Horizon);

            // the power of mat A
            np::Numcpp<T> A_power = np::Numcpp<T>(nx, nx);
            A_power.set_identity();

            // reference: x1 = Phi * x0 + Gamma * U
            for (size_t i = 0; i < N; i++)
            {
                // build a Phi
                auto A_current = A_power;
                for (size_t j = 0; j < nx; j++)
                {
                    for (size_t k = 0; k < nx; k++)
                    {
                        Phi[i * nx + j][k] = A_current[j][k];
                    }
                }

                // build a Gamma mat
                for (size_t j = 0; j <= std::min(i, control_Horizon - 1); j++)
                {
                    auto A_gamma = A_power;
                    for (size_t k = 0; k < i - j; k++)
                    {
                        A_gamma = A * A_gamma;
                    }
                    auto temp = A_gamma * B;

                    for (size_t p = 0; p < nx; p++)
                    {
                        for (size_t q = 0; q < nu; q++)
                        {
                            Gamma[i * nx + p][j * nu + q] = temp[p][q];
                        }
                    }
                }

                A_power = A * A_power;
            }

            np::Numcpp<T> Q_bar(ny * N, ny * N);
            np::Numcpp<T> R_bar(nu * Nu, nu * Nu);
            np::Numcpp<T> S_bar(nu * Nu, nu * Nu);

            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = 0; j < ny; j++)
                {
                    for (size_t k = 0; k < ny; k++)
                    {
                        Q_bar[i * ny + j][i * ny + k] = wy * Q[j][k];
                    }
                }
            }

            for (size_t i = 0; i < Nu; i++)
            {
                for (size_t j = 0; j < nu; j++)
                {
                    for (size_t k = 0; k < nu; k++)
                    {
                        R_bar[i * nu + j][i * nu + k] = wu * R[j][k];
                    }
                }
            }

            // build a u'differentiation weight mat
            for (size_t i = 0; i < Nu - 1; i++)
            {
                for (size_t j = 0; j < nu; j++)
                {
                    S_bar[i * nu + j][i * nu + j] = wdu;
                    S_bar[i * nu + j][(i + 1) * nu + j] = -wdu;
                    S_bar[(i + 1) * nu + j][i * nu + j] = -wdu;
                    S_bar[(i + 1) * nu + j][(i + 1) * nu + j] = wdu;
                }
            }

            // H = Gamma^T * Q_bar * Gamma + R_bar + S_bar
            H = Gamma.transpose() * Q_bar * Gamma + R_bar + S_bar;

            np::Numcpp<T> free_response = Phi * Xk;

            // build trajectory vector of mat
            np::Numcpp<T> Y_ref(ny * N, 1);
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = 0; j < ny; j++)
                {
                    Y_ref[i * ny + j][0] = rk[j][i];
                }
            }

            // F = Gamma^T * Q_bar * (free_response - Y_ref)
            F = Gamma.transpose() * Q_bar * (free_response - Y_ref);

            // non-equal constraint mat
            Ac = np::Numcpp<T>(2 * nu * Nu, nu * Nu);
            bc = np::Numcpp<T>(2 * nu * Nu, 1);

            for (size_t i = 0; i < Nu; i++)
            {
                for (size_t j = 0; j < nu; j++)
                {
                    // max: u <= umax
                    Ac[i * nu + j][i * nu + j] = 1.0;
                    bc[i * nu + j][0] = umax[j][0];

                    // min: -u <= -umin
                    Ac[Nu * nu + i * nu + j][i * nu + j] = -1.0;
                    bc[Nu * nu + i * nu + j][0] = -umin[j][0];
                }
            }
            // equal constraint mat usually none
            Ec = np::Numcpp<T>(2 * nu * Nu, nu * Nu, 0);
            dc = np::Numcpp<T>(2 * nu * Nu, 1, 0);
        }

        // solve the QP problem, return a u0 to use
        np::Numcpp<T> solve()
        {
            build_optimization_problem();

            auto control_sequence = solve_QP();

            // return the first col of control_sequence which is u0
            np::Numcpp<T> u0(control_dim, 1);
            for (size_t i = 0; i < control_dim; i++)
            {
                u0[i][0] = control_sequence[i][0];
            }

            return u0;
        }

        // each step of mpc
        np::Numcpp<T> step(const np::Numcpp<T> &Xk, const np::Numcpp<T> &reference)
        {
            set_current_state(Xk);
            set_reference_trajectory(reference);
            return solve();
        }

        size_t get_prediction_horizon() const { return predict_Horizon; }
        size_t get_control_horizon() const { return control_Horizon; }

    private:
        // QP solver Gradient Descent Method (Newton's method)
        np::Numcpp<T> solve_QP()
        {
            try
            {
                size_t n = control_dim * control_Horizon;
                np::Numcpp<T> U(n, 1, 0.0); // control sequence

                // Simple gradient descent with projection
                T learning_rate = 0.001;
                int max_iterations = 1000;

                for (int iter = 0; iter < max_iterations; iter++)
                {
                    // Compute gradient: H*U + F
                    np::Numcpp<T> gradient = H * U + F;

                    // Gradient descent step
                    np::Numcpp<T> U_new = U - gradient * learning_rate;

                    // Project control constraints
                    for (size_t i = 0; i < control_Horizon; i++)
                    {
                        for (size_t j = 0; j < control_dim; j++)
                        {
                            size_t idx = i * control_dim + j;
                            // Control bounds
                            if (U_new[idx][0] < umin[j][0])
                                U_new[idx][0] = umin[j][0];
                            if (U_new[idx][0] > umax[j][0])
                                U_new[idx][0] = umax[j][0];
                        }
                    }

                    // Check convergence
                    T max_change = 0.0;
                    for (size_t i = 0; i < n; i++)
                    {
                        T change = std::abs(U_new[i][0] - U[i][0]);
                        if (change > max_change)
                            max_change = change;
                    }

                    U = U_new;

                    if (max_change < 1e-6)
                        break;
                }

                return U;
            }
            catch (const std::exception &e)
            {
                std::cerr << "QP solving failed: " << e.what() << std::endl;
                return np::Numcpp<T>(control_dim * control_Horizon, 1, 0.0);
            }
        }
    };

    // MPC_builder
    template <typename T>
    std::shared_ptr<MPC<T>> MPC_builder(
        const np::Numcpp<T> &A, const np::Numcpp<T> &B, const np::Numcpp<T> &C,
        size_t prediction_horizon, size_t control_horizon,
        const np::Numcpp<T> &Q, const np::Numcpp<T> &R, const np::Numcpp<T> &P)
    {
        auto mpc = std::make_shared<MPC<T>>(A, B, C, prediction_horizon, control_horizon);
        mpc->set_weight_matrices(Q, R, P);
        return mpc;
    }
} // namespace mpc

#endif // __MPC__H__