#include <iostream>
#include <armadillo>
#include <cmath>
#include <span>
#include <functional>
#include <vector>
#include <limits>

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace mnfs {

using Function = std::function<double(std::span<const double>)>;
using Gradient = std::function<void(std::span<double>, std::span<const double>)>;
using Hessian = std::function<void(std::span<double>, std::span<const double>)>;

constexpr double eps = 1e-5;
constexpr double eps_inv = 1/1e-5;

auto num_grad(Function func) -> Gradient {
    return [=](std::span<double> out, std::span<const double> params) -> void {

        std::vector<double> tmp(params.begin(), params.end());
        for (std::size_t i = 0; i < params.size(); ++i) {
            double param = params[i];
            tmp[i] = param + eps;
            double upper = func(tmp);
            tmp[i] = param - eps;
            double lower = func(tmp);
            tmp[i] = param;
            out[i] = 0.5 * (upper - lower) * eps_inv;
        }
    };
}

auto num_hess(Gradient grad) -> Hessian {
    return [=](std::span<double> out, std::span<const double> params) -> void {
        std::vector<double> tmp(params.begin(), params.end());
        std::vector<double> grad_upper(params.size());
        std::vector<double> grad_lower(params.size());
        for (std::size_t i = 0; i < params.size(); ++i) {
            double param = params[i];
            tmp[i] = param + eps;
            grad(grad_upper, tmp);
            tmp[i] = param - eps;
            grad(grad_lower, tmp);
            tmp[i] = param;
            for (std::size_t j = 0; j < params.size(); ++j) {
                out[params.size() * i + j] = 0.5 * (grad_upper[j] - grad_lower[j]) * eps_inv;
            }
        }
    };
}

auto edm(Function func, std::span<const double> params) -> double {
    const auto func_grad = num_grad(func);
    const auto func_hess = num_hess(func_grad);

    std::vector<double> grad(params.size());
    std::vector<double> hess(params.size() * params.size());

    func_grad(grad, params);
    func_hess(hess, params);

    arma::vec params_vec{params.data(), params.size()};
    arma::vec grad_vec{grad.data(), params.size(), false};
    arma::mat hess_mat{hess.data(), params.size(), params.size(), false};

    return arma::mat{0.5 * grad_vec.t() * arma::inv(hess_mat) * grad_vec}(0, 0);
}

auto newton_stepper(Function func, double gamma) -> std::function<void(std::span<double> out, std::span<const double> params)> {
    const auto func_grad = num_grad(func);
    const auto func_hess = num_hess(func_grad);
    return [=](std::span<double> out, std::span<const double> params) -> void {
        std::vector<double> grad(params.size());
        std::vector<double> hess(params.size() * params.size());

        func_grad(grad, params);
        func_hess(hess, params);

        arma::vec params_vec{params.data(), params.size()};
        arma::vec out_vec{out.data(), out.size(), false};
        arma::vec grad_vec{grad.data(), params.size(), false};
        arma::mat hess_mat{hess.data(), params.size(), params.size(), false};

        arma::vec rhs = hess_mat * params_vec - gamma * grad_vec;

        solve(out_vec, hess_mat, rhs);
    };
}

} // namespace mnfs

double func(std::span<const double> params) {
    const double x = params[0];
    const double y = params[1];
    return x * x * x * x + y * y * y * y;
}

int main()
{
  std::vector<double> params{2.0, 3.0};

  auto newton_step = mnfs::newton_stepper(func, 1.0);

  double edm = mnfs::edm(func, params);

  fmt::println("init  : {}", params);
  for (int i = 0; i < 20; ++i) {
     newton_step(params, params);
     fmt::println("step {}: {}", i, params);
     edm = mnfs::edm(func, params);
     if (edm < 1e-12) break;
  }
  fmt::println("fval: {}", func(params));

  return 0;
}
