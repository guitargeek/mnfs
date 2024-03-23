#include <mnfs/derivative.h>

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
constexpr double eps_inv = 1 / 1e-5;

constexpr double eps_machine = 4. * std::numeric_limits<double>::epsilon();
constexpr double eps_machine_2 = 2. * std::sqrt(eps_machine);

void get_gradient(std::span<double> out, Function func, std::span<const double> params,
                  std::span<const double> param_errors)
{
   std::vector<double> xVals{params.begin(), params.end()};
   const double centralVal = func(params);

   struct GradientValue {
      std::vector<double> grad;
      std::vector<double> g2;
      std::vector<double> step;
   };

   GradientValue info;
   info.grad.resize(params.size());
   info.g2.resize(params.size());
   info.step.resize(params.size());

   for (unsigned int i = 0; i < params.size(); i++) {

      double error_def = 1.0;

      // TODO: when we support parameters with limits, this logic needs to be
      // updated such that the errors are clipped by the limits if the
      // parameter is near it (see InitialGradientCalculator).
      double vplu = param_errors[i];
      double vmin = param_errors[i];

      double gsmin = 8. * eps_machine_2 * (std::abs(xVals[i]) + eps_machine_2);
      // protect against very small step sizes which can cause dirin to zero and then nan values in grd
      double dirin = std::max(0.5 * (std::abs(vplu) + std::abs(vmin)), gsmin);
      double g2 = 2.0 * error_def / (dirin * dirin);
      double gstep = std::max(gsmin, 0.1 * dirin);
      double grd = g2 * dirin;

      auto funcForParam = [&](double x) -> double {
         if (xVals[i] == x) {
            return centralVal;
         }
         double tmp = xVals[i];
         xVals[i] = x;
         const double out = func(xVals);
         xVals[i] = tmp;
         return out;
      };

      mnfs::DerivativeState state{grd, g2, gstep};
      state = mnfs::update_derivative(state, funcForParam, params[i]);
      info.grad[i] = state.grad;
      info.g2[i] = state.g2;
      info.step[i] = state.step;
   }

   for (std::size_t i = 0; i < params.size(); ++i) {
      out[i] = info.grad[i];
   }
}

auto num_grad(Function func, std::span<const double> param_errors) -> Gradient
{
   return [=](std::span<double> out, std::span<const double> params) -> void {
      get_gradient(out, func, params, param_errors);
   };
}

auto num_hess(Gradient grad) -> Hessian
{
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

auto edm(Function func, Gradient func_grad, std::span<const double> params) -> double
{
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

auto newton_stepper(Function func, Gradient func_grad, double gamma)
   -> std::function<void(std::span<double> out, std::span<const double> params)>
{
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

double func(std::span<const double> params)
{
   const double x = params[0];
   const double y = params[1];
   return x * x * x * x + y * y * y * y;
}

int main()
{
   std::vector<double> params{2.0, 3.0};
   std::vector<double> param_errors{0.1, 0.1};

   const auto func_grad = mnfs::num_grad(func, param_errors);

   auto newton_step = mnfs::newton_stepper(func, func_grad, 1.0);

   double edm = mnfs::edm(func, func_grad, params);

   fmt::println("init  : {}", params);
   for (int i = 0; i < 20; ++i) {
      newton_step(params, params);
      fmt::println("step {}: {}", i, params);
      edm = mnfs::edm(func, func_grad, params);
      if (edm < 1e-12)
         break;
   }
   fmt::println("fval: {}", func(params));

   return 0;
}
