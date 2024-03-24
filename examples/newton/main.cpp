#include <mnfs.hpp>

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
using Gradient = std::function<void(mnfs::GradientValue &, std::span<const double>)>;
using Hessian = std::function<void(std::span<double>, std::span<const double>)>;

constexpr double eps = 1e-5;
constexpr double eps_inv = 1 / 1e-5;

constexpr double eps_machine = 4. * std::numeric_limits<double>::epsilon();
constexpr double eps_machine_2 = 2. * std::sqrt(eps_machine);

void get_gradient(mnfs::GradientValue &out, Function func, std::span<const double> params,
                  std::span<const double> param_errors)
{
   std::vector<double> xVals{params.begin(), params.end()};
   const double centralVal = func(params);

   bool do_init = out.grad.empty();

   if (do_init) {
      out.grad.resize(params.size());
      out.g2.resize(params.size());
      out.step.resize(params.size());
   }

   for (unsigned int i = 0; i < params.size(); i++) {

      if (do_init) {
         double error_def = 1.0;

         // TODO: when we support parameters with limits, this logic needs to be
         // updated such that the errors are clipped by the limits if the
         // parameter is near it (see InitialGradientCalculator).
         double vplu = param_errors[i];
         double vmin = param_errors[i];

         double gsmin = 8. * eps_machine_2 * (std::abs(xVals[i]) + eps_machine_2);
         // protect against very small step sizes which can cause dirin to zero and then nan values in grd
         double dirin = std::max(0.5 * (std::abs(vplu) + std::abs(vmin)), gsmin);
         out.g2[i] = 2.0 * error_def / (dirin * dirin);
         out.step[i] = std::max(gsmin, 0.1 * dirin);
         out.grad[i] = out.g2[i] * dirin;
      }

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

      mnfs::DerivativeState state{out.grad[i], out.g2[i], out.step[i]};
      state = mnfs::update_derivative(state, funcForParam, params[i]);
      out.grad[i] = state.grad;
      out.g2[i] = state.g2;
      out.step[i] = state.step;
   }
}

auto num_grad(Function func, std::span<const double> param_errors) -> Gradient
{
   return [=](mnfs::GradientValue &out, std::span<const double> params) -> void {
      get_gradient(out, func, params, param_errors);
   };
}

auto num_hess(Gradient grad) -> Hessian
{
   return [=](std::span<double> out, std::span<const double> params) -> void {
      std::vector<double> tmp(params.begin(), params.end());
      mnfs::GradientValue grad_upper;
      mnfs::GradientValue grad_lower;
      for (std::size_t i = 0; i < params.size(); ++i) {
         double param = params[i];
         tmp[i] = param + eps;
         grad(grad_upper, tmp);
         tmp[i] = param - eps;
         grad(grad_lower, tmp);
         tmp[i] = param;
         for (std::size_t j = 0; j < params.size(); ++j) {
            out[params.size() * i + j] = 0.5 * (grad_upper.grad[j] - grad_lower.grad[j]) * eps_inv;
         }
      }
   };
}

auto edm(std::span<const double> grad, std::span<const double> inv_hess) -> double
{
   arma::vec grad_vec{grad.data(), grad.size()};
   arma::mat inv_hess_mat{inv_hess.data(), grad.size(), grad.size()};

   return arma::mat{0.5 * grad_vec.t() * inv_hess_mat * grad_vec}(0, 0);
}

void newton_step(std::span<double> out, std::span<const double> grad, std::span<const double> inv_hess)
{
   arma::vec grad_vec{grad.data(), grad.size()};
   arma::mat inv_hess_mat{inv_hess.data(), grad.size(), grad.size()};

   arma::vec out_vec{-inv_hess_mat * grad_vec};
   for (std::size_t i = 0; i < grad.size(); ++i) {
      out[i] = out_vec(i);
   }
}

double gdel(std::span<const double> step, std::span<const double> grad)
{
   arma::vec step_vec{step.data(), step.size()};
   arma::vec grad_vec{grad.data(), grad.size()};

   return arma::vec{step_vec.t() * grad_vec}(0);
}

void update_inv_hessian(std::span<double> inv_hess, std::span<const double> step, std::span<const double> dgrad)
{
   arma::vec s{step.data(), step.size()};
   arma::vec y{dgrad.data(), dgrad.size()};
   arma::mat H{inv_hess.data(), step.size(), step.size()};

   // update of the covarianze matrix (Davidon formula, see Tutorial, par. 4.8 pag 26)
   // in case of delgam > gvg (PHI > 1) use rank one formula
   // see  par 4.10 pag 30
   // ( Tutorial: https://seal.web.cern.ch/seal/documents/minuit/mntutorial.pdf )
   arma::mat tmp{s - H*y};
   arma::mat davidon_update = (tmp * tmp.t()) / arma::mat{y.t() * tmp}(0,0);

   double sty = arma::vec{s.t() * y}(0);
   double sty_inv = 1./sty;
   double ytHy = arma::mat{y.t() * H * y}(0, 0);
   arma::mat bfgs_update = (sty + ytHy)*(s*s.t()) * sty_inv * sty_inv - (H*y*s.t() + s*y.t()*H) * sty_inv;

   //H = H + davidon_update + bfgs_update;
   H = (H + bfgs_update);

   for(std::size_t i = 0; i < inv_hess.size(); ++i) {
      inv_hess[i] = H.memptr()[i];
   }
}

} // namespace mnfs

double func(std::span<const double> params)
{
   const double x = params[0];
   const double y = params[1];
   return x * x * x * x + 2 * y * y * y * y;
}

int main()
{
   std::vector<double> params{5., 6.0};
   std::vector<double> param_errors{0.1, 0.1};
   std::vector<double> hess_value(params.size() * params.size());

   std::vector<double> step{0.0, 0.0};
   std::vector<double> dgrad{0.0, 0.0};

   const auto func_grad = mnfs::num_grad(func, param_errors);
   const auto func_hess = mnfs::num_hess(func_grad);

   mnfs::GradientValue grad_value;
   std::vector<double> inv_hess_value(params.size() * params.size());

   func_grad(grad_value, params);
   mnfs::initialize_inv_hessian(inv_hess_value, grad_value);

   double edm = mnfs::edm(grad_value.grad, inv_hess_value);
   fmt::println("edm    : {}", edm);

   int nIter = 10;
   for (int i = 0; i < nIter; ++i) {
      fmt::println("------- step {} -------", i);

      fmt::println("params : {}", params);
      fmt::println("grad   : {}", grad_value.grad);
      fmt::println("g2     : {}", grad_value.g2);
      fmt::println("step   : {}", grad_value.step);
      fmt::println("inv. hessian: {}", inv_hess_value);

      fmt::println("edm    : {}", edm);

      if(edm < 0.000002) break;

      mnfs::newton_step(step, grad_value.grad, inv_hess_value);

      fmt::println("Newton step   : {}", step);

      std::vector<double> xVals(params.size());
      auto line_func = [&](double alpha) -> double {
         for (std::size_t i = 0; i < params.size(); ++i) {
            xVals[i] = params[i] + alpha * step[i];
         }
         return func(xVals);
      };

      double alpha_min = mnfs::line_search_xmin(params, step);
      double gdel = mnfs::gdel(step, grad_value.grad);
      fmt::println("gdel       : {}", gdel);
      auto ls_result = mnfs::line_search(line_func, alpha_min, gdel);
      fmt::println("line search: {} {}", ls_result.x, ls_result.fval);
      double alpha = ls_result.x;

      for (std::size_t j = 0; j < params.size(); ++j) {
         step[j] = alpha * step[j];
      }

      for (std::size_t j = 0; j < params.size(); ++j) {
         params[j] += step[j];
      }

      for (std::size_t j = 0; j < params.size(); ++j) {
         dgrad[j] = -grad_value.grad[j];
      }

      func_grad(grad_value, params);

      for (std::size_t j = 0; j < params.size(); ++j) {
         dgrad[j] += grad_value.grad[j];
      }

      // Note: it would make sense to switch the two lines. The edm should be
      // estimated with the gradient and Hessian at the same point. But here,
      // the gradient was already updated and the Hessian not. This would fix
      // the weird effect that the edm jumps up after the first iteration,
      // because the Hessian and gradient don't match anymore. However, for
      // Minuit 2 compatibility, the EDM is computed before updating the
      // Hessian for now.
      edm = mnfs::edm(grad_value.grad, inv_hess_value);
      mnfs::update_inv_hessian(inv_hess_value, step, dgrad);
   }
   fmt::println("---------");
   fmt::println("fval   : {}", func(params));

   return 0;
}
