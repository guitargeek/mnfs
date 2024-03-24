#include <mnfs.hpp>

#include <iostream>
#include <cmath>
#include <span>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

double func(std::span<const double> params)
{
   const double x = params[0];
   const double y = params[1];
   return x * x * x * x + y * y * y * y;
}

struct GradientValue {
   std::vector<double> grad;
   std::vector<double> g2;
   std::vector<double> step;
};

using Function = std::function<double(std::span<const double>)>;

constexpr double eps_machine = 4. * std::numeric_limits<double>::epsilon();
constexpr double eps_machine_2 = 2. * std::sqrt(eps_machine);

GradientValue update_gradient(Function func, std::span<const double> params, std::span<const double> param_errors)
{
   std::vector<double> xVals{params.begin(), params.end()};
   const double centralVal = func(params);

   GradientValue out;
   out.grad.resize(params.size());
   out.g2.resize(params.size());
   out.step.resize(params.size());

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
      out.grad[i] = state.grad;
      out.g2[i] = state.g2;
      out.step[i] = state.step;
   }

   return out;
}

int main(int, char *[])
{
   std::vector<double> params{2.0, 3.0};
   std::vector<double> param_errors{0.1, 0.1};

   // GradientValue grad{{20., 20.}, {200., 200.}, {0.01, 0.01}};

   GradientValue grad = update_gradient(func, params, param_errors);

   fmt::println("{}", grad.grad);
   fmt::println("{}", grad.g2);
   fmt::println("{}", grad.step);

   return 0;
}
