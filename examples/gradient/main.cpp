#include <mnfs/derivative.h>

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

GradientValue update_gradient(Function func, std::span<const double> params, GradientValue prev)
{
   std::vector<double> xVals{params.begin(), params.end()};
   const double centralVal = func(params);

   for (unsigned int i = 0; i < params.size(); i++) {

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

      mnfs::DerivativeState state{prev.grad[i], prev.g2[i], prev.step[i]};
      state = mnfs::update_derivative(state, funcForParam, params[i]);
      prev.grad[i] = state.grad;
      prev.g2[i] = state.g2;
      prev.step[i] = state.step;
   }

   return prev;
}

int main(int, char *[])
{
   std::vector<double> params{2.0, 3.0};

   GradientValue grad{{20., 20.}, {200., 200.}, {0.01, 0.01}};

   grad = update_gradient(func, params, grad);

   fmt::println("{}", grad.grad);
   fmt::println("{}", grad.g2);
   fmt::println("{}", grad.step);

   return 0;
}
