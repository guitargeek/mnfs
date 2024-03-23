#ifndef mnfs_derivative_h
#define mnfs_derivative_h

#include <functional>

namespace mnfs {

struct DerivativeOptions {
   double error_level = 1.0;
   unsigned int ncycle = 3;
   double step_tolerance = 0.1;  // strategy 0: 0.5, 1: 0.3, 2: 0.1
   double grad_tolerance = 0.02; // strategy 0: 0.1, 1: 0.05, 2: 0.02
   bool has_limits = false;
};

struct DerivativeState {
   double grad = 0.;
   double g2 = 0.;
   double step = 0.;
};

DerivativeState update_derivative(DerivativeState prev, std::function<double(double)> func, double x,
                                  DerivativeOptions const &opts = {});

} // namespace mnfs

#endif // mnfs_derivative_h
