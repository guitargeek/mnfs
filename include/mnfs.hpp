#ifndef mnfs_hpp
#define mnfs_hpp

#include <functional>
#include <span>

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

struct LineSearchResult {
   double x = 0.;
   double fval = 0.;
};

LineSearchResult line_search(std::function<double(double)> func, double xmin, double gdel);

double line_search_xmin(std::span<const double> params, std::span<const double> step);

} // namespace mnfs

#endif // mnfs_hpp
