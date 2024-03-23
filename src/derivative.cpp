#include <mnfs/derivative.h>

#include <limits>
#include <cmath>

mnfs::DerivativeState mnfs::update_derivative(DerivativeState prev, std::function<double(double)> func, double x,
                                              DerivativeOptions const &opts)
{
   const double val = func(x);

   constexpr double eps_machine = 4. * std::numeric_limits<double>::epsilon();
   constexpr double eps_machine_2 = 2. * std::sqrt(eps_machine);

   const double dfmin = 8. * eps_machine_2 * (val + opts.error_level);
   const double vrysml = 8. * eps_machine * eps_machine;

   double xtf = x;
   const double epspri = eps_machine_2 + std::abs(prev.grad * eps_machine_2);
   double stepb4 = 0.;
   for (unsigned int j = 0; j < opts.ncycle; j++) {
      const double optstp = std::sqrt(dfmin / (std::abs(prev.g2) + epspri));
      double step = std::max(optstp, std::abs(0.1 * prev.step));
      if (opts.has_limits) {
         if (step > 0.5)
            step = 0.5;
      }
      double stpmax = 10. * std::abs(prev.step);
      if (step > stpmax)
         step = stpmax;
      double stpmin = std::max(vrysml, 8. * std::abs(eps_machine_2 * x));
      if (step < stpmin)
         step = stpmin;
      if (std::abs((step - stepb4) / step) < opts.step_tolerance) {
         break;
      }
      prev.step = step;
      stepb4 = step;

      x = xtf + step;
      const double fs1 = func(x);
      x = xtf - step;
      const double fs2 = func(x);
      x = xtf;

      double grdb4 = prev.grad;
      prev.grad = 0.5 * (fs1 - fs2) / step;
      prev.g2 = (fs1 + fs2 - 2. * val) / step / step;

      if (std::abs(grdb4 - prev.grad) / (std::abs(prev.grad) + dfmin / step) < opts.grad_tolerance) {
         break;
      }
   }

   return prev;
}
