#include <mnfs.hpp>

#include "mnfs_common.hpp"

void mnfs::initialize_inv_hessian(std::span<double> out, GradientValue const &grad)
{
   // TODO:
   // * case of initial covariance
   // * case of negative G2 line search
   // * calse of full Hessian in strategy 2

   std::size_t n = grad.grad.size();
   for (std::size_t i = 0; i < n; i++) {
      // if G2 is small better using an arbitrary value (e.g. 1)
      out[n * i + i] = std::abs(grad.g2[i]) > eps_machine ? 1. / grad.g2[i] : 1.0;
   }
}
