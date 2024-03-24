#include <mnfs.hpp>

#include <cmath>

namespace {

constexpr double eps_machine = 4. * std::numeric_limits<double>::epsilon();
constexpr double eps_machine_2 = 2. * std::sqrt(eps_machine);

struct PointXY {
   double x = 0.0;
   double y = 0.0;
};

struct Parabola {
   double a = 0.0;
   double b = 0.0;
   double c = 0.0;
};

// construct the parabola from 3 points p1,p2,p3.
Parabola fit_parabola(const PointXY &p1, const PointXY &p2, const PointXY &p3)
{
   double x1 = p1.x;
   double x2 = p2.x;
   double x3 = p3.x;
   double dx12 = x1 - x2;
   double dx13 = x1 - x3;
   double dx23 = x2 - x3;

   double xm = (x1 + x2 + x3) / 3.;
   x1 -= xm;
   x2 -= xm;
   x3 -= xm;

   double y1 = p1.y;
   double y2 = p2.y;
   double y3 = p3.y;

   double a = y1 / (dx12 * dx13) - y2 / (dx12 * dx23) + y3 / (dx13 * dx23);
   double b = -y1 * (x2 + x3) / (dx12 * dx13) + y2 * (x1 + x3) / (dx12 * dx23) - y3 * (x1 + x2) / (dx13 * dx23);
   double c = y1 - a * x1 * x1 - b * x1;

   c += xm * (xm * a - b);
   b -= 2. * xm * a;

   return {a, b, c};
}

inline double parabola_min(Parabola const &p)
{
   return -p.b / (2. * p.a);
}

} // namespace

double mnfs::line_search_xmin(std::span<const double> params, std::span<const double> step)
{
   double xmin = 0.;
   for (unsigned int i = 0; i < step.size(); i++) {
      if (step[i] == 0)
         continue;
      double ratio = std::abs(params[i] / step[i]);
      if (xmin == 0)
         xmin = ratio;
      if (ratio < xmin)
         xmin = ratio;
   }
   if (std::abs(xmin) < eps_machine)
      xmin = eps_machine;
   xmin *= eps_machine_2;
   return xmin;
}

/**  Perform a line search from position defined by the vector st
       along the direction step, where the length of vector step
       gives the expected position of Minimum.
       fcn is Value of function at the starting position ,
       gdel (if non-zero) is df/dx along step at st.
       Return a parabola point containing Minimum x position and y (function Value)
    - add a flag to control the debug
*/

mnfs::LineSearchResult mnfs::line_search(std::function<double(double)> func, double xmin, double gdel)
{
   const double fval = func(0.);

   //*-*-*-*-*-*-*-*-*-*Perform a line search from position st along step   *-*-*-*-*-*-*-*
   //*-*                =========================================
   //*-* SLAMBG and ALPHA control the maximum individual steps allowed.
   //*-* The first step is always =1. The max length of second step is SLAMBG.
   //*-* The max size of subsequent steps is the maximum previous successful
   //*-*   step multiplied by ALPHA + the size of most recent successful step,
   //*-*   but cannot be smaller than SLAMBG.
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

   double overall = 1000.;
   double undral = -100.;
   double toler = 0.05;
   double slambg = 5.;
   double alpha = 2.;
   int maxiter = 12;
   // start as in Fortran from 1 and count all the time we evaluate the function
   int niter = 1;

   double f0 = fval;
   double f1 = func(1.);
   niter++;
   double fvmin = fval;
   double xvmin = 0.;

   if (f1 < f0) {
      fvmin = f1;
      xvmin = 1.;
   }
   double toler8 = toler;
   double slamax = slambg;
   double flast = f1;
   double slam = 1.;

   bool iterate = false;
   PointXY p0(0., f0);
   PointXY p1(slam, flast);
   double f2 = 0.;
   // quadratic interpolation using the two points p0,p1 and the slope at p0
   do {
      // cut toler8 as function goes up
      iterate = false;

      double denom = 2. * (flast - f0 - gdel * slam) / (slam * slam);

      if (denom != 0) {
         slam = -gdel / denom;
      } else {
         denom = -0.1 * gdel;
         slam = 1.;
      }

      if (slam < 0.) {
         slam = slamax;
      }
      if (slam > slamax) {
         slam = slamax;
      }

      if (slam < toler8) {
         slam = toler8;
      }
      if (slam < xmin) {
         return {xvmin, fvmin};
      }
      if (std::abs(slam - 1.) < toler8 && p1.y < p0.y) {
         return {xvmin, fvmin};
      }
      if (std::abs(slam - 1.) < toler8)
         slam = 1. + toler8;

      f2 = func(slam);

      niter++; // do as in Minuit (count all func evalu)

      if (f2 < fvmin) {
         fvmin = f2;
         xvmin = slam;
      }
      // LM : correct a bug using precision
      if (std::abs(p0.y - fvmin) < std::abs(fvmin) * eps_machine) {
         iterate = true;
         flast = f2;
         toler8 = toler * slam;
         overall = slam - toler8;
         slamax = overall;
         p1 = PointXY(slam, flast);
      }
   } while (iterate && niter < maxiter);
   if (niter >= maxiter) {
      // exhausted max number of iterations
      return {xvmin, fvmin};
   }

   PointXY p2(slam, f2);

   // do now the quadratic interpolation with 3 points
   do {
      slamax = std::max(slamax, alpha * std::abs(xvmin));
      Parabola pb = fit_parabola(p0, p1, p2);
      if (pb.a < eps_machine_2) {
         double slopem = 2. * pb.a * xvmin + pb.b;
         if (slopem < 0.)
            slam = xvmin + slamax;
         else
            slam = xvmin - slamax;
      } else {
         slam = parabola_min(pb);
         if (slam > xvmin + slamax)
            slam = xvmin + slamax;
         if (slam < xvmin - slamax)
            slam = xvmin - slamax;
      }
      if (slam > 0.) {
         if (slam > overall)
            slam = overall;
      } else {
         if (slam < undral)
            slam = undral;
      }

      double f3 = 0.;
      do {
         iterate = false;
         double toler9 = std::max(toler8, std::abs(toler8 * slam));
         // min. of parabola at one point
         if (std::abs(p0.x - slam) < toler9 || std::abs(p1.x - slam) < toler9 || std::abs(p2.x - slam) < toler9) {
            return {xvmin, fvmin};
         }

         // take the step
         f3 = func(slam);
         // if latest point worse than all three previous, cut step
         if (f3 > p0.y && f3 > p1.y && f3 > p2.y) {
            if (slam > xvmin)
               overall = std::min(overall, slam - toler8);
            if (slam < xvmin)
               undral = std::max(undral, slam + toler8);
            slam = 0.5 * (slam + xvmin);
            iterate = true;
            niter++;
         }
      } while (iterate && niter < maxiter);
      if (niter >= maxiter) {
         // exhausted max number of iterations
         return {xvmin, fvmin};
      }

      // find worst previous point out of three and replace
      PointXY p3(slam, f3);
      if (p0.y > p1.y && p0.y > p2.y)
         p0 = p3;
      else if (p1.y > p0.y && p1.y > p2.y)
         p1 = p3;
      else
         p2 = p3;
      if (f3 < fvmin) {
         fvmin = f3;
         xvmin = slam;
      } else {
         if (slam > xvmin)
            overall = std::min(overall, slam - toler8);
         if (slam < xvmin)
            undral = std::max(undral, slam + toler8);
      }

      niter++;
   } while (niter < maxiter);

   return {xvmin, fvmin};
}
