#ifndef mnfs_common_hpp
#define mnfs_common_hpp

#include <cmath>
#include <limits>

namespace mnfs {

constexpr double eps_machine = 4. * std::numeric_limits<double>::epsilon();
constexpr double eps_machine_2 = 2. * std::sqrt(eps_machine);

} // namespace mnfs

#endif // mnfs_common_hpp
