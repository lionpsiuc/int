#include <cstddef>
#include <sys/time.h>

#include "../include/timing.h"

/**
 * @brief Gets the current system time in seconds.
 *
 * @return The current time as a double-precision floating-point number,
 *         representing seconds since the epoch.
 */
double get_current_time() {
  struct timeval current_time;
  gettimeofday(&current_time, NULL);
  return static_cast<double>(current_time.tv_sec) +
         static_cast<double>(current_time.tv_usec) * 1e-6;
}
