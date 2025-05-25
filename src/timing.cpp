#include <cstddef>
#include <sys/time.h>

#include "../include/timing.h"

double get_current_time() {
  struct timeval current_time;
  gettimeofday(&current_time, NULL);
  return static_cast<double>(current_time.tv_sec) +
         static_cast<double>(current_time.tv_usec) * 1e-6;
}
