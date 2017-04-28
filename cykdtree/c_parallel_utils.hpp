#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <float.h>
#include <cstdlib>
#include <cstring>
#include "mpi.h"

uint64_t parallel_partition(double **pts, uint64_t **idx,
                            uint32_t ndim, uint64_t npts);
double parallel_pivot_value(int root, std::vector<int> pool,
                            double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r);
