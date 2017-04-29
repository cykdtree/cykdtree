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

bool in_pool(std::vector<int> pool);
uint64_t parallel_distribute(double **pts, uint64_t **idx,
                             uint32_t ndim, uint64_t npts);
double parallel_pivot_value(std::vector<int> pool,
                            double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r);
int64_t parallel_select(std::vector<int> pool,
                        double *pts, uint64_t *idx,
                        uint32_t ndim, uint32_t d,
                        int64_t l, int64_t r, int64_t n);
