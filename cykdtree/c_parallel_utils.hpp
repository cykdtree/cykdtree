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
                             uint32_t ndim, uint64_t npts,
			     MPI_Comm comm = MPI_COMM_WORLD);
double parallel_pivot_value(double *pts, uint64_t *idx,
                            uint32_t ndim, uint32_t d,
                            int64_t l, int64_t r,
			    MPI_Comm comm = MPI_COMM_WORLD);
int64_t parallel_select(double *pts, uint64_t *idx,
                        uint32_t ndim, uint32_t d,
                        int64_t l, int64_t r, int64_t n,
			double &pivot_val,
			MPI_Comm comm = MPI_COMM_WORLD);
uint32_t parallel_split(double *all_pts, uint64_t *all_idx,
                        uint64_t Lidx, uint64_t n, uint32_t ndim,
                        double *mins, double *maxs,
                        int64_t &split_idx, double &split_val,
                        MPI_Comm comm = MPI_COMM_WORLD);
