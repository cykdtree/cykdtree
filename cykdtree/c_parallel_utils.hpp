#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <float.h>
#include <cstdlib>
#include <cstring>
#include <stddef.h>
#include "mpi.h"
//#define DEBUG
//#define TIMINGS
#ifdef TIMINGS
#include <ctime>
#endif

void debug_msg(bool local_debug, const char *name,
               const char* msg, ...);
double begin_time();
void end_time(double in, const char* name);

typedef struct exch_rec {
  int src;
  int dst;
  uint32_t split_dim;
  double split_val;
  int64_t split_idx;
  uint64_t left_idx;
  uint64_t npts;
  exch_rec();
  exch_rec(int src, int dst, uint32_t split_dim,
           double split_val, int64_t split_idx,
           uint64_t left_idx, uint64_t npts);
  void print();
} exch_rec;
MPI_Datatype init_mpi_exch_type();

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
uint64_t redistribute_split(double **all_pts, uint64_t **all_idx,
                            uint64_t npts, uint32_t ndim,
                            double *mins, double *maxs,
			    int64_t &split_idx, uint32_t &split_dim,
                            double &split_val,
                            MPI_Comm comm = MPI_COMM_WORLD);
