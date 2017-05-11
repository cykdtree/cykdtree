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
#include <cstdarg>
#include "mpi.h"
//#define DEBUG
//#define TIMINGS
#ifdef TIMINGS
#include <ctime>
#endif

extern MPI_Datatype* mpi_type_exch_rec;

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
  void send(int idst, MPI_Comm comm = MPI_COMM_WORLD);
  void recv(int isrc, MPI_Comm comm = MPI_COMM_WORLD);
  void send_vec(int idst, std::vector<exch_rec> st,
		MPI_Comm comm = MPI_COMM_WORLD);
  std::vector<exch_rec> recv_vec(int isrc, 
				 std::vector<exch_rec> st = std::vector<exch_rec>(),
				 MPI_Comm comm = MPI_COMM_WORLD);
} exch_rec;
bool init_mpi_exch_type();
void free_mpi_exch_type(bool free_mpi_type = true);
void print_exch_vec(std::vector<exch_rec> st, MPI_Comm comm = MPI_COMM_WORLD);

class SplitNode {
public:
  int proc;
  exch_rec exch;
  SplitNode *less;
  SplitNode *greater;
  SplitNode(int proc);
  SplitNode(exch_rec exch, SplitNode *less, SplitNode *greater);
  ~SplitNode();
  void send(int idst, MPI_Comm comm = MPI_COMM_WORLD);
  void recv(int isrc, MPI_Comm comm = MPI_COMM_WORLD);
};

bool in_pool(std::vector<int> pool);
uint64_t parallel_distribute(double **pts, uint64_t **idx,
                             uint64_t npts, uint32_t ndim,
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
                            double &split_val, int split_rank = -1,
                            MPI_Comm comm = MPI_COMM_WORLD);
void bcast_bool(bool* arr, uint32_t n, int root,
		MPI_Comm comm = MPI_COMM_WORLD);
int calc_split_rank(int size, bool split_left = true);
int calc_rounds(int &src_round, MPI_Comm comm = MPI_COMM_WORLD);
uint64_t kdtree_parallel_distribute(double **pts, uint64_t **idx,
				    uint64_t npts, uint32_t ndim,
				    double *left_edge, double *right_edge,
                                    bool *periodic_left, bool *periodic_right,
				    exch_rec &src_exch, std::vector<exch_rec> &dst_exch,
				    MPI_Comm comm = MPI_COMM_WORLD);
SplitNode* consolidate_split_tree(exch_rec src_exch, std::vector<exch_rec> dst_exch,
                                  MPI_Comm comm = MPI_COMM_WORLD);

