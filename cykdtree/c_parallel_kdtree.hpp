#include "mpi.h"
#include <vector>
#include <math.h>
#include <stdint.h>
#include "c_kdtree.hpp"

class ParallelKDTree
{
public:
  int rank;
  int size;
  int root;
  int rrank;
  int src = -1;
  std::vector<int> dsts;
  uint32_t ndim;
  uint64_t npts = 0;
  int available = 1;
  int *all_avail = NULL;
  bool is_root = false;
  KDTree *tree = NULL;
  double* all_pts = NULL;
  uint64_t* all_idx = NULL;
  bool *periodic = NULL;
  double *left_edge = NULL;
  double *right_edge = NULL;
  
  ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
		 uint32_t leafsize, double *left_edge0, double *right_edge0,
		 bool *periodic0, bool include_self = true) {
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    all_avail = (int*)malloc(size*sizeof(int));
    // Determine root
    if (pts != NULL) {
      root = rank;
      src = rank;
      is_root = true;
      for (int i = 0; i < size; i++) {
	if (i != rank)
	  MPI_Send(&root, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
      ndim = m;
      npts = n;
    } else {
      MPI_Recv(&root, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // Basic init
    rrank = (rank - root + size) % size;
    MPI_Bcast(&ndim, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&leafsize, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    if (rank == root) {
      available = 0;
      all_pts = pts;
      all_idx = idx;
      periodic = periodic0;
      left_edge = left_edge0;
      right_edge = right_edge0;
    } else {
      periodic = (bool*)malloc(ndim*sizeof(bool));
      left_edge = (double*)malloc(ndim*sizeof(double));
      right_edge = (double*)malloc(ndim*sizeof(double));
      for (uint32_t d = 0; d < ndim; d++)
	periodic[d] = false;
    }
    tree = new KDTree(all_pts, all_idx, npts, ndim, leafsize,
		      left_edge, right_edge,
		      periodic, include_self, false);
    // Partition points until every process has points
    double *exch_mins = (double*)malloc(ndim*sizeof(double));
    double *exch_maxs = (double*)malloc(ndim*sizeof(double));
    double *exch_le = (double*)malloc(ndim*sizeof(double));
    double *exch_re = (double*)malloc(ndim*sizeof(double));
    int nrecv = total_available(true);
    int nsend = 0, nexch = 0;
    int other_rank;
    uint32_t dsplit;
    int64_t split_idx = 0;
    double split_val = 0.0;
    uint64_t npts_send;
    double *pts_send;
    while (nrecv > 0) {
      nsend = size - nrecv;
      nexch = std::min(nrecv, nsend);
      if (available) {
	// Receive a set of points
	if (rrank < (nsend+nexch)) {
	  other_rank = (root + rrank - nsend) % size;
	  MPI_Recv(&(tree->npts), 1, MPI_UNSIGNED_LONG, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  tree->all_pts = (double*)malloc(tree->npts*ndim*sizeof(double));
	  tree->all_idx = (uint64_t*)malloc(tree->npts*sizeof(uint64_t));
	  for (uint64_t i = 0; i < tree->npts; i++)
	    tree->all_idx[i] = i;
	  MPI_Recv(tree->domain_mins, 3, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->domain_maxs, 3, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->domain_left_edge, 3, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->domain_right_edge, 3, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->all_pts, ndim*tree->npts, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  available = 0;
	  src = other_rank;
	}
      } else {
	// Send a subset of points
	if (rrank < nexch) {
	  other_rank = (root + rrank + nsend) % size;
	  dsplit = tree->split(0, npts, tree->domain_mins, tree->domain_maxs,
			       split_idx, split_val);
	  memcpy(exch_mins, tree->domain_mins, ndim*sizeof(double));
	  memcpy(exch_maxs, tree->domain_maxs, ndim*sizeof(double));
	  memcpy(exch_le, tree->domain_left_edge, ndim*sizeof(double));
	  memcpy(exch_re, tree->domain_right_edge, ndim*sizeof(double));
	  exch_mins[dsplit] = split_val;
	  exch_le[dsplit] = split_val;
	  npts_send = tree->npts - split_idx - 1;
	  MPI_Send(&npts_send, 1, MPI_UNSIGNED_LONG, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_mins, 3, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_maxs, 3, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_le, 3, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_re, 3, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  pts_send = (double*)malloc(npts_send*ndim*sizeof(double));
	  for (uint64_t i = 0; i < npts_send; i++) 
	    memcpy(pts_send + ndim*i,
		   tree->all_pts + ndim*(tree->all_idx[i + split_idx + 1]),
		   ndim*sizeof(double));
	  MPI_Send(pts_send, ndim*npts_send, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  free(pts_send);
	  dsts.push_back(other_rank);
	  // Update local info
	  tree->domain_maxs[dsplit] = split_val;
	  tree->domain_right_edge[dsplit] = split_val;
	  tree->npts -= npts_send;
	  if (rank != root) {
	    // tree->all_idx = (uint64_t*)realloc(tree->all_idx,
	    // 				   tree->npts*sizeof(uint64_t));
	    tree->all_pts = (double*)realloc(tree->all_pts,
					     tree->npts*ndim*sizeof(double));
	  }
	}
      }
      nrecv = total_available(true);
    }
    free(exch_mins);
    free(exch_maxs);
    free(exch_le);
    free(exch_re);
  }
  ~ParallelKDTree() {
    free(all_avail);
    if (rank != root) {
      free(tree->all_idx);
      free(tree->all_pts);
      free(periodic);
      free(left_edge);
      free(right_edge);
    }
    delete(tree);
  }

  void build(bool include_self = false) {
    tree->build_tree(include_self);
  }

  int total_available(bool update = false) {
    if (update)
      check_available();
    int out = 0;
    for (int i = 0; i < size; i++)
      out += all_avail[i];
    // MPI_Allreduce(&available, &out, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return out;
  }

  int prev_available(bool update = false) {
    if (update)
      check_available();
    int out = 0;
    for (int i = 0; i < rank; i++)
      out += all_avail[i];
    return out;
  }

  void check_available() {
    MPI_Allgather(&available, 1, MPI_INT,
		  all_avail, 1, MPI_INT,
		  MPI_COMM_WORLD);
  }

  int find_available() {
    int out;
    for (out = 0; out < size; out++) {
      if ((out != rank) && (all_avail[out]))
	return out;
    }
    return -1;
  }


};
