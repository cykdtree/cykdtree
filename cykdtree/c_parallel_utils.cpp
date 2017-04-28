#include "c_utils.hpp"
#include "c_parallel_utils.hpp"

uint64_t parallel_partition(double **pts, uint64_t **idx,
			    uint32_t ndim, uint64_t npts) {
  int size, rank;
  uint64_t npts_local = 0;
  MPI_Comm_size ( MPI_COMM_WORLD, &size);
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
  MPI_Bcast(&ndim, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    uint64_t n_per, n_rem, in_per, n_prev;
    n_per = npts/size;
    n_rem = npts%size;
    n_prev = 0;
    for (int i = 0; i < size; i++) {
      in_per = n_per;
      if (i < (int)(n_rem))
	in_per++;
      if (i == rank)
	npts_local = in_per;
      else {
	MPI_Send(&in_per, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
	MPI_Send((*pts)+ndim*n_prev, ndim*in_per, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
	MPI_Send((*idx)+n_prev, in_per, MPI_UNSIGNED_LONG, i, 2, MPI_COMM_WORLD);
      }
      n_prev += in_per;
    }
  } else {
    MPI_Recv(&npts_local, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    (*pts) = (double*)malloc(ndim*npts_local*sizeof(double));
    (*idx) = (uint64_t*)malloc(npts_local*sizeof(double));
    MPI_Recv(*pts, ndim*npts_local, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(*idx, npts_local, MPI_UNSIGNED_LONG, 0, 2, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }    
  return npts_local;
}

double parallel_pivot_value(int root, std::vector<int> pool,
			    double *pts, uint64_t *idx,
			    uint32_t ndim, uint32_t d,
			    int64_t l, int64_t r) {
  int size, rank, i;
  MPI_Comm_size ( MPI_COMM_WORLD, &size);
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);

  int64_t p = pivot(pts, idx, ndim, d, l, r);
  double pivot_val;
  int64_t npool = (int64_t)(pool.size());

  if (rank == root) {
    std::vector<int>::iterator it;
    double *pts_recv = (double*)malloc(ndim*npool*sizeof(double));
    uint64_t *idx_recv = (uint64_t*)malloc(npool*sizeof(uint64_t));
    for (it = pool.begin(), i = 0; it != pool.end(); it++, i++) {
      if (*it == rank)
	memcpy(pts_recv+ndim*i, pts+ndim*p, ndim*sizeof(double));
      else
	MPI_Recv(pts_recv+ndim*i, ndim, MPI_DOUBLE, *it, *it, MPI_COMM_WORLD,
		 MPI_STATUS_IGNORE);
    }
    int64_t p_tot = pivot(pts_recv, idx_recv, ndim, d, 0, npool-1);
    pivot_val = pts_recv[p_tot*ndim+d];
    free(pts_recv);
    free(idx_recv);
    for (it = pool.begin(); it != pool.end(); it++)
      MPI_Send(&pivot_val, 1, MPI_DOUBLE, *it, rank, MPI_COMM_WORLD);
  } else {
    MPI_Send(pts+p*ndim, ndim, MPI_DOUBLE, root, rank, MPI_COMM_WORLD);
    MPI_Recv(&pivot_val, 1, MPI_DOUBLE, root, root, MPI_COMM_WORLD, 
	     MPI_STATUS_IGNORE);
  }
  return pivot_val;
}

