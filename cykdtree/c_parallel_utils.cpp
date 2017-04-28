#include "c_utils.hpp"
#include "c_parallel_utils.hpp"

double parallel_pivot_value(int root, std::vector<int> pool,
			    double *pts, uint64_t *idx,
			    uint32_t ndim, uint32_t d,
			    int64_t l, int64_t r) {
  int size, rank;
  MPI_Comm_size ( MPI_COMM_WORLD, &size);
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);

  int64_t p = pivot(pts, idx, ndim, d, l, r);
  double pivot;
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
    pivot = pts_recv[p_tot*ndim+d];
    free(pts_recv);
    free(idx_recv);
    for (it = pool.begin(); it != pool.end(); it++)
      MPI_Send(&pivot, 1, MPI_DOUBLE, *it, rank, MPI_COMM_WORLD);
  } else {
    MPI_Send(pts+p*ndim, ndim, MPI_DOUBLE, root, rank, MPI_COMM_WORLD);
    MPI_Recv(&pivot, 1, MPI_DOUBLE, root, root, MPI_COMM_WORLD, 
	     MPI_STATUS_IGNORE);
  }
  return pivot;
}

