#include "c_utils.hpp"
#include "c_parallel_utils.hpp"

bool in_pool(std::vector<int> pool) {
  int rank;
  std::vector<int>::iterator it;
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
  for (it = pool.begin(); it != pool.end(); it++) {
    if (*it == rank)
      return true;
  }
  return false;
}

uint64_t parallel_distribute(double **pts, uint64_t **idx,
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

double parallel_pivot_value(std::vector<int> pool,
			    double *pts, uint64_t *idx,
			    uint32_t ndim, uint32_t d,
			    int64_t l, int64_t r) {
  if (!(in_pool(pool)))
    return 0.0;
  int rank, i;
  double pivot_val = 0.0;
  int64_t npool = (int64_t)(pool.size());
  int root = pool[0];
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);

  // Get local pivot
  int64_t p = pivot(pts, idx, ndim, d, l, r);
  if (npool == 0) {
    if (p >= 0)
      pivot_val = pts[idx[p]*ndim+d];
    return pivot_val;
  }

  if (rank == root) {
    std::vector<int>::iterator it;
    double *pts_recv = (double*)malloc(ndim*npool*sizeof(double));
    uint64_t *idx_recv = (uint64_t*)malloc(npool*sizeof(uint64_t));
    int64_t ptemp;
    // Collect medians from processes in pool
    for (it = pool.begin(), i = 0; it != pool.end(); it++) {
      if (*it == rank)
	ptemp = p;
      else
	MPI_Recv(&ptemp, 1, MPI_LONG, *it, *it, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      if (ptemp >= 0) {
	if (*it == rank)
	  memcpy(pts_recv+ndim*i, pts+ndim*idx[p], ndim*sizeof(double));
	else
	  MPI_Recv(pts_recv+ndim*i, ndim, MPI_DOUBLE, *it, *it, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	idx_recv[i] = (uint64_t)(i);
	i++;
      }
    }
    // Get pivot for points received
    int64_t p_tot = pivot(pts_recv, idx_recv, ndim, d, 0, i-1);
    pivot_val = pts_recv[idx_recv[p_tot]*ndim+d];
    free(pts_recv);
    free(idx_recv);
    // Send pivot value to all processes in pool
    for (it = pool.begin(); it != pool.end(); it++) {
      if (*it == rank)
	continue;
      MPI_Send(&pivot_val, 1, MPI_DOUBLE, *it, rank, MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(&p, 1, MPI_LONG, root, rank, MPI_COMM_WORLD);
    if (p >= 0)
      MPI_Send(pts+idx[p]*ndim, ndim, MPI_DOUBLE, root, rank, MPI_COMM_WORLD);
    MPI_Recv(&pivot_val, 1, MPI_DOUBLE, root, root, MPI_COMM_WORLD, 
	     MPI_STATUS_IGNORE);
  }
  return pivot_val;
}


int64_t parallel_select(std::vector<int> pool,
			double *pts, uint64_t *idx,
			uint32_t ndim, uint32_t d,
			int64_t l, int64_t r, int64_t n) {
  if (!(in_pool(pool)))
    return 0;
  int64_t p, ptot, ptemp;
  double pivot_val;
  std::vector<int>::iterator it;
  int rank, i;
  int64_t npool = (int64_t)(pool.size());
  int root = pool[0];
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);

  while ( 1 ) {
    // if (l == r) {
    //   return l;
    // }

    // Get median of this set
    pivot_val = parallel_pivot_value(pool, pts, idx, ndim, d, l, r);
    if (r < l)
      p = -1;
    else
      p = partition_given_pivot(pts, idx, ndim, d, l, r, pivot_val);

    // Consolidate counts from all processes
    if (rank == root) {
      // Collect
      ptot = 0;
      for (it = pool.begin(); it != pool.end(); it++) {
	if (*it == rank)
	  ptemp = p;
	else
	  MPI_Recv(&ptemp, 1, MPI_LONG, *it, *it, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	if (ptemp >= 0)
	  ptot += ptemp;
      }
      // Distribute
      for (it = pool.begin(); it != pool.end(); it++) {
	if (*it == rank)
	  continue;
	MPI_Send(&ptot, 1, MPI_LONG, *it, rank, MPI_COMM_WORLD);
      }
    } else {
      // Send pivot and receive total count
      MPI_Send(&p, 1, MPI_LONG, root, rank, MPI_COMM_WORLD);
      MPI_Recv(&ptot, 1, MPI_LONG, root, root, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }

    if (n == ptot) { 
      // Return median
      return p;
    } else if (n < ptot) {
      // Exclude right
      

    } else {
      // Exclude left

    }
    // } else if (isEqual(pivot_val, pts[ndim*idx[p]+d])) {
    //   if (n < ptot) {
    // 	r = p - 1;
    //   } else {
    // 	l = p + 1;
    //   }
    // } else {
    //   if (n < ptot) {
    // 	r = p;
    //   } else {
    // 	l = p;
    //   }
    // }
  }
}
