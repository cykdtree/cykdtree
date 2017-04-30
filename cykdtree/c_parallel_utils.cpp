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
			     uint32_t ndim, uint64_t npts,
			     MPI_Comm comm) {
  int size, rank;
  uint64_t npts_local = 0;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);
  MPI_Bcast(&ndim, 1, MPI_UNSIGNED, 0, comm);
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
	MPI_Send(&in_per, 1, MPI_UNSIGNED_LONG, i, 0, comm);
	if (in_per > 0) {
	  MPI_Send((*pts)+ndim*n_prev, ndim*in_per, MPI_DOUBLE, i, 1, comm);
	  MPI_Send((*idx)+n_prev, in_per, MPI_UNSIGNED_LONG, i, 2, comm);
	}
      }
      n_prev += in_per;
    }
  } else {
    MPI_Recv(&npts_local, 1, MPI_UNSIGNED_LONG, 0, 0, comm,
	     MPI_STATUS_IGNORE);
    if (npts_local > 0) {
      (*pts) = (double*)malloc(ndim*npts_local*sizeof(double));
      (*idx) = (uint64_t*)malloc(npts_local*sizeof(double));
      MPI_Recv(*pts, ndim*npts_local, MPI_DOUBLE, 0, 1, comm,
	       MPI_STATUS_IGNORE);
      MPI_Recv(*idx, npts_local, MPI_UNSIGNED_LONG, 0, 2, comm,
	       MPI_STATUS_IGNORE);
    }
  }    
  return npts_local;
}

double parallel_pivot_value(double *pts, uint64_t *idx,
			    uint32_t ndim, uint32_t d,
			    int64_t l, int64_t r,
			    MPI_Comm comm) {
  int size, rank, i, np;
  double pivot_val = 0.0;
  int root = 0;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);

  // Get local pivot
  int64_t p = pivot(pts, idx, ndim, d, l, r);
  if (size <= 1) {
    if (p >= 0)
      pivot_val = pts[idx[p]*ndim+d];
    return pivot_val;
  }

  if (rank == root) {
    double *pts_recv = (double*)malloc(ndim*size*sizeof(double));
    uint64_t *idx_recv = (uint64_t*)malloc(size*sizeof(uint64_t));
    int64_t ptemp;
    // Collect medians from processes in pool
    for (i = 0, np = 0; i < size; i++) {
      if (i == rank)
	ptemp = p;
      else
	MPI_Recv(&ptemp, 1, MPI_LONG, i, i, comm,
                 MPI_STATUS_IGNORE);
      if (ptemp >= 0) {
	if (i == rank)
	  memcpy(pts_recv+ndim*np, pts+ndim*idx[p], ndim*sizeof(double));
	else
	  MPI_Recv(pts_recv+ndim*np, ndim, MPI_DOUBLE, i, i, comm,
		   MPI_STATUS_IGNORE);
	idx_recv[np] = (uint64_t)(np);
	np++;
      }
    }
    // Get pivot for points received
    int64_t p_tot = pivot(pts_recv, idx_recv, ndim, d, 0, np-1);
    pivot_val = pts_recv[idx_recv[p_tot]*ndim+d];
    free(pts_recv);
    free(idx_recv);
  } else {
    MPI_Send(&p, 1, MPI_LONG, root, rank, comm);
    if (p >= 0)
      MPI_Send(pts+idx[p]*ndim, ndim, MPI_DOUBLE, root, rank, comm);
  }
  // Send pivot value to all processes in pool
  MPI_Bcast(&pivot_val, 1, MPI_DOUBLE, root, comm);
  return pivot_val;
}


int64_t parallel_select(double *pts, uint64_t *idx,
			uint32_t ndim, uint32_t d,
			int64_t l0, int64_t r0, int64_t n,
			double &pivot_val, MPI_Comm comm) {
  int64_t p, nl, nl_tot;
  int size, rank;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);
  int64_t l = l0, r = r0;

  p = -1;
  while ( 1 ) {
    // Get median of this set
    pivot_val = parallel_pivot_value(pts, idx, ndim, d, l, r, comm);
    if (l <= r) {
      p = partition_given_pivot(pts, idx, ndim, d, l, r, pivot_val);
      if (pts[ndim*idx[p]+d] > pivot_val)
	p = l - 1;
    }
    nl = p - l0 + 1;

    // Consolidate counts from all processes
    MPI_Allreduce(&nl, &nl_tot, 1, MPI_LONG, MPI_SUM, comm);

    if (n == nl_tot) { 
      // Return median
      return p;
    } else if (p >= l) {
      if (n < nl_tot) {
	// Exclude right
	if (isEqual(pivot_val, pts[ndim*idx[p]+d])) {
	  r = p - 1;
	} else {
	  r = p;
	}
      } else {
	// Exclude left
	if (isEqual(pivot_val, pts[ndim*idx[p]+d])) {
	  l = p + 1;
	} else {
	  l = p;
	}
      }
    }
  }
}


uint32_t parallel_split(uint64_t *orig_idx,
			double *all_pts, uint64_t *all_idx,
			uint64_t Lidx, uint64_t n, uint32_t ndim,
			double *mins, double *maxs,
			int64_t &split_idx, double &split_val,
			MPI_Comm comm) {
  int size, rank, i;
  int root = 0;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);

  // Consolidate number points
  uint64_t ntot = 0;
  MPI_Allreduce(&n, &ntot, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

  // Return immediately if variables empty
  if ((ntot == 0) or (ndim == 0)) {
    split_idx = -1;
    split_val = 0.0;
    return 0;
  }

  // Consolidate mins/maxs
  double *mins_tot = (double*)malloc(ndim*sizeof(double));
  double *maxs_tot = (double*)malloc(ndim*sizeof(double));
  MPI_Allreduce(mins, mins_tot, ndim, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(maxs, maxs_tot, ndim, MPI_DOUBLE, MPI_MAX, comm);

  // Find dimension to split along
  uint32_t dmax, d;
  dmax = 0;
  for (d = 1; d < ndim; d++)
    if ((maxs_tot[d]-mins_tot[d]) > (maxs_tot[dmax]-mins_tot[dmax]))
      dmax = d;
  if (maxs_tot[dmax] == mins_tot[dmax]) {
    // all points singular
    free(mins_tot);
    free(maxs_tot);
    return ndim;
  }

  // Find median along dimension
  int64_t nsel = (ntot/2) + (ntot%2);


  // Free and return
  free(mins_tot);
  free(maxs_tot);
  return dmax;
}
