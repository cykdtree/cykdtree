#include "c_utils.hpp"
#include "c_parallel_utils.hpp"


MPI_Datatype* mpi_type_exch_rec = NULL;

void debug_msg(bool local_debug, const char *name,
               const char* msg, ...) {
#ifdef DEBUG
  if (!(local_debug))
    return;
  int rank, size;
  va_list args;
  MPI_Comm_size ( MPI_COMM_WORLD, &size);
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
  printf("%d: %s: ", rank, name);
  va_start(args, msg);
  vprintf(msg, args);
  va_end(args);
  printf("\n");
#endif
}

double begin_time() {
  double out = 0.0;
#ifdef TIMINGS
  out = ((double)(clock()))/CLOCKS_PER_SEC;
#endif
  return out;
}

void end_time(double in, const char* name) {
#ifdef TIMINGS
  int rank, size;
  MPI_Comm_size ( MPI_COMM_WORLD, &size);
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
  double out = ((double)(clock()))/CLOCKS_PER_SEC;
  // if (rank == 0)
  std::cout << rank << ": " << name << " took " << (out-in) << std::endl;
#endif
}

exch_rec::exch_rec()
  : src(-1), dst(-1), split_dim(0), split_val(0.0), split_idx(-1),
    left_idx(0), npts(0)
{ }
exch_rec::exch_rec(int src, int dst, uint32_t split_dim,
		   double split_val, int64_t split_idx,
		   uint64_t left_idx, uint64_t npts)
  : src(src), dst(dst), split_dim(split_dim), split_val(split_val),
    split_idx(split_idx), left_idx(left_idx), npts(npts)
{ }
void exch_rec::print() {
  printf("src = %d, dst = %d, split_dim = %u, split_val = %f, split_idx = %ld, left_idx = %lu, npts = %lu\n",
         src, dst, split_dim, split_val, split_idx,
         left_idx, npts);
}
void exch_rec::send(int idst, MPI_Comm comm) {
  bool free_mpi_type = init_mpi_exch_type();
  int tag;
  MPI_Comm_rank ( comm, &tag);
  MPI_Send(this, 1, *mpi_type_exch_rec, idst, tag, comm);
  free_mpi_exch_type(free_mpi_type);
}
void exch_rec::recv(int isrc, MPI_Comm comm) {
  bool free_mpi_type = init_mpi_exch_type();
  int tag = isrc;
  MPI_Recv(this, 1, *mpi_type_exch_rec, isrc, tag, comm, MPI_STATUS_IGNORE);
  free_mpi_exch_type(free_mpi_type);
}
void exch_rec::send_vec(int idst, std::vector<exch_rec> st, MPI_Comm comm) {
  bool free_mpi_type = init_mpi_exch_type();
  int tag;
  int nexch = st.size();
  MPI_Comm_rank ( comm, &tag);
  MPI_Send(&nexch, 1, MPI_INT, idst, tag, comm);
  MPI_Send(&st[0], nexch, *mpi_type_exch_rec, idst, tag, comm);
  free_mpi_exch_type(free_mpi_type);
}
  
std::vector<exch_rec> exch_rec::recv_vec(int isrc, 
					 std::vector<exch_rec> st,
					 MPI_Comm comm) {
  bool free_mpi_type = init_mpi_exch_type();
  int tag = isrc;
  int nexch;
  MPI_Recv(&nexch, 1, MPI_INT, isrc, tag, comm, MPI_STATUS_IGNORE);
  st.resize(nexch);
  MPI_Recv(&st[0], nexch, *mpi_type_exch_rec, isrc, tag, comm, MPI_STATUS_IGNORE);
  free_mpi_exch_type(free_mpi_type);
  return st;
}

bool init_mpi_exch_type() {
  if (mpi_type_exch_rec != NULL)
    return false;
  const int nitems = 5;
  int blocklengths[nitems] = {2, 1, 1, 1, 2};
  MPI_Datatype types[nitems] = {MPI_INT, MPI_UNSIGNED, MPI_DOUBLE, MPI_LONG,
                                MPI_UNSIGNED_LONG};
  MPI_Aint offsets[nitems];
  mpi_type_exch_rec = new MPI_Datatype();
  offsets[0] = offsetof(exch_rec, src);
  offsets[1] = offsetof(exch_rec, split_dim);
  offsets[2] = offsetof(exch_rec, split_val);
  offsets[3] = offsetof(exch_rec, split_idx);
  offsets[4] = offsetof(exch_rec, left_idx);
  MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_type_exch_rec);
  MPI_Type_commit(mpi_type_exch_rec);
  return true;
}

void free_mpi_exch_type(bool free_mpi_type) {
  if (free_mpi_type) {
    MPI_Type_free(mpi_type_exch_rec);
    mpi_type_exch_rec = NULL;
  }
}

void print_exch_vec(std::vector<exch_rec> st, MPI_Comm comm) {
  std::vector<exch_rec>::iterator it;
  int rank;
  MPI_Comm_rank ( comm, &rank);
  for (it = st.begin(); it != st.end(); it++) {
    printf("%d: ", rank);
    it->print();
  }
}

SplitNode::SplitNode(int proc)
  : proc(proc), less(NULL), greater(NULL)
{ 
  exch = exch_rec();
}
SplitNode::SplitNode(exch_rec exch, SplitNode *less, SplitNode *greater)
  : proc(-1), exch(exch), less(less), greater(greater)
{ }
SplitNode::~SplitNode() {
  if (less != NULL)
    delete less;
  if (greater != NULL)
    delete greater;
}
void SplitNode::send(int idst, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank ( comm, &rank);
  MPI_Send(&proc, 1, MPI_INT, idst, rank, comm);
  if (proc < 0) {
    exch.send(idst, comm);
    less->send(idst, comm);
    greater->send(idst, comm);
  }
}
void SplitNode::recv(int isrc, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank ( comm, &rank);
  MPI_Recv(&proc, 1, MPI_INT, isrc, isrc, comm, MPI_STATUS_IGNORE);
  if (proc < 0) {
    if (less == NULL)
      less = new SplitNode(-1);
    if (greater == NULL)
      greater = new SplitNode(-1);
    exch.recv(isrc, comm);
    less->recv(isrc, comm);
    greater->recv(isrc, comm);
  }
}


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
			     uint64_t npts, uint32_t ndim,
			     MPI_Comm comm) {
  bool local_debug = true;
  int size, rank;
  uint64_t npts_local = 0;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);
  MPI_Bcast(&ndim, 1, MPI_UNSIGNED, 0, comm);
  if (rank == 0) {
    uint64_t n_per, n_rem, in_per, n_prev;
    n_per = npts/size;
    n_rem = npts%size;
    debug_msg(local_debug, "parallel_distribute",
	      "n_per = %lu, n_rem = %lu", n_per, n_rem);
    n_prev = 0;
    for (int i = 0; i < size; i++) {
      in_per = n_per;
      if (i < (int)(n_rem))
	in_per++;
      if (i == rank) {
	npts_local = in_per;
      } else {
	MPI_Send(&in_per, 1, MPI_UNSIGNED_LONG, i, 0, comm);
	debug_msg(local_debug, "parallel_distribute",
		  "sending %lu points to %d", in_per, i);
	if (in_per > 0) {
	  MPI_Send((*pts)+ndim*n_prev, ndim*in_per, MPI_DOUBLE, i, 1, comm);
	  MPI_Send((*idx)+n_prev, in_per, MPI_UNSIGNED_LONG, i, 2, comm);
	}
      }
      n_prev += in_per;
    }
    // Replicate memory for root and change pointers so that original memory
    // remains untouched on root
    double *temp_pts = (double*)malloc(ndim*npts_local*sizeof(double));
    uint64_t *temp_idx = (uint64_t*)malloc(npts_local*sizeof(uint64_t));
    memcpy(temp_pts, *pts, ndim*npts_local*sizeof(double));
    memcpy(temp_idx, *idx, npts_local*sizeof(uint64_t));
    *pts = temp_pts;
    *idx = temp_idx;
    // // Reduce size on root
    // (*pts) = (double*)realloc(*pts, ndim*npts_local*sizeof(double));
    // (*idx) = (uint64_t*)realloc(*idx, npts_local*sizeof(uint64_t));
  } else {
    MPI_Recv(&npts_local, 1, MPI_UNSIGNED_LONG, 0, 0, comm,
	     MPI_STATUS_IGNORE);
    debug_msg(local_debug, "parallel_distribute",
	      "receiving %lu points from %d", npts_local, 0);
    if (npts_local > 0) {
      (*pts) = (double*)realloc(*pts, ndim*npts_local*sizeof(double));
      (*idx) = (uint64_t*)realloc(*idx, npts_local*sizeof(uint64_t));
      MPI_Recv(*pts, ndim*npts_local, MPI_DOUBLE, 0, 1, comm,
	       MPI_STATUS_IGNORE);
      MPI_Recv(*idx, npts_local, MPI_UNSIGNED_LONG, 0, 2, comm,
	       MPI_STATUS_IGNORE);
    }
  }    
  debug_msg(local_debug, "parallel_distribute", "npts_local = %lu",
	    npts_local);
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
  nl_tot = 0;
  while ( 1 ) {
    // Get median of this set
    pivot_val = parallel_pivot_value(pts, idx, ndim, d, l, r, comm);
    p = partition_given_pivot(pts, idx, ndim, d, l, r, pivot_val);
    if (p < 0)
      p = l - 1;
    nl = p - l0 + 1;

    // Consolidate counts from all processes
    MPI_Allreduce(&nl, &nl_tot, 1, MPI_LONG, MPI_SUM, comm);

    if (n == nl_tot) { 
      // Return median
      return p;
    } else if (l <= r) {
      if (n < nl_tot) {
	// Exclude right
	if (p <= r) {
	  if (isEqual(pivot_val, pts[ndim*idx[p]+d])) {
	    r = p - 1;
	  } else {
	    r = p;
	  }
	}
      } else {
	// Exclude left
	if (p >= l) {
	  if (isEqual(pivot_val, pts[ndim*idx[p]+d])) {
	    l = p + 1;
	  } else {
	    l = p;
	  }
	}
      }
    }
  }
}


uint32_t parallel_split(double *all_pts, uint64_t *all_idx,
			uint64_t Lidx, uint64_t n, uint32_t ndim,
			double *mins, double *maxs,
			int64_t &split_idx, double &split_val,
			MPI_Comm comm) {
  bool local_debug = true;
  int size, rank;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);

  // Consolidate number points
  uint64_t ntot = 0;
  MPI_Allreduce(&n, &ntot, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
  debug_msg(local_debug, "parallel_split", "n = %lu, ntot = %lu", n, ntot);

  // Return immediately if variables empty
  if ((ntot == 0) or (ndim == 0)) {
    debug_msg(local_debug, "parallel_split", "zero points");
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
  split_idx = parallel_select(all_pts, all_idx, ndim, dmax,
			      Lidx, Lidx+n-1, nsel, split_val, comm);

  // Free and return
  free(mins_tot);
  free(maxs_tot);
  return dmax;
}


uint64_t redistribute_split(double **all_pts, uint64_t **all_idx,
			    uint64_t npts, uint32_t ndim,
			    double *mins, double *maxs,
			    int64_t &split_idx, uint32_t &split_dim,
			    double &split_val, int split_rank,
			    MPI_Comm comm) {
  bool local_debug = true;
  int size, rank, split_size, rel_rank;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);
  if (split_rank < 0)
    split_rank = calc_split_rank(size);
  if (split_rank <= (size/2))
    split_size = split_rank;
  else
    split_size = size - split_rank;
  uint64_t x;
  uint64_t nexch, ntemp;
  uint64_t *temp_idx = NULL;
  double *temp_pts = NULL;
  uint64_t *exch_idx = NULL;
  double *exch_pts = NULL;
  std::vector<int> other;
  std::vector<int>::iterator it;
  uint64_t npts_new = 0;

  // Sort
  debug_msg(local_debug, "redistribute_split",
	    "parallel_split, split_rank = %d, split_size = %d",
	    split_rank, split_size);
  uint64_t *sort_idx = (uint64_t*)malloc(npts*sizeof(uint64_t));
  for (uint64_t i = 0; i < npts; i++)
    sort_idx[i] = i;
  split_idx = -1;
  split_dim = 0;
  split_val = 0.0;
  split_dim = parallel_split(*all_pts, sort_idx, 0, npts, ndim,
			     mins, maxs, split_idx, split_val,
			     comm);
  debug_msg(local_debug, "redistribute_split", 
	    "split_idx = %ld, split_dim = %u, split_val = %lf",
	    split_idx, split_dim, split_val);

  // Identify partner process(es)
  // for (int i = 0; i < size; i++) {
  //   if ((i != rank) && ((i%split_size) == rel_rank))
  //     other.push_back(i);
  // }
  // Exchange
  if (rank < split_rank) {
    // LEFT
    rel_rank = rank % split_size;
    for (int i = split_rank; i < size; i++) {
      if (((i - split_rank) % split_size) == rel_rank)
	other.push_back(i);
    }
    debug_msg(local_debug, "redistribute_split",
	      "rel_rank = %d, size(other) = %u", rel_rank, other.size());
    // Put aside points to send
    nexch = npts - (split_idx + 1);
    debug_msg(local_debug, "redistribute_split",
	      "putting aside %lu points to send", nexch);
    exch_idx = (uint64_t*)realloc(exch_idx, nexch*sizeof(uint64_t));
    exch_pts = (double*)realloc(exch_pts, nexch*ndim*sizeof(double));
    for (x = 0; x < nexch; x++) {
      exch_idx[x] = (*all_idx)[sort_idx[split_idx + 1 + x]];
      memcpy(exch_pts+x*ndim, (*all_pts)+sort_idx[split_idx + 1 + x]*ndim,
	     ndim*sizeof(double));
    }
    // Move points
    npts_new = split_idx + 1;
    debug_msg(local_debug, "redistribute_split", "moving %lu points", npts_new);
    temp_idx = (uint64_t*)realloc(temp_idx, npts_new*sizeof(uint64_t));
    temp_pts = (double*)realloc(temp_pts, npts_new*ndim*sizeof(double));
    for (x = 0; x < npts_new; x++) {
      temp_idx[x] = (*all_idx)[sort_idx[x]];
      memcpy(temp_pts+x*ndim, (*all_pts)+sort_idx[x]*ndim, ndim*sizeof(double));
    }
    memcpy(*all_idx, temp_idx, npts_new*sizeof(uint64_t));
    memcpy(*all_pts, temp_pts, npts_new*ndim*sizeof(double));
    free(temp_idx);
    free(temp_pts);
    // Left receives first from all partner processes
    if (rank < split_size) {
      for (it = other.begin(); it != other.end(); it++) {
	MPI_Recv(&ntemp, 1, MPI_UNSIGNED_LONG, *it, *it, comm,
		 MPI_STATUS_IGNORE);
	debug_msg(local_debug, "redistribute_split", "receiving %lu from %d",
		  ntemp, *it);
	(*all_idx) = (uint64_t*)realloc(*all_idx, (npts_new+ntemp)*sizeof(uint64_t));
	(*all_pts) = (double*)realloc(*all_pts, (npts_new+ntemp)*ndim*sizeof(double));
	MPI_Recv((*all_idx)+npts_new, ntemp, MPI_UNSIGNED_LONG, *it, *it, comm,
		 MPI_STATUS_IGNORE);
	MPI_Recv((*all_pts)+npts_new*ndim, ntemp*ndim, MPI_DOUBLE, *it, *it, comm,
		 MPI_STATUS_IGNORE);
	npts_new += ntemp;
      }
    }
    // Left sends second to only the first partner process
    debug_msg(local_debug, "redistribute_split", "sending %lu to %d",
	      nexch, other[0]);
    MPI_Send(&nexch, 1, MPI_UNSIGNED_LONG, other[0], rank, comm);
    MPI_Send(exch_idx, nexch, MPI_UNSIGNED_LONG, other[0], rank, comm);
    MPI_Send(exch_pts, nexch*ndim, MPI_DOUBLE, other[0], rank, comm);
  } else {
    // RIGHT
    rel_rank = (rank - split_rank) % split_size;
    for (int i = 0; i < split_rank; i++) {
      if ((i % split_size) == rel_rank)
	other.push_back(i);
    }
    debug_msg(local_debug, "redistribute_split",
	      "rel_rank = %d, size(other) = %u", rel_rank, other.size());
    // Right sends first to just the first partner process
    nexch = split_idx + 1;
    debug_msg(local_debug, "redistribute_split", "sending %lu to %d",
	      nexch, other[0]);
    MPI_Send(&nexch, 1, MPI_UNSIGNED_LONG, other[0], rank, comm);
    exch_idx = (uint64_t*)realloc(exch_idx, nexch*sizeof(uint64_t));
    exch_pts = (double*)realloc(exch_pts, nexch*ndim*sizeof(double));
    for (x = 0; x < nexch; x++) {
      exch_idx[x] = (*all_idx)[sort_idx[x]];
      memcpy(exch_pts+x*ndim, (*all_pts)+sort_idx[x]*ndim, ndim*sizeof(double));
    }
    MPI_Send(exch_idx, nexch, MPI_UNSIGNED_LONG, other[0], rank, comm);
    MPI_Send(exch_pts, nexch*ndim, MPI_DOUBLE, other[0], rank, comm);
    // Move points
    nexch = npts - (split_idx + 1);
    debug_msg(local_debug, "redistribute_split", "moving %lu points", nexch);
    exch_idx = (uint64_t*)realloc(exch_idx, nexch*sizeof(uint64_t));
    exch_pts = (double*)realloc(exch_pts, nexch*ndim*sizeof(double));
    for (x = 0; x < nexch; x++) {
      exch_idx[x] = (*all_idx)[sort_idx[split_idx + 1 + x]];
      memcpy(exch_pts+x*ndim, (*all_pts)+sort_idx[split_idx + 1 + x]*ndim,
	     ndim*sizeof(double));
    }
    memcpy(*all_idx, exch_idx, nexch*sizeof(uint64_t));
    memcpy(*all_pts, exch_pts, nexch*ndim*sizeof(double));
    npts_new = nexch;
    // Right receives second from all partner processes
    if ((rank - split_rank) < split_size) { 
      for (it = other.begin(); it != other.end(); it++) {
	MPI_Recv(&nexch, 1, MPI_UNSIGNED_LONG, *it, *it, comm,
		 MPI_STATUS_IGNORE);
	debug_msg(local_debug, "redistribute_split", "receiving %lu from %d",
		  nexch, *it);
	(*all_idx) = (uint64_t*)realloc(*all_idx, (npts_new+nexch)*sizeof(uint64_t));
	(*all_pts) = (double*)realloc(*all_pts, (npts_new+nexch)*ndim*sizeof(double));
	MPI_Recv((*all_idx)+npts_new, nexch, MPI_UNSIGNED_LONG, *it, *it,
		 comm, MPI_STATUS_IGNORE);
	MPI_Recv((*all_pts)+npts_new*ndim, nexch*ndim, MPI_DOUBLE, *it, *it,
		 comm, MPI_STATUS_IGNORE);
	npts_new += nexch;
      }
    }
  }
  debug_msg(local_debug, "redistribute_split", "cleanup");

  free(sort_idx);
  free(exch_idx);
  free(exch_pts);
  return npts_new;
}


void bcast_bool(bool* arr, uint32_t n, int root, MPI_Comm comm) {
  int size, rank;
  MPI_Comm_size( comm, &size);
  MPI_Comm_rank( comm, &rank);
  uint32_t d;
  int *dum = (int*)malloc(n*sizeof(int));
  if (root == rank) {
    for (d = 0; d < n; d++)
      dum[d] = (int)(arr[d]);
  }
  MPI_Bcast(dum, n, MPI_INT, root, comm);
  if (root != rank) {
    for (d = 0; d < n; d++)
      arr[d] = (bool)(dum[d]);
  }
  free(dum);
}

int calc_split_rank(int size, bool split_left) {
  int split_rank = size/2;
  if (split_left)
    split_rank += size%2;
  return split_rank;
}

int calc_rounds(int &src_round, MPI_Comm comm) {
  MPI_Comm orig_comm = comm;
  int size, rank, rroot;
  int round = 0, max_round = 0;
  int color = 1;
  MPI_Comm_size ( comm, &size);
  MPI_Comm_rank ( comm, &rank);
  src_round = 0;
  while (size > 1) {
    rroot = calc_split_rank(size);
    color = (color << 1);
    if (rank >= rroot)
      color++;
    if (rank == rroot)
      src_round = (round+1);
    MPI_Comm_split(comm, color, rank, &comm);
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    round++;
  }
  MPI_Allreduce(&round, &max_round, 1, MPI_INT, MPI_MAX, orig_comm);
  return max_round;
}

uint64_t kdtree_parallel_distribute(double **pts, uint64_t **idx,
				    uint64_t npts, uint32_t ndim,
				    double *left_edge, double *right_edge,
				    bool *periodic_left, bool *periodic_right,
				    exch_rec &src_exch, std::vector<exch_rec> &dst_exch,
				    
				    MPI_Comm comm) {
  bool local_debug = true;
  MPI_Comm orig_comm = comm;
  int orig_size, orig_rank;
  int root = 0;
  MPI_Comm_size ( orig_comm, &orig_size);
  MPI_Comm_rank ( orig_comm, &orig_rank);

  // Broadcast values from root
  MPI_Bcast(&ndim, 1, MPI_UNSIGNED, root, orig_comm);
  MPI_Bcast(left_edge, ndim, MPI_DOUBLE, root, orig_comm);
  MPI_Bcast(right_edge, ndim, MPI_DOUBLE, root, orig_comm);
  bcast_bool(periodic_left, ndim, root, orig_comm);
  bcast_bool(periodic_right, ndim, root, orig_comm);

  // Distribute randomly
  npts = parallel_distribute(pts, idx, npts, ndim, comm);
  debug_msg(local_debug, "kdtree_parallel_distribute",
	    "%lu points", npts);

  // Get mins/maxs
  double *mins = min_pts(*pts, npts, ndim);
  double *maxs = max_pts(*pts, npts, ndim);

  // Split until communicator is singular
  int size = orig_size, rank = orig_rank;
  int round = 0;
  int color = 1;
  int lroot, rroot, i;
  uint64_t lnpts, rnpts;
  std::vector<uint64_t> vec_npts = std::vector<uint64_t>(orig_size);
  int64_t split_idx = 0;
  uint32_t split_dim = 0;
  double split_val = 0.0;
  exch_rec this_exch;
  int src, dst;
  uint64_t left_idx = 0;
  src_exch = exch_rec();
  while (size > 1) {
    lroot = 0;
    rroot = calc_split_rank(size);
    debug_msg(local_debug, "kdtree_parallel_distribute",
	      "round %d, comm size now %d, lroot = %d, rroot = %d",
	      round, size, lroot, rroot);

    // Split points between lower/upper processes
    npts = redistribute_split(pts, idx, npts, ndim, mins, maxs,
			      split_idx, split_dim, split_val, 
			      rroot, comm);


    // Construct exchange
    src = orig_rank;
    dst = orig_rank;
    MPI_Bcast(&src, 1, MPI_INT, lroot, comm); // original rank of lroot is src
    MPI_Bcast(&dst, 1, MPI_INT, rroot, comm); // original rank of rroot is dst
    MPI_Allgather(&npts, 1, MPI_UNSIGNED_LONG,
		  &vec_npts[0], 1, MPI_UNSIGNED_LONG,
		  comm);
    for (i = lroot, lnpts = 0; i < rroot; i++) lnpts += vec_npts[i];
    for (i = rroot, rnpts = 0; i < size ; i++) rnpts += vec_npts[i];
    this_exch = exch_rec(src, dst, split_dim, split_val, lnpts - 1,
			 left_idx + lnpts, rnpts);

    // Split communicator and advance round
    if (rank < rroot) {
      // Left split
      color = (color << 1);
      maxs[split_dim] = split_val;
      right_edge[split_dim] = split_val;
      periodic_right[split_dim] = false;
      if (rank == lroot)
	dst_exch.insert(dst_exch.begin(), this_exch); // Smaller splits at front
    } else {
      // Right split
      color = (color << 1) + 1;
      mins[split_dim] = split_val;
      left_edge[split_dim] = split_val;
      periodic_left[split_dim] = false;
      if (rank == rroot)
	src_exch = this_exch;
      left_idx += lnpts;
    }
    debug_msg(local_debug, "kdtree_parallel_distribute",
	      "next color is %d", color);
    MPI_Comm_split(comm, color, rank, &comm);
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    round++;
  }

  // Free and return
  free(mins);
  free(maxs);
  return npts;
}

SplitNode* consolidate_split_tree(exch_rec src_exch,
				  std::vector<exch_rec> dst_exch,
				  MPI_Comm comm) {
  bool local_debug = true;
  int rank;
  MPI_Comm_rank( comm, &rank);
  // Reconstruct tree of splits
  std::vector<exch_rec>::iterator it;
  SplitNode *lnode = new SplitNode(rank);
  SplitNode *rnode = NULL;
  for (it = dst_exch.begin(); it != dst_exch.end(); it++) {
    debug_msg(local_debug, "consolidate_split_tree", 
	      "receiving from %d", it->dst);
    rnode = new SplitNode(-1);
    rnode->recv(it->dst, comm);
    lnode = new SplitNode(*it, lnode, rnode);
  }
  if (src_exch.src >= 0) {
    debug_msg(local_debug, "consolidate_split_tree",
	      "sending to %d", src_exch.src);
    lnode->send(src_exch.src, comm);
    delete lnode; // Don't delete rnode because its free as member of lnode
    lnode = NULL;
    rnode = NULL;
  }
  debug_msg(local_debug, "consolidate_split_tree",
	    "finished");
  return lnode;
}


