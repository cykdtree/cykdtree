#include "mpi.h"
#include <vector>
#include <math.h>
#include <stdint.h>
#include <iostream>
//#define TIMINGS 
#ifdef TIMINGS
#include <ctime>
#endif
//#include "c_kdtree.hpp"

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


void send_leafnode(int dp, Node *node) {
  int i = 0;
  uint32_t j, ndim = node->ndim;
  int *pe = (int*)malloc(ndim*sizeof(int));
  MPI_Send(&(node->ndim), 1, MPI_UNSIGNED, dp, i++, MPI_COMM_WORLD);
  MPI_Send(node->left_edge, ndim, MPI_DOUBLE, dp, i++, MPI_COMM_WORLD);
  MPI_Send(node->right_edge, ndim, MPI_DOUBLE, dp, i++, MPI_COMM_WORLD);
  for (j = 0; j < ndim; j++)
    pe[j] = (int)(node->periodic_left[j]);
  MPI_Send(pe, ndim, MPI_INT, dp, i++, MPI_COMM_WORLD);
  for (j = 0; j < ndim; j++)
    pe[j] = (int)(node->periodic_right[j]);
  MPI_Send(pe, ndim, MPI_INT, dp, i++, MPI_COMM_WORLD);
  for (j = 0; j < ndim; j++) {
    if (node->left_nodes[j] == NULL)
      pe[j] = 0;
    else 
      pe[j] = 1;
  }
  MPI_Send(pe, ndim, MPI_INT, dp, i++, MPI_COMM_WORLD);
  MPI_Send(&(node->leafid), 1, MPI_UNSIGNED, dp, i++, MPI_COMM_WORLD);
  free(pe);
}

Node* recv_leafnode(int sp) {
  int i = 0;
  uint32_t j, ndim;
  uint32_t leafid;
  MPI_Recv(&ndim, 1, MPI_UNSIGNED, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  int *pe = (int*)malloc(ndim*sizeof(int));
  bool *ple = (bool*)malloc(ndim*sizeof(bool));
  bool *pre = (bool*)malloc(ndim*sizeof(bool));
  double *re = (double*)malloc(ndim*sizeof(double));
  double *le = (double*)malloc(ndim*sizeof(double));
  std::vector<Node*> left_nodes;
  for (j = 0; j < ndim; j++)
    left_nodes.push_back(NULL);
  MPI_Recv(le, ndim, MPI_DOUBLE, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(re, ndim, MPI_DOUBLE, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(pe, ndim, MPI_INT, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (j = 0; j < ndim; j++)
    ple[j] = (bool)(pe[j]);
  MPI_Recv(pe, ndim, MPI_INT, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (j = 0; j < ndim; j++)
    pre[j] = (bool)(pe[j]);
  MPI_Recv(pe, ndim, MPI_INT, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (j = 0; j < ndim; j++) {
    if (pe[j] == 1)
      left_nodes[j] = new Node(); // empty place holder
  }
  MPI_Recv(&leafid, 1, MPI_UNSIGNED, sp, i++, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
  Node *node = new Node(ndim, le, re, ple, pre, 0, 0, leafid, left_nodes);
  free(pe);
  free(ple);
  free(pre);
  free(le);
  free(re);
  return node;
}



struct exch_rec {
  int src;
  int dst;
  uint32_t split_dim;
  double split_val;
  int64_t split_idx;
  uint64_t left_idx;
  uint64_t npts;
  exch_rec() {
    src = -1;
    dst = -1;
    split_dim = 0;
    split_val = 0.0;
    split_idx = -1;
    left_idx = 0;
    npts = 0;
  }
};

void print_exch(exch_rec e) {
  printf("src = %d, dst = %d, split_dim = %u, split_val = %f, split_idx = %ld, left_idx = %lu, npts = %lu\n",
	 e.src, e.dst, e.split_dim, e.split_val, e.split_idx,
	 e.left_idx, e.npts);
}

exch_rec init_exch_rec(int src0, int dst0, uint32_t split_dim0, 
		       double split_val0, int64_t split_idx0,
		       uint64_t src_left_idx, uint64_t src_npts) {
  exch_rec out;
  out.src = src0;
  out.dst = dst0;
  out.split_dim = split_dim0;
  out.split_val = split_val0;
  out.split_idx = split_idx0;
  out.left_idx = src_left_idx + split_idx0 + 1;
  out.npts = src_npts - split_idx0 - 1;
  return out;
}

MPI_Datatype init_mpi_exch_type() {
  const int nitems = 5;
  int blocklengths[nitems] = {2, 1, 1, 1, 2};
  MPI_Datatype types[nitems] = {MPI_INT, MPI_UNSIGNED, MPI_DOUBLE, MPI_LONG,
				MPI_UNSIGNED_LONG};
  MPI_Datatype mpi_exch_type;
  MPI_Aint offsets[nitems];
  offsets[0] = offsetof(exch_rec, src);
  offsets[1] = offsetof(exch_rec, split_dim);
  offsets[2] = offsetof(exch_rec, split_val);
  offsets[3] = offsetof(exch_rec, split_idx);
  offsets[4] = offsetof(exch_rec, left_idx);
  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_exch_type);
  MPI_Type_commit(&mpi_exch_type);
  return mpi_exch_type;
}


class ParallelKDTree
{
public:
  // Properties coordinating communication
  int rank;
  int size;
  int root;
  int rrank;
  MPI_Datatype mpi_exch_type;
  int available;
  int *all_avail;
  bool is_root;
  int *leaf2rank;
  std::vector<int> proc_order;
  // Properties specifying source/destination
  std::vector<std::vector<int> > lsplit;
  std::vector<std::vector<int> > rsplit;
  int src;
  std::vector<int> dst;
  std::vector<int> dst_past;
  std::vector<uint32_t> dst_nleaves_begin;
  std::vector<uint32_t> dst_nleaves_final;
  exch_rec src_exch;
  std::vector<exch_rec> dst_exch;
  // Properties that are the same across all processes
  uint32_t ndim;
  uint32_t leafsize;
  double *total_domain_left_edge;
  double *total_domain_right_edge;
  double *total_domain_width;
  bool *total_periodic;
  bool total_any_periodic;
  uint32_t total_num_leaves;
  double *all_lbounds;
  double *all_rbounds;
  // Properties for root node on this process
  KDTree *tree;
  double* all_pts;
  uint64_t* all_idx;
  uint64_t local_npts;
  double *local_domain_left_edge;
  double *local_domain_right_edge;
  bool *local_periodic_left;
  bool *local_periodic_right;
  bool local_any_periodic;
  uint64_t local_left_idx;
  // Convenience properties
  int* dummy;

  uint64_t npts;
  double *leaves_le;
  double *leaves_re;
  std::vector<uint32_t> leaf_count;
  
  ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
		 uint32_t leafsize0, double *left_edge, double *right_edge,
		 bool *periodic0, bool include_self = true) {
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    mpi_exch_type = init_mpi_exch_type();
    src = -1;
    all_avail = NULL;
    tree = NULL;
    local_left_idx = 0;
    total_num_leaves = 0;
    leaf2rank = NULL;
    leaves_le = NULL;
    leaves_re = NULL;
    all_lbounds = NULL;
    all_rbounds = NULL;
    double _t0 = begin_time();
    all_avail = (int*)malloc(size*sizeof(int));
    // Determine root
    if (pts != NULL) {
      root = rank;
      is_root = true;
      for (int i = 0; i < size; i++) {
	if (i != rank)
	  MPI_Send(&root, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
      available = 0;
      all_pts = pts;
      all_idx = idx;
      ndim = m;
      leafsize = leafsize0;
      npts = n;
      local_npts = n;
    } else {
      is_root = false;
      MPI_Recv(&root, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      available = 1;
      all_pts = NULL;
      all_idx = NULL;
      ndim = m;
      leafsize = leafsize0;
      npts = 0;
      local_npts = 0;
    }
    // Get information about global process
    rrank = (rank - root + size) % size;
    MPI_Bcast(&ndim, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&leafsize, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    // Allocate things
    dummy = (int*)malloc(ndim*sizeof(int));
    lsplit = std::vector<std::vector<int> >(ndim);
    rsplit = std::vector<std::vector<int> >(ndim);
    total_domain_left_edge = (double*)malloc(ndim*sizeof(double));
    total_domain_right_edge = (double*)malloc(ndim*sizeof(double));
    total_domain_width = (double*)malloc(ndim*sizeof(double));
    total_periodic = (bool*)malloc(ndim*sizeof(bool));
    total_any_periodic = false;
    local_domain_left_edge = (double*)malloc(ndim*sizeof(double));
    local_domain_right_edge = (double*)malloc(ndim*sizeof(double));
    local_periodic_left = (bool*)malloc(ndim*sizeof(bool));
    local_periodic_right = (bool*)malloc(ndim*sizeof(bool));
    local_any_periodic = false;
    // Domain information
    if (is_root) {
      memcpy(total_domain_left_edge, left_edge, ndim*sizeof(double));
      memcpy(total_domain_right_edge, right_edge, ndim*sizeof(double));
      memcpy(total_periodic, periodic0, ndim*sizeof(bool));
      memcpy(local_domain_left_edge, left_edge, ndim*sizeof(double));
      memcpy(local_domain_right_edge, right_edge, ndim*sizeof(double));
      memcpy(local_periodic_left, periodic0, ndim*sizeof(bool));
      memcpy(local_periodic_right, periodic0, ndim*sizeof(bool));
      for (uint32_t d = 0; d < ndim; d++) {
	dummy[d] = (int)(total_periodic[d]);
	total_domain_width[d] = right_edge[d] - left_edge[d];
	if (total_periodic[d]) {
	  total_any_periodic = true;
	  lsplit[d].push_back(rank);
	  rsplit[d].push_back(rank);
	}
      }
      local_any_periodic = total_any_periodic;
    } else {
      for (uint32_t d = 0; d < ndim; d++) {
	local_periodic_left[d] = false;
	local_periodic_right[d] = false;
      }
      local_any_periodic = false;
    }
    MPI_Bcast(total_domain_left_edge, ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(total_domain_right_edge, ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(total_domain_width, ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(dummy, ndim, MPI_INT, root, MPI_COMM_WORLD);
    if (!(is_root)) {
      for (uint32_t d = 0; d < ndim; d++) {
	total_periodic[d] = (bool)(dummy[d]);
	if (total_periodic[d]) {
	  total_any_periodic = true;
	}
      }
    }
    end_time(_t0, "init");
    set_comm_order();
    // build_tree();
    build_tree0();
  }
  ~ParallelKDTree() {
    delete(tree);
    free(dummy);
    free(all_avail);
    if (rank != root) {
      if (all_idx != NULL)
	free(all_idx);
      if (all_pts != NULL)
	free(all_pts);
    }
    free(total_domain_left_edge);
    free(total_domain_right_edge);
    free(total_domain_width);
    free(total_periodic);
    free(local_domain_left_edge);
    free(local_domain_right_edge);
    free(local_periodic_left);
    free(local_periodic_right);
    if (leaf2rank != NULL)
      free(leaf2rank);
    if (leaves_le != NULL)
      free(leaves_le);
    if (leaves_re != NULL)
      free(leaves_re);
    if (all_lbounds != NULL)
      free(all_lbounds);
    if (all_rbounds != NULL)
      free(all_rbounds);
    MPI_Type_free(&mpi_exch_type);
  }

  void send_exch(int idst, exch_rec st) {
    int tag = rank;
    MPI_Send(&st, 1, mpi_exch_type, idst, tag, MPI_COMM_WORLD);
  }

  exch_rec recv_exch(int isrc) {
    int tag = isrc;
    exch_rec st;
    MPI_Recv(&st, 1, mpi_exch_type, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return st;
  }

  void send_exch_vec(int idst, std::vector<exch_rec> st) {
    int tag = rank;
    int nexch = st.size();
    MPI_Send(&nexch, 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
    MPI_Send(&st[0], nexch, mpi_exch_type, idst, tag, MPI_COMM_WORLD);
  }

  std::vector<exch_rec> recv_exch_vec(int isrc,
				      std::vector<exch_rec> st = std::vector<exch_rec>()) {
    int tag = isrc;
    int nexch;
    MPI_Recv(&nexch, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    st.resize(nexch);
    MPI_Recv(&st[0], nexch, mpi_exch_type, isrc, tag, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    return st;
  }

  void send_node_neighbors(int dp, Node *node) {
    int tag = 4*ndim*(dp*size+rank);
    uint32_t d;
    int j, s;
    uint32_t *ids = NULL;
    for (d = 0; d < ndim; d++) {
      // left
      s = (int)(node->left_neighbors[d].size());
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      for (j = 0; j < s; j++)
	ids[j] = node->left_neighbors[d][j];
      MPI_Send(&s, 1, MPI_INT, dp, tag++, MPI_COMM_WORLD);
      MPI_Send(ids, s, MPI_UNSIGNED, dp, tag++, MPI_COMM_WORLD);
      // right
      s = (int)(node->right_neighbors[d].size());
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      for (j = 0; j < s; j++)
	ids[j] = node->right_neighbors[d][j];
      MPI_Send(&s, 1, MPI_INT, dp, tag++, MPI_COMM_WORLD);
      MPI_Send(ids, s, MPI_UNSIGNED, dp, tag++, MPI_COMM_WORLD);
    }
    if (ids != NULL)
      free(ids);
  }

  void recv_node_neighbors(int sp, Node *node) {
    int tag = 4*ndim*(rank*size+sp);
    uint32_t d;
    int j, s;
    uint32_t *ids = NULL;
    for (d = 0; d < ndim; d++) {
      // left
      MPI_Recv(&s, 1, MPI_INT, sp, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      MPI_Recv(ids, s, MPI_UNSIGNED, sp, tag++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      for (j = 0; j < s; j++)
	node->left_neighbors[d].push_back(ids[j]);
      // right
      MPI_Recv(&s, 1, MPI_INT, sp, tag++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      MPI_Recv(ids, s, MPI_UNSIGNED, sp, tag++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      for (j = 0; j < s; j++)
	node->right_neighbors[d].push_back(ids[j]);
    }
    if (ids != NULL)
      free(ids);
  }

  void add_dst(exch_rec e) {
    rsplit[e.split_dim].clear();
    rsplit[e.split_dim].push_back(e.dst);
  }

  void add_src(exch_rec e) {
    add_split(e, true);
    lsplit[e.split_dim].clear();
    lsplit[e.split_dim].push_back(e.src);
  }

  void add_split(exch_rec e, bool add_self = false) {
    uint32_t i, d;
    if ((e.dst == rank) and (!(add_self)))
      return;
    // printf("%d: Adding ", rank);
    // e.print();
    for (d = 0; d < ndim; d++) {
      // Left
      for (i = 0; i < lsplit[d].size(); i++) {
	if (e.src == lsplit[d][i]) {
	  if (e.split_dim == d) {
	    // Split is along shared dimension, use right of split
	    lsplit[d][i] = e.dst;
	  } else if (e.split_val > tree->domain_right_edge[e.split_dim]) {
	    // Split is farther right than domain, use left of split
	    lsplit[d][i] = e.src;
	  } else if (e.split_val < tree->domain_left_edge[e.split_dim]) {
	    // Split is frather left than domain, use right of split
	    lsplit[d][i] = e.dst;
	  } else {
	    // Use both left and right
	    lsplit[d].push_back(e.dst);
	  }
	}
      }
      // Right
      for (i = 0; i < rsplit[d].size(); i++) {
	if (e.src == rsplit[d][i]) {
	  if (e.split_dim == d) {
	    // Split is along shared dimension, use left of split
	    rsplit[d][i] = e.src;
	  } else if (e.split_val > tree->domain_right_edge[e.split_dim]) {
	    // Split is farther right than domain, use left of split
	    rsplit[d][i] = e.src;
	  } else if (e.split_val < tree->domain_left_edge[e.split_dim]) {
	    // Split is frather left than domain, use right of split
	    rsplit[d][i] = e.dst;
	  } else {
	    // Use both left and right
	    rsplit[d].push_back(e.dst);
	  }
	}
      }
    }
  }

  void add_splits(std::vector<exch_rec> evec, bool add_self = false) {
    std::vector<exch_rec>::iterator it;
    for (it = evec.begin(); it != evec.end(); ++it) {
      add_split(*it);
    }
  }

  void print_neighbors() {
    uint32_t i, d;
    int rank, size;
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    printf("%d: left = [", rank);
    for (d = 0; d < ndim; d++) {
      for (i = 0; i < lsplit[d].size(); i++)
	printf("%d ", (int)(lsplit[d][i]));
      printf(", ");
    }
    printf("]\n");
    printf("%d: right = [", rank);
    for (d = 0; d < ndim; d++) {
      for (i = 0; i < rsplit[d].size(); i++)
	printf("%d ", (int)(rsplit[d][i]));
      printf(", ");
    }
    printf("]\n");
  }


  void send_neighbors(int idst) {
    int np;
    uint32_t d;
    int tag = rank;
    // Left split
    for (d = 0; d < ndim; d++) {
      np = lsplit[d].size();
      MPI_Send(&np, 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
      MPI_Send(&(lsplit[d][0]), np, MPI_INT, idst, tag, MPI_COMM_WORLD);
    }
    // Right split
    for (d = 0; d < ndim; d++) {
      np = rsplit[d].size();
      MPI_Send(&np, 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
      MPI_Send(&(rsplit[d][0]), np, MPI_INT, idst, tag, MPI_COMM_WORLD);
    }
  }

  void recv_neighbors(int isrc) {
    uint32_t d;
    int np;
    int tag = isrc;
    lsplit = std::vector<std::vector<int> >(ndim);
    rsplit = std::vector<std::vector<int> >(ndim);
    // Left split
    for (d = 0; d < ndim; d++) {
      MPI_Recv(&np, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      lsplit[d].resize(np);
      MPI_Recv(&(lsplit[d][0]), np, MPI_INT, isrc, tag, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }
    // Right split
    for (d = 0; d < ndim; d++) {
      MPI_Recv(&np, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      rsplit[d].resize(np);
      MPI_Recv(&(rsplit[d][0]), np, MPI_INT, isrc, tag, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }
  }

  void send_part(exch_rec dst, int *exch_ple, int *exch_pre) {
    // Get variables to send
    for (uint32_t d = 0; d < ndim; d++) {
      exch_ple[d] = (int)(tree->periodic_left[d]);
      exch_pre[d] = (int)(tree->periodic_right[d]);
    }
    double *pts_send;
    // Send variables
    MPI_Send(tree->domain_mins, ndim, MPI_DOUBLE, dst.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(tree->domain_maxs, ndim, MPI_DOUBLE, dst.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(tree->domain_left_edge, ndim, MPI_DOUBLE, dst.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(tree->domain_right_edge, ndim, MPI_DOUBLE, dst.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(exch_ple, ndim, MPI_INT, dst.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(exch_pre, ndim, MPI_INT, dst.dst, rank,
	     MPI_COMM_WORLD);
    // Send points
    uint64_t npts_send = dst.npts;
    pts_send = (double*)malloc(npts_send*ndim*sizeof(double));
    for (uint64_t i = 0; i < npts_send; i++) {
      memcpy(pts_send + ndim*i,
	     all_pts + ndim*(all_idx[i + dst.split_idx + 1]),
	     ndim*sizeof(double));
    }
    MPI_Send(pts_send, ndim*npts_send, MPI_DOUBLE, dst.dst, rank,
	     MPI_COMM_WORLD);
    free(pts_send);
    // Update local info
    tree->domain_maxs[dst.split_dim] = dst.split_val;
    tree->domain_right_edge[dst.split_dim] = dst.split_val;
    tree->periodic_right[dst.split_dim] = false;
    tree->periodic[dst.split_dim] = false;
    tree->domain_width[dst.split_dim] = dst.split_val - tree->domain_left_edge[dst.split_dim];
    tree->npts = dst.split_idx + 1;
    npts = dst.split_idx + 1;
    tree->any_periodic = false;
    for (uint32_t d = 0; d < ndim; d++) {
      if (tree->periodic[d]) {
	tree->any_periodic = true;
      }
    }
    // Send neighbors
    send_neighbors(dst.dst);
    add_dst(dst);
  }

  void recv_part(exch_rec src, int *exch_ple, int *exch_pre) {
    // Receive information about incoming domain
    MPI_Recv(tree->domain_mins, ndim, MPI_DOUBLE, src.src, src.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(tree->domain_maxs, ndim, MPI_DOUBLE, src.src, src.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(tree->domain_left_edge, ndim, MPI_DOUBLE, src.src, src.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(tree->domain_right_edge, ndim, MPI_DOUBLE, src.src, src.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(exch_ple, ndim, MPI_INT, src.src, src.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(exch_pre, ndim, MPI_INT, src.src, src.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Receive points
    all_pts = (double*)malloc(src.npts*ndim*sizeof(double));
    MPI_Recv(all_pts, ndim*src.npts, MPI_DOUBLE, src.src, src.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Create indices
    all_idx = (uint64_t*)malloc(src.npts*sizeof(uint64_t));
    for (uint64_t i = 0; i < src.npts; i++)
      all_idx[i] = i;
    // Update local info
    available = 0;
    src_exch = src;
    npts = src.npts;
    local_npts = src.npts;
    tree->npts = src.npts;
    tree->domain_mins[src.split_dim] = src.split_val;
    tree->domain_left_edge[src.split_dim] = src.split_val;
    exch_ple[src.split_dim] = 0;
    tree->all_pts = all_pts;
    tree->all_idx = all_idx;
    for (uint32_t d = 0; d < ndim; d++) {
      tree->periodic_left[d] = (bool)exch_ple[d];
      tree->periodic_right[d] = (bool)exch_pre[d];
      tree->domain_width[d] = tree->domain_right_edge[d] - tree->domain_left_edge[d];
      if ((tree->periodic_left[d]) && (tree->periodic_right[d])) {
	tree->periodic[d] = true;
	tree->any_periodic = true;
      }
    }
    // Recieve neighbors and previous splits
    recv_neighbors(src.src);
    add_src(src);
  }

  void set_comm_order() {
    double _t0 = begin_time();
    int nrecv = total_available(true);
    int nsend = 0, nexch = 0;
    while (nrecv > 0) {
      nsend = size - nrecv;
      nexch = std::min(nrecv, nsend);
      //printf("%d: nrecv = %d, nsend = %d, nexch = %d\n", rank, nrecv, nsend, nexch);
      if (available) {
	// Get source
	if (rrank < (nsend+nexch)) {
	  src = (root + rrank - nexch) % size;
	  available = false;
	}
      } else {
	// Get destination
	if (rrank < nexch)
	  dst.push_back((root + rrank + nexch) % size);
      }
      nrecv = total_available(true);
    }
    end_time(_t0, "set_comm_order");
  }

  void recv_build_begin(int sp, uint64_t &Lidx, uint64_t &n,
			double *LE, double *RE,
			bool *PLE, bool *PRE,
			double *mins, double *maxs) {
    int tag = sp;
    uint32_t d;
    // Receive domain bounds
    // printf("%d: Receiving beginning build from %d\n", rank, sp);
    MPI_Recv(LE, ndim, MPI_DOUBLE, sp, tag, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    MPI_Recv(RE, ndim, MPI_DOUBLE, sp, tag, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    MPI_Recv(dummy, ndim, MPI_INT, sp, tag, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    for (d = 0; d < ndim; d++)
      PLE[d] = (bool)(dummy[d]);
    MPI_Recv(dummy, ndim, MPI_INT, sp, tag, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    for (d = 0; d < ndim; d++)
      PRE[d] = (bool)(dummy[d]);
    MPI_Recv(mins, ndim, MPI_DOUBLE, sp, tag, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    MPI_Recv(maxs, ndim, MPI_DOUBLE, sp, tag, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    // Receive scalars
    MPI_Recv(&Lidx, 1, MPI_UNSIGNED_LONG, sp, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(&n, 1, MPI_UNSIGNED_LONG, sp, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // Allocate points and create local index
    all_pts = (double*)malloc(ndim*n*sizeof(double));
    all_idx = (uint64_t*)malloc(n*sizeof(uint64_t));
    for (uint64_t i = 0; i < n; i++)
      all_idx[i] = i;
    // Receive points
    MPI_Recv(all_pts, ndim*n, MPI_DOUBLE, sp, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // printf("%d: Received beginning build from %d\n", rank, sp);
  }

  void send_build_begin(int dp, uint64_t Lidx, uint64_t n,
			double *LE, double *RE,
			bool *PLE, bool *PRE,
			double *mins, double *maxs) {
    // printf("%d: Sending beginning build to %d\n", rank, dp);
    int tag = rank;
    uint32_t d;
    int *PLE_i = (int*)malloc(ndim*sizeof(int));
    int *PRE_i = (int*)malloc(ndim*sizeof(int));
    for (d = 0; d < ndim; d++) {
      PLE_i[d] = (int)(PLE[d]);
      PRE_i[d] = (int)(PRE[d]);
    }
    // Send domain bounds
    MPI_Send(LE, ndim, MPI_DOUBLE, dp, tag, MPI_COMM_WORLD);
    MPI_Send(RE, ndim, MPI_DOUBLE, dp, tag, MPI_COMM_WORLD);
    for (d = 0; d < ndim; d++) 
      dummy[d] = (int)(PLE[d]);
    MPI_Send(dummy, ndim, MPI_INT, dp, tag, MPI_COMM_WORLD);
    for (d = 0; d < ndim; d++) 
      dummy[d] = (int)(PRE[d]);
    MPI_Send(dummy, ndim, MPI_INT, dp, tag, MPI_COMM_WORLD);
    MPI_Send(mins, ndim, MPI_DOUBLE, dp, tag, MPI_COMM_WORLD);
    MPI_Send(maxs, ndim, MPI_DOUBLE, dp, tag, MPI_COMM_WORLD);
    // Send scalars
    MPI_Send(&Lidx, 1, MPI_UNSIGNED_LONG, dp, tag, MPI_COMM_WORLD);
    MPI_Send(&n, 1, MPI_UNSIGNED_LONG, dp, tag, MPI_COMM_WORLD);
    // Allocate points and copy over using index
    double *pts_send = (double*)malloc(ndim*n*sizeof(double));
    for (uint64_t i = 0; i < n; i++) {
      memcpy(pts_send + ndim*i,
	     all_pts + ndim*all_idx[Lidx+i],
	     ndim*sizeof(double));
    }
    // Send points
    MPI_Send(pts_send, ndim*n, MPI_DOUBLE, dp, tag, MPI_COMM_WORLD);
    free(pts_send);
    // printf("%d: Sent beginning build to %d\n", rank, dp);
  }

  Node *recv_build_final(int sp, uint64_t Lidx, uint64_t n,
			 double *LE, double *RE,
			 bool *PLE, bool *PRE,
			 std::vector<Node*> r_left_nodes) {
    double _t0 = begin_time();
    // printf("%d: Receiving final build from %d\n", rank, sp);
    int tag = sp;
    // Receive idx
    uint64_t j;
    // uint64_t n, j;
    // MPI_Recv(&n, 1, MPI_UNSIGNED_LONG, sp, tag, MPI_COMM_WORLD, 
    // 	     MPI_STATUS_IGNORE);
    uint64_t *idx_exch = (uint64_t*)malloc(n*sizeof(uint64_t));
    MPI_Recv(idx_exch, n, MPI_UNSIGNED_LONG, sp, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    for (j = 0; j < n; j++)
      idx_exch[j] = all_idx[Lidx + idx_exch[j]];
    memcpy(all_idx + Lidx, idx_exch, n*sizeof(uint64_t));
    free(idx_exch);
    // Receive node & count leaves it adds
    dst_nleaves_begin.push_back(tree->num_leaves);
    Node *rnode = recv_node(sp, Lidx, LE, RE, PLE, PRE, r_left_nodes);
    dst_nleaves_final.push_back(tree->num_leaves);
    dst_past.push_back(sp);
    // printf("%d: Received final build from %d\n", rank, sp);
    end_time(_t0, "recv_build_final");
    return rnode;
  }

  void send_build_final(int dp) {
    // printf("%d: Sending final build to %d\n", rank, dp);
    int tag = rank;
    // Send idx
    // MPI_Send(&(tree->npts), 1, MPI_UNSIGNED_LONG, dp, tag, MPI_COMM_WORLD);
    MPI_Send(all_idx, tree->npts, MPI_UNSIGNED_LONG, dp, tag,
	     MPI_COMM_WORLD);
    // Send node
    send_node(dp, tree->root);
    // printf("%d: Sent final build to %d\n", rank, dp);
  }

  Node* recv_node(int sp, uint64_t prev_Lidx,
		  double *le, double *re, bool *ple, bool *pre,
		  std::vector<Node*> left_nodes) {
    int tag = 0;
    Node *out;
    int is_empty, is_leaf;
    MPI_Recv(&is_empty, 1, MPI_INT, sp, tag++, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
    // Empty node
    if (is_empty) {
      out = new Node();
      return out;
    }
    // Receive properties innernodes and leaf nodes have
    uint32_t d;
    uint64_t Lidx;
    MPI_Recv(&Lidx, 1, MPI_UNSIGNED_LONG, sp, tag++, MPI_COMM_WORLD,
    	     MPI_STATUS_IGNORE);
    // Proceed based on status as leaf
    MPI_Recv(&is_leaf, 1, MPI_INT, sp, tag++, MPI_COMM_WORLD,
	     MPI_STATUS_IGNORE);
    if (is_leaf) {
      uint64_t children;
      // int leafid;
      // Leaf properties
      MPI_Recv(&children, 1, MPI_UNSIGNED_LONG, sp, tag++,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // MPI_Recv(&leafid, 1, MPI_INT, sp, tag++, MPI_COMM_WORLD,
      // 	       MPI_STATUS_IGNORE);
      out = new Node(ndim, le, re, ple, pre, 
		     prev_Lidx + Lidx, children, 
		     tree->num_leaves, left_nodes);
      tree->leaves.push_back(out);
      tree->num_leaves++;
    } else {
      // Innernode properties
      uint32_t sdim;
      double split;
      MPI_Recv(&sdim, 1, MPI_UNSIGNED, sp, tag++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      MPI_Recv(&split, 1, MPI_DOUBLE, sp, tag++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      // Properties to send to children
      double *le_r = (double*)malloc(ndim*sizeof(double));
      double *re_l = (double*)malloc(ndim*sizeof(double));
      bool *ple_r = (bool*)malloc(ndim*sizeof(bool));
      bool *pre_l = (bool*)malloc(ndim*sizeof(bool));
      memcpy(le_r, le, ndim*sizeof(double));
      memcpy(re_l, re, ndim*sizeof(double));
      memcpy(ple_r, ple, ndim*sizeof(double));
      memcpy(pre_l, pre, ndim*sizeof(double));
      le_r[sdim] = split;
      re_l[sdim] = split;
      ple_r[sdim] = false;
      pre_l[sdim] = false;
      std::vector<Node*> greater_left_nodes;
      for (d = 0; d < ndim; d++)
	greater_left_nodes.push_back(left_nodes[d]);
      // Child nodes
      Node *less = recv_node(sp, prev_Lidx,
			     le, re_l, ple, pre_l, left_nodes);
      greater_left_nodes[sdim] = less;
      Node *greater = recv_node(sp, prev_Lidx,
				le_r, re, ple_r, pre, greater_left_nodes);
      out = new Node(ndim, le, re, ple, pre, prev_Lidx+Lidx, sdim, split, 
		     less, greater, left_nodes);
      // Free child properties
      free(le_r);
      free(re_l);
      free(ple_r);
      free(pre_l);
    }
    return out;
  }


  void send_node(int dp, Node *node) {
    int tag = 0;
    int is_empty, is_leaf;
    is_empty = node->is_empty;
    MPI_Send(&is_empty, 1, MPI_INT, dp, tag++, MPI_COMM_WORLD);
    // Empty node
    if (node->is_empty)
      return;
    // Send properties innernodes and leaf nodes have
    MPI_Send(&(node->left_idx), 1, MPI_UNSIGNED_LONG, dp, tag++, MPI_COMM_WORLD);
    // Proceed based on status as leaf
    is_leaf = (int)(node->is_leaf);
    MPI_Send(&is_leaf, 1, MPI_INT, dp, tag++, MPI_COMM_WORLD);
    if (is_leaf) {
      // Leaf properties
      MPI_Send(&(node->children), 1, MPI_UNSIGNED_LONG, dp, tag++,
	       MPI_COMM_WORLD);
      // MPI_Send(&(node->leafid), 1, MPI_INT, dp, tag++, MPI_COMM_WORLD);
    } else {
      // Innernode properties
      MPI_Send(&(node->split_dim), 1, MPI_UNSIGNED, dp, tag++, MPI_COMM_WORLD);
      MPI_Send(&(node->split), 1, MPI_DOUBLE, dp, tag++, MPI_COMM_WORLD);
      // Child nodes
      send_node(dp, node->less);
      send_node(dp, node->greater);
    }
  }

  void build_tree(bool include_self = false) {
    uint32_t d;
    std::vector<Node*> left_nodes;
    for (d = 0; d < ndim; d++)
      left_nodes.push_back(NULL);

    // Create tree
    if (is_root) {
      // Root should already have points etc.
      tree = new KDTree(all_pts, all_idx, local_npts, ndim, leafsize,
			total_domain_left_edge, total_domain_right_edge,
			total_periodic, include_self, true);
    } else {
      // Other processes will receive from source during build
      double *mins = (double*)malloc(ndim*sizeof(double));
      double *maxs = (double*)malloc(ndim*sizeof(double));
      recv_build_begin(src, local_left_idx, local_npts,
		       local_domain_left_edge, local_domain_right_edge,
		       local_periodic_left, local_periodic_right,
		       mins, maxs);
      tree = new KDTree(all_pts, all_idx, local_npts, ndim, leafsize,
			local_domain_left_edge, local_domain_right_edge,
			local_periodic_left, local_periodic_right,
			mins, maxs, include_self, true);
      free(mins);
      free(maxs);
    }
    // Build tree
    double _t0 = begin_time();
    tree->root = build(0, tree->npts,
		       tree->domain_left_edge, tree->domain_right_edge,
		       tree->periodic_left, tree->periodic_right,
		       tree->domain_mins, tree->domain_maxs, left_nodes);
    end_time(_t0, "build_tree");

    // Send root back to source
    if (!(is_root))
      send_build_final(src);

    // Finalize neighbors
    finalize_neighbors(include_self);

    // Send leaf info
    if (is_root)
      total_num_leaves = tree->num_leaves;
    MPI_Bcast(&total_num_leaves, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
  }

  void finalize_neighbors(bool include_self = false) {
    double _t0 = begin_time();
    std::vector<Node*>::iterator it;
    int tag;
    uint32_t leafid;
    // Clear existing neighbors
    if (!(is_root)) 
      tree->clear_neighbors();
    // Receive neighbors from parent
    if (!(is_root)) {
      tag = rank;
      for (it = tree->leaves.begin(); it != tree->leaves.end(); ++it) {
	MPI_Recv(&leafid, 1, MPI_UNSIGNED, src, tag, MPI_COMM_WORLD,
		 MPI_STATUS_IGNORE);
	(*it)->leafid = leafid;
	recv_node_neighbors(src, *it);
      }
      // printf("%d: Received neighbors from %d\n", rank, src);
    }
    // Finalize neighbors locally
    if (is_root)
      tree->finalize_neighbors(include_self);
    // Send neighbors to children
    uint32_t i, l, beg, end;
    for (i = 0; i < dst_past.size(); i++) {
      beg = dst_nleaves_begin[i];
      end = dst_nleaves_final[i];
      tag = dst_past[i];
      for (l = beg; l < end; l++) {
	leafid = tree->leaves[l]->leafid;
	MPI_Ssend(&leafid, 1, MPI_UNSIGNED, dst_past[i], tag,
		  MPI_COMM_WORLD);
	send_node_neighbors(dst_past[i], tree->leaves[l]);
      }
      // printf("%d: Sent neighbors to %d\n", rank, dst_past[i]);
    }
    end_time(_t0, "finalize_neighbors");
  }

  Node* build(uint64_t Lidx, uint64_t n,
	      double *LE, double *RE,
	      bool *PLE, bool *PRE,
	      double *mins, double *maxs,
	      std::vector<Node*> left_nodes) {
    if (dst.size() == 0) {
      // Build tree
      return tree->build(Lidx, n, LE, RE, PLE, PRE,
			 mins, maxs, left_nodes);
    } else {
      int idst = dst[0];
      dst.erase(dst.begin());

      // Split
      uint32_t dmax, d;
      int64_t split_idx = 0;
      double split_val = 0.0;
      dmax = tree->split(Lidx, n, mins, maxs, split_idx, split_val, rank);

      // Determine boundaries
      uint64_t lN = split_idx - Lidx + 1;
      uint64_t rN = n - lN;
      double *lRE = (double*)malloc(ndim*sizeof(double));
      double *rLE = (double*)malloc(ndim*sizeof(double));
      bool *lPRE = (bool*)malloc(ndim*sizeof(bool));
      bool *rPLE = (bool*)malloc(ndim*sizeof(bool));
      double *lmaxs = (double*)malloc(ndim*sizeof(double));
      double *rmins = (double*)malloc(ndim*sizeof(double));
      std::vector<Node*> r_left_nodes;
      memcpy(lmaxs, maxs, ndim*sizeof(double));
      memcpy(rmins, mins, ndim*sizeof(double));
      memcpy(lRE, RE, ndim*sizeof(double));
      memcpy(rLE, LE, ndim*sizeof(double));
      memcpy(lPRE, PRE, ndim*sizeof(double));
      memcpy(rPLE, PLE, ndim*sizeof(double));
      for (d = 0; d < ndim; d++) 
        r_left_nodes.push_back(left_nodes[d]);
      lmaxs[dmax] = split_val;
      rmins[dmax] = split_val;
      lRE[dmax] = split_val;
      rLE[dmax] = split_val;
      lPRE[dmax] = false;
      rPLE[dmax] = false;

      // Send right half to another process
      send_build_begin(idst, Lidx+lN, rN, rLE, RE, rPLE, PRE, 
		       rmins, maxs);
       
      // Build left node
      Node *lnode = build(Lidx, lN, LE, lRE, PLE, lPRE,
			  mins, lmaxs, left_nodes);
      
      // Receive right node
      r_left_nodes[dmax] = lnode;
      Node *rnode = recv_build_final(idst, Lidx+lN, rN,
				     rLE, RE, rPLE, PRE,
				     r_left_nodes);

      // Create innernode
      Node* out = new Node(ndim, LE, RE, PLE, PRE, Lidx, dmax, split_val,
                           lnode, rnode, left_nodes);

      free(lRE);
      free(rLE);
      free(lPRE);
      free(rPLE);
      free(lmaxs);
      free(rmins);
      return out;
    }
  }

  void partition() {
    double _t0 = begin_time();
    double *exch_mins = (double*)malloc(ndim*sizeof(double));
    double *exch_maxs = (double*)malloc(ndim*sizeof(double));
    double *exch_le = (double*)malloc(ndim*sizeof(double));
    double *exch_re = (double*)malloc(ndim*sizeof(double));
    int *exch_ple = (int*)malloc(ndim*sizeof(int));
    int *exch_pre = (int*)malloc(ndim*sizeof(int));
    exch_rec this_exch;
    std::vector<exch_rec> new_splits;
    std::vector<int>::iterator it;
    // Receive from source
    if (src != -1) {
      this_exch = recv_exch(src);
      recv_part(this_exch, exch_ple, exch_pre);
      // Receive splits from parent
      new_splits = recv_exch_vec(src_exch.src);
      // Add splits
      add_splits(new_splits);
      new_splits.clear();
    }
    // Send to destinations
    for (it = dst.begin(); it != dst.end(); it++) {
      // Determine parameters of exchange
      this_exch = split(*it);
      // Send to process
      send_exch(*it, this_exch);
      send_part(this_exch, exch_ple, exch_pre);
      // Receive new splits from children 
      for (uint32_t i = 0; i < dst_exch.size(); i++)
	new_splits = recv_exch_vec(dst_exch[i].dst, new_splits);
      new_splits.push_back(this_exch);
      // Send new splits to parent & receive update back
      if (src_exch.src != -1) {
	send_exch_vec(src_exch.src, new_splits);
	new_splits = recv_exch_vec(src_exch.src);
      }
      // Add new child to list of destinations (at the front)
      dst_exch.insert(dst_exch.begin(), this_exch); // Smaller splits at front
      // Send new splits to children (including the new child)
      for (uint32_t i = 0; i < dst_exch.size(); i++)
	send_exch_vec(dst_exch[i].dst, new_splits);
      // Add splits
      add_splits(new_splits);
      new_splits.clear();
    }
    free(exch_mins);
    free(exch_maxs);
    free(exch_le);
    free(exch_re);
    free(exch_ple);
    free(exch_pre);
    end_time(_t0, "partition");
  }

  exch_rec split(int other_rank) {
    double _t0 = begin_time();
    exch_rec this_exch;
    uint32_t dsplit;
    int64_t split_idx = 0;
    double split_val = 0.0;
    dsplit = tree->split(0, npts, tree->domain_mins, tree->domain_maxs,
			 split_idx, split_val);
    this_exch = init_exch_rec(rank, other_rank, dsplit,
			      split_val, split_idx, local_left_idx, npts);
    end_time(_t0, "split");
    return this_exch;
  }

  void build_tree0(bool include_self = false) {
    // Create trees and partition
    tree = new KDTree(all_pts, all_idx, local_npts, ndim, leafsize,
    		      total_domain_left_edge, total_domain_right_edge,
    		      local_periodic_left, local_periodic_right,
    		      NULL, NULL,
    		      include_self, true);
    partition();
    double _t0 = begin_time();
    // Build, don't include self in all neighbors for now
    tree->build_tree(include_self);
    end_time(_t0, "build_tree0");
    consolidate(include_self);
  }

  void consolidate(bool include_self) {
    double _t0 = begin_time();
    consolidate_order();
    consolidate_leaves();
    // consolidate_leaf_edges();
    consolidate_idx();
    consolidate_neighbors(include_self);
    end_time(_t0, "consolidate");
  }

  void consolidate_bounds() {
    // Send tree bounds to root
    if (rank == root) {
      all_lbounds = (double*)malloc(ndim*size*sizeof(double));
      all_rbounds = (double*)malloc(ndim*size*sizeof(double));
      memcpy(all_lbounds+ndim*rank, tree->domain_left_edge,
	     ndim*sizeof(double));
      memcpy(all_rbounds+ndim*rank, tree->domain_right_edge,
	     ndim*sizeof(double));
      for (int i = 0; i < size; i++) {
	if (rank != i) {
	  MPI_Recv(all_lbounds+ndim*i, ndim, MPI_DOUBLE, i, i, 
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(all_rbounds+ndim*i, ndim, MPI_DOUBLE, i, i, 
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
      }
    } else {
      MPI_Send(tree->domain_left_edge, ndim, MPI_DOUBLE, root, rank,
	       MPI_COMM_WORLD);
      MPI_Send(tree->domain_right_edge, ndim, MPI_DOUBLE, root, rank,
	       MPI_COMM_WORLD);
    }
  }

  void exch_neigh(uint32_t d, std::vector<std::vector<Node*> > lsend,
		  bool p) {
    int i, k;
    uint32_t j;
    int nsend, nrecv;
    Node *node;
    std::vector<Node*>::iterator it;
    nsend = lsend[d].size();
    for (i = 0; i < size; i++) {
      if (i == rank) {
	if (p == tree->periodic_right[d]) {
	  // Receive from right
	  for (j = 0; j < rsplit[d].size(); j++) {
	    if (rsplit[d][j] == rank) {
	      // Add periodic neighbor from this process
	      if (p) {
		for (k = 0; k < nsend; k++) {
		  node = lsend[d][k];
		  node->add_neighbors(tree->root, d);
		}
	      }
	    } else {
	      // Add neighbors from right
	      // printf("%d: Recieving from %d\n", rank, rsplit[d][j]);
	      MPI_Recv(&nrecv, 1, MPI_INT, rsplit[d][j], rsplit[d][j], 
		       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      // printf("%d: Recieving %d from %d\n", rank, nrecv, rsplit[d][j]);
	      for (k = 0; k < nrecv; k++) {
		node = recv_leafnode(rsplit[d][j]);
		node->left_nodes[d] = tree->root;
		if (p) {
		  for (it = tree->leaves.begin(); it != tree->leaves.end(); ++it)
		    add_neighbors_periodic(node, *it);
		} else {
		  node->add_neighbors(tree->root, d);
		}
		// Send neighbors back
		send_node_neighbors(rsplit[d][j], node);
	      }
	    }
	  }
	}
      } else {
	if (p == tree->periodic_left[d]) {
	  // Send to left
	  for (j = 0; j < lsplit[d].size(); j++) {
	    if (lsplit[d][j] == i) {
	      // printf("%d: Sending to %d\n", rank, lsplit[d][j]);
	      MPI_Send(&nsend, 1, MPI_INT, lsplit[d][j], rank,
		       MPI_COMM_WORLD);
	      // printf("%d: Sending %d to %d\n", rank, nsend, lsplit[d][j]);
	      for (k = 0; k < nsend; k++) {
		node = lsend[d][k];
		send_leafnode(lsplit[d][j], node);
		// Recieve neighbors back
		recv_node_neighbors(lsplit[d][j], node);
	      }
	      break;
	    }
	  }
	}
      }
    }
  }


  void consolidate_neighbors(bool include_self) {
    uint32_t d;
    std::vector<Node*>::iterator it;
    std::vector<std::vector<Node*> > leaves_send;
    leaves_send = std::vector<std::vector<Node*> >(ndim);
    // Identify local leaves with missing neighbors
    for (it = tree->leaves.begin();
	 it != tree->leaves.end(); ++it) {
      for (d = 0; d < ndim; d++) {
	if (((*it)->left_nodes[d] == NULL) and (lsplit[d].size() > 0)) {
	  leaves_send[d].push_back(*it);
	}
      }
    }
    // Non-periodic neighbors
    for (d = 0; d < ndim; d++)
      exch_neigh(d, leaves_send, false);
    // Periodic neighbors
    for (d = 0; d < ndim; d++)
      exch_neigh(d, leaves_send, true);
    // Finalize neighbors
    tree->finalize_neighbors(include_self);
  }

  void add_neighbors_periodic(Node *leaf, Node *prev) {
    uint32_t d0;
    for (d0 = 0; d0 < ndim; d0++) {
      tree->add_neighbors_periodic(leaf, prev, d0);
    }
  }

  void consolidate_idx() {
    uint64_t nexch, i, j;
    uint64_t *idx_exch;
    // Receive ids from child processes
    for (i = 0; i < dst_exch.size(); ++i) {
      nexch = dst_exch[i].npts;
      idx_exch = (uint64_t*)malloc(nexch*sizeof(uint64_t));
      MPI_Recv(idx_exch, nexch, MPI_UNSIGNED_LONG, dst_exch[i].dst, 
	       dst_exch[i].dst, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (j = 0; j < nexch; j++)
	idx_exch[j] = all_idx[dst_exch[i].split_idx + 1 + idx_exch[j]];
      memcpy(all_idx + dst_exch[i].split_idx + 1,
	     idx_exch, nexch*sizeof(uint64_t));
      free(idx_exch);
    }
    // Send ids to parent process
    if (src_exch.src != -1) 
      MPI_Send(all_idx, local_npts, MPI_UNSIGNED_LONG, src_exch.src,
	       rank, MPI_COMM_WORLD);
  }

  void consolidate_order() {
    int nexch, nprev;
    uint32_t i;
    // Add self
    proc_order.resize(size);
    proc_order[0] = rank;
    nprev = 1;
    // Receive processes from child
    for (i = 0; i < dst_exch.size(); ++i) {
      MPI_Recv(&nexch, 1, MPI_INT, dst_exch[i].dst, dst_exch[i].dst,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      // proc_order.resize(nprev+nexch);
      MPI_Recv(&proc_order[nprev], nexch, MPI_INT, dst_exch[i].dst,
	       dst_exch[i].dst, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      nprev += nexch;
    }
    // Send processes to parent
    if (src_exch.src != -1) {
      MPI_Send(&nprev, 1, MPI_INT, src_exch.src, rank, MPI_COMM_WORLD);
      MPI_Send(&proc_order[0], nprev, MPI_INT, src_exch.src, rank,
	       MPI_COMM_WORLD);
    }
    // Broadcast to all processors
    nprev = size;
    // MPI_Bcast(&nprev, 1, MPI_INT, root, MPI_COMM_WORLD);
    // printf("%d: nprev = %d, size = %d\n", rank, nprev, size);
    // proc_order.resize(nprev);
    MPI_Bcast(&proc_order[0], nprev, MPI_INT, root, MPI_COMM_WORLD);
  }

  void consolidate_leaves() {
    leaf_count.resize(size);
    uint32_t local_count = tree->num_leaves;
    uint32_t total_count = 0;
    std::vector<Node*>::iterator it;
    int j;
    MPI_Allgather(&local_count, 1, MPI_UNSIGNED,
		  &leaf_count[0], 1, MPI_UNSIGNED, 
		  MPI_COMM_WORLD);
    // Count all leaves
    total_num_leaves = 0;
    for (j = 0; j < size; j++) {
      if (proc_order[j] == rank)
	total_count = total_num_leaves;
      total_num_leaves += leaf_count[j];
    }
    // printf("%d: nprev = %d\n", rank, total_count);
    for (it = tree->leaves.begin(); it != tree->leaves.end(); ++it)
      (*it)->update_ids(total_count);
  }

  void consolidate_leaf_edges() {
    int nprev, j;
    uint32_t i;
    // Leaf edges
    leaves_le = (double*)malloc(total_num_leaves*ndim*sizeof(double));
    leaves_re = (double*)malloc(total_num_leaves*ndim*sizeof(double));
    nprev = 0;
    if (rank == root) {
      for (j = 0; j < size; j++) {
    	if (j == rank) {
    	  for (i = 0; i < leaf_count[j]; i++, nprev++) {
    	    memcpy(leaves_le + ndim*nprev, tree->leaves[i]->left_edge, ndim*sizeof(double));
    	    memcpy(leaves_re + ndim*nprev, tree->leaves[i]->right_edge, ndim*sizeof(double));
    	  }
    	} else {
    	  for (i = 0; i < leaf_count[j]; i++, nprev++) {
    	    MPI_Recv(leaves_le + ndim*nprev, ndim, MPI_DOUBLE, j, j,
    		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	    MPI_Recv(leaves_re + ndim*nprev, ndim, MPI_DOUBLE, j, j,
    		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    	  }
    	}
      }
    } else {
      for (i = 0; i < tree->num_leaves; i++) {
    	MPI_Send(tree->leaves[i]->left_edge, ndim, MPI_DOUBLE, root, rank,
    		 MPI_COMM_WORLD);
    	MPI_Send(tree->leaves[i]->right_edge, ndim, MPI_DOUBLE, root, rank,
    		 MPI_COMM_WORLD);
      }
    }
    MPI_Bcast(leaves_le, total_num_leaves*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(leaves_re, total_num_leaves*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
  }

  void consolidate_leaves0() {
    uint32_t i, nprev;
    int dst;
    uint32_t local_count = 0, total_count = 0, child_count = 0;
    std::vector<Node*>::iterator it;
    // Wait for max leafid from parent process and update local ids
    if (src_exch.src != -1) {
      MPI_Recv(&total_count, 1, MPI_UNSIGNED, src_exch.src, rank,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (it = tree->leaves.begin();
	   it != tree->leaves.end(); ++it) {
	(*it)->update_ids(total_count);
      }
    }
    local_count = tree->num_leaves;
    total_count += tree->num_leaves;
    // Send max leaf id on this process to child processes updating
    // count along the way
    leaf2rank = (int*)malloc(local_count*sizeof(int));
    for (i = 0; i < local_count; i++)
      leaf2rank[i] = rank;
    for (i = 0; i < dst_exch.size(); ++i) {
      dst = dst_exch[i].dst;
      MPI_Send(&total_count, 1, MPI_UNSIGNED, dst, dst, MPI_COMM_WORLD);
      MPI_Recv(&child_count, 1, MPI_UNSIGNED, dst, dst, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      leaf2rank = (int*)realloc(leaf2rank, (local_count+child_count)*sizeof(int));
      MPI_Recv(leaf2rank+local_count, child_count, MPI_INT, dst, dst,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      local_count += child_count;
      total_count += child_count;
    }
    // Send final count back to source
    if (src_exch.src != -1) {
      MPI_Send(&local_count, 1, MPI_UNSIGNED, src_exch.src, rank,
	       MPI_COMM_WORLD);
      MPI_Send(leaf2rank, local_count, MPI_INT, src_exch.src, rank,
	       MPI_COMM_WORLD);
    }
    // Consolidate count
    if (rank == root)
      total_num_leaves = total_count;
    MPI_Bcast(&total_num_leaves, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    leaf2rank = (int*)realloc(leaf2rank, total_num_leaves*sizeof(int));
    MPI_Bcast(leaf2rank, total_num_leaves, MPI_INT, root, MPI_COMM_WORLD);
    // Consolidate left/right edges of all leaves
    // TODO: This could be done using Gatherv...
    leaves_le = (double*)malloc(total_num_leaves*ndim*sizeof(double));
    leaves_re = (double*)malloc(total_num_leaves*ndim*sizeof(double));
    nprev = 0;
    if (rank == root) {
      for (i = 0; i < tree->num_leaves; i++, nprev++) {
	memcpy(leaves_le + ndim*nprev, tree->leaves[i]->left_edge, ndim*sizeof(double));
	memcpy(leaves_re + ndim*nprev, tree->leaves[i]->right_edge, ndim*sizeof(double));
      }
      for (i = tree->num_leaves; i < total_num_leaves; i++, nprev++) {
	MPI_Recv(leaves_le + ndim*nprev, ndim, MPI_DOUBLE, leaf2rank[i],
		 leaf2rank[i], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(leaves_re + ndim*nprev, ndim, MPI_DOUBLE, leaf2rank[i],
		 leaf2rank[i], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    } else {
      for (i = 0; i < tree->num_leaves; i++, nprev++) {
	MPI_Send(tree->leaves[i]->left_edge, ndim, MPI_DOUBLE, root, rank,
		 MPI_COMM_WORLD);
	MPI_Send(tree->leaves[i]->right_edge, ndim, MPI_DOUBLE, root, rank,
		 MPI_COMM_WORLD);
      }
    }
    MPI_Bcast(leaves_le, total_num_leaves*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(leaves_re, total_num_leaves*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
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

  double* wrap_pos(double* pos) {
    // TODO: verify that wrapping occurs for any point
    uint32_t d;
    double* wrapped_pos = (double*)malloc(ndim*sizeof(double));
    for (d = 0; d < ndim; d++) {
      if (total_periodic[d]) {
        if (pos[d] < total_domain_left_edge[d]) {
          wrapped_pos[d] = total_domain_right_edge[d] - fmod((total_domain_right_edge[d] - pos[d]),
							     total_domain_width[d]);
        } else {
          wrapped_pos[d] = total_domain_left_edge[d] + fmod((pos[d] - total_domain_left_edge[d]),
							    total_domain_width[d]);
        }
      } else {
        wrapped_pos[d] = pos[d];
      }
    }
    return wrapped_pos;
  }

  Node* search(double* pos0)
  {
    // Wrap positions
    double* pos;
    if (rank == root) {
      if (total_any_periodic) {
	pos = wrap_pos(pos0); // allocates new array
      } else {
	pos = pos0;
      }
    } else {
      pos = (double*)malloc(ndim*sizeof(double));
    }
    MPI_Bcast(pos, ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    Node* out = tree->search(pos);
    // if (rank == root) {
    //   if (out == NULL) {
    // 	int src;
    // 	MPI_Recv(&src, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
    // 		 MPI_STATUS_IGNORE);
    // 	out = recv_leafnode(src);
    //   }
    // } else {
    //   if (out != NULL) {
    // 	MPI_Send(&rank, 1, MPI_INT, root, 0, MPI_COMM_WORLD);
    // 	send_leafnode(root, out);
    // 	out = NULL;
    //   }
    // }
    if (rank == root) {
      if (total_any_periodic)
	free(pos);
    } else {
      free(pos);
    }
    return out;
  }

  std::vector<uint32_t> get_neighbor_ids(double* pos)
  {
    Node* leaf;
    std::vector<uint32_t> neighbors;
    leaf = search(pos);
    if (leaf != NULL)
      neighbors = leaf->all_neighbors;
    return neighbors;
  }


};
