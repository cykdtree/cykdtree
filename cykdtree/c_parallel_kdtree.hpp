#include "mpi.h"
#include <vector>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <cstdarg>
#include "c_parallel_utils.hpp"
//#include "c_kdtree.hpp"

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
  std::vector<uint32_t> leaf_count;
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
  int src_round;
  int nrounds;
  std::vector<exch_rec> my_splits;
  std::vector<exch_rec> all_splits;
  // Properties that are the same across all processes
  uint32_t ndim;
  uint32_t leafsize;
  double *total_domain_left_edge;
  double *total_domain_right_edge;
  double *total_domain_width;
  bool *total_periodic;
  bool total_any_periodic;
  uint32_t total_num_leaves;
  bool include_self;
  // Properties of original data received by this process
  uint64_t inter_npts;
  double *inter_domain_left_edge;
  double *inter_domain_right_edge;
  double *inter_domain_mins;
  double *inter_domain_maxs;
  bool *inter_periodic_left;
  bool *inter_periodic_right;
  bool inter_any_periodic;
  // Properties for root node on this process
  KDTree *tree;
  double* all_pts;
  uint64_t* all_idx;
  uint64_t local_npts;
  double *local_domain_left_edge;
  double *local_domain_right_edge;
  double *local_domain_mins;
  double *local_domain_maxs;
  bool *local_periodic_left;
  bool *local_periodic_right;
  bool local_any_periodic;
  uint64_t local_left_idx;

  // Convenience properties
  int* dummy;
  
  ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
		 uint32_t leafsize0, double *left_edge, double *right_edge,
		 bool *periodic0, bool include_self0 = true) {
    bool local_debug = true;
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    debug_msg(local_debug, "ParallelKDTree", "init");
    mpi_exch_type = init_mpi_exch_type();
    src = -1;
    all_avail = NULL;
    tree = NULL;
    local_left_idx = 0;
    total_num_leaves = 0;
    leaf2rank = NULL;
    double _t0 = begin_time();
    all_avail = (int*)malloc(size*sizeof(int));
    include_self = include_self0;
    nrounds = 0;
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
      inter_npts = n;
      local_npts = n;
      src_round = nrounds;
    } else {
      is_root = false;
      MPI_Recv(&root, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      available = 1;
      all_pts = NULL;
      all_idx = NULL;
      ndim = m;
      leafsize = leafsize0;
      inter_npts = 0;
      local_npts = 0;
      src_round = -1;
    }
    nrounds++;
    // Get information about global process
    rrank = (rank - root + size) % size;
    MPI_Bcast(&ndim, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&leafsize, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    // Allocate things
    dummy = (int*)malloc(ndim*sizeof(int));
    lsplit = std::vector<std::vector<int> >(ndim);
    rsplit = std::vector<std::vector<int> >(ndim);
    // Total properties
    total_domain_left_edge = (double*)malloc(ndim*sizeof(double));
    total_domain_right_edge = (double*)malloc(ndim*sizeof(double));
    total_domain_width = (double*)malloc(ndim*sizeof(double));
    total_periodic = (bool*)malloc(ndim*sizeof(bool));
    total_any_periodic = false;
    // Intermediate properties
    inter_domain_left_edge = (double*)malloc(ndim*sizeof(double));
    inter_domain_right_edge = (double*)malloc(ndim*sizeof(double));
    if (!(is_root)) {
      inter_domain_mins = (double*)malloc(ndim*sizeof(double));
      inter_domain_maxs = (double*)malloc(ndim*sizeof(double));
    }
    inter_periodic_left = (bool*)malloc(ndim*sizeof(bool));
    inter_periodic_right = (bool*)malloc(ndim*sizeof(bool));
    inter_any_periodic = false;
    // Local properties
    local_domain_left_edge = (double*)malloc(ndim*sizeof(double));
    local_domain_right_edge = (double*)malloc(ndim*sizeof(double));
    local_domain_mins = (double*)malloc(ndim*sizeof(double));
    local_domain_maxs = (double*)malloc(ndim*sizeof(double));
    local_periodic_left = (bool*)malloc(ndim*sizeof(bool));
    local_periodic_right = (bool*)malloc(ndim*sizeof(bool));
    local_any_periodic = false;
    // Domain information
    if (is_root) {
      memcpy(total_domain_left_edge, left_edge, ndim*sizeof(double));
      memcpy(total_domain_right_edge, right_edge, ndim*sizeof(double));
      memcpy(total_periodic, periodic0, ndim*sizeof(bool));
      memcpy(inter_domain_left_edge, left_edge, ndim*sizeof(double));
      memcpy(inter_domain_right_edge, right_edge, ndim*sizeof(double));
      inter_domain_mins = min_pts(all_pts, inter_npts, ndim);
      inter_domain_maxs = max_pts(all_pts, inter_npts, ndim);
      memcpy(inter_periodic_left, periodic0, ndim*sizeof(bool));
      memcpy(inter_periodic_right, periodic0, ndim*sizeof(bool));
      memcpy(local_domain_left_edge, left_edge, ndim*sizeof(double));
      memcpy(local_domain_right_edge, right_edge, ndim*sizeof(double));
      memcpy(local_domain_mins, inter_domain_mins, ndim*sizeof(double));
      memcpy(local_domain_maxs, inter_domain_maxs, ndim*sizeof(double));
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
      inter_any_periodic = total_any_periodic;
      local_any_periodic = total_any_periodic;
    } else {
      for (uint32_t d = 0; d < ndim; d++) {
	inter_periodic_left[d] = false;
	inter_periodic_right[d] = false;
	local_periodic_left[d] = false;
	local_periodic_right[d] = false;
      }
      inter_any_periodic = false;
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
    debug_msg(local_debug, "ParallelKDTree", "finished bcast");
    set_comm_order();
    build_tree();
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
    free(inter_domain_left_edge);
    free(inter_domain_right_edge);
    free(inter_domain_mins);
    free(inter_domain_maxs);
    free(inter_periodic_left);
    free(inter_periodic_right);
    free(local_domain_left_edge);
    free(local_domain_right_edge);
    free(local_domain_mins);
    free(local_domain_maxs);
    free(local_periodic_left);
    free(local_periodic_right);
    if (leaf2rank != NULL)
      free(leaf2rank);
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
    uint32_t i;
    rsplit[e.split_dim].clear();
    rsplit[e.split_dim].push_back(e.dst);
    for (i = 0; i < lsplit[e.split_dim].size(); i++) {
      if (lsplit[e.split_dim][i] == e.src)
	lsplit[e.split_dim][i] = e.dst;
    }
  }

  void add_src(exch_rec e) {
    uint32_t i, d;
    lsplit[e.split_dim].clear();
    lsplit[e.split_dim].push_back(e.src);
    for (d = 0; d < ndim; d++) {
      if (e.split_dim == d)
	continue;
      // Left
      for (i = 0; i < lsplit[d].size(); i++) {
	if (e.src == lsplit[d][i]) {
	  lsplit[d][i] = e.dst;
	}
      }
      // Right
      for (i = 0; i < rsplit[d].size(); i++) {
	if (e.src == rsplit[d][i]) {
	  rsplit[d][i] = e.dst;
	}
      }
    }
  }

  void add_split(exch_rec e) {
    uint32_t i, d;
    // if (rank == 0) {
    //   printf("Before\n");
    //   e.print();
    //   print_neighbors();
    // }
    if (e.dst == rank) {
      // Source
      add_src(e);
    } else if (e.src == rank) {
      // Destination
      add_dst(e);
    } else {
      // Another process
      for (d = 0; d < ndim; d++) {
	// Left
	for (i = 0; i < lsplit[d].size(); i++) {
	  if (e.src == lsplit[d][i]) {
	    if (e.split_dim == d) {
	      // Split is along shared dimension, use right of split
	      lsplit[d][i] = e.dst;
	    // } else if (e.split_val > local_domain_right_edge[e.split_dim]) {
	    //   // Split is farther right than domain, use left of split
	    //   lsplit[d][i] = e.src;
	    // } else if (e.split_val < local_domain_left_edge[e.split_dim]) {
	    //   // Split is farther left than domain, use right of split
	    //   lsplit[d][i] = e.dst;
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
	    // } else if (e.split_val > local_domain_right_edge[e.split_dim]) {
	    //   // Split is farther right than domain, use left of split
	    //   rsplit[d][i] = e.src;
	    // } else if (e.split_val < local_domain_left_edge[e.split_dim]) {
	    //   // Split is frather left than domain, use right of split
	    //   rsplit[d][i] = e.dst;
	    } else {
	      // Use both left and right
	      rsplit[d].push_back(e.dst);
	    }
	  }
	}
      }
    }
    // if (rank == 0) {
    //   printf("After\n");
    //   e.print();
    //   print_neighbors();
    // }
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

  exch_rec send_part(int other_rank) {
    bool local_debug = true;
    uint32_t d;
    double *pts_send;
    debug_msg(local_debug, "send_part", "sending to %d", other_rank);
    // Split
    exch_rec this_exch = split_local(other_rank);
    // Send exchange record
    debug_msg(local_debug, "send_part", "send_exch");
    send_exch(other_rank, this_exch);
    // Send variables
    debug_msg(local_debug, "send_part", "sending domain properties");
    MPI_Send(local_domain_mins, ndim, MPI_DOUBLE, this_exch.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(local_domain_maxs, ndim, MPI_DOUBLE, this_exch.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(local_domain_left_edge, ndim, MPI_DOUBLE, this_exch.dst, rank,
	     MPI_COMM_WORLD);
    MPI_Send(local_domain_right_edge, ndim, MPI_DOUBLE, this_exch.dst, rank,
	     MPI_COMM_WORLD);
    for (d = 0; d < ndim; d++)
      dummy[d] = (int)(local_periodic_left[d]);
    MPI_Send(dummy, ndim, MPI_INT, this_exch.dst, rank,
	     MPI_COMM_WORLD);
    for (d = 0; d < ndim; d++)
      dummy[d] = (int)(local_periodic_right[d]);
    MPI_Send(dummy, ndim, MPI_INT, this_exch.dst, rank,
	     MPI_COMM_WORLD);
    // Send points
    debug_msg(local_debug, "send_part", "sending points");
    uint64_t npts_send = this_exch.npts;
    pts_send = (double*)malloc(npts_send*ndim*sizeof(double));
    for (uint64_t i = 0; i < npts_send; i++) {
      memcpy(pts_send + ndim*i,
	     all_pts + ndim*(all_idx[i + this_exch.split_idx + 1]),
	     ndim*sizeof(double));
    }
    MPI_Send(pts_send, ndim*npts_send, MPI_DOUBLE, this_exch.dst, rank,
	     MPI_COMM_WORLD);
    free(pts_send);
    // Update local info
    debug_msg(local_debug, "send_part", "updating local properties");
    dst_exch.insert(dst_exch.begin(), this_exch); // Smaller splits at front
    local_domain_maxs[this_exch.split_dim] = this_exch.split_val;
    local_domain_right_edge[this_exch.split_dim] = this_exch.split_val;
    local_periodic_right[this_exch.split_dim] = false;
    local_npts = this_exch.split_idx + 1;
    local_any_periodic = false;
    for (uint32_t d = 0; d < ndim; d++) {
      if ((local_periodic_left[d]) and (local_periodic_right[d])) {
	local_any_periodic = true;
	break;
      }
    }
    // Return
    return this_exch;
  }

  void recv_part(int other_rank) {
    bool local_debug = true;
    uint32_t d;
    debug_msg(local_debug, "recv_part", "receiving from %d", other_rank);
    exch_rec this_exch = recv_exch(other_rank);
    // Receive information about incoming domain
    debug_msg(local_debug, "recv_part", "receiving domain properties");
    MPI_Recv(local_domain_mins, ndim, MPI_DOUBLE, this_exch.src, this_exch.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_domain_maxs, ndim, MPI_DOUBLE, this_exch.src, this_exch.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_domain_left_edge, ndim, MPI_DOUBLE, this_exch.src, this_exch.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_domain_right_edge, ndim, MPI_DOUBLE, this_exch.src, this_exch.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(dummy, ndim, MPI_INT, this_exch.src, this_exch.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (d = 0; d < ndim; d++)
      local_periodic_left[d] = (bool)(dummy[d]);
    MPI_Recv(dummy, ndim, MPI_INT, this_exch.src, this_exch.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (d = 0; d < ndim; d++)
      local_periodic_right[d] = (bool)(dummy[d]);
    // Receive points
    debug_msg(local_debug, "recv_part", "receiving points");
    all_pts = (double*)malloc(this_exch.npts*ndim*sizeof(double));
    MPI_Recv(all_pts, ndim*this_exch.npts, MPI_DOUBLE, this_exch.src, this_exch.src,
	     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Create indices
    debug_msg(local_debug, "recv_part", "creating index");
    all_idx = (uint64_t*)malloc(this_exch.npts*sizeof(uint64_t));
    for (uint64_t i = 0; i < this_exch.npts; i++)
      all_idx[i] = i;
    // Update local info
    debug_msg(local_debug, "recv_part", "updating local properties");
    src_exch = this_exch;
    local_npts = this_exch.npts;
    local_domain_mins[this_exch.split_dim] = this_exch.split_val;
    local_domain_left_edge[this_exch.split_dim] = this_exch.split_val;
    local_periodic_left[this_exch.split_dim] = false;
    for (d = 0; d < ndim; d++) {
      if ((local_periodic_left[d]) && (local_periodic_right[d])) {
	local_any_periodic = true;
	break;
      }
    }
    // Update intermediate things
    debug_msg(local_debug, "recv_part", "updating inter properties");
    inter_npts = local_npts;
    inter_any_periodic = local_any_periodic;
    memcpy(inter_domain_mins, local_domain_mins, ndim*sizeof(double));
    memcpy(inter_domain_maxs, local_domain_maxs, ndim*sizeof(double));
    memcpy(inter_domain_left_edge, local_domain_left_edge, ndim*sizeof(double));
    memcpy(inter_domain_right_edge, local_domain_right_edge, ndim*sizeof(double));
    memcpy(inter_periodic_left, local_periodic_left, ndim*sizeof(bool));
    memcpy(inter_periodic_right, local_periodic_right, ndim*sizeof(bool));
  }

  void set_comm_order() {
    bool local_debug = true;
    int nrecv = total_available(true);
    int nsend = 0, nexch = 0;
    while (nrecv > 0) {
      nsend = size - nrecv;
      nexch = std::min(nrecv, nsend);
      debug_msg(local_debug, "set_comm_order",
		"nrecv = %d, nsend = %d, nexch = %d", rank, nrecv, nsend, nexch);
      if (available) {
	// Get source
	if (rrank < (nsend+nexch)) {
	  src = (root + rrank - nsend) % size;
	  available = false;
	  src_round = nrounds;
	  debug_msg(local_debug, "set_comm_order", "src = %d, round = %d",
		    src, src_round);
	}
      } else {
	// Get destination
	if (rrank < nexch)
	  dst.push_back((root + rrank + nsend) % size);
      }
      nrecv = total_available(true);
      nrounds++;
    }
  }

  Node* recv_node(int sp, KDTree *this_tree, uint64_t prev_Lidx,
		  double *le, double *re, bool *ple, bool *pre,
		  std::vector<Node*> left_nodes) {
    // TODO: Add neighbors
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
		     this_tree->num_leaves, left_nodes);
      this_tree->leaves.push_back(out);
      this_tree->num_leaves++;
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
      Node *less = recv_node(sp, this_tree, prev_Lidx,
			     le, re_l, ple, pre_l, left_nodes);
      greater_left_nodes[sdim] = less;
      Node *greater = recv_node(sp, this_tree, prev_Lidx,
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

  Node* build(KDTree *new_tree, uint64_t Lidx, uint64_t n,
	      double *LE, double *RE,
	      bool *PLE, bool *PRE,
	      double *mins, double *maxs,
	      std::vector<Node*> left_nodes, uint32_t dst_count = 0) {
    // Return root if no more splits happened
    if (dst_count >= dst_exch.size()) {
      std::vector<Node*>::iterator it;
      for (it = tree->leaves.begin(); it != tree->leaves.end(); it++) {
	new_tree->leaves.push_back(*it);
	new_tree->num_leaves++;
      }
      return tree->root;
    }

    // Get split info and advance the count
    exch_rec idst = dst_exch[dst_exch.size() - dst_count - 1];

    // Determine boundaries
    uint32_t d;
    uint64_t lN = idst.split_idx - Lidx + 1;
    // uint64_t rN = n - lN;
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
    lmaxs[idst.split_dim] = idst.split_val;
    rmins[idst.split_dim] = idst.split_val;
    lRE[idst.split_dim] = idst.split_val;
    rLE[idst.split_dim] = idst.split_val;
    lPRE[idst.split_dim] = false;
    rPLE[idst.split_dim] = false;

    // Build left node & receive right node
    Node *lnode = build(new_tree, Lidx, lN, LE, lRE, PLE, lPRE,
			mins, lmaxs, left_nodes, dst_count+1);
    r_left_nodes[idst.split_dim] = lnode;
    Node *rnode = recv_node(idst.dst, new_tree, Lidx+lN, rLE, RE, rPLE, PRE, 
			    r_left_nodes);

    // Create innernode
    Node* out = new Node(ndim, LE, RE, PLE, PRE, Lidx, idst.split_dim, 
			 idst.split_val, lnode, rnode, left_nodes);

    free(lRE);
    free(rLE);
    free(lPRE);
    free(rPLE);
    free(lmaxs);
    free(rmins);
    return out;
  }

  void build_tree() {
    // Create trees and partition
    partition();
    double _t0 = begin_time();
    // Build, don't include self in all neighbors for now
    tree->build_tree(all_pts, include_self);
    debug_msg(true, "build_tree", "num_leaves = %u", tree->num_leaves);
    end_time(_t0, "build_tree");
    consolidate();
  }

  void partition() {
    bool local_debug = true;
    double _t0 = begin_time();
    exch_rec this_exch;
    std::vector<int>::iterator it;
    debug_msg(local_debug, "partition", "begin");
    // Receive from source
    if (src != -1) 
      recv_part(src);
    // Send to destinations
    int i;
    for (i = 0; i < nrounds; i++)
      my_splits.push_back(exch_rec());
    for (i = 0; i < (int)(dst.size()); i++)
      my_splits[src_round + 1 + i] = send_part(dst[i]);
    // Initialize tree at local
    debug_msg(local_debug, "partition", "init_tree");
    tree = new KDTree(all_pts, all_idx, local_npts, ndim, leafsize,
		      local_domain_left_edge, local_domain_right_edge,
		      local_periodic_left, local_periodic_right,
		      local_domain_mins, local_domain_maxs,
		      include_self, true);
    end_time(_t0, "partition");
  }

  exch_rec split_local(int other_rank) {
    double _t0 = begin_time();
    exch_rec this_exch;
    uint32_t dsplit;
    int64_t split_idx = 0;
    double split_val = 0.0;
    dsplit = split(all_pts, all_idx, 0, local_npts, ndim,
		   local_domain_mins, local_domain_maxs,
		   split_idx, split_val);
    // dsplit = tree->split(0, local_npts, local_domain_mins, local_domain_maxs,
    // 			 split_idx, split_val);
    this_exch = exch_rec(rank, other_rank, dsplit, split_val, split_idx,
			 local_left_idx + split_idx + 1,
			 local_npts - split_idx - 1);
    end_time(_t0, "split");
    return this_exch;
  }

  KDTree* consolidate_tree() {
    double _t0 = begin_time();
    KDTree* out = NULL;
    uint32_t d;
    std::vector<Node*> left_nodes;
    for (d = 0; d < ndim; d++)
      left_nodes.push_back(NULL);
    // TODO: Add self as neighbor on root for periodic domain?
    // Initialize tree
    out = new KDTree(all_pts, all_idx, inter_npts, ndim, leafsize,
		     inter_domain_left_edge, inter_domain_right_edge,
		     inter_periodic_left, inter_periodic_right,
		     inter_domain_mins, inter_domain_maxs,
		     include_self, true);
    // Consolidate nodes
    double _tb = begin_time();
    out->root = build(out, 0, out->npts,
    		      out->domain_left_edge, out->domain_right_edge,
    		      out->periodic_left, out->periodic_right,
    		      out->domain_mins, out->domain_maxs, left_nodes);
    end_time(_tb, "total build");
    // Send root back to source
    if (src_exch.src != -1)
      send_node(src_exch.src, out->root);
    // Consolidate idx
    out->finalize_neighbors(include_self);
    consolidate_idx();
    end_time(_t0, "consolidate_tree");
    return out;
  }

  void consolidate() {
    double _t0 = begin_time();
    consolidate_order();
    consolidate_splits();
    consolidate_leaves();
    consolidate_neighbors();
    debug_msg(true, "consolidate", "Waiting for all processes to finish");
    MPI_Barrier(MPI_COMM_WORLD);
    end_time(_t0, "consolidate");
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
    // Broadcast order from root to all processors
    nprev = size;
    MPI_Bcast(&proc_order[0], nprev, MPI_INT, root, MPI_COMM_WORLD);
  }
  
  void consolidate_splits() {
    bool local_debug = true;
    double _t0 = begin_time();
    int nsplits = nrounds*size;
    all_splits = std::vector<exch_rec>(nsplits);
    int i, j0, j, idst;
    // int prank = 0;
    // Gather all splits
    debug_msg(local_debug, "consolidate_splits", "gathering splits");
    MPI_Allgather(&(my_splits[0]), nrounds, mpi_exch_type,
		  &(all_splits[0]), nrounds, mpi_exch_type,
		  MPI_COMM_WORLD);
    // Init left right for root based on periodicity
    debug_msg(local_debug, "consolidate_splits", "initializing l/r splits");
    if (src_exch.src == -1) {
      lsplit = std::vector<std::vector<int> >(ndim);
      rsplit = std::vector<std::vector<int> >(ndim);
      for (uint32_t d = 0; d < ndim; d++) {
	if (total_periodic[d]) {
	  lsplit[d].push_back(rank);
	  rsplit[d].push_back(rank);
	}
      }
    } else {
      recv_neighbors(src_exch.src);
      // if (rank == prank) {
      // 	print_neighbors();
      // 	printf("%d: Round %d\n", rank, src_round);
      // 	for (j0 = 0; j0 < size; j0++) {
      // 	  j = proc_order[j0];
      // 	  all_splits[j0*nrounds + src_round].print();
      // 	}
      // }
      for (j0 = 0; j0 < size; j0++) {
	j = proc_order[j0];
	if (all_splits[j*nrounds + src_round].src != -1)
	  add_split(all_splits[j*nrounds + src_round]);
      }
    }
    // if (rank == prank) {
    //   printf("%d: Init\n", rank);
    //   print_neighbors();
    // }
    // Construct l/r neighbors based on splits
    debug_msg(local_debug, "consolidate_splits", "adding splits");
    for (i = (src_round + 1), idst=0; i < nrounds; i++, idst++) {
      // if (rank == prank) {
      // 	printf("%d: Round %d\n", rank, i);
      // 	for (j0 = 0; j0 < size; j0++) {
      // 	  j = proc_order[j0];
      // 	  all_splits[j0*nrounds + i].print();
      // 	}
      // }
      if (idst < (int)(dst.size())) {
	send_neighbors(all_splits[rank*nrounds + i].dst);
      }
      for (j0 = 0; j0 < size; j0++) {
	j = proc_order[j0];
	if (all_splits[j*nrounds + i].src != -1)
	  add_split(all_splits[j*nrounds + i]);
      }
      // if (rank == prank) {
      // 	printf("%d: After round %d\n", rank, i);
      // 	print_neighbors();
      // }
    }
    end_time(_t0, "consolidate_splits");
  }

  void consolidate_leaves() {
    bool local_debug = true;
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
      total_num_leaves += leaf_count[proc_order[j]];
    }
    debug_msg(local_debug, "consolidate_leaves", "num_leaves = %u, nprev = %u", 
	      tree->num_leaves, total_count);
    for (it = tree->leaves.begin(); it != tree->leaves.end(); ++it)
      (*it)->update_ids(total_count);
  }

  void consolidate_neighbors() {
    bool local_debug = true;
    uint32_t d;
    std::vector<Node*>::iterator it;
    std::vector<std::vector<Node*> > leaves_send;
    leaves_send = std::vector<std::vector<Node*> >(ndim);
    // Identify local leaves with missing neighbors
    debug_msg(local_debug, "consolidate_neighbors",
	      "identifying missing neighbors");
    for (it = tree->leaves.begin();
	 it != tree->leaves.end(); ++it) {
      for (d = 0; d < ndim; d++) {
	if (((*it)->left_nodes[d] == NULL) and (lsplit[d].size() > 0)) {
	  leaves_send[d].push_back(*it);
	}
      }
    }
    // Non-periodic neighbors
    debug_msg(local_debug, "consolidate_neighbors",
	      "exchanging non-periodic neighbors");
    for (d = 0; d < ndim; d++)
      exch_neigh(d, leaves_send, false);
    // Periodic neighbors
    debug_msg(local_debug, "consolidate_neighbors",
	      "exchanging periodic neighbors");
    for (d = 0; d < ndim; d++)
      exch_neigh(d, leaves_send, true);
    // Finalize neighbors
    debug_msg(local_debug, "consolidate_neighbors",
	      "finalizing neighbors");
    tree->finalize_neighbors(include_self);
  }

  void exch_neigh(uint32_t d, std::vector<std::vector<Node*> > lsend,
		  bool p) {
    bool local_debug = true;
    int i0, i, k;
    uint32_t j, d0;
    int nsend, nrecv;
    Node *node;
    std::vector<Node*>::iterator it;
    nsend = lsend[d].size();
    for (i0 = 0; i0 < size; i0++) {
      i = proc_order[i0];
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
	      debug_msg(local_debug, "exch_neigh", "Receiving from %d", 
			rsplit[d][j]);
	      MPI_Recv(&nrecv, 1, MPI_INT, rsplit[d][j], rsplit[d][j], 
		       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	      debug_msg(local_debug, "exch_neigh", "Receiving %d from %d", 
			nrecv, rsplit[d][j]);
	      for (k = 0; k < nrecv; k++) {
		node = recv_leafnode(rsplit[d][j]);
		if (node->is_left_node(tree->root, d)) {
		  node->left_nodes[d] = tree->root;
		  if (p) {
		    for (it = tree->leaves.begin(); it != tree->leaves.end(); ++it)
		      for (d0 = 0; d0 < ndim; d0++)
			tree->add_neighbors_periodic(node, *it, d0);
		  } else {
		    node->add_neighbors(tree->root, d);
		  }
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
              debug_msg(local_debug, "exch_neigh", "Sending to %d",
                        lsplit[d][j]);
	      MPI_Send(&nsend, 1, MPI_INT, lsplit[d][j], rank,
		       MPI_COMM_WORLD);
              debug_msg(local_debug, "exch_neigh", "Sending %d to %d",
                        nsend, lsplit[d][j]);
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
      MPI_Send(all_idx, inter_npts, MPI_UNSIGNED_LONG, src_exch.src,
	       rank, MPI_COMM_WORLD);
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
    Node* out = tree->search(pos, true);
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


  void consolidate_edges(double *leaves_le, double *leaves_re) {
    int nprev, j, j0;
    uint32_t i;
    // Leaf edges
    nprev = 0;
    if (rank == root) {
      for (j0 = 0; j0 < size; j0++) {
	j = proc_order[j0];
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

  void consolidate_process_bounds(double *all_lbounds, double *all_rbounds) {
    // Send tree bounds to root
    if (rank == root) {
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
    MPI_Bcast(all_lbounds, size*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(all_rbounds, size*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
  }

};
