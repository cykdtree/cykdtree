#include "mpi.h"
#include <vector>
#include <math.h>
#include <stdint.h>
//#include "c_kdtree.hpp"

class ExchangeRecord
{
public:
  int src = -1;
  int dst = -1;
  uint32_t split_dim = 0;
  double split_val = 0.0;
  ExchangeRecord() {}
  ExchangeRecord(int src0, int dst0, uint32_t split_dim0, double split_val0) {
    src = src0;
    dst = dst0;
    split_dim = split_dim0;
    split_val = split_val0;
  }

  void print() {
    printf("src=%d, dst=%d, split_dim=%d, split_val=%f\n",
	   src, dst, split_dim, split_val);
  }
};

void send_exch(int idst, int tag, ExchangeRecord *e) {
  // int rank, size;
  // MPI_Comm_size ( MPI_COMM_WORLD, &size);
  // MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
  // printf("%d: Sending exchange to %d (tag %d)\n", rank, idst, tag);
  MPI_Send(&(e->src), 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
  MPI_Send(&(e->dst), 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
  MPI_Send(&(e->split_dim), 1, MPI_UNSIGNED, idst, tag, MPI_COMM_WORLD);
  MPI_Send(&(e->split_val), 1, MPI_DOUBLE, idst, tag, MPI_COMM_WORLD);
}

ExchangeRecord* recv_exch(int isrc, int tag) {
  // int rank, size;
  // MPI_Comm_size ( MPI_COMM_WORLD, &size);
  // MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
  // printf("%d: Receiving exchange from %d (tag %d)\n", rank, isrc, tag);
  int src;
  int dst;
  uint32_t split_dim;
  double split_val;
  MPI_Recv(&src, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&dst, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&split_dim, 1, MPI_UNSIGNED, isrc, tag, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
  MPI_Recv(&split_val, 1, MPI_DOUBLE, isrc, tag, MPI_COMM_WORLD,
	   MPI_STATUS_IGNORE);
  ExchangeRecord *e = new ExchangeRecord(src, dst, split_dim, split_val);
  return e;
}


class ParallelKDTree
{
public:
  int rank;
  int size;
  int root;
  int rrank;
  std::vector<std::vector<int>> lsplit;
  std::vector<std::vector<int>> rsplit;
  ExchangeRecord *src_exch = NULL;
  std::vector<ExchangeRecord*> dst_exch;
  uint32_t ndim;
  uint64_t npts = 0;
  uint64_t npts_orig;
  int available = 1;
  int *all_avail = NULL;
  bool is_root = false;
  KDTree *tree = NULL;
  double* all_pts = NULL;
  uint64_t* all_idx = NULL;
  bool *periodic = NULL;
  bool *periodic_left = NULL;
  bool *periodic_right = NULL;
  double *domain_left_edge = NULL;
  double *domain_right_edge = NULL;
  double *domain_width;
  uint64_t left_idx = 0;
  std::vector<Node*> leaves;
  uint32_t tot_num_leaves = 0;
  int *leaf2rank = NULL;
  double *leaves_le = NULL;
  double *leaves_re = NULL;
  
  ParallelKDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
		 uint32_t leafsize, double *left_edge, double *right_edge,
		 bool *periodic0, bool include_self = true) {
    MPI_Comm_size ( MPI_COMM_WORLD, &size);
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank);
    all_avail = (int*)malloc(size*sizeof(int));
    // Determine root
    if (pts != NULL) {
      root = rank;
      is_root = true;
      for (int i = 0; i < size; i++) {
	if (i != rank)
	  MPI_Send(&root, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
      ndim = m;
      npts = n;
    } else {
      MPI_Recv(&root, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
    }
    // Get information about global process
    rrank = (rank - root + size) % size;
    MPI_Bcast(&ndim, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    MPI_Bcast(&leafsize, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    // Domain information
    lsplit = std::vector<std::vector<int>>(ndim);
    rsplit = std::vector<std::vector<int>>(ndim);
    domain_left_edge = (double*)malloc(ndim*sizeof(double));
    domain_right_edge = (double*)malloc(ndim*sizeof(double));
    domain_width = (double*)malloc(ndim*sizeof(double));
    periodic_left = (bool*)malloc(ndim*sizeof(bool));
    periodic_right = (bool*)malloc(ndim*sizeof(bool));
    if (rank == root) {
      available = 0;
      all_pts = pts;
      all_idx = idx;
      for (uint32_t d = 0; d < ndim; d++) {
	domain_left_edge[d] = left_edge[d];
	domain_right_edge[d] = right_edge[d];
	domain_width[d] = right_edge[d] - left_edge[d];
	periodic_left[d] = periodic0[d];
	periodic_right[d] = periodic0[d];
      }
    } else {
      for (uint32_t d = 0; d < ndim; d++) {
	periodic_left[d] = false;
	periodic_right[d] = false;
      }
    }
    MPI_Bcast(domain_left_edge, ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(domain_right_edge, ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(domain_width, ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    // Create trees and partition
    tree = new KDTree(all_pts, all_idx, npts, ndim, leafsize, 0,
		      domain_left_edge, domain_right_edge,
		      periodic_left, periodic_right,
		      include_self, true);
    partition();
    build();
    consolidate(include_self);
  }
  ~ParallelKDTree() {
    delete(tree);
    free(all_avail);
    if (rank != root) {
      if (all_idx != NULL)
	free(all_idx);
      if (all_pts != NULL)
	free(all_pts);
    }
    free(domain_left_edge);
    free(domain_right_edge);
    free(domain_width);
    free(periodic_left);
    free(periodic_right);
    if (leaf2rank != NULL)
      free(leaf2rank);
    if (leaves_le != NULL)
      free(leaves_le);
    if (leaves_re != NULL)
      free(leaves_re);
  }

  void send_node(int dp, Node *node) {
    int i = 0;
    uint32_t j;
    int *pe = (int*)malloc(ndim*sizeof(int));
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

  Node* recv_node(int sp) {
    int i = 0;
    uint32_t j;
    uint32_t leafid;
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

  void send_node_neighbors(int dp, Node *node) {
    uint32_t d;
    int i = 0, j, s;
    uint32_t *ids = NULL;
    for (d = 0; d < ndim; d++) {
      // left
      s = (int)(node->left_neighbors[d].size());
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      for (j = 0; j < s; j++)
	ids[j] = node->left_neighbors[d][j];
      MPI_Send(&s, 1, MPI_INT, dp, i++, MPI_COMM_WORLD);
      MPI_Send(ids, s, MPI_UNSIGNED, dp, i++, MPI_COMM_WORLD);
      // right
      s = (int)(node->right_neighbors[d].size());
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      for (j = 0; j < s; j++)
	ids[j] = node->right_neighbors[d][j];
      MPI_Send(&s, 1, MPI_INT, dp, i++, MPI_COMM_WORLD);
      MPI_Send(ids, s, MPI_UNSIGNED, dp, i++, MPI_COMM_WORLD);
    }
    if (ids != NULL)
      free(ids);
  }

  void recv_node_neighbors(int sp, Node *node) {
    uint32_t d;
    int i = 0, j, s;
    uint32_t *ids = NULL;
    for (d = 0; d < ndim; d++) {
      // left
      MPI_Recv(&s, 1, MPI_INT, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      MPI_Recv(ids, s, MPI_UNSIGNED, sp, i++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      for (j = 0; j < s; j++)
	node->left_neighbors[d].push_back(ids[j]);
      // right
      MPI_Recv(&s, 1, MPI_INT, sp, i++, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      ids = (uint32_t*)realloc(ids, s*sizeof(uint32_t));
      MPI_Recv(ids, s, MPI_UNSIGNED, sp, i++, MPI_COMM_WORLD,
	       MPI_STATUS_IGNORE);
      for (j = 0; j < s; j++)
	node->right_neighbors[d].push_back(ids[j]);
    }
    if (ids != NULL)
      free(ids);
  }

  void add_dst(ExchangeRecord* e) {
    rsplit[e->split_dim].clear();
    rsplit[e->split_dim].push_back(e->dst);
  }

  void add_src(ExchangeRecord* e) {
    lsplit[e->split_dim].clear();
    lsplit[e->split_dim].push_back(e->src);
  }

  void add_split(ExchangeRecord* e) {
    uint32_t i, d;
    if (e->dst == rank)
      return;
    // printf("%d: Adding ", rank);
    // e->print();
    for (d = 0; d < ndim; d++) {
      // Left
      for (i = 0; i < lsplit[d].size(); i++) {
	if (e->src == lsplit[d][i]) {
	  if (e->split_dim == d) {
	    // Split is along shared dimension, use right of split
	    lsplit[d][i] = e->dst;
	  } else if (e->split_val > tree->domain_right_edge[e->split_dim]) {
	    // Split is farther right than domain, use left of split
	    lsplit[d][i] = e->src;
	  } else if (e->split_val < tree->domain_left_edge[e->split_dim]) {
	    // Split is frather left than domain, use right of split
	    lsplit[d][i] = e->dst;
	  } else {
	    // Use both left and right
	    lsplit[d].push_back(e->dst);
	  }
	}
      }
      // Right
      for (i = 0; i < rsplit[d].size(); i++) {
	if (e->src == rsplit[d][i]) {
	  if (e->split_dim == d) {
	    // Split is along shared dimension, use left of split
	    rsplit[d][i] = e->src;
	  } else if (e->split_val > tree->domain_right_edge[e->split_dim]) {
	    // Split is farther right than domain, use left of split
	    rsplit[d][i] = e->src;
	  } else if (e->split_val < tree->domain_left_edge[e->split_dim]) {
	    // Split is frather left than domain, use right of split
	    rsplit[d][i] = e->dst;
	  } else {
	    // Use both left and right
	    rsplit[d].push_back(e->dst);
	  }
	}
      }
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


  void send_neighbors(int idst, int tag) {
    int np, i;
    uint32_t d;
    // Left split
    for (d = 0; d < ndim; d++) {
      np = lsplit[d].size();
      MPI_Send(&np, 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
      for (i = 0; i < np; i++)
	MPI_Send(&(lsplit[d][i]), 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
    }
    // Right split
    for (d = 0; d < ndim; d++) {
      np = rsplit[d].size();
      MPI_Send(&np, 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
      for (i = 0; i < np; i++)
	MPI_Send(&(rsplit[d][i]), 1, MPI_INT, idst, tag, MPI_COMM_WORLD);
    }
  }

  void recv_neighbors(int isrc, int tag) {
    uint32_t d;
    int i, np, e;
    lsplit = std::vector<std::vector<int>>(ndim);
    rsplit = std::vector<std::vector<int>>(ndim);
    // Left split
    for (d = 0; d < ndim; d++) {
      MPI_Recv(&np, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (i = 0; i < np; i++) {
      MPI_Recv(&e, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      lsplit[d].push_back(e);
      }
    }
    // Right split
    for (d = 0; d < ndim; d++) {
      MPI_Recv(&np, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (i = 0; i < np; i++) {
	MPI_Recv(&e, 1, MPI_INT, isrc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	rsplit[d].push_back(e);
      }
    }
  }

  void partition() {
    // Partition points until every process has points
    double *exch_mins = (double*)malloc(ndim*sizeof(double));
    double *exch_maxs = (double*)malloc(ndim*sizeof(double));
    double *exch_le = (double*)malloc(ndim*sizeof(double));
    double *exch_re = (double*)malloc(ndim*sizeof(double));
    int *exch_ple = (int*)malloc(ndim*sizeof(int));
    int *exch_pre = (int*)malloc(ndim*sizeof(int));
    int nrecv = total_available(true);
    int nsend = 0, nexch = 0;
    int other_rank;
    uint32_t dsplit;
    int64_t split_idx = 0;
    double split_val = 0.0;
    uint64_t left_idx_send;
    uint64_t npts_send;
    double *pts_send;
    ExchangeRecord *this_exch;
    ExchangeRecord *some_exch;
    std::vector<ExchangeRecord*> new_splits;
    int n_new_split;
    // uint64_t *idx_send;
    while (nrecv > 0) {
      nsend = size - nrecv;
      nexch = std::min(nrecv, nsend);
      if (available) {
	// Receive a set of points
	if (rrank < (nsend+nexch)) {
	  // Get information about split that creates this domain
	  other_rank = (root + rrank - nsend) % size;
	  this_exch = recv_exch(other_rank, rank);
	  // Receive information about incoming domain
	  MPI_Recv(&(dsplit), 1, MPI_UNSIGNED, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&(left_idx), 1, MPI_UNSIGNED_LONG, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(&(npts), 1, MPI_UNSIGNED_LONG, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->domain_mins, ndim, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->domain_maxs, ndim, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->domain_left_edge, ndim, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(tree->domain_right_edge, ndim, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(exch_ple, ndim, MPI_INT, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  MPI_Recv(exch_pre, ndim, MPI_INT, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  // Receive points
	  all_pts = (double*)malloc(npts*ndim*sizeof(double));
	  MPI_Recv(all_pts, ndim*npts, MPI_DOUBLE, other_rank, rank,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  // Create indices
	  all_idx = (uint64_t*)malloc(npts*sizeof(uint64_t));
	  for (uint64_t i = 0; i < npts; i++)
	    all_idx[i] = i;
	  // Update local info
	  available = 0;
	  src_exch = this_exch;
	  npts_orig = npts;
	  tree->npts = npts;
	  tree->left_idx = left_idx;
	  tree->all_pts = all_pts;
	  tree->all_idx = all_idx;
	  for (uint32_t d = 0; d < ndim; d++) {
	    tree->periodic_left[d] = (bool)exch_ple[d];
	    tree->periodic_right[d] = (bool)exch_pre[d];
	    tree->domain_width[d] = tree->domain_right_edge[d]-tree->domain_left_edge[d];
	    if ((tree->periodic_left[d]) && (tree->periodic_right[d])) {
	      tree->periodic[d] = true;
	      tree->any_periodic = true;
	    }
	  }
	  // Recieve neighbors and previous splits
	  recv_neighbors(other_rank, rank);
	  add_src(this_exch);
	  MPI_Recv(&n_new_split, 1, MPI_INT, src_exch->src, src_exch->src,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  for (int j = 0; j < n_new_split; j++) {
	    some_exch = recv_exch(src_exch->src, src_exch->src);
	    new_splits.push_back(some_exch);
	  }
	}
      } else {
	// Send a subset of points
	if (rrank < nexch) {
	  // Determine parameters of exchange
	  other_rank = (root + rrank + nsend) % size;
	  dsplit = tree->split(0, npts, tree->domain_mins, tree->domain_maxs,
			       split_idx, split_val);
	  this_exch = new ExchangeRecord(rank, other_rank, dsplit, split_val);
	  send_exch(other_rank, other_rank, this_exch);
	  // Get variables to send
	  memcpy(exch_mins, tree->domain_mins, ndim*sizeof(double));
	  memcpy(exch_maxs, tree->domain_maxs, ndim*sizeof(double));
	  memcpy(exch_le, tree->domain_left_edge, ndim*sizeof(double));
	  memcpy(exch_re, tree->domain_right_edge, ndim*sizeof(double));
	  for (uint32_t d = 0; d < ndim; d++) {
	    exch_ple[d] = (int)(tree->periodic_left[d]);
	    exch_pre[d] = (int)(tree->periodic_right[d]);
	  }
	  exch_mins[dsplit] = split_val;
	  exch_le[dsplit] = split_val;
	  exch_ple[dsplit] = 0;
	  npts_send = npts - split_idx - 1;
	  left_idx_send = left_idx + split_idx + 1;
	  MPI_Send(&dsplit, 1, MPI_UNSIGNED, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(&left_idx_send, 1, MPI_UNSIGNED_LONG, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(&npts_send, 1, MPI_UNSIGNED_LONG, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_mins, ndim, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_maxs, ndim, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_le, ndim, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_re, ndim, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_ple, ndim, MPI_INT, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  MPI_Send(exch_pre, ndim, MPI_INT, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  // Send points
	  pts_send = (double*)malloc(npts_send*ndim*sizeof(double));
	  for (uint64_t i = 0; i < npts_send; i++) {
	    memcpy(pts_send + ndim*i,
		   all_pts + ndim*(all_idx[i + split_idx + 1]),
		   ndim*sizeof(double));
	  }
	  MPI_Send(pts_send, ndim*npts_send, MPI_DOUBLE, other_rank, other_rank,
		   MPI_COMM_WORLD);
	  free(pts_send);
	  // Send neighbors
	  send_neighbors(other_rank, other_rank);
	  add_dst(this_exch);
	  // Receive new splits from children 
	  for (uint32_t i = 0; i < dst_exch.size(); i++) {
	    MPI_Recv(&n_new_split, 1, MPI_INT, dst_exch[i]->dst,
		     dst_exch[i]->dst, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    for (int j = 0; j < n_new_split; j++) {
	      some_exch = recv_exch(dst_exch[i]->dst, dst_exch[i]->dst);
	      // add_split(some_exch);
	      new_splits.push_back(some_exch);
	    }
	  }
	  new_splits.push_back(this_exch);
	  n_new_split = new_splits.size();
	  // Send new splits to parent
	  if (src_exch != NULL) {
	    MPI_Send(&n_new_split, 1, MPI_INT, src_exch->src, rank,
		     MPI_COMM_WORLD);
	    for (int j = 0; j < n_new_split; j++) {
	      send_exch(src_exch->src, rank, new_splits[j]);
	    }
	  }
	  // Receive new splits from parent
	  if (src_exch != NULL) {
	    new_splits.clear();
	    MPI_Recv(&n_new_split, 1, MPI_INT, src_exch->src, src_exch->src,
		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    for (int j = 0; j < n_new_split; j++) {
	      some_exch = recv_exch(src_exch->src, src_exch->src);
	      new_splits.push_back(some_exch);
	    }
	  }
	  // Add new child to list of destinations (at the front)
	  dst_exch.insert(dst_exch.begin(), this_exch); // Smaller splits at front
	  // Send new splits to children (including the new child)
	  for (uint32_t i = 0; i < dst_exch.size(); i++) {
	    MPI_Send(&n_new_split, 1, MPI_INT, dst_exch[i]->dst, rank,
	  	     MPI_COMM_WORLD);
	    for (int j = 0; j < n_new_split; j++) {
	      send_exch(dst_exch[i]->dst, rank, new_splits[j]);
	    }
	  }	
	  // Update local info
	  tree->domain_maxs[dsplit] = split_val;
	  tree->domain_right_edge[dsplit] = split_val;
	  tree->periodic_right[dsplit] = false;
	  tree->periodic[dsplit] = false;
	  tree->domain_width[dsplit] = split_val - tree->domain_left_edge[dsplit];
	  tree->npts -= npts_send;
	  npts -= npts_send;
	  tree->any_periodic = false;
	  for (uint32_t d = 0; d < ndim; d++) {
	    if (tree->periodic[d]) {
	      tree->any_periodic = true;
	    }
	  }
	}
      }
      // Add splits
      n_new_split = new_splits.size();
      for (int j = 0; j < n_new_split; j++) 
	add_split(new_splits[j]);
      new_splits.clear();
      nrecv = total_available(true);
    }
    // print_neighbors();
    free(exch_mins);
    free(exch_maxs);
    free(exch_le);
    free(exch_re);
    free(exch_ple);
    free(exch_pre);
  }

  void build(bool include_self = false) {
    // Build, don't include self in all neighbors for now
    tree->build_tree(include_self);
    leaves = tree->leaves;
  }

  void consolidate(bool include_self) {
    consolidate_leaves();
    consolidate_idx();
    consolidate_neighbors(include_self);
    leaves = tree->leaves;
  }

  void consolidate_splits() {
    // Receive splits from child processes
    // list of all splits that occured after
    // determine neighbors from those splits
    // Send splits to parent process
    // Receive splits from parent processes
    // Send splits from parent to children
  }

  void consolidate_neighbors(bool include_self) {
    Node *node;
    std::vector<Node*>::iterator it;
    std::vector<Node*> leaves_send;
    std::vector<Node*> leaves_recv;
    std::vector<int>::iterator dst;
    std::vector<uint64_t> dst_nexch;
    uint64_t i;
    uint64_t nexch, j;
    // Receive nodes from child processes
    for (i = 0; i < dst_exch.size(); ++i) {
      nexch = 0;
      MPI_Recv(&nexch, 1, MPI_UNSIGNED_LONG, dst_exch[i]->dst, dst_exch[i]->dst,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      dst_nexch.push_back(nexch);
      for (j = 0; j < nexch; j++) {
	node = recv_node(dst_exch[i]->dst);
	if (node->left_nodes[dst_exch[i]->split_dim] == NULL) {
	  node->left_nodes[dst_exch[i]->split_dim] = tree->root;
	  node->add_neighbors(tree->root, dst_exch[i]->split_dim);
	}
	leaves_recv.push_back(node);
      }
    }
    // Send nodes to parent processes
    if (src_exch != NULL) {
      // Local leaves
      for (it = tree->leaves.begin();
	   it != tree->leaves.end(); ++it) {
	if ((*it)->left_nodes[src_exch->split_dim] == NULL)
	  leaves_send.push_back(*it);
      }
      // Child leaves
      for (it = leaves_recv.begin();
	   it != leaves_recv.end(); ++it) {
	if ((*it)->left_nodes[src_exch->split_dim] == NULL)
	  leaves_send.push_back(*it);
      }
      nexch = (uint64_t)(leaves_send.size());
      MPI_Send(&nexch, 1, MPI_UNSIGNED_LONG, src_exch->src, rank,
	       MPI_COMM_WORLD);
      for (it = leaves_send.begin();
	   it != leaves_send.end(); ++it) {
	send_node(src_exch->src, *it);
      } 
    }
    // Receive neighbors from parent process
    if (src_exch != NULL) {
      for (it = leaves_send.begin();
	   it != leaves_send.end(); ++it) {
	recv_node_neighbors(src_exch->src, *it);
      } 
    }
    // Send neighbors to child processes
    uint64_t c = 0;
    for (i = 0; i < dst_exch.size(); ++i) {
      for (j = 0; j < dst_nexch[i]; ++j, ++c) {
	send_node_neighbors(dst_exch[i]->dst, leaves_recv[c]);
      }
    }
    // TODO: handle periodic neighbors
    // Finalize neighbors
    tree->finalize_neighbors(include_self);
  }

  void consolidate_idx() {
    uint64_t left_idx_exch, nexch, i, j;
    uint64_t *idx_exch;
    // Receive ids from child processes
    for (i = 0; i < dst_exch.size(); ++i) {
      MPI_Recv(&left_idx_exch, 1, MPI_UNSIGNED_LONG, dst_exch[i]->dst,
	       dst_exch[i]->dst, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&nexch, 1, MPI_UNSIGNED_LONG, dst_exch[i]->dst, dst_exch[i]->dst,
	       MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      idx_exch = (uint64_t*)malloc(nexch*sizeof(uint64_t));
      MPI_Recv(idx_exch, nexch, MPI_UNSIGNED_LONG, dst_exch[i]->dst, 
	       dst_exch[i]->dst, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (j = 0; j < nexch; j++) {
	idx_exch[j] = all_idx[left_idx_exch-left_idx+idx_exch[j]];
      }
      memcpy(all_idx+left_idx_exch-left_idx, idx_exch, nexch*sizeof(uint64_t));
      free(idx_exch);
    }
    // Send ids to parent process
    if (src_exch != NULL) {
      MPI_Send(&left_idx, 1, MPI_UNSIGNED_LONG, src_exch->src, rank,
	       MPI_COMM_WORLD);
      MPI_Send(&npts_orig, 1, MPI_UNSIGNED_LONG, src_exch->src, rank,
	       MPI_COMM_WORLD);
      MPI_Send(all_idx, npts_orig, MPI_UNSIGNED_LONG, src_exch->src,
	       rank, MPI_COMM_WORLD);
    }
  }

  void consolidate_leaves() {
    uint32_t i, nprev;
    int dst;
    uint32_t local_count = 0, total_count = 0, child_count = 0;
    std::vector<Node*>::iterator it;
    // Wait for max leafid from parent process and update local ids
    if (src_exch != NULL) {
      MPI_Recv(&total_count, 1, MPI_UNSIGNED, src_exch->src, rank,
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
      dst = dst_exch[i]->dst;
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
    if (src_exch != NULL) {
      MPI_Send(&local_count, 1, MPI_UNSIGNED, src_exch->src, rank,
	       MPI_COMM_WORLD);
      MPI_Send(leaf2rank, local_count, MPI_INT, src_exch->src, rank,
	       MPI_COMM_WORLD);
    }
    // Consolidate count
    if (rank == root)
      tot_num_leaves = total_count;
    MPI_Bcast(&tot_num_leaves, 1, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    leaf2rank = (int*)realloc(leaf2rank, tot_num_leaves*sizeof(int));
    MPI_Bcast(leaf2rank, tot_num_leaves, MPI_INT, root, MPI_COMM_WORLD);
    // Consolidate left/right edges of all leaves
    // TODO: This could be done using Gatherv...
    leaves_le = (double*)malloc(tot_num_leaves*ndim*sizeof(double));
    leaves_re = (double*)malloc(tot_num_leaves*ndim*sizeof(double));
    nprev = 0;
    if (rank == root) {
      for (i = 0; i < tree->num_leaves; i++, nprev++) {
	memcpy(leaves_le + ndim*nprev, tree->leaves[i]->left_edge, ndim*sizeof(double));
	memcpy(leaves_re + ndim*nprev, tree->leaves[i]->right_edge, ndim*sizeof(double));
      }
      for (i = tree->num_leaves; i < tot_num_leaves; i++, nprev++) {
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
    MPI_Bcast(leaves_le, tot_num_leaves*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Bcast(leaves_re, tot_num_leaves*ndim, MPI_DOUBLE, root, MPI_COMM_WORLD);
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

  Node* search(double* pos0)
  {
    Node* out = tree->search(pos0);
    // if (rank == root) {
    //   if (out == NULL) {
    // 	int src;
    // 	MPI_Recv(&src, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
    // 		 MPI_STATUS_IGNORE);
    // 	out = recv_node(src);
    //   }
    // } else {
    //   if (out != NULL) {
    // 	MPI_Send(&rank, 1, MPI_INT, root, 0, MPI_COMM_WORLD);
    // 	send_node(root, out);
    // 	out = NULL;
    //   }
    // }
    return out;
  }


};
