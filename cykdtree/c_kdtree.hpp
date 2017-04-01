#include <vector>
#include <algorithm>
#include <array>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include "c_utils.hpp"

class Node
{
public:
  bool is_empty;
  bool is_leaf;
  uint32_t leafid;
  uint32_t ndim;
  double *left_edge;
  double *right_edge;
  uint64_t left_idx;
  uint64_t children;
  bool *periodic_left;
  bool *periodic_right;
  std::vector<std::vector<uint32_t>> left_neighbors;
  std::vector<std::vector<uint32_t>> right_neighbors;
  std::vector<uint32_t> all_neighbors;
  std::vector<Node*> left_nodes;
  // innernode parameters
  uint32_t split_dim;
  double split;
  Node *less;
  Node *greater;
  // empty node constructor
  Node() {
    is_empty = true;
  }
  // innernode constructor
  Node(uint32_t ndim0, double *le, double *re, bool *ple, bool *pre,
       uint64_t Lidx, uint32_t sdim0, double split0, Node *lnode, Node *gnode,
       std::vector<Node*> left_nodes0)
  {
    is_empty = false;
    is_leaf = false;
    leafid = 4294967295;
    ndim = ndim0;
    left_idx = Lidx;

    split_dim = sdim0;
    split = split0;
    less = lnode;
    greater = gnode;
    children = lnode->children + gnode->children;

    left_edge = (double*)malloc(ndim*sizeof(double));
    right_edge = (double*)malloc(ndim*sizeof(double));
    periodic_left = (bool*)malloc(ndim*sizeof(bool));
    periodic_right = (bool*)malloc(ndim*sizeof(bool));
    for (uint32_t d = 0; d < ndim; d++) {
      left_edge[d] = le[d];
      right_edge[d] = re[d];
      periodic_left[d] = ple[d];
      periodic_right[d] = pre[d];
      left_nodes.push_back(left_nodes0[d]);
    }

    left_neighbors = std::vector<std::vector<uint32_t>>(ndim);
    right_neighbors = std::vector<std::vector<uint32_t>>(ndim);
  }
  // leafnode constructor
  Node(uint32_t ndim0, double *le, double *re, bool *ple, bool *pre,
       uint64_t Lidx, uint64_t n, int leafid0,
       std::vector<Node*> left_nodes0)
  {
    is_empty = false;
    is_leaf = true;
    leafid = leafid0;
    ndim = ndim0;
    split = 0.0;
    split_dim = 0;
    left_idx = Lidx;

    children = n;

    left_edge = (double*)malloc(ndim*sizeof(double));
    right_edge = (double*)malloc(ndim*sizeof(double));
    periodic_left = (bool*)malloc(ndim*sizeof(bool));
    periodic_right = (bool*)malloc(ndim*sizeof(bool));
    for (uint32_t d = 0; d < ndim; d++) {
      left_edge[d] = le[d];
      right_edge[d] = re[d];
      periodic_left[d] = ple[d];
      periodic_right[d] = pre[d];
      left_nodes.push_back(left_nodes0[d]);
    }

    left_neighbors = std::vector<std::vector<uint32_t>>(ndim);
    right_neighbors = std::vector<std::vector<uint32_t>>(ndim);

    for (uint32_t d = 0; d < ndim; d++) {
      if ((left_nodes[d] != NULL) && (!(left_nodes[d]->is_empty)))
    	add_neighbors(left_nodes[d], d);
    }
  }
  ~Node() {
    free(left_edge);
    free(right_edge);
    free(periodic_left);
    free(periodic_right);
  }

  void update_ids(uint32_t add_to) {
    leafid += add_to;
    uint32_t i;
    for (uint32_t d = 0; d < ndim; d++) {
      for (i = 0; i < left_neighbors[d].size(); i++)
	left_neighbors[d][i] += add_to;
      for (i = 0; i < right_neighbors[d].size(); i++)
	right_neighbors[d][i] += add_to;
    }
    for (i = 0; i < all_neighbors.size(); i++)
      all_neighbors[i] += add_to;
  }

  void add_neighbors(Node* curr, uint32_t dim) {
    if (curr->is_leaf) {
      left_neighbors[dim].push_back(curr->leafid);
      curr->right_neighbors[dim].push_back(leafid);
    } else {
      if (curr->split_dim == dim) {
	add_neighbors(curr->greater, dim);
      } else {
	if (curr->split > this->right_edge[curr->split_dim]) 
	  add_neighbors(curr->less, dim);
	else if (curr->split < this->left_edge[curr->split_dim])
	  add_neighbors(curr->greater, dim);
	else {
	  add_neighbors(curr->less, dim);
	  add_neighbors(curr->greater, dim);
	}
      }
    }
  }

  void select_unique_neighbors() {
    if (not is_leaf)
      return;

    uint32_t d;
    std::vector<uint32_t>::iterator last;
    for (d = 0; d < ndim; d++) {
      // left
      std::sort(left_neighbors[d].begin(), left_neighbors[d].end());
      last = std::unique(left_neighbors[d].begin(), left_neighbors[d].end());
      left_neighbors[d].erase(last, left_neighbors[d].end());
      // right
      std::sort(right_neighbors[d].begin(), right_neighbors[d].end());
      last = std::unique(right_neighbors[d].begin(), right_neighbors[d].end());
      right_neighbors[d].erase(last, right_neighbors[d].end());
    }
  }

  void join_neighbors(bool include_self = true) {
    if (not is_leaf)
      return;

    uint32_t d;
    std::vector<uint32_t>::iterator last;
    // Create concatenated vector and remove duplicates
    all_neighbors = left_neighbors[0];
    for (d = 1; d < ndim; d++) 
      all_neighbors.insert(all_neighbors.end(), left_neighbors[d].begin(), left_neighbors[d].end());
    for (d = 0; d < ndim; d++)
      all_neighbors.insert(all_neighbors.end(), right_neighbors[d].begin(), right_neighbors[d].end());
    if (include_self)
      all_neighbors.push_back(leafid);
    
    // Get unique
    std::sort(all_neighbors.begin(), all_neighbors.end());
    last = std::unique(all_neighbors.begin(), all_neighbors.end());
    all_neighbors.erase(last, all_neighbors.end());

  }

  bool check_overlap(Node other, uint32_t dim) {
    if (other.right_edge[dim] < left_edge[dim])
      return false;
    else if (other.left_edge[dim] > right_edge[dim])
      return false;
    else
      return true;
  }

};

class KDTree
{
public:
  bool is_partial;
  double* all_pts;
  uint64_t* all_idx;
  uint64_t npts;
  uint32_t ndim;
  uint64_t left_idx;
  bool *periodic_left;
  bool *periodic_right;
  uint32_t leafsize;
  double* domain_left_edge;
  double* domain_right_edge;
  double* domain_width;
  bool* periodic;
  bool any_periodic;
  double* domain_mins;
  double* domain_maxs;
  uint32_t num_leaves;
  std::vector<Node*> leaves;
  Node* root;
  double* leaves_le = NULL;
  double* leaves_re = NULL;

  // KDTree() {}
  KDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m,
	 uint32_t leafsize0, uint64_t left_idx0,
	 double *left_edge, double *right_edge,
	 bool *periodic_left0, bool *periodic_right0,
	 bool include_self = true, bool dont_build = false)
  {
    is_partial = true;
    left_idx = left_idx0;

    all_pts = pts;
    all_idx = idx;
    npts = n;
    ndim = m;
    leafsize = leafsize0;
    domain_left_edge = left_edge;
    domain_right_edge = right_edge;
    periodic_left = periodic_left0;
    periodic_right = periodic_right0;
    periodic = (bool*)malloc(ndim*sizeof(bool));
    num_leaves = 0;

    domain_mins = min_pts(pts, n, m);
    domain_maxs = max_pts(pts, n, m);

    any_periodic = false;
    for (uint32_t d = 0; d < ndim; d++) {
      periodic[d] = false;
      if ((periodic_left[d]) && (periodic_right[d])) {
	periodic[d] = true;
      }
      if (periodic[d]) {
	any_periodic = true;
	break;
      }
    }

    domain_width = (double*)malloc(ndim*sizeof(double));
    for (uint32_t d = 0; d < ndim; d++) {
      domain_width[d] = domain_right_edge[d] - domain_left_edge[d];
    }

    if (!(dont_build))
      build_tree(include_self);

  }
  KDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m, uint32_t leafsize0, 
	 double *left_edge, double *right_edge, bool *periodic0,
	 bool include_self = true, bool dont_build = false)
  {
    is_partial = false;
    left_idx = 0;

    all_pts = pts;
    all_idx = idx;
    npts = n;
    ndim = m;
    leafsize = leafsize0;
    domain_left_edge = left_edge;
    domain_right_edge = right_edge;
    periodic_left = (bool*)malloc(ndim*sizeof(bool));
    periodic_right = (bool*)malloc(ndim*sizeof(bool));
    periodic = periodic0;
    num_leaves = 0;

    domain_mins = min_pts(pts, n, m);
    domain_maxs = max_pts(pts, n, m);

    any_periodic = false;
    for (uint32_t d = 0; d < ndim; d++) {
      if (periodic[d]) {
	periodic_left[d] = true;
	periodic_right[d] = true;
	any_periodic = true;
	break;
      }
    }

    domain_width = (double*)malloc(ndim*sizeof(double));
    for (uint32_t d = 0; d < ndim; d++) {
      domain_width[d] = domain_right_edge[d] - domain_left_edge[d];
    }

    if (!(dont_build))
      build_tree(include_self);

  }
  ~KDTree()
  {
    free(domain_width);
    free(domain_mins);
    free(domain_maxs);
    free(root);
    if (is_partial) {
      free(periodic);
    } else {
      free(periodic_left);
      free(periodic_right);
    }
    if (leaves_le != NULL)
      free(leaves_le);
    if (leaves_re != NULL)
      free(leaves_re);
  }

  void consolidate_edges() {
    leaves_le = (double*)malloc(num_leaves*ndim*sizeof(double));
    leaves_re = (double*)malloc(num_leaves*ndim*sizeof(double));
    for (uint32_t k = 0; k < num_leaves; k++) {
      memcpy(leaves_le+leaves[k]->leafid,
             leaves[k]->left_edge,
             ndim*sizeof(double));
      memcpy(leaves_re+ndim*leaves[k]->leafid,
             leaves[k]->right_edge,
             ndim*sizeof(double));
    }
  }

  void build_tree(bool include_self = true) {
    uint32_t d;
    double *LE = (double*)malloc(ndim*sizeof(double));
    double *RE = (double*)malloc(ndim*sizeof(double));
    bool *PLE = (bool*)malloc(ndim*sizeof(bool));
    bool *PRE = (bool*)malloc(ndim*sizeof(bool));
    double *mins = (double*)malloc(ndim*sizeof(double));
    double *maxs = (double*)malloc(ndim*sizeof(double));
    std::vector<Node*> left_nodes;

    for (d = 0; d < ndim; d++) {
      LE[d] = domain_left_edge[d];
      RE[d] = domain_right_edge[d];
      PLE[d] = periodic_left[d];
      PRE[d] = periodic_right[d];
      mins[d] = domain_mins[d];
      maxs[d] = domain_maxs[d];
      left_nodes.push_back(NULL);
    }

    root = build(0, npts, LE, RE, PLE, PRE,
		 mins, maxs, left_nodes);

    free(LE);
    free(RE);
    free(PLE);
    free(PRE);
    free(mins);
    free(maxs);

    // Finalize neighbors
    finalize_neighbors(include_self);

  }

  void finalize_neighbors(bool include_self = true) {
    uint32_t d;

    // Add periodic neighbors
    if (any_periodic)
      set_neighbors_periodic();

    // Remove duplicate neighbors
    for (d = 0; d < num_leaves; d++) {
      leaves[d]->select_unique_neighbors();
      leaves[d]->join_neighbors(include_self);
    }
  }

  void set_neighbors_periodic() 
  {
    uint32_t d0;
    Node* leaf;
    Node *prev;
    uint64_t i, j;

    // Periodic neighbors
    for (i = 0; i < num_leaves; i++) {
      leaf = leaves[i];
      for (d0 = 0; d0 < ndim; d0++) {
	if (not leaf->periodic_left[d0]) 
	  continue;
	for (j = i; j < num_leaves; j++) {
	  prev = leaves[j];
	  if (not prev->periodic_right[d0])
	    continue;
	  add_neighbors_periodic(leaf, prev, d0);
	}
      }
    }
  }

  void add_neighbors_periodic(Node *leaf, Node *prev, uint32_t d0) {
    uint32_t d, ndim_escape;
    bool match;
    if (not leaf->periodic_left[d0])
      return;
    if (not prev->periodic_right[d0])
      return;
    match = true;
    ndim_escape = 0;
    for (d = 0; d < ndim; d++) {
      if (d == d0)
	continue;
      if (leaf->left_edge[d] >= prev->right_edge[d]) {
	if (!(leaf->periodic_right[d] && prev->periodic_left[d])) {
	  match = false;
	  break;
	} else {
	  ndim_escape++;
	}
      }
      if (leaf->right_edge[d] <= prev->left_edge[d]) {
	if (!(prev->periodic_right[d] && leaf->periodic_left[d])) {
	  match = false;
	  break;
	} else {
	  ndim_escape++;
	}
      }
    }
    if ((match) and (ndim_escape < (ndim-1))) {
      // printf("%d: %d, %d (%d)\n", d0, leaf->leafid, prev->leafid, ndim_escape);
      leaf->left_neighbors[d0].push_back(prev->leafid);
      prev->right_neighbors[d0].push_back(leaf->leafid);
    }
  }

  uint32_t split(uint64_t Lidx, uint64_t n,
		 double *mins, double *maxes,
		 int64_t &split_idx, double &split_val) {
    // Find dimension to split along
    uint32_t dmax, d;
    dmax = 0;
    for (d = 1; d < ndim; d++) 
      if ((maxes[d]-mins[d]) > (maxes[dmax]-mins[dmax]))
	dmax = d;
    if (maxes[dmax] == mins[dmax]) {
      // all points singular
      return ndim;
    }
      
    // Find median along dimension
    int64_t stop = n-1;
    select(all_pts, all_idx, ndim, dmax, Lidx, stop+Lidx, (stop/2)+Lidx);
    split_idx = (stop/2)+Lidx;
    split_val = all_pts[ndim*all_idx[split_idx] + dmax];

    return dmax;
  }

  Node* build(uint64_t Lidx, uint64_t n, 
	      double *LE, double *RE, 
	      bool *PLE, bool *PRE,
	      double *mins, double *maxes,
	      std::vector<Node*> left_nodes)
  {
    // Create leaf
    if (n < leafsize) {
      Node* out = new Node(ndim, LE, RE, PLE, PRE, Lidx, n, num_leaves, 
			   left_nodes);
      num_leaves++;
      leaves.push_back(out);
      return out;
    } else {
      // Split
      uint32_t dmax, d;
      int64_t split_idx = 0;
      double split_val = 0.0;
      dmax = split(Lidx, n, mins, maxes, split_idx, split_val);
      if (maxes[dmax] == mins[dmax]) {
	// all points singular
	Node* out = new Node(ndim, LE, RE, PLE, PRE, Lidx, n, num_leaves,
			     left_nodes);
	num_leaves++;
	leaves.push_back(out);
	return out;
      }
      
      // Get new boundaries
      uint64_t Nless = split_idx-Lidx+1;
      uint64_t Ngreater = n - Nless;
      double *lessmaxes = (double*)malloc(ndim*sizeof(double));
      double *lessright = (double*)malloc(ndim*sizeof(double));
      bool *lessPRE = (bool*)malloc(ndim*sizeof(bool));
      double *greatermins = (double*)malloc(ndim*sizeof(double));
      double *greaterleft = (double*)malloc(ndim*sizeof(double));
      bool *greaterPLE = (bool*)malloc(ndim*sizeof(bool));
      std::vector<Node*> greater_left_nodes;
      for (d = 0; d < ndim; d++) {
	lessmaxes[d] = maxes[d];
	lessright[d] = RE[d];
	lessPRE[d] = PRE[d];
	greatermins[d] = mins[d];
	greaterleft[d] = LE[d];
	greaterPLE[d] = PLE[d];
	greater_left_nodes.push_back(left_nodes[d]);
      }
      lessmaxes[dmax] = split_val;
      lessright[dmax] = split_val;
      lessPRE[dmax] = false;
      greatermins[dmax] = split_val;
      greaterleft[dmax] = split_val;
      greaterPLE[dmax] = false;

      // Build less and greater nodes
      Node* less = build(Lidx, Nless, LE, lessright, PLE, lessPRE,
			 mins, lessmaxes, left_nodes);
      greater_left_nodes[dmax] = less;
      Node* greater = build(Lidx+Nless, Ngreater, greaterleft, RE,
			    greaterPLE, PRE,
			    greatermins, maxes, greater_left_nodes);

      // Create innernode referencing child nodes
      Node* out = new Node(ndim, LE, RE, PLE, PRE, Lidx, dmax, split_val,
			   less, greater, left_nodes);
      
      free(lessright);
      free(greaterleft);
      free(lessPRE);
      free(greaterPLE);
      free(lessmaxes);
      free(greatermins);
      return out;
    } 
  }	 

  double* wrap_pos(double* pos) {
    uint32_t d;
    double* wrapped_pos = (double*)malloc(ndim*sizeof(double));
    for (d = 0; d < ndim; d++) {
      if (periodic[d]) {
	if (pos[d] < domain_left_edge[d]) {
	  wrapped_pos[d] = domain_right_edge[d] - fmod((domain_right_edge[d] - pos[d]),domain_width[d]);
	} else {
	  wrapped_pos[d] = domain_left_edge[d] + fmod((pos[d] - domain_left_edge[d]),domain_width[d]);
	}
      } else {
	wrapped_pos[d] = pos[d];
      }
    }
    return wrapped_pos;
  }

  Node* search(double* pos0)
  {
    uint32_t i;
    Node* out = NULL;
    bool valid;
    // Wrap positions
    double* pos;
    if (any_periodic) 
      pos = wrap_pos(pos0); // allocates new array
    else
      pos = pos0;
    // Ensure that pos is in root, early return NULL if it's not
    valid = true;
    for (i = 0; i < ndim; i++) {
      if (pos[i] < root->left_edge[i]) {
	valid = false;
	break;
      }
      if (pos[i] >= root->right_edge[i]) {
	valid = false;
	break;
      }
    }
    // Traverse tree looking for leaf containing pos
    if (valid) {
      out = root;
      while (!(out->is_leaf)) {
	if (pos[out->split_dim] < out->split)
	  out = out->less;
	else
	  out = out->greater;
      }
    }

    if (any_periodic)
      free(pos);
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


