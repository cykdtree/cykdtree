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
  // innernode constructor
  Node(uint32_t ndim0, double *le, double *re, uint64_t Lidx, 
       uint32_t sdim0, double split0, Node *lnode, Node *gnode,
       std::vector<Node*> left_nodes0)
  {
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
      periodic_left[d] = false;
      periodic_right[d] = false;
      left_nodes.push_back(left_nodes0[d]);
    }

    left_neighbors = std::vector<std::vector<uint32_t>>(ndim);
    right_neighbors = std::vector<std::vector<uint32_t>>(ndim);
  }
  // leafnode constructor
  Node(uint32_t ndim0, double *le, double *re, 
       uint64_t Lidx, uint64_t n, int leafid0,
       std::vector<Node*> left_nodes0)
  {
    is_leaf = true;
    leafid = leafid0;
    ndim = ndim0;
    split=0.0;
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
      periodic_left[d] = false;
      periodic_right[d] = false;
      left_nodes.push_back(left_nodes0[d]);
    }

    left_neighbors = std::vector<std::vector<uint32_t>>(ndim);
    right_neighbors = std::vector<std::vector<uint32_t>>(ndim);

    for (uint32_t d = 0; d < ndim; d++) {
      if (left_nodes[d] != NULL)
    	add_neighbors(left_nodes[d], d);
    }
  }
  ~Node() {
    free(left_edge);
    free(right_edge);
    free(periodic_left);
    free(periodic_right);
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

};

class KDTree
{
public:
  double* all_pts;
  uint64_t* all_idx;
  uint64_t npts;
  uint32_t ndim;
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

  // KDTree() {}
  KDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m, uint32_t leafsize0, 
	 double *left_edge, double *right_edge, bool *periodic0,
	 bool include_self = true)
  {
    uint32_t d;
    all_pts = pts;
    all_idx = idx;
    npts = n;
    ndim = m;
    leafsize = leafsize0;
    domain_left_edge = left_edge;
    domain_right_edge = right_edge;
    periodic = periodic0;
    num_leaves = 0;

    domain_mins = min_pts(pts, n, m);
    domain_maxs = max_pts(pts, n, m);

    any_periodic = false;
    for (d = 0; d < ndim; d++) {
      if (periodic[d]) {
	any_periodic = true;
	break;
      }
    }

    domain_width = (double*)malloc(ndim*sizeof(double));
    double *LE = (double*)malloc(ndim*sizeof(double));
    double *RE = (double*)malloc(ndim*sizeof(double));
    double *mins = (double*)malloc(ndim*sizeof(double));
    double *maxs = (double*)malloc(ndim*sizeof(double));
    std::vector<Node*> left_nodes;

    for (d = 0; d < ndim; d++) {
      domain_width[d] = right_edge[d] - left_edge[d];
      LE[d] = left_edge[d];
      RE[d] = right_edge[d];
      mins[d] = domain_mins[d];
      maxs[d] = domain_maxs[d];
      left_nodes.push_back(NULL);
    }

    root = build(0, n, LE, RE, mins, maxs, left_nodes);

    free(LE);
    free(RE);
    free(mins);
    free(maxs);

    // Add periodic neighbors
    if (any_periodic)
      set_neighbors_periodic();

    // Remove duplicate neighbors
    for (d = 0; d < num_leaves; d++) {
      leaves[d]->select_unique_neighbors();
      leaves[d]->join_neighbors(include_self);
    }

  }
  ~KDTree()
  {
    free(domain_width);
    free(domain_mins);
    free(domain_maxs);
    free(root);
  }

  void set_neighbors_periodic() 
  {
    uint32_t d, d0;
    Node* leaf;
    Node *prev;
    uint64_t i, j;
    bool match;

    // Periodicity in each dimension
    for (i = 0; i < num_leaves; i++) {
      leaf = leaves[i];
      for (d = 0; d < ndim; d++) {
	if (periodic[d]) {
	  if (leaf->left_neighbors[d].size() == 0)
	    leaf->periodic_left[d] = true;
	  if (leaf->right_neighbors[d].size() == 0)
	    leaf->periodic_right[d] = true;
	}
      }
    }

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
	  match = true;
	  for (d = 0; d < ndim; d++) {
	    if (d == d0)
	      continue;
	    if (leaf->left_edge[d] > prev->right_edge[d]) {
	      if (!(leaf->periodic_right[d] && prev->periodic_left[d])) {
		match = false;
		break;
	      }
	    }
	    if (leaf->right_edge[d] < prev->left_edge[d]) {
	      if (!(prev->periodic_right[d] && leaf->periodic_left[d])) {
		match = false;
		break;
	      }
	    }
	  }
	  if (match) {
	    leaf->left_neighbors[d0].push_back(prev->leafid);
	    prev->right_neighbors[d0].push_back(leaf->leafid);
	  }
	}
      }
    }
  }

  Node* build(uint64_t Lidx, uint64_t n, 
	      double *LE, double *RE, 
	      double *mins, double *maxes,
	      std::vector<Node*> left_nodes)
  {
    // Create leaf
    if (n < leafsize) {
      Node* out = new Node(ndim, LE, RE, Lidx, n, num_leaves, 
			   left_nodes);
      num_leaves++;
      leaves.push_back(out);
      return out;
    } else {
      // Find dimension to split along
      uint32_t dmax, d;
      dmax = 0;
      for (d = 1; d < ndim; d++) 
	if ((maxes[d]-mins[d]) > (maxes[dmax]-mins[dmax]))
	  dmax = d;
      if (maxes[dmax] == mins[dmax]) {
	// all points singular
	Node* out = new Node(ndim, LE, RE, Lidx, n, num_leaves,
			     left_nodes);
	num_leaves++;
	leaves.push_back(out);
	return out;
      }
      
      // Find median along dimension
      int64_t stop = n-1;
      int64_t med = (n/2) + (n%2);

      // Version using pointer to all points and index
      med = select(all_pts, all_idx, ndim, dmax, Lidx, stop+Lidx, (stop/2)+Lidx);
      med = (stop/2)+Lidx;
      uint64_t Nless = med-Lidx+1;
      uint64_t Ngreater = n - Nless;
      double split;
      if ((n%2) == 0) {
	split = all_pts[ndim*all_idx[med] + dmax];
      } else {
	split = all_pts[ndim*all_idx[med] + dmax];
      }

      // Get new boundaries
      double *lessmaxes = (double*)malloc(ndim*sizeof(double));
      double *lessright = (double*)malloc(ndim*sizeof(double));
      double *greatermins = (double*)malloc(ndim*sizeof(double));
      double *greaterleft = (double*)malloc(ndim*sizeof(double));
      std::vector<Node*> greater_left_nodes;
      for (d = 0; d < ndim; d++) {
	lessmaxes[d] = maxes[d];
	lessright[d] = RE[d];
	greatermins[d] = mins[d];
	greaterleft[d] = LE[d];
	greater_left_nodes.push_back(left_nodes[d]);
      }
      lessmaxes[dmax] = split;
      lessright[dmax] = split;
      greatermins[dmax] = split;
      greaterleft[dmax] = split;

      // Build less and greater nodes
      Node* less = build(Lidx, Nless, LE, lessright, mins, lessmaxes, 
			 left_nodes);
      greater_left_nodes[dmax] = less;
      Node* greater = build(Lidx+Nless, Ngreater, greaterleft, RE, greatermins, 
			    maxes, greater_left_nodes);

      // Create innernode referencing child nodes
      Node* out = new Node(ndim, LE, RE, Lidx, dmax, split, less, greater,
			   left_nodes);
      
      free(lessright);
      free(greaterleft);
      free(lessmaxes);
      free(greatermins);
      return out;
    } 
  }	 

  double* wrap_pos(double* pos) {
    uint32_t d;
    double* wrapped_pos = (double*)malloc(ndim*sizeof(double));
    for (d = 0; d < ndim; d++) {
      if (periodic[d])
	wrapped_pos[d] = domain_left_edge[d] + fmod((pos[d] - domain_left_edge[d]),domain_width[d]);
      else
	wrapped_pos[d] = pos[d];
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


