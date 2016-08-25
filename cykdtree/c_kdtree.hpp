#include <vector>
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
  std::vector<double> left_edge;
  std::vector<double> right_edge;
  uint64_t left_idx;
  uint64_t children;
  std::vector<bool> periodic_left;
  std::vector<bool> periodic_right;
  std::vector<std::vector<uint32_t>> left_neighbors;
  std::vector<std::vector<uint32_t>> right_neighbors;
  // innernode parameters
  uint32_t split_dim;
  double split;
  Node *less;
  Node *greater;
  // innernode constructor
  Node(uint32_t ndim0, std::vector<double> le, std::vector<double> re, uint64_t Lidx, 
       uint32_t sdim0, double split0, Node *lnode, Node *gnode)
  {
    is_leaf = false;
    leafid = 4294967295;
    ndim = ndim0;
    left_edge = le;
    right_edge = re;
    left_idx = Lidx;

    split_dim = sdim0;
    split = split0;
    less = lnode;
    greater = gnode;
    children = lnode->children + gnode->children;

    for (uint32_t d = 0; d < ndim; d++) {
      periodic_left.push_back(false);
      periodic_right.push_back(true);
    }

    left_neighbors = std::vector<std::vector<uint32_t>>(ndim);
    right_neighbors = std::vector<std::vector<uint32_t>>(ndim);
  }
  // leafnode constructor
  Node(uint32_t ndim0, std::vector<double> le, std::vector<double> re, 
       uint64_t Lidx, uint64_t n, int leafid0)
  {
    is_leaf = true;
    leafid = leafid0;
    ndim = ndim0;
    left_edge = le;
    right_edge = re;
    left_idx = Lidx;

    children = n;

    for (uint32_t d = 0; d < ndim; d++) {
      periodic_left.push_back(false);
      periodic_right.push_back(true);
    }

    left_neighbors = std::vector<std::vector<uint32_t>>(ndim);
    right_neighbors = std::vector<std::vector<uint32_t>>(ndim);
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
  bool periodic;
  double* domain_mins;
  double* domain_maxs;
  uint32_t num_leaves;
  std::vector<Node*> leaves;
  Node* root;

  // KDTree() {}
  KDTree(double *pts, uint64_t *idx, uint64_t n, uint32_t m, uint32_t leafsize0, 
	 double *left_edge, double *right_edge, bool periodic0=false)
  {
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

    std::vector<double> LE;
    std::vector<double> RE;
    std::vector<double> mins;
    std::vector<double> maxs;
    for (uint32_t d = 0; d < m; d++) {
      LE.push_back(left_edge[d]);
      RE.push_back(right_edge[d]);
      mins.push_back(domain_mins[d]);
      maxs.push_back(domain_maxs[d]);
    }

    root = build(0, n, LE, RE, mins, maxs);

    set_neighbors();
  }
  ~KDTree()
  {
    free(domain_mins);
    free(domain_maxs);
    free(root);
  }

  void set_neighbors()
  {
    uint32_t d;
    Node* leaf;
    Node* prev;
    uint64_t i, j;
    bool match;

    // Periodicity in each dimension
    if (periodic) {
      for (i = 0; i < num_leaves; i++) {
    	leaf = leaves[i];
    	for (d = 0; d < ndim; d++) {
    	  leaf->periodic_left[d] = isEqual(leaf->left_edge[d], domain_left_edge[d]);
    	  leaf->periodic_right[d] = isEqual(leaf->right_edge[d], domain_right_edge[d]);
    	}
      }
    }

    // Neighbors
    for (i = 0; i < num_leaves; i++) {
      leaf = leaves[i];
      for (j = 0; j <= i; j++) {
	prev = leaves[j];
	match = true;
	for (d = 0; d < ndim; d++) {
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
	  for (d = 0; d < ndim; d++) {
	    if (isEqual(leaf->left_edge[d], prev->right_edge[d])) {
	      leaf->left_neighbors[d].push_back(prev->leafid);
	      prev->right_neighbors[d].push_back(leaf->leafid);
	    } else if (isEqual(leaf->right_edge[d], prev->left_edge[d])) {
	      leaf->right_neighbors[d].push_back(prev->leafid);
	      prev->left_neighbors[d].push_back(leaf->leafid);
	    }
	    if (periodic) {
	      if (leaf->periodic_right[d] && prev->periodic_left[d]) {
		leaf->right_neighbors[d].push_back(prev->leafid);
		prev->left_neighbors[d].push_back(leaf->leafid);
	      }
	      if (prev->periodic_right[d] && leaf->periodic_left[d]) {
		leaf->left_neighbors[d].push_back(prev->leafid);
		prev->right_neighbors[d].push_back(leaf->leafid);
	      }
	    }		
	  }
	}
      }
    }
  }

  Node* build(uint64_t Lidx, uint64_t n, 
	      std::vector<double> LE, std::vector<double> RE, 
	      std::vector<double> mins, std::vector<double> maxes)
  {
    if (n < leafsize) {
      Node* out = new Node(ndim, LE, RE, Lidx, n, num_leaves);
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
	Node* out = new Node(ndim, LE, RE, Lidx, n, num_leaves);
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
	// split = (all_pts[ndim*all_idx[med] + dmax] + 
	// 	 all_pts[ndim*all_idx[med+1] + dmax])/2.0;
      } else {
	split = all_pts[ndim*all_idx[med] + dmax];
      }

      // Get new boundaries
      std::vector<double> lessmaxes;
      std::vector<double> lessright;
      std::vector<double> greatermins;
      std::vector<double> greaterleft;
      for (d = 0; d < ndim; d++) {
	lessmaxes.push_back(maxes[d]);
	lessright.push_back(RE[d]);
	greatermins.push_back(mins[d]);
	greaterleft.push_back(LE[d]);
      }
      lessmaxes[dmax] = split;
      lessright[dmax] = split;
      greatermins[dmax] = split;
      greaterleft[dmax] = split;

      // Build less and greater nodes
      Node* less = build(Lidx, Nless, LE, lessright, mins, lessmaxes);
      Node* greater = build(Lidx+Nless, Ngreater, greaterleft, RE, greatermins, maxes);

      // Create innernode referencing child nodes
      Node* out = new Node(ndim, LE, RE, Lidx, dmax, split, less, greater);
      return out;
    } 
  }	 

  Node* search(double* pos)
  {
    uint32_t i;
    bool thisChild;
    Node* out;
    // Ensure that pos is in root, return NULL if it's not
    thisChild = true;
    for (i = 0; i < ndim; i++) {
      if (pos[i] < root->left_edge[i]) {
	thisChild = false;
	break;
      }
      if (pos[i] >= root->right_edge[i]) {
	thisChild = false;
	break;
      }
    }
    if (thisChild) {
      out = root;
    } else {
      out = NULL;
      return out;
    }
    // Traverse tree looking for leaf containing pos
    while (!(out->is_leaf)) {
      // Less
      thisChild = true;
      for (i = 0; i < ndim; i++) {
	if (pos[i] < out->less->left_edge[i]) {
	  thisChild = false;
	  break;
	}
	if (pos[i] >= out->less->right_edge[i]) {
	  thisChild = false;
	  break;
	}
      }
      if (thisChild) {
	out = out->less;
	continue;
      }
      // Greater
      thisChild = true;
      for (i = 0; i < ndim; i++) {
	if (pos[i] < out->greater->left_edge[i]) {
	  thisChild = false;
	  break;
	}
	if (pos[i] >= out->greater->right_edge[i]) {
	  thisChild = false;
	  break;
	}
      }
      if (thisChild) {
	out = out->greater;
	continue;
      }
    }
    return out;
  }

};


