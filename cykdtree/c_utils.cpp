#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include "c_utils.hpp"
#include <float.h>

bool isEqual(double f1, double f2) {
  return (fabs(f1 - f2) <= FLT_EPSILON);
}

double* max_pts(double *pts, uint64_t n, uint32_t m)
{
  double* max = (double*)std::malloc(m*sizeof(double));
  uint32_t d;
  for (d = 0; d < m; d++) max[d] = -DBL_MAX; // pts[d];
  for (uint64_t i = 0; i < n; i++) {
    for (d = 0; d < m; d++) {
      if (pts[m*i + d] > max[d])
        max[d] = pts[m*i + d];
    }
  }
  return max;
}

double* min_pts(double *pts, uint64_t n, uint32_t m)
{
  double* min = (double*)std::malloc(m*sizeof(double));
  uint32_t d;
  for (d = 0; d < m; d++) min[d] = DBL_MAX; // pts[d];
  for (uint64_t i = 0; i < n; i++) {
    for (d = 0; d < m; d++) {
      if (pts[m*i + d] < min[d])
        min[d] = pts[m*i + d];
    }
  }
  return min;
}

// http://www.comp.dit.ie/rlawlor/Alg_DS/sorting/quickSort.c 
void quickSort(double *pts, uint64_t *idx,
               uint32_t ndim, uint32_t d,
               int64_t l, int64_t r)
{
  int64_t j;
  if( l < r )
    {
      // divide and conquer
      j = partition(pts, idx, ndim, d, l, r, (l+r)/2);
      quickSort(pts, idx, ndim, d, l, j-1);
      quickSort(pts, idx, ndim, d, j+1, r);
    }
}

void insertSort(double *pts, uint64_t *idx,
                uint32_t ndim, uint32_t d,
                int64_t l, int64_t r)
{
  int64_t i, j;
  uint64_t t;

  if (r <= l) return;
  for (i = l+1; i <= r; i++) {
    t = idx[i];
    j = i - 1;
    while ((j >= l) && (pts[ndim*idx[j]+d] > pts[ndim*t+d])) {
      idx[j+1] = idx[j];
      j--;
    }
    idx[j+1] = t;
  }
}

int64_t pivot(double *pts, uint64_t *idx,
              uint32_t ndim, uint32_t d,
              int64_t l, int64_t r)
{ 
  if (r < l) {
    return -1;
  } else if (r == l) {
    return l;
  } else if ((r - l) < 5) {
    insertSort(pts, idx, ndim, d, l, r);
    return (l+r)/2;
  }

  int64_t i, subr, m5;
  uint64_t t;
  int64_t nsub = 0;
  for (i = l; i <= r; i+=5) {
    subr = i + 4;
    if (subr > r) subr = r;

    insertSort(pts, idx, ndim, d, i, subr);
    m5 = (i+subr)/2;
    t = idx[m5]; idx[m5] = idx[l + nsub]; idx[l + nsub] = t;

    nsub++;
  }
  return pivot(pts, idx, ndim, d, l, l+nsub-1);
  // return select(pts, idx, ndim, d, l, l+nsub-1, (nsub/2)+(nsub%2));
}

int64_t partition_given_pivot(double *pts, uint64_t *idx,
			      uint32_t ndim, uint32_t d,
			      int64_t l, int64_t r, double pivot) {
  // If all less than pivot, j will remain r
  // If all greater than pivot, j will be l-1
  if (r < l)
    return -1;
  int64_t i, j;
  uint64_t t;
  for (i = l, j = r; i <= j; ) {
    if ((pts[ndim*idx[i]+d] > pivot) && (pts[ndim*idx[j]+d] <= pivot)) {
      t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    if (pts[ndim*idx[i]+d] <= pivot) i++;
    if (pts[ndim*idx[j]+d] > pivot) j--;
  }

  return j;
}

int64_t partition(double *pts, uint64_t *idx,
                  uint32_t ndim, uint32_t d,
                  int64_t l, int64_t r, int64_t p)
{ 
  double pivot;
  int64_t j;
  uint64_t t;
  if (r < l)
    return -1;
  pivot = pts[ndim*idx[p]+d];
  t = idx[p]; idx[p] = idx[l]; idx[l] = t;

  j = partition_given_pivot(pts, idx, ndim, d, l+1, r, pivot);

  t = idx[l]; idx[l] = idx[j]; idx[j] = t;

  return j;
}

// https://en.wikipedia.org/wiki/Median_of_medians
int64_t select(double *pts, uint64_t *idx,
               uint32_t ndim, uint32_t d,
               int64_t l0, int64_t r0, int64_t n)
{
  int64_t p;
  int64_t l = l0, r = r0;

  while ( 1 ) {
    if (l == r) return l;

    p = pivot(pts, idx, ndim, d, l, r);
    p = partition(pts, idx, ndim, d, l, r, p);
    if (p < 0) 
      return -1;
    else if (n == (p-l0+1)) {
      return p;
    } else if (n < (p-l0+1)) {
      r = p - 1;
    } else {
      l = p + 1;
    }
  }
}

uint32_t split(double *all_pts, uint64_t *all_idx,
               uint64_t Lidx, uint64_t n, uint32_t ndim,
               double *mins, double *maxes,
               int64_t &split_idx, double &split_val) {
  // Return immediately if variables empty
  if ((n == 0) or (ndim == 0)) {
    split_idx = -1;
    split_val = 0.0;
    return 0;
  }

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
  int64_t nsel = (n/2) + (n%2);
  split_idx = select(all_pts, all_idx, ndim, dmax, Lidx, Lidx+n-1, nsel);
  split_val = all_pts[ndim*all_idx[split_idx] + dmax];

  return dmax;
}



