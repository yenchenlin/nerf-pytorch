#include "searchsorted_cpu_wrapper.h"
#include <stdio.h>

template<typename scalar_t>
int eval(scalar_t val, scalar_t *a, int64_t row, int64_t col, int64_t ncol, bool side_left)
{
    /* Evaluates whether a[row,col] < val <= a[row, col+1]*/

    if (col == ncol - 1)
    {
      // special case: we are on the right border
      if (a[row * ncol + col] <= val){
        return 1;}
      else {
        return -1;}
    }
    bool is_lower;
    bool is_next_higher;

    if (side_left) {
      // a[row, col] < v <= a[row, col+1]
      is_lower = (a[row * ncol + col] < val);
      is_next_higher = (a[row*ncol + col + 1] >= val);
    } else {
      // a[row, col] <= v < a[row, col+1]
      is_lower = (a[row * ncol + col] <= val);
      is_next_higher = (a[row * ncol + col + 1] > val);
    }
    if (is_lower && is_next_higher) {
        // we found the right spot
        return 0;
    } else if (is_lower) {
    	// answer is on the right side
        return 1;
    } else {
    	// answer is on the left side
        return -1;
    }
}

template<typename scalar_t>
int64_t binary_search(scalar_t*a, int64_t row, scalar_t val, int64_t ncol, bool side_left)
{
  /* Look for the value `val` within row `row` of matrix `a`, which
  has `ncol` columns.

  the `a` matrix is assumed sorted in increasing order, row-wise

  returns:
  * -1 if `val` is smaller than the smallest value found within that row of `a`
  * `ncol` - 1 if `val` is larger than the largest element of that row of `a`
  * Otherwise, return the column index `res` such that:
    - a[row, col] < val <= a[row, col+1]. (if side_left), or
    - a[row, col] < val <= a[row, col+1] (if not side_left).
   */

  //start with left at 0 and right at number of columns of a
  int64_t right = ncol;
  int64_t left = 0;

  while (right >= left) {
      // take the midpoint of current left and right cursors
      int64_t mid = left + (right-left)/2;

      // check the relative position of val: are we good here ?
      int rel_pos = eval(val, a, row, mid, ncol, side_left);
      // we found the point
      if(rel_pos == 0) {
          return mid;
      } else if (rel_pos > 0) {
        if (mid==ncol-1){return ncol-1;}
        // the answer is on the right side
        left = mid;
      } else {
        if (mid==0){return -1;}
        right = mid;
      }
  }
  return -1;
}

void searchsorted_cpu_wrapper(
    at::Tensor a,
    at::Tensor v,
    at::Tensor res,
    bool side_left)
{

  // Get the dimensions
  auto nrow_a = a.size(/*dim=*/0);
  auto ncol_a = a.size(/*dim=*/1);
  auto nrow_v = v.size(/*dim=*/0);
  auto ncol_v = v.size(/*dim=*/1);

  auto nrow_res = fmax(nrow_a, nrow_v);

  //auto acc_v = v.accessor<float, 2>();
  //auto acc_res = res.accessor<float, 2>();

  AT_DISPATCH_ALL_TYPES(a.type(), "searchsorted cpu", [&] {

      scalar_t* a_data = a.data_ptr<scalar_t>();
      scalar_t* v_data = v.data_ptr<scalar_t>();
      int64_t* res_data = res.data<int64_t>();

      for (int64_t row = 0; row < nrow_res; row++)
      {
          for (int64_t col = 0; col < ncol_v; col++)
          {
              // get the value to look for
              int64_t row_in_v = (nrow_v == 1) ? 0 : row;
              int64_t row_in_a = (nrow_a == 1) ? 0 : row;

              int64_t idx_in_v = row_in_v * ncol_v + col;
              int64_t idx_in_res = row * ncol_v + col;

              // apply binary search
              res_data[idx_in_res] = (binary_search(a_data, row_in_a, v_data[idx_in_v], ncol_a, side_left) + 1);
          }
      }
      });
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("searchsorted_cpu_wrapper", &searchsorted_cpu_wrapper, "searchsorted (CPU)");
  }
