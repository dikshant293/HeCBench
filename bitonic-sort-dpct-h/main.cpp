//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Bitonic Sort: this algorithm converts a randomized sequence of numbers into
// a bitonic sequence (two ordered sequences), and then merge these two ordered
// sequences into a ordered sequence. Bitonic sort algorithm is briefly
// described as followed:
//
// - First, it decomposes the randomized sequence of size 2**n into 2**(n-1)
// pairs where each pair consists of 2 consecutive elements. Note that each pair
// is a bitonic sequence.
// - Step 0: for each pair (sequence of size 2), the two elements are swapped so
// that the two consecutive pairs form  a bitonic sequence in increasing order,
// the next two pairs form the second bitonic sequence in decreasing order, the
// next two pairs form the third bitonic sequence in  increasing order, etc, ...
// . At the end of this step, we have 2**(n-1) bitonic sequences of size 2, and
// they follow an order increasing, decreasing, increasing, .., decreasing.
// Thus, they form 2**(n-2) bitonic sequences of size 4.
// - Step 1: for each new 2**(n-2) bitonic sequences of size 4, (each new
// sequence consists of 2 consecutive previous sequences), it swaps the elements
// so that at the end of step 1, we have 2**(n-2) bitonic sequences of size 4,
// and they follow an order: increasing, decreasing, increasing, ...,
// decreasing. Thus, they form 2**(n-3) bitonic sequences of size 8.
// - Same logic applies until we reach the last step.
// - Step n: at this last step, we have one bitonic sequence of size 2**n. The
// elements in the sequence are swapped until we have a sequence in increasing
// oder.
//
// In this implementation, a randomized sequence of size 2**n is given (n is a
// positive number). Unified Shared Memory (USM) is used for data management. At
// each stage, a part of step, the host redefines the ordered sequenes and sends
// data to the kernel. The kernel swaps the elements accordingly in parallel.
//
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <iostream>
#include <limits>

using namespace std;

#define BLOCK_SIZE 256

void bs (const int seq_len, const int two_power, int *a,
         sycl::nd_item<3> item_ct1) {

  int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
          item_ct1.get_local_id(2);

  // Assign the bitonic sequence number.
  int seq_num = i / seq_len;

  // Variable used to identified the swapped element.
  int swapped_ele = -1;

  // Because the elements in the first half in the bitonic
  // sequence may swap with elements in the second half,
  // only the first half of elements in each sequence is
  // required (seq_len/2).
  int h_len = seq_len / 2;

  if (i < (seq_len * seq_num) + h_len) swapped_ele = i + h_len;

  // Check whether increasing or decreasing order.
  int odd = seq_num / two_power;

  // Boolean variable used to determine "increasing" or
  // "decreasing" order.
  bool increasing = ((odd % 2) == 0);

  // Swap the elements in the bitonic sequence if needed
  if (swapped_ele != -1) {
    if (((a[i] > a[swapped_ele]) && increasing) ||
	((a[i] < a[swapped_ele]) && !increasing)) {
      int temp = a[i];
      a[i] = a[swapped_ele];
      a[swapped_ele] = temp;
    }
  }
}

void ParallelBitonicSort(int data_gpu[], int n) {

  // n: the exponent used to set the array size. Array size = power(2, n)
  int size = sycl::pown((float)(2), n);

  // SYCL buffer allocated for device 
  int *a;
  dpct::dpct_malloc((void **)&a, sizeof(int) * size);
  dpct::dpct_memcpy(a, data_gpu, sizeof(int) * size, dpct::host_to_device);

  // step from 0, 1, 2, ...., n-1
  for (int step = 0; step < n; step++) {
    // for each step s, stage goes s, s-1, ..., 0
    for (int stage = step; stage >= 0; stage--) {
      // In each state, construct a number (num_seq) of bitonic sequences of
      // size seq_len (2, 4, ...) num_seq stores the number of bitonic sequences
      // at each stage. seq_len stores the length of the bitonic sequence at
      // each stage.
      int seq_len = sycl::pown((float)(2), stage + 1);
#if DEBUG
      int num_seq = pow(2, (n - stage - 1));  // Used for debug purpose.
      std::cout << "step num:" << step << " stage num:" << stage
                << " num_seq:" << num_seq << "(" << seq_len << ") => ";
#endif
      // Constant used in the kernel: 2**(step-stage).
      int two_power = 1 << (step - stage);
      {
        dpct::buffer_t a_buf_ct2 = dpct::get_buffer(a);
        dpct::get_default_queue().submit([&](sycl::handler &cgh) {
          auto a_acc_ct2 =
              a_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, size / BLOCK_SIZE) *
                                    sycl::range<3>(1, 1, BLOCK_SIZE),
                                sycl::range<3>(1, 1, BLOCK_SIZE)),
              [=](sycl::nd_item<3> item_ct1) {
                bs(seq_len, two_power, (int *)(&a_acc_ct2[0]), item_ct1);
              });
        });
      }
      dpct::get_current_device().queues_wait_and_throw();
    }  // end stage
  }    // end step
  dpct::dpct_memcpy(data_gpu, a, sizeof(int) * size, dpct::device_to_host);
  dpct::dpct_free(a);
}

// Loop over the bitonic sequences at each stage in serial.
void SwapElements(int step, int stage, int num_sequence, int seq_len,
                  int *array) {
  for (int seq_num = 0; seq_num < num_sequence; seq_num++) {
    int odd = seq_num / (sycl::pown((float)(2), (step - stage)));
    bool increasing = ((odd % 2) == 0);

    int h_len = seq_len / 2;

    // For all elements in a bitonic sequence, swap them if needed
    for (int i = seq_num * seq_len; i < seq_num * seq_len + h_len; i++) {
      int swapped_ele = i + h_len;

      if (((array[i] > array[swapped_ele]) && increasing) ||
          ((array[i] < array[swapped_ele]) && !increasing)) {
        int temp = array[i];
        array[i] = array[swapped_ele];
        array[swapped_ele] = temp;
      }
    }  // end for all elements in a sequence
  }    // end all sequences
}

// Function sorts an array in serial using bitonic sort algorithm. The size of
// the array is indicated by the exponent n: the array size is 2 ** n.
inline void BitonicSort(int a[], int n) {
  // n: the exponent indicating the array size = 2 ** n.

  // step from 0, 1, 2, ...., n-1
  for (int step = 0; step < n; step++) {
    // for each step s, stage goes s, s-1,..., 0
    for (int stage = step; stage >= 0; stage--) {
      // Sequences (same size) are formed at each stage.
      int num_sequence = sycl::pown((float)(2), (n - stage - 1));
      // The length of the sequences (2, 4, ...).
      int sequence_len = sycl::pown((float)(2), stage + 1);

      SwapElements(step, stage, num_sequence, sequence_len, a);
    }
  }
}

// Function showing the array.
void DisplayArray(int a[], int array_size) {
  for (int i = 0; i < array_size; ++i) std::cout << a[i] << " ";
  std::cout << "\n";
}

void Usage(std::string prog_name, int exponent) {
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: " << prog_name << " n k \n\n";
  std::cout << " n: Integer exponent presenting the size of the input array. "
               "The number of element in\n";
  std::cout << "    the array must be power of 2 (e.g., 1, 2, 4, ...). Please "
               "enter the corresponding\n";
  std::cout << "    exponent betwwen 0 and " << exponent - 1 << ".\n";
  std::cout << " k: Seed used to generate a random sequence.\n";
}

int main(int argc, char *argv[]) {
  int n, seed, size;
  int exp_max = log2(std::numeric_limits<int>::max());

  // Read parameters.
  try {
    n = std::stoi(argv[1]);

    // Verify the boundary of acceptance.
    if (n < 0 || n >= exp_max) {
      Usage(argv[0], exp_max);
      return -1;
    }

    seed = std::stoi(argv[2]);
    size = sycl::pown((float)(2), n);
  } catch (...) {
    Usage(argv[0], exp_max);
    return -1;
  }

  std::cout << "\nArray size: " << size << ", seed: " << seed << "\n";

  // Memory allocated for host access only.
  int *data_cpu = (int *)malloc(size * sizeof(int));

  // Memory allocated to store gpu results
  int *data_gpu = (int *)malloc(size * sizeof(int));

  // Initialize the array randomly using a seed.
  srand(seed);

  for (int i = 0; i < size; i++) {
    data_gpu[i] = data_cpu[i] = rand() % 1000;
  }


#if DEBUG
  std::cout << "\ndata before:\n";
  DisplayArray(data_gpu, size);
#endif

  // Start timer
  ParallelBitonicSort(data_gpu, n);

#if DEBUG
  std::cout << "\ndata after sorting using parallel bitonic sort:\n";
  DisplayArray(data_gpu, size);
#endif

  // Bitonic sort in CPU (serial)
  BitonicSort(data_cpu, n);

  // Verify both bitonic sort algorithms in kernel and in CPU.
  bool pass = true;
  for (int i = 0; i < size - 1; i++) {
    // Validate the sequence order is increasing in both kernel and CPU.
    if ((data_gpu[i] > data_gpu[i + 1]) || (data_gpu[i] != data_cpu[i])) {
      pass = false;
      break;
    }
  }

  // Clean CPU memory.
  free(data_cpu);
  free(data_gpu);

  if (!pass) {
    std::cout << "\nFailed!\n";
    return -2;
  }

  std::cout << "\nSuccess!\n";
  return 0;
}
