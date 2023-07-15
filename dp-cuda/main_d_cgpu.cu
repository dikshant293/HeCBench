
 /* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// *********************************************************************
// A simple demo application that implements a
// vector dot product computation between 2 float arrays. 
//
// Runs computations with on the GPU device and then checks results 
// against basic host CPU/C++ computation.
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>
#include "shrUtils.h"
#include <omp.h>

// Forward Declarations
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);

__global__
void dot_product(const float *__restrict__ a,
                 const float *__restrict__ b,
                       float *__restrict__ c,
#ifdef ASYNC
                 const int streamIdx,
                 const int pivot,
#endif
                 const int n,
                 const int iKWeight)
{
  int iGID = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef ASYNC
  iGID += pivot +streamIdx*n;
  if (iGID < pivot +(streamIdx+1)*n) {
#else
  if (iGID < n) {
#endif

    int iInOffset = iGID << 2;
    for (int k = 0; k < iKWeight; k++) 
    c[iGID] = a[iInOffset    ] * b[iInOffset    ] +
              a[iInOffset + 1] * b[iInOffset + 1] +
              a[iInOffset + 2] * b[iInOffset + 2] +
              a[iInOffset + 3] * b[iInOffset + 3];
  }
}

int main(int argc, char **argv)
{
#ifdef ASYNC
  if (argc >= 9) {
#else
  if (argc != 5) {
#endif
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int iNumElements = atoi(argv[1]);
  const int iNumIterations = atoi(argv[2]);
  const int iKWeight = atoi(argv[3]);
  // set and log Global and Local work size dimensions
  int szLocalWorkSize = atoi(argv[4]);
#ifdef ASYNC
  const int ncustreams = atoi(argv[5]);
  const int nhostthreads  = atoi(argv[6]);
  const int fraction  = atoi(argv[7]); //partition for the computation cpu-gpu
  const int numElementsCPU = iNumElements/fraction;
  const int numElementsGPU = iNumElements - numElementsCPU;
  const int numElements_stream = numElementsGPU/ncustreams;
  const size_t src_size_cpu = numElementsCPU *4;
  const size_t dst_gpu_size_bytes = numElementsGPU * sizeof(float);
  const size_t src_size_stream = numElements_stream *4;
  const size_t src_size_bytes_stream  = src_size_stream* sizeof(float);
#endif
  // rounded up to the nearest multiple of the LocalWorkSize
  int szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  

  const size_t src_size = szGlobalWorkSize * 4;
  const size_t src_size_bytes = src_size * sizeof(float);

  const size_t dst_size = szGlobalWorkSize;
  const size_t dst_size_bytes = dst_size * sizeof(float);

  // Allocate and initialize host arrays
float* srcA;
float* srcB;

#ifdef ASYNC
  cudaMallocHost (&srcA, src_size_bytes);
  cudaMallocHost (&srcB, src_size_bytes);
#else
  srcA = (float*) malloc (src_size_bytes);
  srcB = (float*) malloc (src_size_bytes);
#endif
  float*  dst = (float*) malloc (dst_size_bytes);

  float* Golden = (float*) malloc (sizeof(float) * iNumElements);
  shrFillArray(srcA, 4 * iNumElements);
  shrFillArray(srcB, 4 * iNumElements);

  float *d_srcA;
  float *d_srcB;
  float *d_dst; 

  cudaMalloc((void**)&d_srcA, src_size_bytes);
  cudaMalloc((void**)&d_srcB, src_size_bytes);
  cudaMalloc((void**)&d_dst, dst_size_bytes);

  //printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n# of Work Groups \t\t= %d\n\n", 
      //szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 
#ifdef ASYNC
  dim3 grid (numElements_stream % szLocalWorkSize + numElements_stream/szLocalWorkSize); 
#else
  dim3 grid (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize); 
#endif
  dim3 block (szLocalWorkSize);

  //cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

#ifdef ASYNC
  cudaStream_t custream[ncustreams];
  for (int ics=0; ics<ncustreams; ics++ )
    cudaStreamCreate(&custream[ics]);
#endif

  for (int i = 0; i < iNumIterations; i++) {
#ifdef ASYNC
    //cudaMemcpyAsync(d_srcA, srcA, src_size_bytes, cudaMemcpyHostToDevice, custream[0]);
    //cudaMemcpyAsync(d_srcB, srcB, src_size_bytes, cudaMemcpyHostToDevice, custream[1]);
    //cudaDeviceSynchronize();

    DotProductHost ((const float*)srcA, (const float*)srcB, (float*)dst, numElementsCPU);
    #pragma omp parallel num_threads( nhostthreads)
    {
       //cudaStream_t custream;
       //cudaStreamCreate(&custream);
       //int tid = omp_get_thread_num();
       //int offset = tid*ncustreams_thread;
    #pragma omp for nowait
    for (int k=0; k<ncustreams; k++){
   //      cudaStream_t custream;
   //      cudaStreamCreate(&custream);
         size_t offset = src_size_cpu +k*src_size_stream;
         cudaMemcpyAsync(&d_srcA[offset], &srcA[offset], src_size_bytes_stream, cudaMemcpyHostToDevice, custream[k]);
         cudaMemcpyAsync(&d_srcB[offset], &srcB[offset], src_size_bytes_stream, cudaMemcpyHostToDevice, custream[k]);
         cudaDeviceSynchronize();

       dot_product<<<grid, block, 0, custream[k] >>>(d_srcA, d_srcB, d_dst, k, numElementsCPU, numElements_stream, iKWeight);

      // cudaStreamDestroy(custream);
    }
    }

#else
    cudaMemcpy(d_srcA, srcA, src_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcB, srcB, src_size_bytes, cudaMemcpyHostToDevice);
    dot_product<<<grid, block>>>(d_srcA, d_srcB, d_dst, iNumElements, iKWeight);
#endif

}
  cudaDeviceSynchronize();
#ifdef ASYNC
  cudaMemcpy(&dst[numElementsCPU], &d_dst[numElementsCPU], dst_gpu_size_bytes, cudaMemcpyDeviceToHost);
  for (int ics=0; ics<ncustreams; ics++ )
    cudaStreamDestroy(custream[ics]);
#else
  cudaMemcpy(dst, d_dst, dst_size_bytes, cudaMemcpyDeviceToHost);
#endif

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  //printf("Average execution time %f (s)\n", (time * 1e-9f) / iNumIterations);
  printf("%f\n", (time * 1e-9f) );

  // Compute and compare results for golden-host and report errors and pass/fail
  //printf("Comparing against Host/C++ computation...\n\n"); 
  DotProductHost ((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
  shrBOOL bMatch = shrComparefet((const float*)Golden, (const float*)dst, (unsigned int)iNumElements, 0.0f, 0);
  //printf("\nGPU Result %s CPU Result\n", (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

#ifdef ASYNC
  cudaFreeHost(srcA);
  cudaFreeHost(srcB);
#endif

  cudaFree(d_dst);
  cudaFree(d_srcA);
  cudaFree(d_srcB);

#ifndef ASYNC
  free(srcA);
  free(srcB);
#endif

  free(dst);
  free(Golden);
  return EXIT_SUCCESS;
}

// "Golden" Host processing dot product function for comparison purposes
// *********************************************************************
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
{
  int i, j, k;
#pragma omp parallel for
  for (i = 0, j = 0; i < iNumElements; i++) 
  {
    pfResult[i] = 0.0f;
    for (k = 0; k < 4; k++, j++) 
    {
      pfResult[i] += pfData1[j] * pfData2[j]; 
    } 
  }
}
