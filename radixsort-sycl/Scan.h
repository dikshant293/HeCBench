/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#ifndef _SCAN_H_
#define _SCAN_H_

#include "common.h"

#define MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE 1024
#define MAX_LOCAL_GROUP_SIZE 256
static const int WORKGROUP_SIZE = 256;
static const unsigned int   MAX_BATCH_ELEMENTS = 64 * 1048576;
static const unsigned int MIN_SHORT_ARRAY_SIZE = 4;
static const unsigned int MAX_SHORT_ARRAY_SIZE = 4 * WORKGROUP_SIZE;
static const unsigned int MIN_LARGE_ARRAY_SIZE = 8 * WORKGROUP_SIZE;
static const unsigned int MAX_LARGE_ARRAY_SIZE = 4 * WORKGROUP_SIZE * WORKGROUP_SIZE;

unsigned int factorRadix2(unsigned int& log2L, unsigned int L);

void scanExclusiveLarge(
    queue &q,
    buffer<unsigned int> &d_Dst,
    buffer<unsigned int> &d_Src,
    buffer<unsigned int> &d_Buf,
    const unsigned int batchSize,
    const unsigned int arrayLength, 
    const unsigned int numElements);
#endif
