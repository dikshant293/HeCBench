#!/bin/bash

# ml cuda/11.7.1
# ml cuda/11.5.2
 ml cuda

 export USE_ASYNC=$1
 export USE_OMP=$2

 make clean
 make -f Makefile
