#!/bin/bash


 CMP=$1

 case ${CMP} in
	 xl)
	 echo "xl compiler" 
	 ;;	 
	 nvc)
	 echo "nvcc compiler" 
	 ml nvhpc/22.11
         ;;	 
	 clang)
	 echo "llvm clang compiler" 
	 ml cuda/11.7.1
         module use /sw/summit/modulefiles/ums/stf010/Core
         ml llvm/16.0.0-20230110
	 #ml llvm/17.0.0-20230126
	 #ml llvm/17.0.0-20230306
	 ;;
	 gcc)
	 echo "gcc compiler" 
	 #ml gcc/12.1.0
	 ml gcc/11.2.0
	 ;;
	 *)
	 echo "unknown compiler" 
	 ;;
 esac	 

 export USE_ASYNC=$2

 make -f Makefile.${CMP} clean
 make -f Makefile.${CMP}
