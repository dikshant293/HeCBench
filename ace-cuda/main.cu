#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

//define the data set size (cubic volume)
#define DATAXSIZE 600
#define DATAYSIZE 600
#define DATAZSIZE 600

// square
#define SQ(x) ((x)*(x))

__device__
double dFphi(double phi, double u, double lambda)
{
  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));
}

__device__
double GradientX(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);
}

__device__
double GradientY(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);
}

__device__
double GradientZ(double phi[][DATAYSIZE][DATAXSIZE], 
                 double dx, double dy, double dz, int x, int y, int z)
{
  return (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);
}

__device__
double Divergence(double phix[][DATAYSIZE][DATAXSIZE], 
                  double phiy[][DATAYSIZE][DATAXSIZE],
                  double phiz[][DATAYSIZE][DATAXSIZE], 
                  double dx, double dy, double dz, int x, int y, int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) + 
         GradientY(phiy,dx,dy,dz,x,y,z) +
         GradientZ(phiz,dx,dy,dz,x,y,z);
}

__device__
double Laplacian(double phi[][DATAYSIZE][DATAXSIZE],
                 double dx, double dy, double dz, int x, int y, int z)
{
  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0 * phi[x][y][z]) / SQ(dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0 * phi[x][y][z]) / SQ(dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0 * phi[x][y][z]) / SQ(dz);
  return phixx + phiyy + phizz;
}

__device__
double An(double phix, double phiy, double phiz, double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
    return ((1.0 - 3.0 * epsilon) * (1.0 + (((4.0 * epsilon) / (1.0-3.0*epsilon))*
           ((SQ(phix)*SQ(phix)+SQ(phiy)*SQ(phiy)+SQ(phiz)*SQ(phiz)) /
           ((SQ(phix)+SQ(phiy)+SQ(phiz))*(SQ(phix)+SQ(phiy)+SQ(phiz)))))));
  }
  else
  {
    return (1.0-((5.0/3.0)*epsilon));
  }
}

__device__
double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{
  return (W0*An(phix,phiy,phiz,epsilon));
}

__device__
double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{
  return tau0 * SQ(An(phix,phiy,phiz,epsilon));
}

__device__
double dFunc(double l, double m, double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0){
    return (((l*l*l*(SQ(m)+SQ(n)))-(l*(SQ(m)*SQ(m)+SQ(n)*SQ(n)))) /
            ((SQ(l)+SQ(m)+SQ(n))*(SQ(l)+SQ(m)+SQ(n))));
  }
  else
  {
    return 0.0;
  }
}

__global__
void calculateForce(double phi[][DATAYSIZE][DATAXSIZE], 
                    double Fx[][DATAYSIZE][DATAXSIZE],
                    double Fy[][DATAYSIZE][DATAXSIZE],
                    double Fz[][DATAYSIZE][DATAXSIZE],
                    double dx, double dy, double dz,
                    double epsilon, double W0, double tau0)
{

  unsigned iz = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned ix = blockIdx.z*blockDim.z + threadIdx.z;

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
      (iz < (DATAZSIZE-1)) && (ix > (0)) && 
      (iy > (0)) && (iz > (0))) {

    double phix = GradientX(phi,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phi,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phi,dx,dy,dz,ix,iy,iz);
    double sqGphi = SQ(phix) + SQ(phiy) + SQ(phiz);
    double c = 16.0 * W0 * epsilon;
    double w = Wn(phix,phiy,phiz,epsilon,W0);
    double w2 = SQ(w);
    

    Fx[ix][iy][iz] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
    Fy[ix][iy][iz] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
    Fz[ix][iy][iz] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
  }
  else
  {
    Fx[ix][iy][iz] = 0.0;
    Fy[ix][iy][iz] = 0.0;
    Fz[ix][iy][iz] = 0.0;
  }

}

// device function to set the 3D volume
__global__
void allenCahn(double phinew[][DATAYSIZE][DATAXSIZE], 
               double phiold[][DATAYSIZE][DATAXSIZE],
               double uold[][DATAYSIZE][DATAXSIZE],
               double Fx[][DATAYSIZE][DATAXSIZE],
               double Fy[][DATAYSIZE][DATAXSIZE],
               double Fz[][DATAYSIZE][DATAXSIZE],
               double epsilon, double W0, double tau0, double lambda,
               double dt, double dx, double dy, double dz)
{
  unsigned iz = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned ix = blockIdx.z*blockDim.z + threadIdx.z;

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
      (iz < (DATAZSIZE-1)) && (ix > (0)) && 
      (iy > (0)) && (iz > (0))) {

    double phix = GradientX(phiold,dx,dy,dz,ix,iy,iz);
    double phiy = GradientY(phiold,dx,dy,dz,ix,iy,iz);
    double phiz = GradientZ(phiold,dx,dy,dz,ix,iy,iz); 

    phinew[ix][iy][iz] = phiold[ix][iy][iz] + 
     (dt / taun(phix,phiy,phiz,epsilon,tau0)) * 
     (Divergence(Fx,Fy,Fz,dx,dy,dz,ix,iy,iz) - 
      dFphi(phiold[ix][iy][iz], uold[ix][iy][iz],lambda));
  }
}

__global__
void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAXSIZE])
{
  unsigned iz = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned ix = blockIdx.z*blockDim.z + threadIdx.z;

  if (ix == 0){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (ix == DATAXSIZE-1){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iy == 0){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iy == DATAYSIZE-1){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iz == 0){
    phinew[ix][iy][iz] = -1.0;
  }
  else if (iz == DATAZSIZE-1){
    phinew[ix][iy][iz] = -1.0;
  }
}

__global__
void thermalEquation(double unew[][DATAYSIZE][DATAXSIZE],
                     double uold[][DATAYSIZE][DATAXSIZE],
                     double phinew[][DATAYSIZE][DATAXSIZE],
                     double phiold[][DATAYSIZE][DATAXSIZE],
                     double D, double dt, double dx, double dy, double dz)
{
  unsigned iz = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned ix = blockIdx.z*blockDim.z + threadIdx.z;

  if ((ix < (DATAXSIZE-1)) && (iy < (DATAYSIZE-1)) && 
      (iz < (DATAZSIZE-1)) && (ix > (0)) && 
      (iy > (0)) && (iz > (0))){
    unew[ix][iy][iz] = uold[ix][iy][iz] + 
      0.5*(phinew[ix][iy][iz]- phiold[ix][iy][iz]) +
      dt * D * Laplacian(uold,dx,dy,dz,ix,iy,iz);
  }
}

__global__
void boundaryConditionsU(double unew[][DATAYSIZE][DATAXSIZE], double delta)
{
  unsigned iz = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned ix = blockIdx.z*blockDim.z + threadIdx.z;

  if (ix == 0){
    unew[ix][iy][iz] =  -delta;
  }
  else if (ix == DATAXSIZE-1){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iy == 0){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iy == DATAYSIZE-1){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iz == 0){
    unew[ix][iy][iz] =  -delta;
  }
  else if (iz == DATAZSIZE-1){
    unew[ix][iy][iz] =  -delta;
  }
}

__global__
void swapGrid(double cnew[][DATAYSIZE][DATAXSIZE],
              double cold[][DATAYSIZE][DATAXSIZE])
{
  unsigned iz = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned iy = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned ix = blockIdx.z*blockDim.z + threadIdx.z;

  if ((ix < (DATAXSIZE)) && 
      (iy < (DATAYSIZE)) &&
      (iz < (DATAZSIZE))) {
    double tmp = cnew[ix][iy][iz];
    cnew[ix][iy][iz] = cold[ix][iy][iz];
    cold[ix][iy][iz] = tmp;
  }
}

void initializationPhi(double phi[][DATAYSIZE][DATAXSIZE], double r0)
{
  #pragma omp parallel for collapse(3)
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0){
          phi[idx][idy][idz] = 1.0;
        }
        else
        {
          phi[idx][idy][idz] = -1.0;
        }
      }
    }
  }
}

void initializationU(double u[][DATAYSIZE][DATAXSIZE], double r0, double delta)
{
  #pragma omp parallel for collapse(3)
  for (int idx = 0; idx < DATAXSIZE; idx++) {
    for (int idy = 0; idy < DATAYSIZE; idy++) {
      for (int idz = 0; idz < DATAZSIZE; idz++) {
        double r = std::sqrt(SQ(idx-0.5*DATAXSIZE) + SQ(idy-0.5*DATAYSIZE) + SQ(idz-0.5*DATAZSIZE));
        if (r < r0) {
          u[idx][idy][idz] = 0.0;
        }
        else
        {
          u[idx][idy][idz] = -delta * (1.0 - std::exp(-(r-r0)));
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  typedef double nRarray[DATAYSIZE][DATAXSIZE];
  const int num_steps = atoi(argv[1]);  //6000;
  const double dx = 0.4;
  const double dy = 0.4;
  const double dz = 0.4;
  const double dt = 0.01;
  const double delta = 0.8;
  const double r0 = 5.0;
  const double epsilon = 0.07;
  const double W0 = 1.0;
  const double beta0 = 0.0;
  const double D = 2.0;
  const double d0 = 0.5;
  const double a1 = 1.25 / std::sqrt(2.0);
  const double a2 = 0.64;
  const double lambda = (W0*a1)/(d0);
  const double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));

  // overall data set sizes
  const int nx = DATAXSIZE;
  const int ny = DATAYSIZE;
  const int nz = DATAZSIZE;
  // pointers for data set storage via malloc
  nRarray *phi_host; // storage for result stored on host
  nRarray *u_host;
  nRarray *d_phiold;  // storage for result computed on device
  nRarray *d_phinew;
  nRarray *d_uold;
  nRarray *d_unew;
  nRarray *d_Fx;
  nRarray *d_Fy;
  nRarray *d_Fz;
  // allocate storage for data set

  phi_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double));
  u_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double));

  // compute result

  initializationPhi(phi_host,r0);
  initializationU(u_host,r0,delta);

  double clock_d = double(clock()) / CLOCKS_PER_SEC;

  // define the chunk sizes that each threadblock will work on
  dim3 grid ((DATAZSIZE+7)/8, (DATAYSIZE+7)/8, (DATAXSIZE+3)/4);
  dim3 block (8, 8, 4);

  // allocate GPU device buffers
  cudaMalloc((void **) &d_phiold, (nx*ny*nz)*sizeof(double));
  cudaMalloc((void **) &d_phinew, (nx*ny*nz)*sizeof(double));
  cudaMalloc((void **) &d_uold, (nx*ny*nz)*sizeof(double));
  cudaMalloc((void **) &d_unew, (nx*ny*nz)*sizeof(double));
  cudaMalloc((void **) &d_Fx, (nx*ny*nz)*sizeof(double));
  cudaMalloc((void **) &d_Fy, (nx*ny*nz)*sizeof(double));
  cudaMalloc((void **) &d_Fz, (nx*ny*nz)*sizeof(double));

  cudaMemcpy(d_phiold, phi_host, ((nx*ny*nz)*sizeof(double)), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uold, u_host, ((nx*ny*nz)*sizeof(double)), cudaMemcpyHostToDevice);

  int t = 0;

  while (t <= num_steps) {

    calculateForce<<<grid, block>>>(d_phiold,d_Fx,d_Fy,d_Fz,
                                    dx,dy,dz,epsilon,W0,tau0);

    allenCahn<<<grid, block>>>(d_phinew,d_phiold,d_uold,
                               d_Fx,d_Fy,d_Fz,
                               epsilon,W0,tau0,lambda,
                               dt,dx,dy,dz);

    boundaryConditionsPhi<<<grid, block>>>(d_phinew);

    thermalEquation<<<grid, block>>>(d_unew,d_uold,d_phinew,d_phiold,
                                     D,dt,dx,dy,dz);

    boundaryConditionsU<<<grid, block>>>(d_unew,delta);

    swapGrid<<<grid, block>>>(d_phinew, d_phiold);

    swapGrid<<<grid, block>>>(d_unew, d_uold);

    t++;
  }

  cudaMemcpy(phi_host, d_phiold, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
  cudaMemcpy(u_host, d_uold, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);

  cudaFree(d_phiold);
  cudaFree(d_phinew);
  cudaFree(d_uold);
  cudaFree(d_unew);
  cudaFree(d_Fx);
  cudaFree(d_Fy);
  cudaFree(d_Fz);

  clock_d = double(clock()) / CLOCKS_PER_SEC - clock_d; 
  printf("Offload time = %.3fms\n", clock_d*1e3);

#ifdef DEBUG
  for (int idx = 0; idx < nx; idx++)
    for (int idy = 0; idy < ny; idy++)
      for (int idz = 0; idz < nz; idz++)
        printf("%lf %lf\n", phi_host[idx][idy][idz], u_host[idx][idy][idz]);
#endif

  free(phi_host);
  free(u_host);

  return 0;
}
