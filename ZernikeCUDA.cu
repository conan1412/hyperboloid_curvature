#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 
#include<iostream>
#include<math.h>
#include<iostream>

using namespace std;

__constant__ float indice[36][2] = {
		{0,0},{1,-1},{1,1},{2,-2},{2,0},{2,2},{3,-3},{3,-1},{3,1},{3,3},
		{4,-4},{4,-2},{4,0},{4,2},{4,4},
		{5,-5},{5,-3},{5,-1},{5,1},{5,3},{5,5},
		{6,-6},{6,-4},{6,-2},{6,0},{6,2},{6,4},{6,6},
		{7,-7},{7,-5},{7,-3},{7,-1},{7,1},{7,3},{7,5},{7,7}
};


// zernke径向公式CUDA计算
__device__ float zernike_radial_CUDA(float r, float n, float m)
{
	float radial;
	if (n == m)
	{
		radial = pow(r, n);
	}
	else if (((n - m) - 2) < 0.000001)
	{
		radial = n * zernike_radial_CUDA(r, n, n) - (n - 1) * zernike_radial_CUDA(r, n - 2, n - 2);
	}
	else
	{
		float H3 = (-4 * ((m + 4) - 2) * ((m + 4) - 3)) / ((n + (m + 4) - 2) * (n - (m + 4) + 4));
		float H2 = (H3 * (n + (m + 4)) * (n - (m + 4) + 2)) / (4 * ((m + 4) - 1)) + ((m + 4) - 2);
		float H1 = ((m + 4) * ((m + 4) - 1) / 2) - (m + 4) * H2 + (H3 * (n + (m + 4) + 2) * (n - (m + 4))) / (8);
		radial = H1 * zernike_radial_CUDA(r, n, m + 4) + (H2 + H3 / pow(r, 2)) * zernike_radial_CUDA(r, n, m + 2);
	}

	return radial;
}

// zernike多项式
__device__ float zernike_CUDA(float r, float t, float n, float m)
{
	float zern;
	if (m < 0)
	{
		zern = -zernike_radial_CUDA(r, n, -m) * sin(m * t);
	}
	else
	{
		zern = zernike_radial_CUDA(r, n, m) * cos(m * t);
	}
	return zern;
}


/*核函数（设备运行函数）*/
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
	{
		C[i] = A[i] + B[i] + 10;
	}
}


__global__ void zernikeMat_CUDA(float* zernikeMatrix, int n, int col)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	
	if ((i < col) && (j < col) )
	{
		float center_x = (col - 1) / 2;
		float center_y = (col - 1) / 2;
		float radius_x = (col - center_x);
		float radius_y = (col - center_y);
		
		if (sqrt(pow(((float)i - center_y), 2) / pow(radius_y, 2) + pow(((float)j - center_x), 2) / pow(radius_x, 2)) > 1)
		{
			zernikeMatrix[k * col * col + i * col + j] = 0;
		}
		else
		{
			int size = (col - 1) / 2;
			float delta = 1.0 / (float)size;

			float rx = 0 - delta * (float)size + (i - 1) * delta;
			float ry = 0 + delta * (float)size - (j - 1) * delta;

			

			float r = sqrt(rx * rx + ry * ry);
			float theta = atan2(rx, ry);
			float result = zernike_CUDA(r, theta, indice[k][0], indice[k][1]);
			if (isnan(result))
			{
				zernikeMatrix[k * col * col + i * col + j] = 0;
			}
			else
			{
				zernikeMatrix[k * col * col + i * col + j] = result;
			}
			
		}
		
	}
}

extern "C" void zernike_fit_cuda(int nZernike, int sizeZernike, float* h_zernikMatrix)
{
	/*cout << "nZernike: " << nZernike << endl;
	cout << "sizeZernike: " << sizeZernike << endl;
	cout << "ceil(sizeZernike / 16): " << ceil(sizeZernike / 16) << endl;*/
	

	int numElements = nZernike * sizeZernike * sizeZernike;//一维矩阵大小
	size_t size = numElements * sizeof(float);	
	float* d_zernikMatrix = NULL;
	dim3 threadsPerBlock(16, 16, 1);

	dim3 blocksPerGrid(ceil(sizeZernike / 16)+1, ceil(sizeZernike / 16)+1, nZernike);
	cudaMalloc((void**)&d_zernikMatrix, size);
	zernikeMat_CUDA << < blocksPerGrid, threadsPerBlock >> > (d_zernikMatrix, nZernike, sizeZernike);
	cudaMemcpy(h_zernikMatrix, d_zernikMatrix, size, cudaMemcpyDeviceToHost);
	cudaFree(d_zernikMatrix);

}

/*主机函数*/
extern "C" void test1(int num)
{
	/*生成主机数组内存 h_A, h_B, h_C*/
	int numElements = num;
	size_t size = numElements * sizeof(float);
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	float* h_C = (float*)malloc(size);
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (float)RAND_MAX;
		h_B[i] = rand() / (float)RAND_MAX;
	}

	/*生成设备内存 d_A，d_B，d_C */
	float* d_A = NULL;
	cudaMalloc((void**)&d_A, size);
	float* d_B = NULL;
	cudaMalloc((void**)&d_B, size);
	float* d_C = NULL;
	cudaMalloc((void**)&d_C, size);

	/*将主机内存数据复制到设备内存 h_A--d_A，h_B--d_B */
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	/*设置设备的线程数，并调用核函数*/
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd << < blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
	cudaGetLastError();

	/*将设备内存数据复制到主机内存 d_C--h_C */
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	/* 释放设备内存 d_A d_B d_C */
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	/* 结果验证 */
	std::cout << "A[0]: " << (float)h_A[0] << " B[0]: " << (float)h_B[0] << " 结果C[0] = A[i] + B[i] + 10: " << (float)h_C[0] << std::endl;
	std::cout << "A[1]: " << (float)h_A[1] << " B[1]: " << (float)h_B[1] << " 结果C[1] = A[i] + B[i] + 10: " << (float)h_C[1] << std::endl;
	std::cout << "A[2]: " << (float)h_A[2] << " B[2]: " << (float)h_B[2] << " 结果C[2] = A[i] + B[i] + 10: " << (float)h_C[2] << std::endl;

	/* 释放主机内存 h_A h_B h_C */
	free(h_A);
	free(h_B);
	free(h_C);

}