// src/Fractal.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__global__ void mandelbrotKernel(
    unsigned char* image, int width, int height, int maxIterations,
    double xMin, double xMax, double yMin, double yMax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    double x0 = xMin + (xMax - xMin) * i / width;
    double y0 = yMin + (yMax - yMin) * j / height;

    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < maxIterations) {
        double xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        ++iter;
    }

    double t = iter < maxIterations ? (double)iter / maxIterations : 0.0;

    unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
    unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
    unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

    int idx = (j * width + i) * 3;
    image[idx + 0] = r;
    image[idx + 1] = g;
    image[idx + 2] = b;
}
