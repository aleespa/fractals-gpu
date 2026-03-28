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
        y = 2.0 * x * y + y0;
        x = xtemp;
        ++iter;
    }

    double nu = iter;
    if (iter < maxIterations) {
        double mag = sqrt(x * x + y * y);
        nu = iter + 1 - log(log(mag)) / log(2.0);
	}

    double t = nu < maxIterations ? (double)nu / maxIterations : 0.0;

    unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
    unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
    unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

    int idx = (j * width + i) * 3;
    image[idx + 0] = r;
    image[idx + 1] = g;
    image[idx + 2] = b;
}

__global__ void juliaKernel(
    unsigned char* image, int width, int height, int maxIterations,
    double xMin, double xMax, double yMin, double yMax, double cx, double cy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    double x = xMin + (xMax - xMin) * i / width;
    double y = yMin + (yMax - yMin) * j / height;

    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < maxIterations) {
        double xtemp = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xtemp;
        ++iter;
    }

    double nu = iter;
    if (iter < maxIterations) {
        double mag = sqrt(x * x + y * y);
        nu = iter + 1 - log(log(mag)) / log(2.0);
    }

    double t = nu < maxIterations ? (double)nu / maxIterations : 0.0;

    unsigned char r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
    unsigned char g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
    unsigned char b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

    int idx = (j * width + i) * 3;
    image[idx + 0] = r;
    image[idx + 1] = g;
    image[idx + 2] = b;
}
