// src/Fractal.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

__device__ float3 paletteColor(const float* palT, const float* palRGB, int palSize, float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    int i = 0;
    for (int k = 0; k < palSize - 1; ++k) {
        if (palT[k] <= t) i = k;
    }
    int j = min(i + 1, palSize - 1);
    float span = palT[j] - palT[i];
    float frac = (span > 0.0f) ? (t - palT[i]) / span : 0.0f;

    float3 a = {palRGB[i * 3 + 0], palRGB[i * 3 + 1], palRGB[i * 3 + 2]};
    float3 b = {palRGB[j * 3 + 0], palRGB[j * 3 + 1], palRGB[j * 3 + 2]};
    return {
        a.x + frac * (b.x - a.x),
        a.y + frac * (b.y - a.y),
        a.z + frac * (b.z - a.z)
    };
}

__global__ void mandelbrotKernel(
    unsigned char* image, int width, int height, int maxIterations,
    double xMin, double xMax, double yMin, double yMax,
    const float* palT, const float* palRGB, int palSize) {
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

    float3 color = paletteColor(palT, palRGB, palSize, (float)t);
    unsigned char r = (unsigned char)(color.x * 255.0f);
    unsigned char g = (unsigned char)(color.y * 255.0f);
    unsigned char b = (unsigned char)(color.z * 255.0f);

    int idx = (j * width + i) * 3;
    image[idx + 0] = r;
    image[idx + 1] = g;
    image[idx + 2] = b;
}

__global__ void juliaKernel(
    unsigned char* image, int width, int height, int maxIterations,
    double xMin, double xMax, double yMin, double yMax, double cx, double cy,
    const float* palT, const float* palRGB, int palSize) {
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

    float3 color = paletteColor(palT, palRGB, palSize, (float)t);
    unsigned char r = (unsigned char)(color.x * 255.0f);
    unsigned char g = (unsigned char)(color.y * 255.0f);
    unsigned char b = (unsigned char)(color.z * 255.0f);

    int idx = (j * width + i) * 3;
    image[idx + 0] = r;
    image[idx + 1] = g;
    image[idx + 2] = b;
}
