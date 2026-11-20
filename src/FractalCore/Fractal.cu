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

// Iterates z -> z^2 + c from z0 and returns the smooth-coloured palette value.
__device__ float3 escapeColor(double x, double y, double cx, double cy, int maxIterations,
                              const float* palT, const float* palRGB, int palSize) {
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

    double t = nu < maxIterations ? nu / maxIterations : 0.0;
    return paletteColor(palT, palRGB, palSize, (float)t);
}

// Renders one pixel, averaging an ss x ss grid of samples inside it (ss = 1: single sample).
// julia == false: Mandelbrot (z0 = 0, c = point); julia == true: z0 = point, c = (cx, cy).
__device__ void renderPixel(unsigned char* image, int i, int j, int width, int height,
                            int maxIterations, double xMin, double xMax, double yMin,
                            double yMax, bool julia, double cx, double cy, int ss,
                            const float* palT, const float* palRGB, int palSize) {
    double dx = (xMax - xMin) / width;
    double dy = (yMax - yMin) / height;
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int sy = 0; sy < ss; ++sy) {
        for (int sx = 0; sx < ss; ++sx) {
            // ss == 1 keeps the original top-left pixel sampling
            double ox = ss > 1 ? (sx + 0.5) / ss : 0.0;
            double oy = ss > 1 ? (sy + 0.5) / ss : 0.0;
            double px = xMin + dx * (i + ox);
            double py = yMin + dy * (j + oy);
            float3 c = julia
                ? escapeColor(px, py, cx, cy, maxIterations, palT, palRGB, palSize)
                : escapeColor(0.0, 0.0, px, py, maxIterations, palT, palRGB, palSize);
            acc.x += c.x;
            acc.y += c.y;
            acc.z += c.z;
        }
    }

    float inv = 1.0f / (ss * ss);
    int idx = (j * width + i) * 3;
    image[idx + 0] = (unsigned char)(fminf(acc.x * inv, 1.0f) * 255.0f);
    image[idx + 1] = (unsigned char)(fminf(acc.y * inv, 1.0f) * 255.0f);
    image[idx + 2] = (unsigned char)(fminf(acc.z * inv, 1.0f) * 255.0f);
}

__global__ void mandelbrotKernel(
    unsigned char* image, int width, int height, int maxIterations,
    double xMin, double xMax, double yMin, double yMax,
    const float* palT, const float* palRGB, int palSize, int ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    renderPixel(image, i, j, width, height, maxIterations, xMin, xMax, yMin, yMax,
                false, 0.0, 0.0, ss, palT, palRGB, palSize);
}

__global__ void juliaKernel(
    unsigned char* image, int width, int height, int maxIterations,
    double xMin, double xMax, double yMin, double yMax, double cx, double cy,
    const float* palT, const float* palRGB, int palSize, int ss) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;
    renderPixel(image, i, j, width, height, maxIterations, xMin, xMax, yMin, yMax,
                true, cx, cy, ss, palT, palRGB, palSize);
}
