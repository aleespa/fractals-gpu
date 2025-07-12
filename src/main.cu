#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

extern __global__ void mandelbrotKernel(unsigned char* image, int width, int height, int maxIterations,
                                        double xMin, double xMax, double yMin, double yMax);

int main() {
    int width = 1080;
    int height = 1080;
    int maxIterations = 500;
    double xMin = -2.0, xMax = 1.0;
    double yMin = -1.5, yMax = 1.5;

    size_t imageSize = width * height * 3;
    unsigned char* d_image;
    cudaMalloc(&d_image, imageSize);
    cudaMemset(d_image, 0, imageSize);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Generating fractal on GPU..." << std::endl;
    mandelbrotKernel<<<gridSize, blockSize>>>(d_image, width, height, maxIterations, xMin, xMax, yMin, yMax);
    cudaDeviceSynchronize();

    std::vector<unsigned char> image(imageSize);
    cudaMemcpy(image.data(), d_image, imageSize, cudaMemcpyDeviceToHost);

    if (stbi_write_png("mandelbrot_cuda.png", width, height, 3, image.data(), width * 3)) {
        std::cout << "Fractal saved as mandelbrot_cuda.png" << std::endl;
    } else {
        std::cout << "Failed to save image!" << std::endl;
    }

    cudaFree(d_image);
    return 0;
}
