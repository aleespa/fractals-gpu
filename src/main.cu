#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

extern __global__ void mandelbrotKernel(unsigned char* image, int width, int height, int maxIterations,
                                        double xMin, double xMax, double yMin, double yMax);

extern __global__ void juliaKernel(unsigned char* image, int width, int height, int maxIterations,
                                   double xMin, double xMax, double yMin, double yMax, double cx, double cy);

int main(int argc, char* argv[]) {
    int type = 0; // 0 for mandelbrot, 1 for julia
    int width = 3840;
    int height = 2160;
    int maxIterations = 400;
	double xCenter = -1.4002, yCenter = 0.0; double zoom = 0.10;
    double cx = -0.4, cy = 0.6;
    std::string outputFilename = "mandelbrot.png";

    if (argc >= 2) type = std::atoi(argv[1]);
    if (argc >= 3) width = std::atoi(argv[2]);
    if (argc >= 4) height = std::atoi(argv[3]);
    if (argc >= 5) maxIterations = std::atoi(argv[4]);
    if (argc >= 6) xCenter = std::atof(argv[5]);
    if (argc >= 7) yCenter = std::atof(argv[6]);
    if (argc >= 8) zoom = std::atof(argv[7]);
    if (argc >= 9) cx = std::atof(argv[8]);
    if (argc >= 10) cy = std::atof(argv[9]);
    if (argc >= 11) outputFilename = argv[10];


	double aspectRatio = static_cast<double>(width) / height;
    double xMin = xCenter - zoom, xMax = xCenter + zoom;
    double yMin = yCenter - zoom / aspectRatio, yMax = yCenter + zoom / aspectRatio;

    size_t imageSize = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    unsigned char* d_image;
    cudaMalloc(reinterpret_cast<void**>(&d_image), imageSize);
    cudaMemset(d_image, 0, imageSize);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Generating fractal on GPU..." << std::endl;
    if (type == 0) {
        mandelbrotKernel<<<gridSize, blockSize>>>(d_image, width, height, maxIterations, xMin, xMax, yMin, yMax);
    } else {
        juliaKernel<<<gridSize, blockSize>>>(d_image, width, height, maxIterations, xMin, xMax, yMin, yMax, cx, cy);
    }
    cudaDeviceSynchronize();

    std::vector<unsigned char> image(imageSize);
    cudaMemcpy(image.data(), d_image, imageSize, cudaMemcpyDeviceToHost);

    if (stbi_write_png(outputFilename.c_str(), width, height, 3, image.data(), width * 3)) {
        std::cout << "Fractal saved in " << outputFilename << std::endl;
    } else {
        std::cout << "Failed to save image!" << std::endl;
    }

    cudaFree(d_image);
    return 0;
}
