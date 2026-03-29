#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

extern __global__ void mandelbrotKernel(unsigned char *image, int width,
                                        int height, int maxIterations,
                                        double xMin, double xMax, double yMin,
                                        double yMax, const float *palT,
                                        const float *palRGB, int palSize);

extern __global__ void juliaKernel(unsigned char *image, int width, int height,
                                   int maxIterations, double xMin, double xMax,
                                   double yMin, double yMax, double cx,
                                   double cy, const float *palT,
                                   const float *palRGB, int palSize);

int main(int argc, char *argv[]) {
  int type = 0; // 0 for mandelbrot, 1 for julia
  int width = 3840;
  int height = 2160;
  int maxIterations = 400;
  double xCenter = -1.4002, yCenter = 0.0;
  double zoom = 0.10;
  double cx = -0.4, cy = 0.6;
  std::string outputFilename = "mandelbrot.png";

  if (argc >= 2)
    type = std::atoi(argv[1]);
  if (argc >= 3)
    width = std::atoi(argv[2]);
  if (argc >= 4)
    height = std::atoi(argv[3]);
  if (argc >= 5)
    maxIterations = std::atoi(argv[4]);
  if (argc >= 6)
    xCenter = std::atof(argv[5]);
  if (argc >= 7)
    yCenter = std::atof(argv[6]);
  if (argc >= 8)
    zoom = std::atof(argv[7]);
  if (argc >= 9)
    cx = std::atof(argv[8]);
  if (argc >= 10)
    cy = std::atof(argv[9]);
  if (argc >= 11)
    outputFilename = argv[10];

  // Parse palette stops from argv[11] onward: pairs of (t_float, hexcolor)
  std::vector<float> palT;
  std::vector<float> palRGB;

  for (int a = 11; a + 1 < argc; a += 2) {
    float t = std::stof(argv[a]);
    unsigned int hex = (unsigned int)std::stoul(argv[a + 1], nullptr, 16);
    palT.push_back(t);
    palRGB.push_back(((hex >> 16) & 0xFF) / 255.0f);
    palRGB.push_back(((hex >> 8) & 0xFF) / 255.0f);
    palRGB.push_back((hex & 0xFF) / 255.0f);
  }

  // Fallback default palette if none supplied
  if (palT.empty()) {
    float defT[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float defRGB[] = {
        0.0f,  0.0f,  0.0f,  // #000000
        0.13f, 0.19f, 0.78f, // #2130C8
        0.55f, 0.0f,  0.13f, // #8C0021
        1.0f,  0.6f,  0.0f,  // #FF9900
        1.0f,  1.0f,  1.0f   // #FFFFFF
    };
    for (int i = 0; i < 5; ++i) {
      palT.push_back(defT[i]);
      palRGB.push_back(defRGB[i * 3 + 0]);
      palRGB.push_back(defRGB[i * 3 + 1]);
      palRGB.push_back(defRGB[i * 3 + 2]);
    }
  }

  int paletteSize = (int)palT.size();

  // Print all parameters for debugging
  std::cout << "=== Render Parameters ===" << std::endl;
  std::cout << "Type:          " << (type == 0 ? "Mandelbrot" : "Julia")
            << std::endl;
  std::cout << "Resolution:    " << width << " x " << height << std::endl;
  std::cout << "Max Iters:     " << maxIterations << std::endl;
  std::cout << "Center:        (" << xCenter << ", " << yCenter << ")"
            << std::endl;
  std::cout << "Zoom:          " << zoom << std::endl;
  if (type == 1)
    std::cout << "Julia C:       (" << cx << ", " << cy << ")" << std::endl;
  std::cout << "Output:        " << outputFilename << std::endl;

  std::cout << "Palette stops: " << paletteSize << std::endl;
  for (int i = 0; i < paletteSize; ++i) {
    int r = (int)std::round(palRGB[i * 3 + 0] * 255);
    int g = (int)std::round(palRGB[i * 3 + 1] * 255);
    int b = (int)std::round(palRGB[i * 3 + 2] * 255);
    std::cout << "  [" << i << "] t=" << palT[i] << "  #" << std::hex
              << std::uppercase << std::setfill('0') << std::setw(2) << r
              << std::setw(2) << g << std::setw(2) << b << std::dec
              << std::endl;
  }
  std::cout << "=========================" << std::endl;

  float *d_palT;
  float *d_palRGB;
  cudaMalloc(reinterpret_cast<void **>(&d_palT), paletteSize * sizeof(float));
  cudaMalloc(reinterpret_cast<void **>(&d_palRGB),
             paletteSize * 3 * sizeof(float));
  cudaMemcpy(d_palT, palT.data(), paletteSize * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_palRGB, palRGB.data(), paletteSize * 3 * sizeof(float),
             cudaMemcpyHostToDevice);

  double aspectRatio = static_cast<double>(width) / height;
  double xMin = xCenter - zoom, xMax = xCenter + zoom;
  double yMin = yCenter - zoom / aspectRatio,
         yMax = yCenter + zoom / aspectRatio;

  size_t imageSize =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  unsigned char *d_image;
  cudaMalloc(reinterpret_cast<void **>(&d_image), imageSize);
  cudaMemset(d_image, 0, imageSize);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  std::cout << "Generating fractal on GPU..." << std::endl;
  if (type == 0) {
    mandelbrotKernel<<<gridSize, blockSize>>>(
        d_image, width, height, maxIterations, xMin, xMax, yMin, yMax, d_palT,
        d_palRGB, paletteSize);
  } else {
    juliaKernel<<<gridSize, blockSize>>>(d_image, width, height, maxIterations,
                                         xMin, xMax, yMin, yMax, cx, cy, d_palT,
                                         d_palRGB, paletteSize);
  }
  cudaDeviceSynchronize();

  std::vector<unsigned char> image(imageSize);
  cudaMemcpy(image.data(), d_image, imageSize, cudaMemcpyDeviceToHost);

  if (stbi_write_png(outputFilename.c_str(), width, height, 3, image.data(),
                     width * 3)) {
    std::cout << "Fractal saved in " << outputFilename << std::endl;
  } else {
    std::cout << "Failed to save image!" << std::endl;
  }

  cudaFree(d_image);
  cudaFree(d_palT);
  cudaFree(d_palRGB);
  return 0;
}
