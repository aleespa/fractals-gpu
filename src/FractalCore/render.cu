#include "fractal_core.h"
#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <vector>

extern __global__ void mandelbrotKernel(unsigned char *image, int width,
                                        int height, int maxIterations,
                                        double xMin, double xMax, double yMin,
                                        double yMax, const float *palT,
                                        const float *palRGB, int palSize,
                                        int ss);

extern __global__ void juliaKernel(unsigned char *image, int width, int height,
                                   int maxIterations, double xMin, double xMax,
                                   double yMin, double yMax, double cx,
                                   double cy, const float *palT,
                                   const float *palRGB, int palSize, int ss);

static thread_local std::string g_lastError;
static std::mutex g_renderMutex;

void fc_set_error(const std::string &msg) { g_lastError = msg; }

extern "C" FC_API const char *fc_last_error(void) { return g_lastError.c_str(); }

namespace {
struct DevBuf {
  void *p = nullptr;
  ~DevBuf() {
    if (p)
      cudaFree(p);
  }
};
} // namespace

extern "C" FC_API int32_t fc_render(const FcParams *p, uint8_t *out,
                                    int32_t format,
                                    const volatile int32_t *cancelFlag) {
  g_lastError.clear();
  if (!p || !out) {
    g_lastError = "Null parameter or output buffer";
    return FC_ERR_INVALID;
  }
  if (p->width < 1 || p->width > 16384 || p->height < 1 || p->height > 16384) {
    g_lastError = "Width and height must be within 1..16384";
    return FC_ERR_INVALID;
  }
  if (p->maxIterations < 1) {
    g_lastError = "maxIterations must be >= 1";
    return FC_ERR_INVALID;
  }
  if (p->type != 0 && p->type != 1) {
    g_lastError = "Invalid fractal type";
    return FC_ERR_INVALID;
  }
  if (format != FC_FORMAT_RGB8 && format != FC_FORMAT_BGRA8) {
    g_lastError = "Invalid output format";
    return FC_ERR_INVALID;
  }
  if (p->paletteSize < 0 ||
      (p->paletteSize > 0 && (!p->palT || !p->palRGB))) {
    g_lastError = "Invalid palette";
    return FC_ERR_INVALID;
  }

  std::lock_guard<std::mutex> lock(g_renderMutex);
  cudaGetLastError(); // clear stale error state from earlier calls

  const int width = p->width, height = p->height;
  const int maxIterations = p->maxIterations;
  const int type = p->type;
  int ss = p->ss < 1 ? 1 : (p->ss > 8 ? 8 : p->ss);

  std::vector<float> palT, palRGB;
  if (p->paletteSize == 0) {
    float defT[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float defRGB[] = {
        0.0f,  0.0f,  0.0f,  // #000000
        0.13f, 0.19f, 0.78f, // #2130C8
        0.55f, 0.0f,  0.13f, // #8C0021
        1.0f,  0.6f,  0.0f,  // #FF9900
        1.0f,  1.0f,  1.0f   // #FFFFFF
    };
    palT.assign(defT, defT + 5);
    palRGB.assign(defRGB, defRGB + 15);
  } else {
    palT.assign(p->palT, p->palT + p->paletteSize);
    palRGB.assign(p->palRGB, p->palRGB + p->paletteSize * 3);
  }
  const int paletteSize = (int)palT.size();

  auto cudaFail = [&](cudaError_t e) {
    g_lastError = cudaGetErrorString(e);
    return (int32_t)FC_ERR_CUDA;
  };

  DevBuf dPalT, dPalRGB, dImage;
  cudaError_t e;
  if ((e = cudaMalloc(&dPalT.p, paletteSize * sizeof(float))) != cudaSuccess)
    return cudaFail(e);
  if ((e = cudaMalloc(&dPalRGB.p, paletteSize * 3 * sizeof(float))) !=
      cudaSuccess)
    return cudaFail(e);
  if ((e = cudaMemcpy(dPalT.p, palT.data(), paletteSize * sizeof(float),
                      cudaMemcpyHostToDevice)) != cudaSuccess)
    return cudaFail(e);
  if ((e = cudaMemcpy(dPalRGB.p, palRGB.data(),
                      paletteSize * 3 * sizeof(float),
                      cudaMemcpyHostToDevice)) != cudaSuccess)
    return cudaFail(e);
  float *d_palT = (float *)dPalT.p;
  float *d_palRGB = (float *)dPalRGB.p;

  double aspectRatio = static_cast<double>(width) / height;
  double xMin = p->xCenter - p->zoom, xMax = p->xCenter + p->zoom;
  double yMin = p->yCenter - p->zoom / aspectRatio,
         yMax = p->yCenter + p->zoom / aspectRatio;

  size_t imageSize =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  if ((e = cudaMalloc(&dImage.p, imageSize)) != cudaSuccess) {
    g_lastError =
        "could not allocate " + std::to_string(imageSize) + " bytes for the image: " + cudaGetErrorString(e);
    return FC_ERR_CUDA;
  }
  unsigned char *d_image = (unsigned char *)dImage.p;
  cudaMemset(d_image, 0, imageSize);

  // Render in horizontal bands so no single kernel launch runs long enough
  // to trigger the Windows GPU watchdog (TDR, ~2 s) on large/supersampled
  // renders. Each band is rendered as its own "image" with a y-range slice.
  const int bandRows = 32;
  dim3 blockSize(16, 16);
  double dyPerRow = (yMax - yMin) / height;

  for (int row0 = 0; row0 < height; row0 += bandRows) {
    int rows = (height - row0 < bandRows) ? height - row0 : bandRows;
    double bandYMin = yMin + dyPerRow * row0;
    double bandYMax = bandYMin + dyPerRow * rows;
    unsigned char *bandImage = d_image + static_cast<size_t>(row0) * width * 3;
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);
    if (type == 0) {
      mandelbrotKernel<<<gridSize, blockSize>>>(
          bandImage, width, rows, maxIterations, xMin, xMax, bandYMin,
          bandYMax, d_palT, d_palRGB, paletteSize, ss);
    } else {
      juliaKernel<<<gridSize, blockSize>>>(
          bandImage, width, rows, maxIterations, xMin, xMax, bandYMin,
          bandYMax, p->cx, p->cy, d_palT, d_palRGB, paletteSize, ss);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess)
      err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
      return cudaFail(err);
    if (cancelFlag && *cancelFlag != 0) {
      g_lastError = "Cancelled";
      return FC_CANCELLED;
    }
  }

  if (format == FC_FORMAT_RGB8) {
    e = cudaMemcpy(out, d_image, imageSize, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
      return cudaFail(e);
  } else {
    std::vector<unsigned char> rgb(imageSize);
    e = cudaMemcpy(rgb.data(), d_image, imageSize, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess)
      return cudaFail(e);
    size_t n = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < n; ++i) {
      out[i * 4 + 0] = rgb[i * 3 + 2];
      out[i * 4 + 1] = rgb[i * 3 + 1];
      out[i * 4 + 2] = rgb[i * 3 + 0];
      out[i * 4 + 3] = 255;
    }
  }
  return FC_OK;
}
