#define _CRT_SECURE_NO_WARNINGS
#include "fractal_core.h"
#include <string>

#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void fc_set_error(const std::string &msg);

extern "C" FC_API int32_t fc_write_png(const char *pathUtf8, int32_t width,
                                       int32_t height, const uint8_t *rgb) {
  if (!pathUtf8 || !rgb || width < 1 || height < 1) {
    fc_set_error("Invalid arguments to fc_write_png");
    return FC_ERR_INVALID;
  }
  if (!stbi_write_png(pathUtf8, width, height, 3, rgb, width * 3)) {
    fc_set_error(std::string("Failed to write ") + pathUtf8);
    return FC_ERR_IO;
  }
  return FC_OK;
}
