#pragma once
#include <stdint.h>
#ifdef FRACTALCORE_EXPORTS
#define FC_API __declspec(dllexport)
#else
#define FC_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif

enum { FC_OK = 0, FC_CANCELLED = 1, FC_ERR_INVALID = 2, FC_ERR_CUDA = 3, FC_ERR_IO = 4 };
enum { FC_FORMAT_RGB8 = 0, FC_FORMAT_BGRA8 = 1 };   /* BGRA8: alpha = 255 */

typedef struct FcParams {
    int32_t type;            /* 0 Mandelbrot, 1 Julia */
    int32_t width, height;   /* 1..16384 */
    int32_t maxIterations;   /* >= 1 */
    double  xCenter, yCenter;
    double  zoom;            /* half-width in the complex plane; half-height = zoom / (w/h) */
    double  cx, cy;          /* Julia constant */
    int32_t ss;              /* clamped to 1..8 */
    int32_t paletteSize;     /* 0 => built-in 5-stop default palette */
    const float* palT;       /* paletteSize entries, sorted ascending */
    const float* palRGB;     /* paletteSize*3 entries, 0..1 */
} FcParams;

/* out: width*height*3 (RGB8) or *4 (BGRA8) bytes, top row first.
   cancelFlag may be NULL; non-zero => stop after the current band, return FC_CANCELLED.
   Calls are serialized internally (one GPU job at a time). */
FC_API int32_t fc_render(const FcParams* p, uint8_t* out, int32_t format,
                         const volatile int32_t* cancelFlag);
/* RGB8 buffer -> PNG via stb_image_write. pathUtf8 is UTF-8. */
FC_API int32_t fc_write_png(const char* pathUtf8, int32_t width, int32_t height,
                            const uint8_t* rgb);
/* Message for the last non-OK result on this thread ("" if none). */
FC_API const char* fc_last_error(void);

#ifdef __cplusplus
}
#endif
