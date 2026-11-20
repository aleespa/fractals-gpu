using System.Runtime.InteropServices;
using System.Text;

namespace FractalExplorer.Core;

public enum PixelFormat { Rgb8 = 0, Bgra8 = 1 }

public sealed record RenderRequest(FractalType Type, int Width, int Height, int MaxIterations,
    double X, double Y, double Zoom, double Cx, double Cy, int Supersample,
    float[] PaletteT, float[] PaletteRgb);

public sealed class FractalRenderException(string message) : Exception(message);

public static unsafe partial class NativeRenderer
{
    private const int FcOk = 0, FcCancelled = 1;

    [StructLayout(LayoutKind.Sequential)]
    internal struct FcParams
    {
        public int Type, Width, Height, MaxIterations;
        public double XCenter, YCenter, Zoom, Cx, Cy;
        public int Ss, PaletteSize;
        public float* PalT;
        public float* PalRgb;
    }

    [LibraryImport("FractalCore", EntryPoint = "fc_render")]
    private static partial int FcRender(FcParams* p, byte* output, int format, int* cancelFlag);

    [LibraryImport("FractalCore", EntryPoint = "fc_write_png")]
    private static partial int FcWritePng(byte* pathUtf8, int width, int height, byte* rgb);

    [LibraryImport("FractalCore", EntryPoint = "fc_last_error")]
    private static partial byte* FcLastError();

    public static int NativeParamsSize => sizeof(FcParams);

    public static int BytesPerPixel(PixelFormat f) => f == PixelFormat.Bgra8 ? 4 : 3;

    private static string LastError(string fallback)
    {
        var s = Marshal.PtrToStringUTF8((IntPtr)FcLastError());
        return string.IsNullOrEmpty(s) ? fallback : s;
    }

    private static FractalRenderException NotFound(DllNotFoundException e) =>
        new($"FractalCore.dll not found (build the native projects first): {e.Message}");

    public static bool Render(RenderRequest req, PixelFormat format, Span<byte> output, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(req);
        long need = (long)Math.Max(req.Width, 0) * Math.Max(req.Height, 0) * BytesPerPixel(format);
        if (output.Length < need)
            throw new ArgumentException($"Output buffer too small: need {need} bytes, got {output.Length}.", nameof(output));
        int n = Math.Min(req.PaletteT.Length, req.PaletteRgb.Length / 3);

        int* flag = (int*)NativeMemory.AllocZeroed(sizeof(int));
        try
        {
            using var reg = ct.Register(static p => Volatile.Write(ref *(int*)(nint)p!, 1), (nint)flag);
            fixed (float* pt = req.PaletteT)
            fixed (float* prgb = req.PaletteRgb)
            fixed (byte* o = output)
            {
                var p = new FcParams
                {
                    Type = (int)req.Type, Width = req.Width, Height = req.Height, MaxIterations = req.MaxIterations,
                    XCenter = req.X, YCenter = req.Y, Zoom = req.Zoom, Cx = req.Cx, Cy = req.Cy,
                    Ss = req.Supersample, PaletteSize = n,
                    PalT = n > 0 ? pt : null, PalRgb = n > 0 ? prgb : null,
                };
                int rc;
                try { rc = FcRender(&p, o, (int)format, flag); }
                catch (DllNotFoundException e) { throw NotFound(e); }
                if (rc == FcOk) return true;
                if (rc == FcCancelled) return false;
                throw new FractalRenderException(LastError($"fc_render failed with code {rc}"));
            }
        }
        finally
        {
            NativeMemory.Free(flag);
        }
    }

    public static void WritePng(string path, int width, int height, ReadOnlySpan<byte> rgb)
    {
        long need = (long)width * height * 3;
        if (width <= 0 || height <= 0 || rgb.Length < need)
            throw new ArgumentException("Invalid dimensions or RGB buffer too small.", nameof(rgb));
        byte[] p = Encoding.UTF8.GetBytes(path + "\0");
        fixed (byte* pp = p)
        fixed (byte* d = rgb)
        {
            int rc;
            try { rc = FcWritePng(pp, width, height, d); }
            catch (DllNotFoundException e) { throw NotFound(e); }
            if (rc != FcOk) throw new FractalRenderException(LastError($"fc_write_png failed with code {rc}"));
        }
    }
}
