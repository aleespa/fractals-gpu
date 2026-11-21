using System.Diagnostics;
using FractalExplorer.Core;
using Xunit;

namespace FractalExplorer.Tests;

public class NativeRendererTests
{
    private static RenderRequest Req(FractalType t, int w, int h, int iter, double x, double y, double zoom,
        double cx, double cy, int ss, bool defaultPalette)
    {
        if (defaultPalette)
        {
            var (pt, prgb) = Palette.CreateDefault().ToNative();
            return new RenderRequest(t, w, h, iter, x, y, zoom, cx, cy, ss, pt, prgb);
        }
        return new RenderRequest(t, w, h, iter, x, y, zoom, cx, cy, ss, Array.Empty<float>(), Array.Empty<float>());
    }

    private static string TempDir()
    {
        var d = Path.Combine(Path.GetTempPath(), "fc_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(d);
        return d;
    }

    [Fact]
    public void Render_InsideSet_IsBlack()
    {
        var buf = new byte[3];
        var r = NativeRenderer.Render(Req(FractalType.Mandelbrot, 1, 1, 200, -0.5, 0, 0.001, 0, 0, 1, true),
            PixelFormat.Rgb8, buf, CancellationToken.None);
        Assert.True(r);
        Assert.Equal(new byte[] { 0, 0, 0 }, buf);
    }

    [Fact]
    public void Render_Bgra_IsRgbSwizzled()
    {
        var req = Req(FractalType.Julia, 37, 23, 100, 0, 0, 1.5, 0.285, 0.01, 1, true);
        var rgb = new byte[37 * 23 * 3];
        var bgra = new byte[37 * 23 * 4];
        Assert.True(NativeRenderer.Render(req, PixelFormat.Rgb8, rgb, CancellationToken.None));
        Assert.True(NativeRenderer.Render(req, PixelFormat.Bgra8, bgra, CancellationToken.None));
        for (int i = 0; i < 37 * 23; i++)
        {
            Assert.Equal(rgb[i * 3 + 2], bgra[i * 4 + 0]);
            Assert.Equal(rgb[i * 3 + 1], bgra[i * 4 + 1]);
            Assert.Equal(rgb[i * 3 + 0], bgra[i * 4 + 2]);
            Assert.Equal(255, bgra[i * 4 + 3]);
        }
    }

    [Fact]
    public void Render_MatchesCli_OddSize()
    {
        var dir = TempDir();
        var cliPng = Path.Combine(dir, "cli.png");
        var dllPng = Path.Combine(dir, "dll.png");
        var psi = new ProcessStartInfo(Path.Combine(AppContext.BaseDirectory, "CppFractals.exe"))
        {
            WorkingDirectory = AppContext.BaseDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var a in new[] { "1", "1001", "333", "250", "0", "0", "1.2", "0.285", "0.01", cliPng, "--ss", "3" })
            psi.ArgumentList.Add(a);
        using (var p = Process.Start(psi)!)
        {
            p.StandardOutput.ReadToEnd();
            p.StandardError.ReadToEnd();
            p.WaitForExit();
            Assert.Equal(0, p.ExitCode);
        }

        var req = Req(FractalType.Julia, 1001, 333, 250, 0, 0, 1.2, 0.285, 0.01, 3, false);
        var rgb = new byte[1001 * 333 * 3];
        Assert.True(NativeRenderer.Render(req, PixelFormat.Rgb8, rgb, CancellationToken.None));
        NativeRenderer.WritePng(dllPng, 1001, 333, rgb);
        Assert.Equal(File.ReadAllBytes(cliPng), File.ReadAllBytes(dllPng));
        Directory.Delete(dir, true);
    }

    [Fact]
    public void Render_InvalidSize_Throws()
    {
        var req = Req(FractalType.Mandelbrot, 0, 10, 100, 0, 0, 1, 0, 0, 1, true);
        var ex = Assert.Throws<FractalRenderException>(() =>
            NativeRenderer.Render(req, PixelFormat.Rgb8, new byte[30], CancellationToken.None));
        Assert.Contains("width", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Render_PreCancelled_ReturnsCancelled()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var req = Req(FractalType.Mandelbrot, 256, 256, 100, -0.5, 0, 1.5, 0, 0, 1, true);
        Assert.False(NativeRenderer.Render(req, PixelFormat.Rgb8, new byte[256 * 256 * 3], cts.Token));
    }

    [Fact]
    public void Render_BufferTooSmall_Throws()
    {
        var req = Req(FractalType.Mandelbrot, 10, 10, 100, -0.5, 0, 1.5, 0, 0, 1, true);
        Assert.Throws<ArgumentException>(() =>
            NativeRenderer.Render(req, PixelFormat.Rgb8, new byte[10], CancellationToken.None));
    }

    [Fact]
    public void WritePng_NonAsciiPath()
    {
        var dir = TempDir();
        var sub = Path.Combine(dir, "Imágenes ñ");
        Directory.CreateDirectory(sub);
        var path = Path.Combine(sub, "out.png");
        NativeRenderer.WritePng(path, 2, 2, new byte[12]);
        var head = File.ReadAllBytes(path).Take(8).ToArray();
        Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, head);
        Directory.Delete(dir, true);
    }

    [Fact]
    public void Cli_WritesNonAsciiOutputPath()
    {
        var dir = TempDir();
        var path = Path.Combine(dir, "salida é ñ.png");
        var psi = new ProcessStartInfo(Path.Combine(AppContext.BaseDirectory, "CppFractals.exe"))
        {
            WorkingDirectory = dir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var a in new[] { "0", "16", "16", "50", "-0.5", "0", "1.5", "0", "0", path })
            psi.ArgumentList.Add(a);
        using (var p = Process.Start(psi)!)
        {
            p.StandardOutput.ReadToEnd();
            p.StandardError.ReadToEnd();
            p.WaitForExit();
            Assert.Equal(0, p.ExitCode);
        }
        Assert.True(File.Exists(path), "CLI did not create " + path);
        Assert.Single(Directory.GetFiles(dir));
        Directory.Delete(dir, true);
    }

    [Fact]
    public void RenderWithProgress_ReportsMonotonicToOne()
    {
        var req = Req(FractalType.Mandelbrot, 64, 100, 100, -0.5, 0, 1.5, 0, 0, 1, true);
        var vals = new List<double>();
        Assert.True(NativeRenderer.Render(req, PixelFormat.Rgb8, new byte[64 * 100 * 3], vals.Add, CancellationToken.None));
        Assert.Equal(4, vals.Count);
        for (int i = 1; i < vals.Count; i++) Assert.True(vals[i] >= vals[i - 1]);
        Assert.Equal(1.0, vals[^1]);
    }

    [Fact]
    public void RenderWithProgress_CancelFromCallback_ReturnsFalse()
    {
        var req = Req(FractalType.Mandelbrot, 64, 256, 100, -0.5, 0, 1.5, 0, 0, 1, true);
        using var cts = new CancellationTokenSource();
        int count = 0;
        bool ok = NativeRenderer.Render(req, PixelFormat.Rgb8, new byte[64 * 256 * 3], _ => { count++; cts.Cancel(); }, cts.Token);
        Assert.False(ok);
        Assert.True(count < 8);
    }

    [Fact]
    public void RenderWithProgress_ThrowingCallback_CancelsWithoutCrash()
    {
        var req = Req(FractalType.Mandelbrot, 64, 256, 100, -0.5, 0, 1.5, 0, 0, 1, true);
        bool ok = NativeRenderer.Render(req, PixelFormat.Rgb8, new byte[64 * 256 * 3],
            _ => throw new InvalidOperationException("boom"), CancellationToken.None);
        Assert.False(ok);
    }
}

public class FcParamsLayoutTests
{
    [Fact]
    public void FcParams_Size_Is80() => Assert.Equal(80, NativeRenderer.NativeParamsSize);
}
