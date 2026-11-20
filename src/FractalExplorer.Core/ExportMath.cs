using System.Globalization;

namespace FractalExplorer.Core;

public static class ExportMath
{
    public const int MinSize = 16, MaxSize = 16384;

    public static bool IsValidSize(int w, int h) =>
        w >= MinSize && w <= MaxSize && h >= MinSize && h <= MaxSize;

    public static double AdjustZoomForAspect(double zoom, int w, int h, double viewAspect)
    {
        double exportAspect = (double)w / h;
        return exportAspect > viewAspect ? zoom * exportAspect / viewAspect : zoom;
    }

    public static string DefaultFileName(FractalType type, int w, int h, DateTime local) =>
        string.Create(CultureInfo.InvariantCulture,
            $"{type.ToString().ToLowerInvariant()}_{w}x{h}_{local:yyyyMMdd_HHmmss}.png");

    public static int SupersampleForIndex(int index) => index switch { 1 => 2, 2 => 4, _ => 1 };
}
