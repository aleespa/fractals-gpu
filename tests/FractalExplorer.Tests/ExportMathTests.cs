using FractalExplorer.Core;
using Xunit;

namespace FractalExplorer.Tests;

public class ExportMathTests
{
    [Fact]
    public void AdjustZoom_WiderExportWidens()
    {
        Assert.Equal(0.1 * (3840 / 2160.0), ExportMath.AdjustZoomForAspect(0.1, 3840, 2160, 1.0), 1e-12);
        Assert.Equal(0.1, ExportMath.AdjustZoomForAspect(0.1, 1000, 1000, 2.0), 1e-12);
    }

    [Fact]
    public void IsValidSize_Bounds()
    {
        Assert.True(ExportMath.IsValidSize(16, 16384));
        Assert.False(ExportMath.IsValidSize(15, 100));
        Assert.False(ExportMath.IsValidSize(100, 16385));
    }

    [Fact]
    public void DefaultFileName_Format()
    {
        Assert.Equal("julia_7680x4320_20261120_153005.png",
            ExportMath.DefaultFileName(FractalType.Julia, 7680, 4320, new DateTime(2026, 11, 20, 15, 30, 5)));
    }
}
