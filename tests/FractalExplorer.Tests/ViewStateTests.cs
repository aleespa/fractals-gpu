using System.Globalization;
using FractalExplorer.Core;
using Xunit;

namespace FractalExplorer.Tests;

public class ViewStateTests
{
    [Fact]
    public void Pan_MatchesAppJs()
    {
        var v = ViewState.CreateDefault();
        v.Pan(100, 50, 1000, 500);
        Assert.Equal(-1.4002 - 0.02, v.X, 1e-12);
        Assert.Equal(-0.01, v.Y, 1e-12);
    }

    [Fact]
    public void ZoomAt_CenterNotch_ZoomsIn()
    {
        var v = ViewState.CreateDefault();
        double f = v.ZoomAt(120, 500, 250, 1000, 500);
        Assert.Equal(0.10 * 0.85, v.Zoom, 1e-12);
        Assert.Equal(-1.4002, v.X, 1e-12);
        Assert.Equal(0, v.Y, 1e-12);
        Assert.Equal(1 / 0.85, f, 1e-12);
    }

    static (double, double) PointAt(ViewState v, double px, double py, double w, double h)
    {
        double aspect = w / h;
        return (v.X + ((px - w / 2) / w) * 2 * v.Zoom,
                v.Y + ((py - h / 2) / h) * (2 * v.Zoom / aspect));
    }

    [Fact]
    public void ZoomAt_KeepsPointUnderCursorFixed()
    {
        var v = ViewState.CreateDefault();
        var before = PointAt(v, 800, 100, 1000, 500);
        v.ZoomAt(-240, 800, 100, 1000, 500);
        var after = PointAt(v, 800, 100, 1000, 500);
        Assert.Equal(before.Item1, after.Item1, 1e-12);
        Assert.Equal(before.Item2, after.Item2, 1e-12);
    }

    [Fact]
    public void Reset_JuliaUsesJuliaView()
    {
        var v = new ViewState { Type = FractalType.Julia, X = 3, Y = 4, Zoom = 9, MaxIterations = 5, Cx = 1, Cy = 2 };
        v.Reset();
        Assert.Equal(0, v.X); Assert.Equal(0, v.Y); Assert.Equal(1.5, v.Zoom);
        Assert.Equal(400, v.MaxIterations); Assert.Equal(-0.4, v.Cx); Assert.Equal(0.6, v.Cy);
        Assert.Equal(FractalType.Julia, v.Type);
    }

    [Fact]
    public void ZoomText_Formats()
    {
        var v = ViewState.CreateDefault();
        Assert.Equal("10x", v.ZoomText(new CultureInfo("en-US")));
        v.Zoom = 0.0001;
        Assert.Equal("10,000x", v.ZoomText(new CultureInfo("en-US")));
    }

    [Fact]
    public void PreviewTransform_ZoomMatchesAppJs()
    {
        var t = new PreviewTransform();
        t.ApplyPan(10, 0);
        t.ApplyZoom(100, 0, 2);
        Assert.Equal(-80, t.TranslateX, 1e-12);
        Assert.Equal(2, t.Scale, 1e-12);
    }
}
