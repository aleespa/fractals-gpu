using Xunit;
using FractalExplorer.Core;
using System.Text.RegularExpressions;

namespace FractalExplorer.Tests;

public class SavePlannerTests
{
    private static string Tmp() { var d = Path.Combine(Path.GetTempPath(), "fe_" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(d); return d; }

    [Fact]
    public void UniquePath_AppendsCounter()
    {
        var d = Tmp();
        try
        {
            Assert.Equal(Path.Combine(d, "a.png"), SavePlanner.UniquePath(d, "a.png"));
            File.WriteAllText(Path.Combine(d, "a.png"), "");
            File.WriteAllText(Path.Combine(d, "a (2).png"), "");
            Assert.Equal(Path.Combine(d, "a (3).png"), SavePlanner.UniquePath(d, "a.png"));
        }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void QuickSavePath_CreatesFolder()
    {
        var d = Tmp();
        try
        {
            var folder = Path.Combine(d, "new");
            var t = new DateTime(2026, 11, 20, 15, 30, 5);
            var p = SavePlanner.QuickSavePath(new AppSettings { SaveFolder = folder, SaveWidth = 100, SaveHeight = 50 }, FractalType.Julia, t);
            Assert.True(Directory.Exists(folder));
            Assert.Equal(ExportMath.DefaultFileName(FractalType.Julia, 100, 50, t), Path.GetFileName(p));
        }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void QuickSavePath_UncreatableFolder_Throws()
    {
        var d = Tmp();
        try
        {
            var f = Path.Combine(d, "file.txt");
            File.WriteAllText(f, "x");
            Assert.ThrowsAny<IOException>(() =>
                SavePlanner.QuickSavePath(new AppSettings { SaveFolder = Path.Combine(f, "sub") }, FractalType.Mandelbrot, DateTime.Now));
        }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void BuildRequest_WidensZoomForWiderExport()
    {
        var v = ViewState.CreateDefault(); v.Zoom = 0.1;
        var r = SavePlanner.BuildRequest(v, (new float[] { 0, 1 }, new float[6]), new SaveSpec(2000, 1000, 4, 777, "x.png"), 1.0);
        Assert.Equal(0.2, r.Zoom, 12);
        Assert.Equal(4, r.Supersample);
        Assert.Equal(777, r.MaxIterations);
        Assert.Equal(2000, r.Width);
    }
}
