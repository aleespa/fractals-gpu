using Xunit;
using FractalExplorer.Core;

namespace FractalExplorer.Tests;

public class SettingsStoreTests
{
    private static string Tmp() { var d = Path.Combine(Path.GetTempPath(), "fe_" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(d); return d; }

    [Fact]
    public void Load_MissingFile_ReturnsDefaults()
    {
        var d = Tmp();
        try { Assert.Equal(AppSettings.Default, new SettingsStore(Path.Combine(d, "s.json")).Load()); }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void SaveThenLoad_RoundTrips()
    {
        var d = Tmp();
        try
        {
            var s = new AppSettings { SaveFolder = @"C:\x", SaveWidth = 1000, SaveHeight = 500, SaveSmoothingIndex = 2, Theme = AppTheme.Dark };
            var store = new SettingsStore(Path.Combine(d, "sub", "s.json"));
            store.Save(s);
            Assert.Equal(s, store.Load());
        }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void Load_CorruptJson_ReturnsDefaults()
    {
        var d = Tmp();
        try
        {
            var p = Path.Combine(d, "s.json");
            File.WriteAllText(p, "{not json");
            Assert.Equal(AppSettings.Default, new SettingsStore(p).Load());
        }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void Load_UnknownAndMissingFields_FillsDefaults()
    {
        var d = Tmp();
        try
        {
            var p = Path.Combine(d, "s.json");
            File.WriteAllText(p, "{\"saveWidth\":1234,\"bogus\":1}");
            var s = new SettingsStore(p).Load();
            Assert.Equal(1234, s.SaveWidth);
            Assert.Equal(2160, s.SaveHeight);
            Assert.Equal(AppTheme.System, s.Theme);
        }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void Load_InvalidValues_Normalized()
    {
        var d = Tmp();
        try
        {
            var p = Path.Combine(d, "s.json");
            File.WriteAllText(p, "{\"saveWidth\":5,\"saveSmoothingIndex\":9}");
            var s = new SettingsStore(p).Load();
            Assert.Equal(3840, s.SaveWidth);
            Assert.Equal(2160, s.SaveHeight);
            Assert.Equal(1, s.SaveSmoothingIndex);
        }
        finally { Directory.Delete(d, true); }
    }

    [Fact]
    public void Save_UnwritableDirectory_DoesNotThrow()
    {
        var d = Tmp();
        try
        {
            var f = Path.Combine(d, "file.txt");
            File.WriteAllText(f, "x");
            new SettingsStore(Path.Combine(f, "settings.json")).Save(AppSettings.Default);
        }
        finally { Directory.Delete(d, true); }
    }
}
