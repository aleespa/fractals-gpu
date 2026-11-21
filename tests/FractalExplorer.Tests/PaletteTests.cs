using System.Globalization;
using FractalExplorer.Core;
using Xunit;

namespace FractalExplorer.Tests;

public class PaletteTests
{
    [Fact]
    public void Default_HasFiveSortedStops()
    {
        var p = Palette.CreateDefault();
        Assert.Equal(new[] { 0, .25, .5, .75, 1 }, p.Stops.Select(s => s.T).ToArray());
        Assert.Equal(new uint[] { 0x000000, 0x2130C8, 0x8C0021, 0xFF9900, 0xFFFFFF }, p.Stops.Select(s => s.Rgb).ToArray());
    }

    [Fact]
    public void Sample_Midway_Lerps()
        => Assert.Equal(0x111864u, Palette.CreateDefault().Sample(0.125));

    [Fact]
    public void AddAt_ClampsAndUsesSampledColor()
    {
        var p = Palette.CreateDefault();
        Assert.Equal(0.001, p.AddAt(0.0).T);
        uint expected = p.Sample(0.6);
        var s = p.AddAt(0.6);
        Assert.Equal(expected, s.Rgb);
        Assert.Equal(7, p.Stops.Count);
    }

    [Fact]
    public void Move_PinnedIgnored_UnpinnedClampedAndSorted()
    {
        var p = Palette.CreateDefault();
        p.Move(p.Stops[0], 0.5);
        Assert.Equal(0.0, p.Stops[0].T);
        var s = p.Stops[1];
        p.Move(s, 0.9);
        Assert.Equal(3, p.Stops.ToList().IndexOf(s));
        p.Move(s, 2.0);
        Assert.Equal(0.999, s.T);
    }

    [Fact]
    public void Remove_RejectsPinnedAndLastTwo()
    {
        var p = Palette.CreateDefault();
        Assert.False(p.Remove(p.Stops[0]));
        Assert.False(p.Remove(p.Stops[4]));
        for (int i = 0; i < 3; i++) Assert.True(p.Remove(p.Stops[1]));
        Assert.Equal(2, p.Stops.Count);
        var s = p.AddAt(0.5);
        Assert.True(p.Remove(s));
    }

    [Fact]
    public void ToNative_MatchesCliParsing()
    {
        var p = Palette.CreateDefault();
        p.AddAt(1.0 / 3);
        var (t, rgb) = p.ToNative();
        Assert.Equal(float.Parse("0.333333", CultureInfo.InvariantCulture), t[2]);
        Assert.Equal(new[] { 0x21 / 255f, 0x30 / 255f, 0xC8 / 255f }, rgb[3..6]);
    }

    [Fact]
    public void Mutations_RaiseChanged()
    {
        var p = Palette.CreateDefault();
        int n = 0;
        p.Changed += (_, _) => n++;
        var s = p.AddAt(0.6); Assert.Equal(1, n);
        p.Move(s, 0.7); Assert.Equal(2, n);
        p.SetColor(s, 0x123456); Assert.Equal(3, n);
        p.Remove(s); Assert.Equal(4, n);
        p.ResetToDefault(); Assert.Equal(5, n);
    }
}
