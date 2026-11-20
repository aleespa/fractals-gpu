using System.Globalization;
using FractalExplorer.Core;
using Xunit;

namespace FractalExplorer.Tests;

public class CliCommandTests
{
    const string Expected = @"C:\app\CppFractals.exe 0 5000 5000 400 -1.4002 0 0.1 -0.4 0.6 output_202611201800045.png 0.000000 000000 0.250000 2130C8 0.500000 8C0021 0.750000 FF9900 1.000000 FFFFFF";
    static readonly DateTime Utc = new(2026, 11, 20, 18, 0, 4, 567, DateTimeKind.Utc);

    [Fact]
    public void Build_DefaultView() =>
        Assert.Equal(Expected, CliCommand.Build(@"C:\app\CppFractals.exe", ViewState.CreateDefault(), Palette.CreateDefault(), Utc));

    [Fact]
    public void Build_QuotesPathWithSpaces() =>
        Assert.StartsWith(@"& ""C:\My Apps\CppFractals.exe"" 0 5000",
            CliCommand.Build(@"C:\My Apps\CppFractals.exe", ViewState.CreateDefault(), Palette.CreateDefault(), Utc));

    [Fact]
    public void Build_UsesInvariantCulture_UnderSpanishCulture()
    {
        var old = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("es-ES");
            Assert.Equal(Expected, CliCommand.Build(@"C:\app\CppFractals.exe", ViewState.CreateDefault(), Palette.CreateDefault(), Utc));
        }
        finally { CultureInfo.CurrentCulture = old; }
    }
}
