using System.Globalization;

namespace FractalExplorer.Core;

public static class CliCommand
{
    public static string Build(string exePath, ViewState v, Palette p, DateTime utcNow)
    {
        var inv = CultureInfo.InvariantCulture;
        string exe = exePath.Contains(' ') ? $"& \"{exePath}\"" : exePath;
        string stamp = utcNow.ToString("yyyyMMddHHmmss", inv) + (utcNow.Millisecond / 100).ToString(inv);
        var parts = new List<string>
        {
            exe, ((int)v.Type).ToString(inv), "5000", "5000", v.MaxIterations.ToString(inv),
            v.X.ToString("R", inv), v.Y.ToString("R", inv), v.Zoom.ToString("R", inv),
            v.Cx.ToString("R", inv), v.Cy.ToString("R", inv), $"output_{stamp}.png",
        };
        parts.AddRange(p.ToCliArgs());
        return string.Join(' ', parts);
    }
}
