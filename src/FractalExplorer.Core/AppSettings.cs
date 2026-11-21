namespace FractalExplorer.Core;

public enum AppTheme { System, Light, Dark }

public sealed record AppSettings
{
    private static string DefaultFolder() =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyPictures), "Fractal Explorer");

    public string SaveFolder { get; init; } = DefaultFolder();
    public int SaveWidth { get; init; } = 3840;
    public int SaveHeight { get; init; } = 2160;
    public int SaveSmoothingIndex { get; init; } = 1;
    public AppTheme Theme { get; init; } = AppTheme.System;

    public static AppSettings Default => new();

    public AppSettings Normalized()
    {
        bool sizeOk = ExportMath.IsValidSize(SaveWidth, SaveHeight);
        return this with
        {
            // A relative folder would resolve against the process working directory: use the default.
            SaveFolder = !string.IsNullOrWhiteSpace(SaveFolder) && Path.IsPathFullyQualified(SaveFolder)
                ? SaveFolder : DefaultFolder(),
            SaveWidth = sizeOk ? SaveWidth : 3840,
            SaveHeight = sizeOk ? SaveHeight : 2160,
            SaveSmoothingIndex = SaveSmoothingIndex is >= 0 and <= 2 ? SaveSmoothingIndex : 1,
            Theme = Enum.IsDefined(Theme) ? Theme : AppTheme.System,
        };
    }
}

public sealed class SettingsStore
{
    private static readonly System.Text.Json.JsonSerializerOptions Options = new(System.Text.Json.JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() },
    };

    private readonly string _path;
    public SettingsStore(string filePath) => _path = filePath;

    public static string DefaultPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "FractalExplorer", "settings.json");

    public AppSettings Load()
    {
        try
        {
            if (!File.Exists(_path)) return AppSettings.Default;
            var s = System.Text.Json.JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(_path), Options);
            return (s ?? AppSettings.Default).Normalized();
        }
        catch { return AppSettings.Default; }
    }

    public void Save(AppSettings s)
    {
        try
        {
            var dir = Path.GetDirectoryName(Path.GetFullPath(_path));
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
            var tmp = _path + ".tmp";
            File.WriteAllText(tmp, System.Text.Json.JsonSerializer.Serialize(s, Options));
            File.Move(tmp, _path, overwrite: true);
        }
        catch { }
    }
}
