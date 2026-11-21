using System.Reflection;
using FractalExplorer.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace FractalExplorer.Panes;

/// <summary>Settings page: save folder/size/smoothing, theme, about.</summary>
public sealed partial class SettingsView : UserControl
{
    private readonly MainWindow _owner;
    private bool _syncing;

    public SettingsView(MainWindow owner)
    {
        _owner = owner;
        _syncing = true;   // XAML initialisation raises selection/value events
        InitializeComponent();
        var v = Assembly.GetExecutingAssembly().GetName().Version;
        VersionText.Text = v is null ? "Fractal Explorer" : $"Fractal Explorer {v.Major}.{v.Minor}.{Math.Max(v.Build, 0)}";
        _syncing = false;
        SyncFromState();
    }

    /// <summary>Pushes the owner's settings into every control without triggering change handlers.</summary>
    public void SyncFromState()
    {
        bool was = _syncing;
        _syncing = true;
        try
        {
            var s = _owner.Settings;
            FolderText.Text = s.SaveFolder;
            SizeCombo.SelectedIndex = (s.SaveWidth, s.SaveHeight) switch
            {
                (3840, 2160) => 0,
                (7680, 4320) => 1,
                _ => 2,
            };
            WidthBox.Value = s.SaveWidth;
            HeightBox.Value = s.SaveHeight;
            CustomSizePanel.Visibility = SizeCombo.SelectedIndex == 2 ? Visibility.Visible : Visibility.Collapsed;
            SizeError.Visibility = Visibility.Collapsed;
            SmoothCombo.SelectedIndex = s.SaveSmoothingIndex;
            ThemeRadios.SelectedIndex = (int)s.Theme;
        }
        finally
        {
            _syncing = was;
        }
    }

    private async void ChangeFolder_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var picker = new Windows.Storage.Pickers.FolderPicker();
            picker.FileTypeFilter.Add("*");
            WinRT.Interop.InitializeWithWindow.Initialize(picker, WinRT.Interop.WindowNative.GetWindowHandle(_owner));
            var folder = await picker.PickSingleFolderAsync();
            if (folder is null) return;
            _owner.UpdateSettings(s => s with { SaveFolder = folder.Path });
            SyncFromState();
        }
        catch (Exception ex)
        {
            _owner.ShowError(ex.Message, "Could not choose folder");
        }
    }

    private void OpenFolder_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            string folder = _owner.Settings.SaveFolder;
            Directory.CreateDirectory(folder);
            System.Diagnostics.Process.Start("explorer.exe", $"\"{folder}\"");
        }
        catch (Exception ex)
        {
            _owner.ShowError(ex.Message, "Could not open folder");
        }
    }

    private void SizeCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_syncing) return;
        int i = SizeCombo.SelectedIndex;
        CustomSizePanel.Visibility = i == 2 ? Visibility.Visible : Visibility.Collapsed;
        SizeError.Visibility = Visibility.Collapsed;
        if (i == 0) ApplySize(3840, 2160);
        else if (i == 1) ApplySize(7680, 4320);
        else
        {
            WidthBox.Value = _owner.Settings.SaveWidth;
            HeightBox.Value = _owner.Settings.SaveHeight;
        }
    }

    private void SizeBox_ValueChanged(NumberBox sender, NumberBoxValueChangedEventArgs args)
    {
        if (_syncing) return;
        double w = WidthBox.Value, h = HeightBox.Value;
        bool ok = !double.IsNaN(w) && !double.IsNaN(h) && w == Math.Floor(w) && h == Math.Floor(h)
                  && w <= int.MaxValue && h <= int.MaxValue && ExportMath.IsValidSize((int)w, (int)h);
        SizeError.Visibility = ok ? Visibility.Collapsed : Visibility.Visible;
        if (ok) ApplySize((int)w, (int)h);
    }

    private void ApplySize(int w, int h) =>
        _owner.UpdateSettings(s => s with { SaveWidth = w, SaveHeight = h });

    private void SmoothCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_syncing || SmoothCombo.SelectedIndex < 0) return;
        int i = SmoothCombo.SelectedIndex;
        _owner.UpdateSettings(s => s with { SaveSmoothingIndex = i });
    }

    private void ThemeRadios_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_syncing || ThemeRadios.SelectedIndex < 0) return;
        var theme = (AppTheme)ThemeRadios.SelectedIndex;
        _owner.UpdateSettings(s => s with { Theme = theme });
    }
}
