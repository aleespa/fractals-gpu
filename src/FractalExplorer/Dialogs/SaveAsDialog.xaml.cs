using FractalExplorer.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace FractalExplorer.Dialogs;

/// <summary>
/// "Save as" dialog: size, smoothing and iterations, then a PNG file picker.
/// <see cref="Result"/> is set only when the user confirmed a valid size and chose a file.
/// </summary>
public sealed partial class SaveAsDialog : ContentDialog
{
    private readonly FractalType _type;
    private readonly int _viewIterations;
    private readonly IntPtr _hwnd;

    /// <summary>The confirmed save, or null if the dialog was cancelled.</summary>
    public SaveSpec? Result { get; private set; }

    public SaveAsDialog(AppSettings settings, FractalType type, int viewIterations, IntPtr hwnd)
    {
        _type = type;
        _viewIterations = viewIterations;
        _hwnd = hwnd;
        InitializeComponent();

        SizeCombo.SelectedIndex = (settings.SaveWidth, settings.SaveHeight) switch
        {
            (3840, 2160) => 0,
            (7680, 4320) => 1,
            _ => 2,
        };
        WidthBox.Value = settings.SaveWidth;
        HeightBox.Value = settings.SaveHeight;
        CustomSizePanel.Visibility = SizeCombo.SelectedIndex == 2 ? Visibility.Visible : Visibility.Collapsed;
        SmoothCombo.SelectedIndex = settings.SaveSmoothingIndex;
        IterBox.PlaceholderText = $"Same as view ({viewIterations.ToString(System.Globalization.CultureInfo.InvariantCulture)})";
        IterBox.Value = double.NaN;
    }

    private void SizeCombo_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (CustomSizePanel is null) return;   // raised during InitializeComponent
        CustomSizePanel.Visibility = SizeCombo.SelectedIndex == 2 ? Visibility.Visible : Visibility.Collapsed;
        SizeError.Visibility = Visibility.Collapsed;
    }

    private void SizeBox_ValueChanged(NumberBox sender, NumberBoxValueChangedEventArgs args)
    {
        if (SizeError is null) return;
        SizeError.Visibility = Visibility.Collapsed;
    }

    private bool TryGetSize(out int w, out int h)
    {
        switch (SizeCombo.SelectedIndex)
        {
            case 0: w = 3840; h = 2160; return true;
            case 1: w = 7680; h = 4320; return true;
        }
        w = h = 0;
        double wv = WidthBox.Value, hv = HeightBox.Value;
        if (double.IsNaN(wv) || double.IsNaN(hv) || wv != Math.Floor(wv) || hv != Math.Floor(hv)
            || wv < ExportMath.MinSize || wv > ExportMath.MaxSize || hv < ExportMath.MinSize || hv > ExportMath.MaxSize)
            return false;
        w = (int)wv;
        h = (int)hv;
        return ExportMath.IsValidSize(w, h);
    }

    private int Iterations()
    {
        double v = IterBox.Value;
        if (double.IsNaN(v) || v <= 0) return _viewIterations;
        return (int)Math.Max(1, Math.Min(Math.Round(v), int.MaxValue));
    }

    private async void OnPrimaryButtonClick(ContentDialog sender, ContentDialogButtonClickEventArgs args)
    {
        PickerError.Visibility = Visibility.Collapsed;
        if (!TryGetSize(out int w, out int h))
        {
            SizeError.Visibility = Visibility.Visible;
            args.Cancel = true;
            return;
        }
        SizeError.Visibility = Visibility.Collapsed;

        int ss = ExportMath.SupersampleForIndex(SmoothCombo.SelectedIndex);
        int iter = Iterations();

        var deferral = args.GetDeferral();
        try
        {
            var picker = new Windows.Storage.Pickers.FileSavePicker();
            WinRT.Interop.InitializeWithWindow.Initialize(picker, _hwnd);
            picker.FileTypeChoices.Add("PNG image", new List<string> { ".png" });
            picker.SuggestedFileName = Path.GetFileNameWithoutExtension(ExportMath.DefaultFileName(_type, w, h, DateTime.Now));
            picker.SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary;
            var file = await picker.PickSaveFileAsync();
            if (file is null)
            {
                args.Cancel = true;   // picker cancelled: keep the dialog open
                return;
            }
            Result = new SaveSpec(w, h, ss, iter, file.Path);
        }
        catch (Exception ex)
        {
            PickerError.Text = ex.Message;
            PickerError.Visibility = Visibility.Visible;
            args.Cancel = true;
        }
        finally
        {
            deferral.Complete();
        }
    }
}
