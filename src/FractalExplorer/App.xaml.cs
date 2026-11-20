using Microsoft.UI.Xaml;

namespace FractalExplorer;

public partial class App : Application
{
    private Window? _window;

    public App()
    {
        InitializeComponent();
        UnhandledException += OnUnhandledException;
    }

    /// <summary>
    /// Last-chance handler: log, keep the app alive and surface the message in the
    /// main window's error InfoBar when it is reachable.
    /// </summary>
    private void OnUnhandledException(object sender, Microsoft.UI.Xaml.UnhandledExceptionEventArgs e)
    {
        System.Diagnostics.Debug.WriteLine($"Unhandled exception: {e.Exception}");
        e.Handled = true;
        try
        {
            if (_window is MainWindow mw)
            {
                string msg = e.Message;
                if (mw.DispatcherQueue.HasThreadAccess) mw.ShowError(msg, "Unexpected error");
                else mw.DispatcherQueue.TryEnqueue(() => mw.ShowError(msg, "Unexpected error"));
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Could not show unhandled exception: {ex}");
        }
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new MainWindow();
        _window.Activate();
    }
}
