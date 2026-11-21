namespace FractalExplorer.Core;

public enum RenderLane { Interactive, Save }

/// <summary>
/// Serializes GPU jobs in managed code so superseded interactive renders
/// are dropped while still waiting, before they ever reach the GPU.
/// </summary>
public sealed class RenderQueue
{
    private readonly SemaphoreSlim _slot = new(1, 1);
    private readonly object _lock = new();
    private CancellationTokenSource? _latestWaitingInteractive;

    public async Task<T> RunAsync<T>(RenderLane lane, Func<CancellationToken, T> job, CancellationToken ct)
    {
        // Cancels this job while it is still waiting (caller ct or superseded).
        using var waitCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        if (lane == RenderLane.Interactive)
        {
            lock (_lock)
            {
                _latestWaitingInteractive?.Cancel();
                _latestWaitingInteractive = waitCts;
            }
        }

        try
        {
            await _slot.WaitAsync(waitCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            ClearLatest(waitCts);
            throw;
        }

        // Job now owns the slot: it is running, no longer supersedable.
        ClearLatest(waitCts);
        try
        {
            ct.ThrowIfCancellationRequested();
            return await Task.Run(() => job(ct), CancellationToken.None).ConfigureAwait(false);
        }
        finally
        {
            _slot.Release();
        }
    }

    private void ClearLatest(CancellationTokenSource cts)
    {
        lock (_lock)
        {
            if (ReferenceEquals(_latestWaitingInteractive, cts)) _latestWaitingInteractive = null;
        }
    }
}
