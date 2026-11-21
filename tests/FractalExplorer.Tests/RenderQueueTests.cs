using FractalExplorer.Core;
using Xunit;

namespace FractalExplorer.Tests;

public class RenderQueueTests
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(5);

    private static TaskCompletionSource NewGate() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static TaskCompletionSource<int> NewSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    [Fact]
    public async Task JobsRunOneAtATime()
    {
        var q = new RenderQueue();
        var gate = NewGate();
        var firstStarted = NewGate();
        var secondStarted = NewGate();
        int current = 0, max = 0;

        void Enter() { int c = Interlocked.Increment(ref current); int m; while (c > (m = Volatile.Read(ref max))) Interlocked.CompareExchange(ref max, c, m); }

        var t1 = q.RunAsync(RenderLane.Save, _ =>
        {
            Enter(); firstStarted.TrySetResult();
            gate.Task.Wait(Timeout);
            Interlocked.Decrement(ref current);
            return 1;
        }, CancellationToken.None);
        await firstStarted.Task.WaitAsync(Timeout);

        var t2 = q.RunAsync(RenderLane.Save, _ =>
        {
            Enter(); secondStarted.TrySetResult();
            Interlocked.Decrement(ref current);
            return 2;
        }, CancellationToken.None);

        await Task.Delay(100);
        Assert.False(secondStarted.Task.IsCompleted);

        gate.SetResult();
        Assert.Equal(1, await t1.WaitAsync(Timeout));
        Assert.Equal(2, await t2.WaitAsync(Timeout));
        Assert.Equal(1, max);
    }

    [Fact]
    public async Task SupersededWaitingJob_IsCancelledWithoutRunning()
    {
        var q = new RenderQueue();
        var gate = NewGate();
        var saveStarted = NewGate();
        var order = new List<string>();
        bool aInvoked = false;

        var save = q.RunAsync(RenderLane.Save, _ =>
        {
            saveStarted.TrySetResult();
            gate.Task.Wait(Timeout);
            lock (order) order.Add("save");
            return 0;
        }, CancellationToken.None);
        await saveStarted.Task.WaitAsync(Timeout);

        var a = q.RunAsync(RenderLane.Interactive, _ => { aInvoked = true; return 1; }, CancellationToken.None);
        var b = q.RunAsync(RenderLane.Interactive, _ => { lock (order) order.Add("B"); return 2; }, CancellationToken.None);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => a.WaitAsync(Timeout));
        Assert.True(a.IsCanceled);

        gate.SetResult();
        await save.WaitAsync(Timeout);
        Assert.Equal(2, await b.WaitAsync(Timeout));
        Assert.False(aInvoked);
        Assert.Equal(new[] { "save", "B" }, order);
    }

    [Fact]
    public async Task SaveJobs_AreNotSuperseded()
    {
        var q = new RenderQueue();
        var gate = NewGate();
        var started = NewGate();
        var ran = new List<string>();

        var i = q.RunAsync(RenderLane.Interactive, _ =>
        {
            started.TrySetResult();
            gate.Task.Wait(Timeout);
            return 0;
        }, CancellationToken.None);
        await started.Task.WaitAsync(Timeout);

        var s1 = q.RunAsync(RenderLane.Save, _ => { lock (ran) ran.Add("S1"); return 1; }, CancellationToken.None);
        var s2 = q.RunAsync(RenderLane.Save, _ => { lock (ran) ran.Add("S2"); return 2; }, CancellationToken.None);

        gate.SetResult();
        await i.WaitAsync(Timeout);
        Assert.Equal(1, await s1.WaitAsync(Timeout));
        Assert.Equal(2, await s2.WaitAsync(Timeout));
        Assert.Equal(new[] { "S1", "S2" }, ran);
    }

    [Fact]
    public async Task CancelledWhileWaiting_NeverRuns()
    {
        var q = new RenderQueue();
        var gate = NewGate();
        var started = NewGate();
        bool invoked = false;

        var blocker = q.RunAsync(RenderLane.Save, _ =>
        {
            started.TrySetResult();
            gate.Task.Wait(Timeout);
            return 0;
        }, CancellationToken.None);
        await started.Task.WaitAsync(Timeout);

        using var cts = new CancellationTokenSource();
        var t = q.RunAsync(RenderLane.Interactive, _ => { invoked = true; return 1; }, cts.Token);
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => t.WaitAsync(Timeout));
        Assert.True(t.IsCanceled);

        gate.SetResult();
        await blocker.WaitAsync(Timeout);
        Assert.False(invoked);
    }

    [Fact]
    public async Task JobException_PropagatesAndQueueContinues()
    {
        var q = new RenderQueue();
        var bad = q.RunAsync<int>(RenderLane.Save, _ => throw new FractalRenderException("x"), CancellationToken.None);
        var ex = await Assert.ThrowsAsync<FractalRenderException>(() => bad.WaitAsync(Timeout));
        Assert.Equal("x", ex.Message);

        var next = q.RunAsync(RenderLane.Interactive, _ => 7, CancellationToken.None);
        Assert.Equal(7, await next.WaitAsync(Timeout));
    }
}
