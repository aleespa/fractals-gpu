const express = require('express');
const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// Serve static UI files
app.use(express.static('public'));

console.log("SERVER VERSION 5.0 - READY");

const exePath = path.join(__dirname, 'x64', 'Release', 'CppFractals.exe');
const tmpDir = path.join(__dirname, 'tmp');

function num(v, def, parse) {
    if (v === undefined) return def;
    const n = parse(v);
    return Number.isFinite(n) ? n : def;
}

function parseView(q) {
    return {
        type: num(q.type, 0, parseInt),
        width: num(q.w, 1280, parseInt),
        height: num(q.h, 720, parseInt),
        maxIterations: num(q.iter, 400, parseInt),
        xCenter: num(q.x, -1.4002, parseFloat),
        yCenter: num(q.y, 0.0, parseFloat),
        zoom: num(q.z, 0.10, parseFloat),
        cx: num(q.cx, -0.4, parseFloat),
        cy: num(q.cy, 0.6, parseFloat),
        // Flat comma list [t, hex, t, hex, ...] passed straight through as argv
        paletteArgs: (q.pal || '').split(',').filter(s => s.length > 0),
        ss: 1
    };
}

/**
 * Runs CppFractals.exe for the given view and calls done(err, pngPath).
 * Args match the order in main.cu:
 * <type> <width> <height> <maxIter> <x> <y> <zoom> <cx> <cy> <output> [--ss N] [t hex ...]
 */
function renderFractal(v, done) {
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir);
    const outputFilename = path.join(tmpDir, `fractal_${Date.now()}_${Math.random().toString(36).substring(7)}.png`);
    const args = [
        v.type, v.width, v.height, v.maxIterations, v.xCenter, v.yCenter,
        v.zoom, v.cx, v.cy
    ].map(String);
    args.push(outputFilename);
    if (v.ss > 1) args.push('--ss', String(v.ss));
    args.push(...v.paletteArgs);

    console.log(`Executing: ${exePath} ${args.join(' ')}`);
    execFile(exePath, args, (error, stdout, stderr) => {
        console.log(`--- C++ STDOUT ---
${stdout}
------------------`);
        if (stderr) console.error(`--- C++ STDERR ---
${stderr}
------------------`);
        if (error) {
            console.error(`Execution error: ${error.message}`);
            fs.unlink(outputFilename, () => {});
            return done(stderr.trim() || error.message);
        }
        done(null, outputFilename);
    });
}

function sendAndDelete(res, file, options) {
    const cleanup = () => fs.unlink(file, (err) => {
        if (err) console.error(`Failed to delete temp file: ${err}`);
    });
    const cb = (err) => {
        if (err) console.error(`Error sending file: ${err.message}`);
        cleanup();
    };
    if (options && options.downloadName) res.download(file, options.downloadName, cb);
    else res.sendFile(file, cb);
}

app.get('/api/fractal', (req, res) => {
    // Prevent browser caching
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.setHeader('Surrogate-Control', 'no-store');

    renderFractal(parseView(req.query), (err, file) => {
        if (err) {
            return res.status(500).json({ error: "Failed to generate fractal. Make sure CppFractals.exe is compiled to x64/Release." });
        }
        sendAndDelete(res, file);
    });
});

// High-quality export of the current view, sent as a file download.
// Extra params: va = aspect ratio (w/h) of the on-screen view, ss = samples per side.
let exportInProgress = false;
app.get('/api/export', (req, res) => {
    const v = parseView(req.query);
    if (!(v.width >= 16 && v.width <= 16384 && v.height >= 16 && v.height <= 16384)) {
        return res.status(400).json({ error: 'Width and height must be between 16 and 16384.' });
    }
    v.ss = Math.min(Math.max(num(req.query.ss, 2, parseInt), 1), 4);

    // Keep everything framed on screen visible: zoom is the half-width, so when
    // the export is relatively wider than the view, grow the half-width to keep
    // the view's half-height.
    const viewAspect = num(req.query.va, v.width / v.height, parseFloat);
    const exportAspect = v.width / v.height;
    if (viewAspect > 0 && exportAspect > viewAspect) {
        v.zoom = v.zoom * exportAspect / viewAspect;
    }

    if (exportInProgress) {
        return res.status(429).json({ error: 'An export is already rendering.' });
    }
    exportInProgress = true;
    const started = Date.now();
    renderFractal(v, (err, file) => {
        exportInProgress = false;
        if (err) return res.status(500).json({ error: `Render failed: ${err}` });
        console.log(`Export ${v.width}x${v.height} ss=${v.ss} rendered in ${Date.now() - started} ms`);
        const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').slice(0, 15);
        const kind = v.type === 1 ? 'julia' : 'mandelbrot';
        sendAndDelete(res, file, { downloadName: `${kind}_${v.width}x${v.height}_${stamp}.png` });
    });
});

app.listen(port, () => {
    console.log(`Mandelbrot Explorer is running! http://localhost:${port}`);
});
