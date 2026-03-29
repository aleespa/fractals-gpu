const express = require('express');
const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// Serve static UI files
app.use(express.static('public'));

console.log("SERVER VERSION 5.0 - READY");

app.get('/api/fractal', (req, res) => {
    // Prevent browser caching
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.setHeader('Surrogate-Control', 'no-store');

    // Parse parameters
    const type = req.query.type !== undefined ? parseInt(req.query.type) : 0;
    const width = req.query.w !== undefined ? parseInt(req.query.w) : 1280;
    const height = req.query.h !== undefined ? parseInt(req.query.h) : 720;
    const maxIterations = req.query.iter !== undefined ? parseInt(req.query.iter) : 400;
    const xCenter = req.query.x !== undefined ? parseFloat(req.query.x) : -1.4002;
    const yCenter = req.query.y !== undefined ? parseFloat(req.query.y) : 0.0;
    const zoom = req.query.z !== undefined ? parseFloat(req.query.z) : 0.10;
    const cx = req.query.cx !== undefined ? parseFloat(req.query.cx) : -0.4;
    const cy = req.query.cy !== undefined ? parseFloat(req.query.cy) : 0.6;
    const palRaw = req.query.pal || '';
    
    // Ensure temporary output directory exists
    const tmpDir = path.join(__dirname, 'tmp');
    if (!fs.existsSync(tmpDir)) {
        fs.mkdirSync(tmpDir);
    }
    
    const outputFilename = path.join(tmpDir, `fractal_${Date.now()}_${Math.random().toString(36).substring(7)}.png`);
    const exePath = path.join(__dirname, 'x64', 'Release', 'CppFractals.exe');
    
    // Build palette args: split flat comma list into [t, hex, t, hex, ...]
    const paletteArgs = palRaw ? palRaw.split(',').filter(s => s.length > 0) : [];

    // Arguments match the order in main.cu:
    // <type> <width> <height> <maxIter> <x> <y> <zoom> <cx> <cy> <output> [t hex ...]
    const args = [
        type.toString(),
        width.toString(),
        height.toString(),
        maxIterations.toString(),
        xCenter.toString(),
        yCenter.toString(),
        zoom.toString(),
        cx.toString(),
        cy.toString(),
        outputFilename,
        ...paletteArgs
    ];
    
    // Execute the CUDA generator
    console.log(`Executing: ${exePath} ${args.join(' ')}`);
    execFile(exePath, args, (error, stdout, stderr) => {
        console.log(`--- C++ STDOUT ---\n${stdout}\n------------------`);
        if (stderr) console.error(`--- C++ STDERR ---\n${stderr}\n------------------`);
        
        if (error) {
            console.error(`Execution error: ${error.message}`);
            return res.status(500).json({ error: "Failed to generate fractal. Make sure CppFractals.exe is compiled to x64/Release." });
        }
        
        // Send generated image to the client
        res.sendFile(outputFilename, (err) => {
            if (err) {
                console.error(`Error sending file: ${err.message}`);
            }
            // Fire and forget deletion
            fs.unlink(outputFilename, (unlinkErr) => {
                if (unlinkErr) console.error(`Failed to delete temp file: ${unlinkErr}`);
            });
        });
    });
});

app.listen(port, () => {
    console.log(`Mandelbrot Explorer is running! http://localhost:${port}`);
});
