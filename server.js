const express = require('express');
const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

// Serve static UI files
app.use(express.static('public'));

app.get('/api/fractal', (req, res) => {
    // Parse parameters
    const type = parseInt(req.query.type) || 0;
    const width = parseInt(req.query.w) || 1280;
    const height = parseInt(req.query.h) || 720;
    const maxIterations = parseInt(req.query.iter) || 400;
    const xCenter = parseFloat(req.query.x) || -1.4002;
    const yCenter = parseFloat(req.query.y) || 0.0;
    const zoom = parseFloat(req.query.z) || 0.10;
    const cx = parseFloat(req.query.cx) || -0.4;
    const cy = parseFloat(req.query.cy) || 0.6;
    
    // Ensure temporary output directory exists
    const tmpDir = path.join(__dirname, 'tmp');
    if (!fs.existsSync(tmpDir)) {
        fs.mkdirSync(tmpDir);
    }
    
    const outputFilename = path.join(tmpDir, `fractal_${Date.now()}_${Math.random().toString(36).substring(7)}.png`);
    const exePath = path.join(__dirname, 'x64', 'Release', 'CppFractals.exe');
    
    // Arguments match the order in main.cu:
    // <type> <width> <height> <maxIterations> <xCenter> <yCenter> <zoom> <cx> <cy> <outputFilename>
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
        outputFilename
    ];
    
    // Execute the CUDA generator
    execFile(exePath, args, (error, stdout, stderr) => {
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
