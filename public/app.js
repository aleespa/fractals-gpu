// Initial state matching the server defaults
let state = {
    type: 0,
    x: -1.4002,
    y: 0.0,
    zoom: 0.10,
    iter: 400,
    cx: -0.4,
    cy: 0.6
};

let isDragging = false;
let startDragX = 0;
let startDragY = 0;
let lastImageX = 0;
let lastImageY = 0;

let fetchTimeout = null;
let currentAbortController = null;

let visTranslateX = 0;
let visTranslateY = 0;
let prevVisTranslateX = 0;
let prevVisTranslateY = 0;
let visScale = 1;

const imgElement = document.getElementById('fractal-img');
const containerElement = document.getElementById('fractal-container');
const loaderElement = document.getElementById('loader');

function updateTransform() {
    imgElement.style.transform = `translate(${visTranslateX}px, ${visTranslateY}px) scale(${visScale})`;
}

// Controls
const typeSelect = document.getElementById('type-select');
const juliaControls = document.getElementById('julia-controls');
const iterSlider = document.getElementById('iterations');
const cxSlider = document.getElementById('cx');
const cySlider = document.getElementById('cy');

const iterVal = document.getElementById('iter-val');
const cxVal = document.getElementById('cx-val');
const cyVal = document.getElementById('cy-val');

const statX = document.getElementById('stat-x');
const statY = document.getElementById('stat-y');
const statZ = document.getElementById('stat-z');
const copyCmdBtn = document.getElementById('copy-cmd-btn');
const resetBtn = document.getElementById('reset-btn');

function updateStatsUI() {
    iterVal.textContent = state.iter;
    cxVal.textContent = state.cx.toFixed(3);
    cyVal.textContent = state.cy.toFixed(3);
    statX.textContent = state.x.toFixed(6);
    statY.textContent = state.y.toFixed(6);
    statZ.textContent = state.zoom.toFixed(6);
}

async function fetchFractal() {
    if (currentAbortController) {
        currentAbortController.abort();
    }
    currentAbortController = new AbortController();

    const width = window.innerWidth;
    const height = window.innerHeight;

    // Use query params
    const params = new URLSearchParams({
        type: state.type,
        w: width,
        h: height,
        iter: state.iter,
        x: state.x,
        y: state.y,
        z: state.zoom,
        cx: state.cx,
        cy: state.cy
    });

    const url = `/api/fractal?${params.toString()}`;

    imgElement.classList.add('loading');
    loaderElement.classList.add('active');

    try {
        const response = await fetch(url, { signal: currentAbortController.signal });
        if (!response.ok) throw new Error("Failed to load image");
        
        const blob = await response.blob();
        const objectURL = URL.createObjectURL(blob);
        
        imgElement.onload = () => {
            URL.revokeObjectURL(objectURL);
            imgElement.classList.remove('loading');
            loaderElement.classList.remove('active');

            // Reset visual transforms now that we have the exact frame
            visTranslateX = 0;
            visTranslateY = 0;
            visScale = 1;
            updateTransform();
        };
        imgElement.src = objectURL;
        updateStatsUI();

    } catch (e) {
        if (e.name !== 'AbortError') {
            console.error(e);
            imgElement.classList.remove('loading');
            loaderElement.classList.remove('active');
        }
    }
}

// Debounce fetches for sliders
function debouncedFetch() {
    clearTimeout(fetchTimeout);
    fetchTimeout = setTimeout(fetchFractal, 150);
}

// Event Listeners
typeSelect.addEventListener('change', (e) => {
    state.type = parseInt(e.target.value);
    if (state.type === 1) {
        juliaControls.style.display = 'block';
    } else {
        juliaControls.style.display = 'none';
    }
    debouncedFetch();
});

iterSlider.addEventListener('input', (e) => {
    state.iter = parseInt(e.target.value);
    iterVal.textContent = state.iter;
    debouncedFetch();
});

cxSlider.addEventListener('input', (e) => {
    state.cx = parseFloat(e.target.value);
    cxVal.textContent = state.cx.toFixed(3);
    debouncedFetch();
});

cySlider.addEventListener('input', (e) => {
    state.cy = parseFloat(e.target.value);
    cyVal.textContent = state.cy.toFixed(3);
    debouncedFetch();
});

resetBtn.addEventListener('click', () => {
    state = { type: state.type, x: -1.4002, y: 0.0, zoom: 0.10, iter: 400, cx: -0.4, cy: 0.6 };
    if (state.type === 1) {
        state.x = 0;
        state.zoom = 1.5;
    }
    iterSlider.value = state.iter;
    cxSlider.value = state.cx;
    cySlider.value = state.cy;
    visTranslateX = 0;
    visTranslateY = 0;
    visScale = 1;
    updateTransform();
    fetchFractal();
});

copyCmdBtn.addEventListener('click', () => {
    // We use full precision for the config variables
    const cmd = `.\\x64\\Release\\CppFractals.exe ${state.type} 5000 5000 ${state.iter} ${state.x} ${state.y} ${state.zoom} ${state.cx} ${state.cy} output.png`;
    navigator.clipboard.writeText(cmd).then(() => {
        const originalText = copyCmdBtn.textContent;
        copyCmdBtn.textContent = 'Copied!';
        setTimeout(() => copyCmdBtn.textContent = originalText, 1500);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
});

// Panning Logic
containerElement.addEventListener('mousedown', (e) => {
    isDragging = true;
    startDragX = e.clientX;
    startDragY = e.clientY;
});

window.addEventListener('mouseup', () => {
    if(isDragging) {
        isDragging = false;
        fetchFractal();
    }
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    const dx = e.clientX - startDragX;
    const dy = e.clientY - startDragY;

    // Reset start coordinates for incremental updates
    startDragX = e.clientX;
    startDragY = e.clientY;

    // Visual transform instantly
    visTranslateX += dx;
    visTranslateY += dy;
    updateTransform();

    // Mathematical coordinate
    const width = window.innerWidth;
    const height = window.innerHeight;
    const aspectRatio = width / height;

    const cordDx = (dx / width) * (2 * state.zoom);
    const cordDy = (dy / height) * (2 * state.zoom / aspectRatio);

    state.x -= cordDx;
    state.y -= cordDy;

    updateStatsUI();
});

// Zoom Logic
containerElement.addEventListener('wheel', (e) => {
    e.preventDefault();
    
    // Smooth scroll zooming
    const zoomDelta = Math.abs(e.deltaY) * 0.001 * 0.5 + 0.1; // proportional scroll
    
    let factor = 1;
    let mathFactor = 1;

    if (e.deltaY > 0) {
        mathFactor = (1 + zoomDelta);
        factor = 1 / mathFactor;
    } else {
        mathFactor = (1 - zoomDelta);
        factor = 1 / mathFactor;
    }

    const W = window.innerWidth;
    const H = window.innerHeight;
    const aspectRatio = W / H;

    const offsetX = e.clientX - W / 2;
    const offsetY = e.clientY - H / 2;

    // Convert mouse position to complex plane coord before zooming
    const cxOffset = (offsetX / W) * (2 * state.zoom);
    const cyOffset = (offsetY / H) * (2 * state.zoom / aspectRatio);
    const mouseCx = state.x + cxOffset;
    const mouseCy = state.y + cyOffset;

    // Apply zoom
    state.zoom *= mathFactor;

    // Keep mouse stationary in complex plane
    const newCxOffset = (offsetX / W) * (2 * state.zoom);
    const newCyOffset = (offsetY / H) * (2 * state.zoom / aspectRatio);
    
    state.x = mouseCx - newCxOffset;
    state.y = mouseCy - newCyOffset;
    
    // Visual smooth CSS shift logic tracking the mouse zoom 
    visTranslateX = offsetX - (offsetX - visTranslateX) * factor;
    visTranslateY = offsetY - (offsetY - visTranslateY) * factor;
    visScale *= factor;

    updateTransform();
    
    updateStatsUI();
    debouncedFetch();
}, { passive: false });

// Initial fetch
fetchFractal();

// Handle window resize
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(fetchFractal, 500);
});
