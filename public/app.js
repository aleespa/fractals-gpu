// ── State ───────────────────────────────────────────────────
let state = {
    type: 0,
    x: -1.4002,
    y: 0.0,
    zoom: 0.10,
    iter: 400,
    cx: -0.4,
    cy: 0.6
};

let palette = [
    { t: 0.0,  hex: "000000" },
    { t: 0.25, hex: "2130C8" },
    { t: 0.5,  hex: "8C0021" },
    { t: 0.75, hex: "FF9900" },
    { t: 1.0,  hex: "FFFFFF" },
];

let isDragging = false;
let startDragX = 0;
let startDragY = 0;

let fetchTimeout = null;
let currentAbortController = null;

let visTranslateX = 0;
let visTranslateY = 0;
let visScale = 1;

// ── DOM refs ────────────────────────────────────────────────
const imgElement       = document.getElementById('fractal-img');
const containerElement = document.getElementById('fractal-container');
const loaderElement    = document.getElementById('loader');
const typeSelect       = document.getElementById('type-select');
const juliaControls    = document.getElementById('julia-controls');
const iterSlider       = document.getElementById('iterations');
const cxSlider         = document.getElementById('cx');
const cySlider         = document.getElementById('cy');
const iterVal          = document.getElementById('iter-val');
const cxVal            = document.getElementById('cx-val');
const cyVal            = document.getElementById('cy-val');
const statX            = document.getElementById('stat-x');
const statY            = document.getElementById('stat-y');
const statZ            = document.getElementById('stat-z');
const copyCmdBtn       = document.getElementById('copy-cmd-btn');
const resetBtn         = document.getElementById('reset-btn');
const gradientBar      = document.getElementById('gradient-bar');
const nativePicker     = document.getElementById('native-color-picker');

// ── Helpers ─────────────────────────────────────────────────
function hexToRgb(hex) {
    const n = parseInt(hex, 16);
    return [(n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF];
}

function hexToRgbStr(hex) {
    const [r, g, b] = hexToRgb(hex);
    return `rgb(${r},${g},${b})`;
}

/** Sample the current gradient at position t ∈ [0,1] and return a hex string */
function sampleGradient(t) {
    const sorted = [...palette].sort((a, b) => a.t - b.t);
    t = Math.max(0, Math.min(1, t));
    let i = 0;
    for (let k = 0; k < sorted.length - 1; k++) {
        if (sorted[k].t <= t) i = k;
    }
    const j = Math.min(i + 1, sorted.length - 1);
    const span = sorted[j].t - sorted[i].t;
    const frac = span > 0 ? (t - sorted[i].t) / span : 0;
    const a = hexToRgb(sorted[i].hex);
    const b = hexToRgb(sorted[j].hex);
    const lerp = (x, y) => Math.round(x + frac * (y - x));
    const r = lerp(a[0], b[0]).toString(16).padStart(2, '0');
    const g = lerp(a[1], b[1]).toString(16).padStart(2, '0');
    const bh = lerp(a[2], b[2]).toString(16).padStart(2, '0');
    return (r + g + bh).toUpperCase();
}

/** Serialize palette to the comma-separated string for the API */
function serializePalette() {
    // ALWAYS sort before sending to backend
    palette.sort((a, b) => a.t - b.t);
    return palette.flatMap(s => [s.t.toFixed(6), s.hex]).join(',');
}

// ── Transform ────────────────────────────────────────────────
function updateTransform() {
    imgElement.style.transform = `translate(${visTranslateX}px, ${visTranslateY}px) scale(${visScale})`;
}

// ── Stats UI ────────────────────────────────────────────────
function updateStatsUI() {
    iterVal.textContent = state.iter;
    cxVal.textContent   = state.cx.toFixed(3);
    cyVal.textContent   = state.cy.toFixed(3);
    statX.textContent   = state.x.toFixed(6);
    statY.textContent   = state.y.toFixed(6);
    statZ.textContent   = parseFloat((1 / state.zoom).toFixed(2)).toLocaleString() + "x";
}

// ── Fetch ────────────────────────────────────────────────────
async function fetchFractal() {
    if (currentAbortController) currentAbortController.abort();
    currentAbortController = new AbortController();

    const width  = window.innerWidth;
    const height = window.innerHeight;

    const params = new URLSearchParams({
        type: state.type,
        w:    width,
        h:    height,
        iter: state.iter,
        x:    state.x,
        y:    state.y,
        z:    state.zoom,
        cx:   state.cx,
        cy:   state.cy,
        pal:  serializePalette()
    });

    const url = `/api/fractal?${params.toString()}`;
    imgElement.classList.add('loading');
    loaderElement.classList.add('active');

    try {
        const response = await fetch(url, { signal: currentAbortController.signal });
        if (!response.ok) throw new Error("Failed to load image");

        const blob      = await response.blob();
        const objectURL = URL.createObjectURL(blob);

        imgElement.onload = () => {
            URL.revokeObjectURL(objectURL);
            imgElement.classList.remove('loading');
            loaderElement.classList.remove('active');
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

function debouncedFetch() {
    clearTimeout(fetchTimeout);
    fetchTimeout = setTimeout(fetchFractal, 150);
}

// ── Gradient bar ─────────────────────────────────────────────
function renderGradientBar() {
    // Sort in-place to keep data-state and UI-state unified
    palette.sort((a, b) => a.t - b.t);

    // Recompute CSS gradient string
    const stops = palette.map(s => `${hexToRgbStr(s.hex)} ${(s.t * 100).toFixed(2)}%`).join(', ');
    gradientBar.style.background = `linear-gradient(to right, ${stops})`;

    // Remove existing handles to avoid event listener ghosting and sync issues
    gradientBar.querySelectorAll('.palette-handle').forEach(el => el.remove());

    // Re-create handles based on sorted state
    palette.forEach((stop) => {
        const handle = document.createElement('div');
        handle.className = 'palette-handle' + (stop.t === 0 || stop.t === 1 ? ' pinned' : '');
        handle.style.left            = `${stop.t * 100}%`;
        handle.style.backgroundColor = '#' + stop.hex;
        handle.dataset.t = stop.t; // store t as a key for lookup

        gradientBar.appendChild(handle);
    });
}

// ── Palette interactions ─────────────────────────────────────
let draggingStop = null; 
let barDragStartX = 0;
let barDragStartY = 0;
let draggedDistance = 0;
let mousedownHandle = null;   // the handle element that received mousedown
let mousedownStop = null;     // the palette stop object for that handle

/** Find palette stop by t with high precision */
function stopByT(t) {
    return palette.find(s => Math.abs(s.t - t) < 1e-9);
}

gradientBar.addEventListener('mousedown', (e) => {
    const handle = e.target.closest('.palette-handle');
    if (handle) {
        e.stopPropagation();
        const t = parseFloat(handle.dataset.t);
        const stop = stopByT(t);
        if (!stop) return;

        barDragStartX = e.clientX;
        barDragStartY = e.clientY;
        draggedDistance = 0;
        mousedownHandle = handle;
        mousedownStop = stop;

        // All handles (including pinned) can be dragged or clicked
        // Only non-pinned handles are actually movable
        if (stop.t !== 0 && stop.t !== 1) {
            draggingStop = stop;
        }
        e.preventDefault();
    }
});

// Click on BAR background to add stop
gradientBar.addEventListener('click', (e) => {
    if (e.target !== gradientBar) return;
    const rect = gradientBar.getBoundingClientRect();
    const t = Math.max(0.001, Math.min(0.999, (e.clientX - rect.left) / rect.width));
    const hex = sampleGradient(t);
    palette.push({ t, hex });
    renderGradientBar();
    debouncedFetch();
});

window.addEventListener('mousemove', (e) => {
    if (!draggingStop) return;
    const dx = e.clientX - barDragStartX;
    const dy = e.clientY - barDragStartY;
    draggedDistance = Math.sqrt(dx * dx + dy * dy);

    const rect = gradientBar.getBoundingClientRect();
    const newT = Math.max(0.001, Math.min(0.999, (e.clientX - rect.left) / rect.width));

    draggingStop.t = newT;
    
    // Live update background for smooth dragging
    const tempSorted = [...palette].sort((a, b) => a.t - b.t);
    const stopsString = tempSorted.map(s => `${hexToRgbStr(s.hex)} ${(s.t * 100).toFixed(2)}%`).join(', ');
    gradientBar.style.background = `linear-gradient(to right, ${stopsString})`;
    
    // Find the current DOM handle and move it manually
    const handle = [...gradientBar.querySelectorAll('.palette-handle')].find(h => stopByT(parseFloat(h.dataset.t)) === draggingStop);
    if (handle) {
        handle.style.left = `${newT * 100}%`;
        handle.dataset.t = newT;
    }
});

window.addEventListener('mouseup', (e) => {
    const wasClick = draggedDistance < 5;
    const clickedStop = mousedownStop;
    const clickedHandle = mousedownHandle;

    // Reset drag state
    if (draggingStop) {
        draggingStop = null;
        renderGradientBar();
        debouncedFetch();
    }

    // If it was a click (not a drag), open the color picker
    // We do this HERE because renderGradientBar() above destroyed the old handle,
    // so we need to find the new handle element for positioning
    if (wasClick && clickedStop) {
        // Find the freshly-rendered handle for this stop
        const freshHandle = [...gradientBar.querySelectorAll('.palette-handle')]
            .find(h => Math.abs(parseFloat(h.dataset.t) - clickedStop.t) < 1e-6);
        if (freshHandle) {
            openColorPicker(clickedStop, freshHandle);
        }
    }

    mousedownHandle = null;
    mousedownStop = null;
    draggedDistance = 0;
});

// Handle deletion
gradientBar.addEventListener('contextmenu', (e) => {
    const handle = e.target.closest('.palette-handle');
    if (!handle) return;
    e.preventDefault();

    const t = parseFloat(handle.dataset.t);
    if (t === 0 || t === 1) return; 
    if (palette.length <= 2) return;

    palette = palette.filter(s => Math.abs(s.t - t) >= 1e-9);
    renderGradientBar();
    debouncedFetch();
});

// Singleton attachment for the native color picker to avoid listener accumulation
let currentActiveStop = null;
let currentActiveHandle = null;

const onPickerInput = (ev) => {
    if (!currentActiveStop || !currentActiveHandle) return;
    currentActiveStop.hex = ev.target.value.slice(1).toUpperCase();
    currentActiveHandle.style.backgroundColor = ev.target.value;
    
    // Live CSS gradient sync
    const tempSorted = [...palette].sort((a, b) => a.t - b.t);
    const stopsStr = tempSorted.map(s => `${hexToRgbStr(s.hex)} ${(s.t * 100).toFixed(2)}%`).join(', ');
    gradientBar.style.background = `linear-gradient(to right, ${stopsStr})`;
    debouncedFetch();
};

const onPickerChange = () => {
    // Re-render fully once color is picked to ensure all state is consistent
    renderGradientBar();
    currentActiveStop = null;
    currentActiveHandle = null;
};

// Initialize picker once
nativePicker.addEventListener('input', onPickerInput);
nativePicker.addEventListener('change', onPickerChange);

function openColorPicker(stop, handle) {
    currentActiveStop = stop;
    currentActiveHandle = handle;
    nativePicker.value = '#' + stop.hex;
    
    const rect = handle.getBoundingClientRect();
    nativePicker.style.left = `${rect.left}px`;
    nativePicker.style.top  = `${rect.bottom + 4}px`;
    
    nativePicker.click();
}

// ── Controls ─────────────────────────────────────────────────
typeSelect.addEventListener('change', (e) => {
    state.type = parseInt(e.target.value);
    juliaControls.style.display = state.type === 1 ? 'block' : 'none';
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
    if (state.type === 1) { state.x = 0; state.zoom = 1.5; }
    iterSlider.value = state.iter;
    cxSlider.value   = state.cx;
    cySlider.value   = state.cy;
    visTranslateX = 0;
    visTranslateY = 0;
    visScale = 1;

    // Reset palette to default
    palette = [
        { t: 0.0,  hex: "000000" },
        { t: 0.25, hex: "2130C8" },
        { t: 0.5,  hex: "8C0021" },
        { t: 0.75, hex: "FF9900" },
        { t: 1.0,  hex: "FFFFFF" },
    ];

    updateTransform();
    updateStatsUI();
    renderGradientBar();
    fetchFractal();
});

copyCmdBtn.addEventListener('click', () => {
    const palArgs = [...palette]
        .sort((a, b) => a.t - b.t)
        .flatMap(s => [s.t.toFixed(6), s.hex])
        .join(' ');
    const cmd = `.\\x64\\Release\\CppFractals.exe ${state.type} 5000 5000 ${state.iter} ${state.x} ${state.y} ${state.zoom} ${state.cx} ${state.cy} output.png ${palArgs}`;
    navigator.clipboard.writeText(cmd).then(() => {
        const orig = copyCmdBtn.textContent;
        copyCmdBtn.textContent = 'Copied!';
        setTimeout(() => copyCmdBtn.textContent = orig, 1500);
    }).catch(err => console.error('Failed to copy text:', err));
});

// ── Panning ──────────────────────────────────────────────────
containerElement.addEventListener('mousedown', (e) => {
    isDragging = true;
    startDragX = e.clientX;
    startDragY = e.clientY;
});

window.addEventListener('mouseup', () => {
    if (isDragging) {
        isDragging = false;
        fetchFractal();
    }
});

window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;

    const dx = e.clientX - startDragX;
    const dy = e.clientY - startDragY;
    startDragX = e.clientX;
    startDragY = e.clientY;

    visTranslateX += dx;
    visTranslateY += dy;
    updateTransform();

    const width       = window.innerWidth;
    const height      = window.innerHeight;
    const aspectRatio = width / height;

    state.x -= (dx / width)  * (2 * state.zoom);
    state.y -= (dy / height) * (2 * state.zoom / aspectRatio);
    updateStatsUI();
});

// ── Zooming ──────────────────────────────────────────────────
containerElement.addEventListener('wheel', (e) => {
    e.preventDefault();

    const zoomDelta  = Math.abs(e.deltaY) * 0.001 * 0.5 + 0.1;
    let mathFactor   = e.deltaY > 0 ? (1 + zoomDelta) : (1 - zoomDelta);
    let factor       = 1 / mathFactor;

    const W           = window.innerWidth;
    const H           = window.innerHeight;
    const aspectRatio = W / H;
    const offsetX     = e.clientX - W / 2;
    const offsetY     = e.clientY - H / 2;

    const cxOffset = (offsetX / W) * (2 * state.zoom);
    const cyOffset = (offsetY / H) * (2 * state.zoom / aspectRatio);
    const mouseCx  = state.x + cxOffset;
    const mouseCy  = state.y + cyOffset;

    state.zoom *= mathFactor;

    state.x = mouseCx - (offsetX / W) * (2 * state.zoom);
    state.y = mouseCy - (offsetY / H) * (2 * state.zoom / aspectRatio);

    visTranslateX = offsetX - (offsetX - visTranslateX) * factor;
    visTranslateY = offsetY - (offsetY - visTranslateY) * factor;
    visScale *= factor;

    updateTransform();
    updateStatsUI();
    debouncedFetch();
}, { passive: false });

// ── Init ─────────────────────────────────────────────────────
renderGradientBar();
fetchFractal();

window.addEventListener('resize', () => {
    clearTimeout(fetchTimeout);
    fetchTimeout = setTimeout(fetchFractal, 500);
});
