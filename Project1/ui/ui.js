// ui/ui.js

export function setupPartAUI(state, onChange) {
    const container = document.getElementById('d-order');
    if (!container || document.getElementById('slider-tx')) return;
    const controlBar = document.createElement('div');
    controlBar.className = 'ctl';
    controlBar.style.marginTop = '15px';
    controlBar.innerHTML = `
        <label>Translate X: <input type="range" id="slider-tx" min="-3" max="3" step="0.1" value="${state.tx}"><span class="val" id="val-tx">${state.tx}</span></label>
        <label>Rotate Y: <input type="range" id="slider-ry" min="-3.14" max="3.14" step="0.05" value="${state.ry}"><span class="val" id="val-ry">${state.ry}</span></label>
        <label>Scale: <input type="range" id="slider-s" min="0.3" max="2.5" step="0.05" value="${state.scale}"><span class="val" id="val-s">${state.scale}</span></label>
    `;
    container.appendChild(controlBar);

    document.getElementById('slider-tx').addEventListener('input', (e) => { state.tx = parseFloat(e.target.value); document.getElementById('val-tx').textContent = state.tx.toFixed(1); onChange(); });
    document.getElementById('slider-ry').addEventListener('input', (e) => { state.ry = parseFloat(e.target.value); document.getElementById('val-ry').textContent = state.ry.toFixed(2); onChange(); });
    document.getElementById('slider-s').addEventListener('input', (e) => { state.scale = parseFloat(e.target.value); document.getElementById('val-s').textContent = state.scale.toFixed(2); onChange(); });
}

export function setupPartBUI(state, onChange) {
    const container = document.getElementById('d-graph');
    if (!container || document.getElementById('slider-b-a1')) return;
    const controlBar = document.createElement('div');
    controlBar.className = 'ctl';
    controlBar.style.marginTop = '10px';
    controlBar.innerHTML = `
        <label>Base Rotation: <input type="range" id="slider-b-a1" min="-3.14" max="3.14" step="0.02" value="${state.a1}"></label>
        <label>Joint 1: <input type="range" id="slider-b-a2" min="-1.5" max="1.5" step="0.02" value="${state.a2}"></label>
        <label>Joint 2: <input type="range" id="slider-b-a3" min="-1.5" max="1.5" step="0.02" value="${state.a3}"></label>
        <button id="btn-axes" class="ctl button">Debug Axes: OFF</button>
        <button id="btn-animate" class="ctl button">Animate</button>
    `;
    container.appendChild(controlBar);

    document.getElementById('slider-b-a1').addEventListener('input', (e) => onChange({ a1: parseFloat(e.target.value) }));
    document.getElementById('slider-b-a2').addEventListener('input', (e) => onChange({ a2: parseFloat(e.target.value) }));
    document.getElementById('slider-b-a3').addEventListener('input', (e) => onChange({ a3: parseFloat(e.target.value) }));

    const btnAxes = document.getElementById('btn-axes');
    btnAxes.addEventListener('click', () => {
        const isAxes = !btnAxes.classList.contains('on');
        btnAxes.classList.toggle('on', isAxes);
        btnAxes.textContent = isAxes ? 'Debug Axes: ON' : 'Debug Axes: OFF';
        onChange({ showAxes: isAxes });
    });

    const btn = document.getElementById('btn-animate');
    btn.addEventListener('click', () => {
        const isAnim = !btn.classList.contains('on');
        btn.classList.toggle('on', isAnim);
        onChange({ isAnimated: isAnim });
    });
}

export function setupPartCUI(cam, onChange) {
    const container = document.getElementById('d-camera');
    if (!container || document.getElementById('slider-cyaw')) return;
    const controlBar = document.createElement('div');
    controlBar.className = 'ctl';
    controlBar.style.marginTop = '10px';
    controlBar.innerHTML = `
        <label>Azimuth: <input type="range" id="slider-cyaw" min="-3.14" max="3.14" step="0.02" value="${cam.cyaw}"></label>
        <label>Height: <input type="range" id="slider-cpitch" min="-0.3" max="1.4" step="0.02" value="${cam.cpitch}"></label>
        <label>Distance: <input type="range" id="slider-cdist" min="2" max="12" step="0.1" value="${cam.cdist}"></label>
        <label>FOV (°): <input type="range" id="slider-fov" min="15" max="100" step="1" value="${cam.fov}"></label>
        <label>Near: <input type="range" id="slider-near" min="0.2" max="4" step="0.1" value="${cam.near}"></label>
        <label>Far: <input type="range" id="slider-far" min="3" max="14" step="0.5" value="${cam.far}"></label>
        <button id="btn-proj-toggle" class="ctl button on">Perspective / Orthographic</button>
    `;
    container.appendChild(controlBar);

    document.getElementById('slider-cyaw').addEventListener('input', (e) => { cam.cyaw = parseFloat(e.target.value); onChange(); });
    document.getElementById('slider-cpitch').addEventListener('input', (e) => { cam.cpitch = parseFloat(e.target.value); onChange(); });
    document.getElementById('slider-cdist').addEventListener('input', (e) => { cam.cdist = parseFloat(e.target.value); onChange(); });
    document.getElementById('slider-fov').addEventListener('input', (e) => { cam.fov = parseFloat(e.target.value); onChange(); });
    document.getElementById('slider-near').addEventListener('input', (e) => { cam.near = parseFloat(e.target.value); onChange(); });
    document.getElementById('slider-far').addEventListener('input', (e) => { cam.far = parseFloat(e.target.value); onChange(); });

    const btnProj = document.getElementById('btn-proj-toggle');
    btnProj.addEventListener('click', () => {
        cam.isPersp = !cam.isPersp;
        btnProj.classList.toggle('on', cam.isPersp);
        onChange();
    });
}

export function setupPartDUI(callbacks) {
    const container = document.getElementById('d-curves');
    if (!container || document.getElementById('btn-curve-type')) return;
    const controlBar = document.createElement('div');
    controlBar.className = 'ctl';
    controlBar.style.marginTop = '10px';
    controlBar.innerHTML = `
        <button id="btn-curve-type" class="ctl button on">Type: Cubic Bézier</button>
        <button id="btn-casteljau" class="ctl button on">De Casteljau: ON</button>
        <button id="btn-anim-curve" class="ctl button">Animate $C(t)$</button>
        <button id="btn-export-json" class="ctl button">Export JSON</button>
    `;
    container.appendChild(controlBar);

    let isBezier = true;
    const btnType = document.getElementById('btn-curve-type');
    btnType.addEventListener('click', () => {
        isBezier = !isBezier;
        btnType.textContent = isBezier ? 'Type: Cubic Bézier' : 'Type: Catmull-Rom';
        callbacks.onToggleType(isBezier ? 'bezier' : 'catmull');
    });

    let showCasteljau = true;
    const btnCas = document.getElementById('btn-casteljau');
    btnCas.addEventListener('click', () => {
        showCasteljau = !showCasteljau;
        btnCas.textContent = showCasteljau ? 'De Casteljau: ON' : 'De Casteljau: OFF';
        btnCas.classList.toggle('on', showCasteljau);
        callbacks.onToggleCasteljau(showCasteljau);
    });

    let isAnim = false;
    const btnAnim = document.getElementById('btn-anim-curve');
    btnAnim.addEventListener('click', () => {
        isAnim = !isAnim;
        btnAnim.classList.toggle('on', isAnim);
        callbacks.onToggleAnim(isAnim);
    });

    document.getElementById('btn-export-json').addEventListener('click', () => callbacks.onExportJSON());
}