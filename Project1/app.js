import { Mat4 } from './mat4.js';
import { SceneNode } from './scene.js';

// Camera & Math Helpers
function lookAt(eye, target, up) {
    const f = norm(sub(target, eye));
    const r = norm(cross(f, up));
    const u = cross(r, f);
    return new Float32Array([
        r[0], u[0], -f[0], 0,
        r[1], u[1], -f[1], 0,
        r[2], u[2], -f[2], 0,
        -dot(r, eye), -dot(u, eye), dot(f, eye), 1
    ]);
}

function perspective(fovY, aspect, near, far) {
    const f = 1.0 / Math.tan(fovY / 2);
    const rangeInv = 1.0 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, far * rangeInv, -1,
        0, 0, near * far * rangeInv, 0
    ]);
}

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ];
}
function norm(v) {
    const len = Math.hypot(...v) || 1;
    return [v[0] / len, v[1] / len, v[2] / len];
}

// 3D Cube Data
const CUBE_VERTICES = new Float32Array([
    // Front
    -0.5, -0.5,  0.5,  1, 0.2, 0.2,   0.5, -0.5,  0.5,  1, 0.2, 0.2,   0.5,  0.5,  0.5,  1, 0.2, 0.2,  -0.5,  0.5,  0.5,  1, 0.2, 0.2,
    // Back
    -0.5, -0.5, -0.5,  0.2, 0.4, 0.9, -0.5,  0.5, -0.5,  0.2, 0.4, 0.9,  0.5,  0.5, -0.5,  0.2, 0.4, 0.9,  0.5, -0.5, -0.5,  0.2, 0.4, 0.9,
    // Top
    -0.5,  0.5, -0.5,  0.2, 0.8, 0.3, -0.5,  0.5,  0.5,  0.2, 0.8, 0.3,  0.5,  0.5,  0.5,  0.2, 0.8, 0.3,  0.5,  0.5, -0.5,  0.2, 0.8, 0.3,
    // Bottom
    -0.5, -0.5, -0.5,  0.9, 0.8, 0.2,  0.5, -0.5, -0.5,  0.9, 0.8, 0.2,  0.5, -0.5,  0.5,  0.9, 0.8, 0.2, -0.5, -0.5,  0.5,  0.9, 0.8, 0.2,
    // Right
     0.5, -0.5, -0.5,  0.6, 0.3, 0.8,  0.5,  0.5, -0.5,  0.6, 0.3, 0.8,  0.5,  0.5,  0.5,  0.6, 0.3, 0.8,  0.5, -0.5,  0.5,  0.6, 0.3, 0.8,
    // Left
    -0.5, -0.5, -0.5,  0.2, 0.8, 0.8, -0.5, -0.5,  0.5,  0.2, 0.8, 0.8, -0.5,  0.5,  0.5,  0.2, 0.8, 0.8, -0.5,  0.5, -0.5,  0.2, 0.8, 0.8,
]);

const CUBE_INDICES = new Uint16Array([
     0,  1,  2,    0,  2,  3,
     4,  5,  6,    4,  6,  7,
     8,  9, 10,    8, 10, 11,
    12, 13, 14,   12, 14, 15,
    16, 17, 18,   16, 18, 19,
    20, 21, 22,   20, 22, 23,
]);

const SHADER_CODE = /* wgsl */ `
struct Uniforms {
    mvpMatrix : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) color : vec3<f32>,
};

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
};

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
`;

async function main() {
    if (!navigator.gpu) {
        alert("WebGPU is not supported.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const format = navigator.gpu.getPreferredCanvasFormat();
    const shaderModule = device.createShaderModule({ code: SHADER_CODE });

    const vertexBuffer = device.createBuffer({
        size: CUBE_VERTICES.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, CUBE_VERTICES);

    const indexBuffer = device.createBuffer({
        size: CUBE_INDICES.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, CUBE_INDICES);

    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: [{
                arrayStride: 24,
                attributes: [
                    { shaderLocation: 0, offset: 0, format: 'float32x3' },
                    { shaderLocation: 1, offset: 12, format: 'float32x3' }
                ]
            }]
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus'
        }
    });

    setupPartA(device, format, pipeline, vertexBuffer, indexBuffer);
    setupPartB(device, format, pipeline, vertexBuffer, indexBuffer);
}

// -------------------------------------------------------------
// PART A: Transformations
// -------------------------------------------------------------
function setupPartA(device, format, pipeline, vertexBuffer, indexBuffer) {
    const canvasTRS = document.getElementById('canvas-trs');
    const canvasSRT = document.getElementById('canvas-srt');
    if (!canvasTRS || !canvasSRT) return;

    const matTRSEl = document.getElementById('mat-trs');
    const matSRTEl = document.getElementById('mat-srt');

    const ctxTRS = canvasTRS.getContext('webgpu');
    const ctxSRT = canvasSRT.getContext('webgpu');
    ctxTRS.configure({ device, format, alphaMode: 'opaque' });
    ctxSRT.configure({ device, format, alphaMode: 'opaque' });

    const uniformBufferTRS = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const uniformBufferSRT = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const bindGroupTRS = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBufferTRS } }]
    });

    const bindGroupSRT = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBufferSRT } }]
    });

    const depthTextureTRS = device.createTexture({ size: [canvasTRS.width, canvasTRS.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const depthTextureSRT = device.createTexture({ size: [canvasSRT.width, canvasSRT.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });

    const state = { tx: 1.5, ry: 0.6, scale: 1.4 };

    function renderPartA() {
        const V = lookAt([3, 3, 5], [0, 0, 0], [0, 1, 0]);
        const P = perspective(Math.PI / 4, canvasTRS.width / canvasTRS.height, 0.1, 100.0);
        const VP = Mat4.multiply(P, V);

        const T = Mat4.translate(state.tx, 0, 0);
        const R = Mat4.rotateY(state.ry);
        const S = Mat4.scale(state.scale, state.scale, state.scale);

        const M_TRS = Mat4.multiply(T, Mat4.multiply(R, S));
        const M_SRT = Mat4.multiply(S, Mat4.multiply(R, T));

        device.queue.writeBuffer(uniformBufferTRS, 0, Mat4.multiply(VP, M_TRS));
        device.queue.writeBuffer(uniformBufferSRT, 0, Mat4.multiply(VP, M_SRT));

        if (matTRSEl) matTRSEl.textContent = Mat4.format(M_TRS);
        if (matSRTEl) matSRTEl.textContent = Mat4.format(M_SRT);

        drawPass(ctxTRS, depthTextureTRS, bindGroupTRS);
        drawPass(ctxSRT, depthTextureSRT, bindGroupSRT);
    }

    function drawPass(ctx, depthTexture, bindGroup) {
        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
        });
        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');
        renderPass.drawIndexed(CUBE_INDICES.length);
        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    setupPartAUI(state, renderPartA);
    renderPartA();
}

function setupPartAUI(state, onChange) {
    const container = document.getElementById('d-order');
    if (!container || document.getElementById('slider-tx')) return;

    const controlBar = document.createElement('div');
    controlBar.className = 'ctl';
    controlBar.style.marginTop = '15px';
    controlBar.innerHTML = `
        <label>Translate X: 
            <input type="range" id="slider-tx" min="-3" max="3" step="0.1" value="${state.tx}">
            <span class="val" id="val-tx">${state.tx}</span>
        </label>
        <label>Rotate Y: 
            <input type="range" id="slider-ry" min="-3.14" max="3.14" step="0.05" value="${state.ry}">
            <span class="val" id="val-ry">${state.ry}</span>
        </label>
        <label>Scale: 
            <input type="range" id="slider-s" min="0.3" max="2.5" step="0.05" value="${state.scale}">
            <span class="val" id="val-s">${state.scale}</span>
        </label>
    `;
    container.appendChild(controlBar);

    document.getElementById('slider-tx').addEventListener('input', (e) => { state.tx = parseFloat(e.target.value); document.getElementById('val-tx').textContent = state.tx.toFixed(1); onChange(); });
    document.getElementById('slider-ry').addEventListener('input', (e) => { state.ry = parseFloat(e.target.value); document.getElementById('val-ry').textContent = state.ry.toFixed(2); onChange(); });
    document.getElementById('slider-s').addEventListener('input', (e) => { state.scale = parseFloat(e.target.value); document.getElementById('val-s').textContent = state.scale.toFixed(2); onChange(); });
}

// -------------------------------------------------------------
// PART B: Scene Graph
// -------------------------------------------------------------
function setupPartB(device, format, pipeline, vertexBuffer, indexBuffer) {
    const canvas = document.getElementById('canvas-graph');
    const treeReadout = document.getElementById('tree-readout');
    if (!canvas) return;

    const ctx = canvas.getContext('webgpu');
    ctx.configure({ device, format, alphaMode: 'opaque' });

    const depthTexture = device.createTexture({ size: [canvas.width, canvas.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });

    // Scene Graph Tree: Root -> Base -> Arm1 -> Arm2 -> Hand
    const root = new SceneNode('root');
    const base = new SceneNode('base');
    const arm1 = new SceneNode('arm 1');
    const arm2 = new SceneNode('arm 2');
    const hand = new SceneNode('hand');

    root.addChild(base);
    base.addChild(arm1);
    arm1.addChild(arm2);
    arm2.addChild(hand);

    // Relative scales for visual presentation
    base.scale = [1.1, 0.3, 1.1];
    arm1.scale = [0.28, 1.2, 0.28];
    arm2.scale = [0.28, 1.0, 0.28];
    hand.scale = [0.5, 0.35, 0.5];

    const nodesToDraw = [
        { node: base, meshOffset: Mat4.identity ? Mat4.create() : Mat4.translate(0,0,0) },
        { node: arm1, meshOffset: Mat4.translate(0, 0.6, 0) },
        { node: arm2, meshOffset: Mat4.translate(0, 0.5, 0) },
        { node: hand, meshOffset: Mat4.translate(0, 0, 0) }
    ];

    // Create uniform buffers for drawing nodes
    nodesToDraw.forEach(item => {
        item.buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        item.bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: item.buffer } }]
        });
    });

    let isAnimated = false;
    let animAngle = 0;
    let a1 = 0.5, a2 = -0.8, a3 = 0.9;

    setupPartBUI(a1, a2, a3, (vals, anim) => {
        if (vals.a1 !== undefined) a1 = vals.a1;
        if (vals.a2 !== undefined) a2 = vals.a2;
        if (vals.a3 !== undefined) a3 = vals.a3;
        if (anim !== undefined) isAnimated = anim;
    });

    function renderPartB() {
        if (isAnimated) {
            animAngle += 0.02;
            a2 = Math.sin(animAngle) * 0.8;
            a3 = Math.cos(animAngle * 1.3) * 1.1;
            const sl2 = document.getElementById('slider-b-a2');
            const sl3 = document.getElementById('slider-b-a3');
            if (sl2) sl2.value = a2;
            if (sl3) sl3.value = a3;
        }

        base.localMatrix = Mat4.multiply(Mat4.translate(0, 0.15, 0), Mat4.rotateY(a1));
        arm1.localMatrix = Mat4.multiply(Mat4.translate(0, 0.15, 0), Mat4.rotateZ(a2));
        arm2.localMatrix = Mat4.multiply(Mat4.translate(0, 1.2, 0), Mat4.rotateZ(a3));
        hand.localMatrix = Mat4.translate(0, 1.0, 0);

        root.updateWorldMatrix(null);

        if (treeReadout) {
            const px = hand.worldMatrix[12].toFixed(2);
            const py = hand.worldMatrix[13].toFixed(2);
            const pz = hand.worldMatrix[14].toFixed(2);
            treeReadout.textContent = 
`root
└─ base      local: R_y(${a1.toFixed(2)})
   └─ arm 1  local: T(0,0.15,0)·R_z(${a2.toFixed(2)})
      └─ arm 2  local: T(0,1.2,0)·R_z(${a3.toFixed(2)})
         └─ hand  local: T(0,1.0,0)

world(hand) = M_base · M_arm1 · M_arm2 · M_hand

${Mat4.format(hand.worldMatrix)}
hand position in world:
  (${px}, ${py}, ${pz})`;
        }

        const V = lookAt([3, 3, 6], [0, 1.2, 0], [0, 1, 0]);
        const P = perspective(Math.PI / 4, canvas.width / canvas.height, 0.1, 100.0);
        const VP = Mat4.multiply(P, V);

        nodesToDraw.forEach(item => {
            const S = Mat4.scale(item.node.scale[0], item.node.scale[1], item.node.scale[2]);
            const finalWorld = Mat4.multiply(item.node.worldMatrix, Mat4.multiply(item.meshOffset, S));
            device.queue.writeBuffer(item.buffer, 0, Mat4.multiply(VP, finalWorld));
        });

        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
        });

        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');

        nodesToDraw.forEach(item => {
            renderPass.setBindGroup(0, item.bindGroup);
            renderPass.drawIndexed(CUBE_INDICES.length);
        });

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(renderPartB);
    }

    renderPartB();
}

function setupPartBUI(a1, a2, a3, onChange) {
    const container = document.getElementById('d-graph');
    if (!container || document.getElementById('slider-b-a1')) return;

    const controlBar = document.createElement('div');
    controlBar.className = 'ctl';
    controlBar.style.marginTop = '10px';
    controlBar.innerHTML = `
        <label>Base Rotation: 
            <input type="range" id="slider-b-a1" min="-3.14" max="3.14" step="0.02" value="${a1}">
        </label>
        <label>Joint 1: 
            <input type="range" id="slider-b-a2" min="-1.5" max="1.5" step="0.02" value="${a2}">
        </label>
        <label>Joint 2: 
            <input type="range" id="slider-b-a3" min="-1.5" max="1.5" step="0.02" value="${a3}">
        </label>
        <button id="btn-animate" class="ctl button">Animate</button>
    `;
    container.appendChild(controlBar);

    document.getElementById('slider-b-a1').addEventListener('input', (e) => onChange({ a1: parseFloat(e.target.value) }));
    document.getElementById('slider-b-a2').addEventListener('input', (e) => onChange({ a2: parseFloat(e.target.value) }));
    document.getElementById('slider-b-a3').addEventListener('input', (e) => onChange({ a3: parseFloat(e.target.value) }));

    let isAnim = false;
    const btn = document.getElementById('btn-animate');
    btn.addEventListener('click', () => {
        isAnim = !isAnim;
        btn.classList.toggle('on', isAnim);
        onChange({}, isAnim);
    });
}

main();