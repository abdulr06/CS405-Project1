import { Mat4 } from './mat4.js';

// Camera Helpers (WebGPU z in [0, 1])
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

// Vector math utilities
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

// 3D Cube Data (Position xyz, Color rgb)
const CUBE_VERTICES = new Float32Array([
    // Front face (Red)
    -0.5, -0.5,  0.5,  1, 0.2, 0.2,
     0.5, -0.5,  0.5,  1, 0.2, 0.2,
     0.5,  0.5,  0.5,  1, 0.2, 0.2,
    -0.5,  0.5,  0.5,  1, 0.2, 0.2,
    // Back face (Blue)
    -0.5, -0.5, -0.5,  0.2, 0.4, 0.9,
    -0.5,  0.5, -0.5,  0.2, 0.4, 0.9,
     0.5,  0.5, -0.5,  0.2, 0.4, 0.9,
     0.5, -0.5, -0.5,  0.2, 0.4, 0.9,
    // Top face (Green)
    -0.5,  0.5, -0.5,  0.2, 0.8, 0.3,
    -0.5,  0.5,  0.5,  0.2, 0.8, 0.3,
     0.5,  0.5,  0.5,  0.2, 0.8, 0.3,
     0.5,  0.5, -0.5,  0.2, 0.8, 0.3,
    // Bottom face (Yellow)
    -0.5, -0.5, -0.5,  0.9, 0.8, 0.2,
     0.5, -0.5, -0.5,  0.9, 0.8, 0.2,
     0.5, -0.5,  0.5,  0.9, 0.8, 0.2,
    -0.5, -0.5,  0.5,  0.9, 0.8, 0.2,
    // Right face (Purple)
     0.5, -0.5, -0.5,  0.6, 0.3, 0.8,
     0.5,  0.5, -0.5,  0.6, 0.3, 0.8,
     0.5,  0.5,  0.5,  0.6, 0.3, 0.8,
     0.5, -0.5,  0.5,  0.6, 0.3, 0.8,
    // Left face (Cyan)
    -0.5, -0.5, -0.5,  0.2, 0.8, 0.8,
    -0.5, -0.5,  0.5,  0.2, 0.8, 0.8,
    -0.5,  0.5,  0.5,  0.2, 0.8, 0.8,
    -0.5,  0.5, -0.5,  0.2, 0.8, 0.8,
]);

const CUBE_INDICES = new Uint16Array([
     0,  1,  2,    0,  2,  3,  // Front
     4,  5,  6,    4,  6,  7,  // Back
     8,  9, 10,    8, 10, 11,  // Top
    12, 13, 14,   12, 14, 15,  // Bottom
    16, 17, 18,   16, 18, 19,  // Right
    20, 21, 22,   20, 22, 23,  // Left
]);

// WGSL Shaders
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
        alert("WebGPU is not supported in this browser.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const canvasTRS = document.getElementById('canvas-trs');
    const canvasSRT = document.getElementById('canvas-srt');
    const matTRSEl = document.getElementById('mat-trs');
    const matSRTEl = document.getElementById('mat-srt');

    const format = navigator.gpu.getPreferredCanvasFormat();
    
    const ctxTRS = canvasTRS.getContext('webgpu');
    const ctxSRT = canvasSRT.getContext('webgpu');
    
    ctxTRS.configure({ device, format, alphaMode: 'opaque' });
    ctxSRT.configure({ device, format, alphaMode: 'opaque' });

    // Create GPU Buffers
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

    // Uniform Buffers for both canvases
    const uniformBufferTRS = device.createBuffer({
        size: 64, // 16 floats * 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformBufferSRT = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Pipeline Setup
    const shaderModule = device.createShaderModule({ code: SHADER_CODE });

    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: [{
                arrayStride: 24, // 6 floats * 4 bytes
                attributes: [
                    { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
                    { shaderLocation: 1, offset: 12, format: 'float32x3' }  // color
                ]
            }]
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fs_main',
            targets: [{ format }]
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus'
        }
    });

    const bindGroupTRS = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBufferTRS } }]
    });

    const bindGroupSRT = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBufferSRT } }]
    });

    // Create Depth Textures
    const depthTextureTRS = device.createTexture({
        size: [canvasTRS.width, canvasTRS.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const depthTextureSRT = device.createTexture({
        size: [canvasSRT.width, canvasSRT.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Transformation Parameters
    const state = { tx: 1.5, ry: 0.6, scale: 1.4 };

    // Build UI Controls
    setupUI(state, render);

    function render() {
        // Camera View & Projection
        const eye = [3, 3, 5];
        const target = [0, 0, 0];
        const up = [0, 1, 0];
        const V = lookAt(eye, target, up);
        const P = perspective(Math.PI / 4, canvasTRS.width / canvasTRS.height, 0.1, 100.0);
        const VP = Mat4.multiply(P, V);

        // Transformation Matrices
        const T = Mat4.translate(state.tx, 0, 0);
        const R = Mat4.rotateY(state.ry);
        const S = Mat4.scale(state.scale, state.scale, state.scale);

        // Order 1: M = T * R * S (Scale first)
        const M_TRS = Mat4.multiply(T, Mat4.multiply(R, S));
        const MVP_TRS = Mat4.multiply(VP, M_TRS);

        // Order 2: M = S * R * T (Translate first, then scale)
        const M_SRT = Mat4.multiply(S, Mat4.multiply(R, T));
        const MVP_SRT = Mat4.multiply(VP, M_SRT);

        // Update Uniform Buffers
        device.queue.writeBuffer(uniformBufferTRS, 0, MVP_TRS);
        device.queue.writeBuffer(uniformBufferSRT, 0, MVP_SRT);

        // Update Matrix Text Display
        if (matTRSEl) matTRSEl.textContent = Mat4.format(M_TRS);
        if (matSRTEl) matSRTEl.textContent = Mat4.format(M_SRT);

        // Render Canvas 1 (TRS)
        drawPass(ctxTRS, depthTextureTRS, bindGroupTRS);

        // Render Canvas 2 (SRT)
        drawPass(ctxSRT, depthTextureSRT, bindGroupSRT);
    }

    function drawPass(ctx, depthTexture, bindGroup) {
        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: ctx.getCurrentTexture().createView(),
                clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            }
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');
        renderPass.drawIndexed(CUBE_INDICES.length);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    // Initial Draw
    render();
}

function setupUI(state, onChange) {
    const container = document.getElementById('d-order');
    if (!container) return;

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

    document.getElementById('slider-tx').addEventListener('input', (e) => {
        state.tx = parseFloat(e.target.value);
        document.getElementById('val-tx').textContent = state.tx.toFixed(1);
        onChange();
    });

    document.getElementById('slider-ry').addEventListener('input', (e) => {
        state.ry = parseFloat(e.target.value);
        document.getElementById('val-ry').textContent = state.ry.toFixed(2);
        onChange();
    });

    document.getElementById('slider-s').addEventListener('input', (e) => {
        state.scale = parseFloat(e.target.value);
        document.getElementById('val-s').textContent = state.scale.toFixed(2);
        onChange();
    });
}

main();