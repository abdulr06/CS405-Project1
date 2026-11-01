import { Mat4 } from '../math/mat4.js';
import { SceneNode } from '../scene/scene.js';
import { setupPartAUI, setupPartBUI, setupPartCUI, setupPartDUI } from '../ui/ui.js';

// -------------------------------------------------------------
// Math & Camera Helpers
// -------------------------------------------------------------
function lookAt(eye, target, up) {
    return Mat4.lookAt(eye, target, up);
}

function perspective(fovY, aspect, near, far) {
    return Mat4.perspective(fovY, aspect, near, far);
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

// -------------------------------------------------------------
// Geometry Data
// -------------------------------------------------------------
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

function createGroundGrid(size = 5, divisions = 10) {
    const verts = [];
    const step = (size * 2) / divisions;
    for (let i = -size; i <= size; i += step) {
        verts.push(-size, 0, i,  size, 0, i);
        verts.push(i, 0, -size,  i, 0, size);
    }
    return new Float32Array(verts);
}

// -------------------------------------------------------------
// WGSL Shaders
// -------------------------------------------------------------
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

const LINE_SHADER = /* wgsl */ `
struct Uniforms {
    mvpMatrix : mat4x4<f32>,
    color : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn vs_main(@location(0) position : vec3<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.mvpMatrix * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return uniforms.color;
}
`;

const TEXTURE_SHADER = /* wgsl */ `
struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0)
    );
    var uv = array<vec2<f32>, 4>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0)
    );

    var output : VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.99, 1.0);
    output.uv = uv[vertexIndex];
    return output;
}

@group(0) @binding(0) var texSampler : sampler;
@group(0) @binding(1) var texData : texture_2d<f32>;

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(texData, texSampler, input.uv);
}
`;

const CURVE_SHADER = /* wgsl */ `
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

// -------------------------------------------------------------
// Main Function
// -------------------------------------------------------------
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
    setupPartC(device, format);
    await setupPartD(device, format);
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

    // Helper to generate distinct buffers and bind groups for multiple objects
    const createObjectData = () => {
        const bufferTRS = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferSRT = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        return {
            bufferTRS,
            bufferSRT,
            bgTRS: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bufferTRS } }] }),
            bgSRT: device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: bufferSRT } }] })
        };
    };

    // Object 1: UI Controlled
    // Object 2 & 3: Static background geometry to satisfy the 3-object requirement
    const objects = [
        { ...createObjectData(), isUI: true },
        { ...createObjectData(), isUI: false, tx: 2.0, ty: 1.0, tz: -2.0, ry: 0.5, scale: 0.6 },
        { ...createObjectData(), isUI: false, tx: -1.5, ty: -0.5, tz: 1.0, ry: -0.8, scale: 0.4 }
    ];

    const depthTextureTRS = device.createTexture({ size: [canvasTRS.width, canvasTRS.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const depthTextureSRT = device.createTexture({ size: [canvasSRT.width, canvasSRT.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });

    const state = { tx: -2.0, ry: 1.16, scale: 0.95 };

    function renderPartA() {
        const V = lookAt([3, 3, 5], [0, 0, 0], [0, 1, 0]);
        const P = perspective(Math.PI / 4, canvasTRS.width / canvasTRS.height, 0.1, 100.0);
        const VP = Mat4.multiply(P, V);

        objects.forEach((obj) => {
            const tx = obj.isUI ? state.tx : obj.tx;
            const ty = obj.isUI ? 0 : obj.ty;
            const tz = obj.isUI ? 0 : obj.tz;
            const ry = obj.isUI ? state.ry : obj.ry;
            const s = obj.isUI ? state.scale : obj.scale;

            const T = Mat4.translate(tx, ty, tz);
            const R = Mat4.rotateY(ry);
            const S = Mat4.scale(s, s, s);

            const M_TRS = Mat4.multiply(T, Mat4.multiply(R, S));
            const M_SRT = Mat4.multiply(S, Mat4.multiply(R, T));

            device.queue.writeBuffer(obj.bufferTRS, 0, Mat4.multiply(VP, M_TRS));
            device.queue.writeBuffer(obj.bufferSRT, 0, Mat4.multiply(VP, M_SRT));

            // Only update the matrix readout panel for the object actively controlled by the UI
            if (obj.isUI) {
                if (matTRSEl) matTRSEl.textContent = Mat4.format(M_TRS);
                if (matSRTEl) matSRTEl.textContent = Mat4.format(M_SRT);
            }
        });

        drawPass(ctxTRS, depthTextureTRS, objects.map(o => o.bgTRS));
        drawPass(ctxSRT, depthTextureSRT, objects.map(o => o.bgSRT));
    }

    function drawPass(ctx, depthTexture, bindGroups) {
        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
        });
        
        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');
        
        // Dispatch a draw call for every bind group in the array
        bindGroups.forEach(bg => {
            renderPass.setBindGroup(0, bg);
            renderPass.drawIndexed(CUBE_INDICES.length);
        });
        
        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    setupPartAUI(state, renderPartA);
    renderPartA();
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

    // --- NEW: Line Pipeline for Debug Axes ---
    const shaderModule = device.createShaderModule({ code: SHADER_CODE });
    const linePipeline = device.createRenderPipeline({
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
        fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'line-list' },
        depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' }
    });

    // X (Red), Y (Green), Z (Blue) axes geometry
    const AXES_VERTICES = new Float32Array([
        0,0,0, 1,0.2,0.2,   1.5,0,0, 1,0.2,0.2,
        0,0,0, 0.2,1,0.2,   0,1.5,0, 0.2,1,0.2,
        0,0,0, 0.2,0.2,1,   0,0,1.5, 0.2,0.2,1
    ]);
    const axesVertexBuffer = device.createBuffer({
        size: AXES_VERTICES.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(axesVertexBuffer, 0, AXES_VERTICES);

    const root = new SceneNode('root');
    const base = new SceneNode('base');
    const arm1 = new SceneNode('arm 1');
    const arm2 = new SceneNode('arm 2');
    const hand = new SceneNode('hand');

    root.addChild(base);
    base.addChild(arm1);
    arm1.addChild(arm2);
    arm2.addChild(hand);

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

    nodesToDraw.forEach(item => {
        // Buffer for the mesh (Cubes)
        item.buffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        item.bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: item.buffer } }]
        });
        
        // --- NEW: Buffer for the local axes ---
        item.axesBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        item.axesBindGroup = device.createBindGroup({
            layout: linePipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: item.axesBuffer } }]
        });
    });

    let isAnimated = false;
    let showAxes = false; // State for debug axes
    let animAngle = 0;
    let a1 = 3.14, a2 = 0.71, a3 = -0.77;

    // Hook up the imported setupPartBUI
    setupPartBUI({ a1, a2, a3 }, (vals) => {
        if (vals.a1 !== undefined) a1 = vals.a1;
        if (vals.a2 !== undefined) a2 = vals.a2;
        if (vals.a3 !== undefined) a3 = vals.a3;
        if (vals.isAnimated !== undefined) isAnimated = vals.isAnimated;
        if (vals.showAxes !== undefined) showAxes = vals.showAxes;
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

        // Update uniform buffers
        nodesToDraw.forEach(item => {
            const S = Mat4.scale(item.node.scale[0], item.node.scale[1], item.node.scale[2]);
            const finalWorld = Mat4.multiply(item.node.worldMatrix, Mat4.multiply(item.meshOffset, S));
            
            device.queue.writeBuffer(item.buffer, 0, Mat4.multiply(VP, finalWorld));
            
            if (showAxes) {
                // Apply ONLY the worldMatrix to axes to represent local origin/orientation
                device.queue.writeBuffer(item.axesBuffer, 0, Mat4.multiply(VP, item.node.worldMatrix));
            }
        });

        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
        });

        // 1. Draw Cubes
        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.setIndexBuffer(indexBuffer, 'uint16');
        nodesToDraw.forEach(item => {
            renderPass.setBindGroup(0, item.bindGroup);
            renderPass.drawIndexed(CUBE_INDICES.length);
        });

        // 2. Draw Debug Axes
        if (showAxes) {
            renderPass.setPipeline(linePipeline);
            renderPass.setVertexBuffer(0, axesVertexBuffer);
            nodesToDraw.forEach(item => {
                renderPass.setBindGroup(0, item.axesBindGroup);
                renderPass.draw(6); // 6 vertices = 3 lines
            });
        }

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(renderPartB);
    }

    renderPartB();
}


// -------------------------------------------------------------
// PART C: Camera and Projection
// -------------------------------------------------------------
function setupPartC(device, format) {
    const canvasWorld = document.getElementById('canvas-cam-world');
    const canvasView = document.getElementById('canvas-cam-view');
    const matInfo = document.getElementById('mat-camera');
    if (!canvasWorld || !canvasView) return;

    const ctxWorld = canvasWorld.getContext('webgpu');
    const ctxView = canvasView.getContext('webgpu');
    ctxWorld.configure({ device, format, alphaMode: 'opaque' });
    ctxView.configure({ device, format, alphaMode: 'opaque' });

    const shaderModule = device.createShaderModule({ code: SHADER_CODE });
    const lineShaderModule = device.createShaderModule({ code: LINE_SHADER });

    const cubePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vs_main',
            buffers: [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }]
        },
        fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' }
    });

    const linePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: lineShaderModule, entryPoint: 'vs_main', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
        fragment: { module: lineShaderModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'line-list' },
        depthStencil: { depthWriteEnabled: true, depthCompare: 'less', format: 'depth24plus' }
    });

    const cubeVB = device.createBuffer({ size: CUBE_VERTICES.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(cubeVB, 0, CUBE_VERTICES);

    const cubeIB = device.createBuffer({ size: CUBE_INDICES.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(cubeIB, 0, CUBE_INDICES);

    const gridData = createGroundGrid(4, 8);
    const gridVB = device.createBuffer({ size: gridData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(gridVB, 0, gridData);

    const lineUB = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const lineBG = device.createBindGroup({ layout: linePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: lineUB } }] });

    const depthWorld = device.createTexture({ size: [canvasWorld.width, canvasWorld.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const depthView = device.createTexture({ size: [canvasView.width, canvasView.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });

    const boxes = [
        { pos: [-2, 0.5, -1], scale: [1.0, 1.0, 1.0] },
        { pos: [0.6, 0.8, 0.4], scale: [1.6, 1.6, 1.6] },
        { pos: [2.2, 0.4, -1.6], scale: [0.8, 0.8, 0.8] }
    ];

    boxes.forEach(b => {
        b.bufView = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        b.bgView = device.createBindGroup({ layout: cubePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b.bufView } }] });

        b.bufWorld = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        b.bgWorld = device.createBindGroup({ layout: cubePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: b.bufWorld } }] });
    });

    const cam = {
        cyaw: 0.5, cpitch: 0.3, cdist: 6.5,
        fov: 50, near: 1.0, far: 9.0, isPersp: true,
        wyaw: 0.8, wpitch: 0.45, wdist: 12.0
    };

    let isDragging = false, lastX = 0, lastY = 0;
    canvasWorld.addEventListener('pointerdown', (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
    window.addEventListener('pointermove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        cam.wyaw -= dx * 0.01;
        cam.wpitch = Math.max(-1.3, Math.min(1.35, cam.wpitch + dy * 0.008));
        lastX = e.clientX; lastY = e.clientY;
        renderPartC();
    });
    window.addEventListener('pointerup', () => { isDragging = false; });

    setupPartCUI(cam, renderPartC);

    function getCamEye() {
        return [
            cam.cdist * Math.cos(cam.cpitch) * Math.sin(cam.cyaw),
            0.5 + cam.cdist * Math.sin(cam.cpitch),
            cam.cdist * Math.cos(cam.cpitch) * Math.cos(cam.cyaw)
        ];
    }

    function renderPartC() {
        const eye = getCamEye();
        const target = [0, 0.5, 0];
        const up = [0, 1, 0];

        const Vc = Mat4.lookAt(eye, target, up);
        const aspect = canvasView.width / canvasView.height;
        const Pc = cam.isPersp ? 
            Mat4.perspective(cam.fov * Math.PI / 180, aspect, cam.near, cam.far) : 
            Mat4.ortho(4, aspect, cam.near, cam.far);

        const worldEye = [
            cam.wdist * Math.cos(cam.wpitch) * Math.sin(cam.wyaw),
            1.5 + cam.wdist * Math.sin(cam.wpitch),
            cam.wdist * Math.cos(cam.wpitch) * Math.cos(cam.wyaw)
        ];
        const Vw = Mat4.lookAt(worldEye, [0, 0.5, 0], [0, 1, 0]);
        const Pw = Mat4.perspective(0.8, aspect, 0.1, 60);
        const VPw = Mat4.multiply(Pw, Vw);

        if (matInfo) {
            matInfo.textContent = 
`eye: (${eye[0].toFixed(2)}, ${eye[1].toFixed(2)}, ${eye[2].toFixed(2)})
target: (${target[0].toFixed(2)}, ${target[1].toFixed(2)}, ${target[2].toFixed(2)})
V = lookAt(eye, target, up)
${Mat4.format(Vc)}
${cam.isPersp ? `P = perspective(${cam.fov}°, aspect, ${cam.near}, ${cam.far})` : `P = orthographic(height, aspect, ${cam.near}, ${cam.far})`}
${Mat4.format(Pc)}`;
        }

        const VPc = Mat4.multiply(Pc, Vc);

        drawScene(ctxView, depthView, VPc, false);
        drawWorldView(ctxWorld, depthWorld, VPw, eye, target, up, aspect);
    }

    function drawScene(ctx, depthTexture, VP, isWorldPass) {
        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTexture.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
        });

        renderPass.setPipeline(linePipeline);
        renderPass.setVertexBuffer(0, gridVB);
        const lineUniforms = new Float32Array(20);
        lineUniforms.set(VP, 0);
        lineUniforms.set([0.3, 0.3, 0.35, 1.0], 16);
        device.queue.writeBuffer(lineUB, 0, lineUniforms);
        renderPass.setBindGroup(0, lineBG);
        renderPass.draw(gridData.length / 3);

        renderPass.setPipeline(cubePipeline);
        renderPass.setVertexBuffer(0, cubeVB);
        renderPass.setIndexBuffer(cubeIB, 'uint16');

        boxes.forEach(b => {
            const M = Mat4.multiply(Mat4.translate(...b.pos), Mat4.scale(...b.scale));
            const buf = isWorldPass ? b.bufWorld : b.bufView;
            const bg = isWorldPass ? b.bgWorld : b.bgView;

            device.queue.writeBuffer(buf, 0, Mat4.multiply(VP, M));
            renderPass.setBindGroup(0, bg);
            renderPass.drawIndexed(CUBE_INDICES.length);
        });

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    function drawWorldView(ctx, depthTexture, VPw, eye, target, up, aspect) {
        drawScene(ctx, depthTexture, VPw, true);

        const f = norm(sub(target, eye));
        const r = norm(cross(f, up));
        const u = cross(r, f);

        const corner = (d, sx, sy) => {
            const hh = cam.isPersp ? Math.tan(cam.fov * Math.PI / 360) * d : 2.0;
            const hw = hh * aspect;
            return [
                eye[0] + f[0] * d + r[0] * sx * hw + u[0] * sy * hh,
                eye[1] + f[1] * d + r[1] * sx * hw + u[1] * sy * hh,
                eye[2] + f[2] * d + r[2] * sx * hw + u[2] * sy * hh
            ];
        };

        const nr = [corner(cam.near, -1, -1), corner(cam.near, 1, -1), corner(cam.near, 1, 1), corner(cam.near, -1, 1)];
        const fr = [corner(cam.far, -1, -1), corner(cam.far, 1, -1), corner(cam.far, 1, 1), corner(cam.far, -1, 1)];

        const frustumLines = new Float32Array([
            ...nr[0], ...nr[1],  ...nr[1], ...nr[2],  ...nr[2], ...nr[3],  ...nr[3], ...nr[0],
            ...fr[0], ...fr[1],  ...fr[1], ...fr[2],  ...fr[2], ...fr[3],  ...fr[3], ...fr[0],
            ...nr[0], ...fr[0],  ...nr[1], ...fr[1],  ...nr[2], ...fr[2],  ...nr[3], ...fr[3],
            ...eye, ...target
        ]);

        const lineVB = device.createBuffer({ size: frustumLines.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(lineVB, 0, new Float32Array(frustumLines));

        const frustumUniforms = new Float32Array(20);
        frustumUniforms.set(VPw, 0);
        frustumUniforms.set([0.85, 0.35, 0.15, 1.0], 16);
        device.queue.writeBuffer(lineUB, 0, frustumUniforms);

        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'load', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTexture.createView(), depthLoadOp: 'load', depthStoreOp: 'store' }
        });

        renderPass.setPipeline(linePipeline);
        renderPass.setVertexBuffer(0, lineVB);
        renderPass.setBindGroup(0, lineBG);
        renderPass.draw(frustumLines.length / 3);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
    }

    renderPartC();
}


// -------------------------------------------------------------
// PART D: Curves Over Painting
// -------------------------------------------------------------
function evalBezier(p0, p1, p2, p3, t) {
    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    const uuu = uu * u;
    const ttt = tt * t;

    return [
        uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0],
        uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
    ];
}

function evalBezierTangent(p0, p1, p2, p3, t) {
    const u = 1 - t;
    return [
        3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0]),
        3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
    ];
}

function evalCatmullRom(p0, p1, p2, p3, t) {
    const t2 = t * t;
    const t3 = t2 * t;

    const f0 = -0.5 * t3 + t2 - 0.5 * t;
    const f1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    const f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    const f3 = 0.5 * t3 - 0.5 * t2;

    return [
        p0[0] * f0 + p1[0] * f1 + p2[0] * f2 + p3[0] * f3,
        p0[1] * f0 + p1[1] * f1 + p2[1] * f2 + p3[1] * f3
    ];
}

async function setupPartD(device, format) {
    const canvas = document.getElementById('canvas-curves');
    const jsonOut = document.getElementById('json-readout');
    if (!canvas) return;

    const ctx = canvas.getContext('webgpu');
    ctx.configure({ device, format, alphaMode: 'opaque' });

    let texture;
    if (window.KAVUNLU_SRC) {
        const img = new Image();
        img.src = window.KAVUNLU_SRC;
        await img.decode();
        const bitmap = await createImageBitmap(img);

        texture = device.createTexture({
            size: [bitmap.width, bitmap.height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        device.queue.copyExternalImageToTexture(
            { source: bitmap },
            { texture: texture },
            [bitmap.width, bitmap.height]
        );
    }

    const texShaderModule = device.createShaderModule({ code: TEXTURE_SHADER });
    const texPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: texShaderModule, entryPoint: 'vs_main' },
        fragment: { module: texShaderModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'triangle-strip' },
        depthStencil: { depthWriteEnabled: true, depthCompare: 'less-equal', format: 'depth24plus' }
    });

    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    let texBG;
    if (texture) {
        texBG = device.createBindGroup({
            layout: texPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: sampler },
                { binding: 1, resource: texture.createView() }
            ]
        });
    }

    const rawPoints = [
        [0.10, 0.75], [0.22, 0.30], [0.38, 0.25], [0.50, 0.60],
        [0.62, 0.95], [0.78, 0.70], [0.90, 0.35]
    ];

    let curveType = 'bezier';
    let showCasteljau = true;
    let isAnimating = false;
    let animT = 0;
    let draggedIdx = -1;

    function enforceC1() {
        rawPoints[4][0] = 2 * rawPoints[3][0] - rawPoints[2][0];
        rawPoints[4][1] = 2 * rawPoints[3][1] - rawPoints[2][1];
    }
    enforceC1();

    setupPartDUI({
        onToggleType: (t) => { curveType = t; renderPartD(); },
        onToggleCasteljau: (v) => { showCasteljau = v; renderPartD(); },
        onToggleAnim: (v) => { 
            isAnimating = v; 
            if (isAnimating) requestAnimationFrame(animLoop);
            else renderPartD();
        },
        onExportJSON: () => exportJSON()
    });

    canvas.addEventListener('pointerdown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) / canvas.width;
        const my = (e.clientY - rect.top) / canvas.height;

        let minDist = 0.08;
        draggedIdx = -1;
        rawPoints.forEach((p, i) => {
            const d = Math.hypot(p[0] - mx, p[1] - my);
            if (d < minDist) {
                minDist = d;
                draggedIdx = i;
            }
        });
    });

    window.addEventListener('pointermove', (e) => {
        if (draggedIdx < 0) return;
        const rect = canvas.getBoundingClientRect();
        const mx = Math.max(0, Math.min(1, (e.clientX - rect.left) / canvas.width));
        const my = Math.max(0, Math.min(1, (e.clientY - rect.top) / canvas.height));

        const dx = mx - rawPoints[draggedIdx][0];
        const dy = my - rawPoints[draggedIdx][1];

        rawPoints[draggedIdx] = [mx, my];

        if (draggedIdx === 3) {
            rawPoints[2][0] += dx; rawPoints[2][1] += dy;
            rawPoints[4][0] += dx; rawPoints[4][1] += dy;
        } else if (draggedIdx === 2) {
            enforceC1();
        } else if (draggedIdx === 4) {
            rawPoints[2][0] = 2 * rawPoints[3][0] - rawPoints[4][0];
            rawPoints[2][1] = 2 * rawPoints[3][1] - rawPoints[4][1];
        }

        exportJSON();
        renderPartD();
    });

    window.addEventListener('pointerup', () => { draggedIdx = -1; });

    function exportJSON() {
        const data = {
            type: curveType,
            c1_continuous: true,
            control_points: rawPoints.map(p => ({ x: Number(p[0].toFixed(3)), y: Number(p[1].toFixed(3)) }))
        };
        if (jsonOut) jsonOut.textContent = JSON.stringify(data, null, 2);
    }
    exportJSON();

    const curveShaderModule = device.createShaderModule({ code: CURVE_SHADER });
    const curvePipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { 
            module: curveShaderModule, 
            entryPoint: 'vs_main', 
            buffers: [{ 
                arrayStride: 24, 
                attributes: [
                    { shaderLocation: 0, offset: 0, format: 'float32x3' },
                    { shaderLocation: 1, offset: 12, format: 'float32x3' }
                ] 
            }] 
        },
        fragment: { module: curveShaderModule, entryPoint: 'fs_main', targets: [{ format }] },
        primitive: { topology: 'line-list' },
        depthStencil: { depthWriteEnabled: false, depthCompare: 'always', format: 'depth24plus' }
    });

    const depthTex = device.createTexture({ size: [canvas.width, canvas.height], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });

    function animLoop() {
        if (!isAnimating) return;
        animT = (animT + 0.008) % 2.0;
        renderPartD();
        requestAnimationFrame(animLoop);
    }

    function renderPartD() {
        const lineVerts = [];
        const toNDC = (p, c) => [p[0] * 2 - 1, 1 - p[1] * 2, 0, c[0], c[1], c[2]];

        const colCtrl = [0.6, 0.6, 0.6];       // Gray handles & polygon
        const colCas1 = [0.2, 0.8, 0.9];       // Cyan level 1
        const colCas2 = [0.9, 0.3, 0.8];       // Magenta level 2
        const colCurve = [0.95, 0.75, 0.2];    // Gold main curve
        const colTan = [0.2, 0.9, 0.3];        // Green tangent marker

        const aspect = canvas.width / canvas.height;

        const addHandleMarker = (p, isAnchor = false) => {
            const rx = isAnchor ? 0.014 : 0.010;
            const ry = rx * aspect;

            const c0 = [p[0] - rx, p[1] - ry];
            const c1 = [p[0] + rx, p[1] - ry];
            const c2 = [p[0] + rx, p[1] + ry];
            const c3 = [p[0] - rx, p[1] + ry];

            lineVerts.push(...toNDC(c0, colCtrl), ...toNDC(c1, colCtrl));
            lineVerts.push(...toNDC(c1, colCtrl), ...toNDC(c2, colCtrl));
            lineVerts.push(...toNDC(c2, colCtrl), ...toNDC(c3, colCtrl));
            lineVerts.push(...toNDC(c3, colCtrl), ...toNDC(c0, colCtrl));
            lineVerts.push(...toNDC([p[0] - rx, p[1]], colCtrl), ...toNDC([p[0] + rx, p[1]], colCtrl));
            lineVerts.push(...toNDC([p[0], p[1] - ry], colCtrl), ...toNDC([p[0], p[1] + ry], colCtrl));
        };

        // 1. Control Polygon & Handles
        for (let i = 0; i < rawPoints.length - 1; i++) {
            lineVerts.push(...toNDC(rawPoints[i], colCtrl), ...toNDC(rawPoints[i + 1], colCtrl));
        }

        rawPoints.forEach((p, idx) => {
            const isAnchor = (idx === 0 || idx === 3 || idx === 6);
            addHandleMarker(p, isAnchor);
        });

        // 2. Curve Sampling
        const samples = 60;
        if (curveType === 'bezier') {
            let prev = evalBezier(rawPoints[0], rawPoints[1], rawPoints[2], rawPoints[3], 0);
            for (let i = 1; i <= samples; i++) {
                const curr = evalBezier(rawPoints[0], rawPoints[1], rawPoints[2], rawPoints[3], i / samples);
                lineVerts.push(...toNDC(prev, colCurve), ...toNDC(curr, colCurve));
                prev = curr;
            }
            prev = evalBezier(rawPoints[3], rawPoints[4], rawPoints[5], rawPoints[6], 0);
            for (let i = 1; i <= samples; i++) {
                const curr = evalBezier(rawPoints[3], rawPoints[4], rawPoints[5], rawPoints[6], i / samples);
                lineVerts.push(...toNDC(prev, colCurve), ...toNDC(curr, colCurve));
                prev = curr;
            }
        } else {
            let prev = evalCatmullRom(rawPoints[0], rawPoints[0], rawPoints[1], rawPoints[2], 0);
            for (let i = 1; i <= samples * 2; i++) {
                const segT = (i / (samples * 2)) * 2;
                const curr = segT <= 1 ?
                    evalCatmullRom(rawPoints[0], rawPoints[1], rawPoints[2], rawPoints[3], segT) :
                    evalCatmullRom(rawPoints[1], rawPoints[2], rawPoints[3], rawPoints[4], segT - 1);
                lineVerts.push(...toNDC(prev, colCurve), ...toNDC(curr, colCurve));
                prev = curr;
            }
        }

        // 3. Marker & Tangent Vector
        const segIndex = animT < 1.0 ? 0 : 1;
        const localT = animT % 1.0;
        let pos, tan;

        if (segIndex === 0) {
            pos = evalBezier(rawPoints[0], rawPoints[1], rawPoints[2], rawPoints[3], localT);
            tan = evalBezierTangent(rawPoints[0], rawPoints[1], rawPoints[2], rawPoints[3], localT);
        } else {
            pos = evalBezier(rawPoints[3], rawPoints[4], rawPoints[5], rawPoints[6], localT);
            tan = evalBezierTangent(rawPoints[3], rawPoints[4], rawPoints[5], rawPoints[6], localT);
        }

        const normTan = norm([tan[0], tan[1], 0]);
        const tanTip = [pos[0] + normTan[0] * 0.08, pos[1] + normTan[1] * 0.08];
        lineVerts.push(...toNDC(pos, colTan), ...toNDC(tanTip, colTan));

        // 4. De Casteljau Lines
        if (showCasteljau && curveType === 'bezier') {
            const p = segIndex === 0 ?
                [rawPoints[0], rawPoints[1], rawPoints[2], rawPoints[3]] :
                [rawPoints[3], rawPoints[4], rawPoints[5], rawPoints[6]];

            const q0 = [p[0][0] + localT * (p[1][0] - p[0][0]), p[0][1] + localT * (p[1][1] - p[0][1])];
            const q1 = [p[1][0] + localT * (p[2][0] - p[1][0]), p[1][1] + localT * (p[2][1] - p[1][1])];
            const q2 = [p[2][0] + localT * (p[3][0] - p[2][0]), p[2][1] + localT * (p[3][1] - p[2][1])];

            const r0 = [q0[0] + localT * (q1[0] - q0[0]), q0[1] + localT * (q1[1] - q0[1])];
            const r1 = [q1[0] + localT * (q2[0] - q1[0]), q1[1] + localT * (q2[1] - q1[1])];

            lineVerts.push(...toNDC(q0, colCas1), ...toNDC(q1, colCas1), ...toNDC(q1, colCas1), ...toNDC(q2, colCas1));
            lineVerts.push(...toNDC(r0, colCas2), ...toNDC(r1, colCas2));
        }

        const lineVB = device.createBuffer({ size: Math.max(128, lineVerts.length * 4), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(lineVB, 0, new Float32Array(lineVerts));

        const curveUB = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const curveBG = device.createBindGroup({ layout: curvePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: curveUB } }] });

        const curveUniforms = new Float32Array(16);
        curveUniforms.set(Mat4.create(), 0);
        device.queue.writeBuffer(curveUB, 0, curveUniforms);

        const commandEncoder = device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0.1, g: 0.1, b: 0.12, a: 1.0 }, loadOp: 'clear', storeOp: 'store' }],
            depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'store' }
        });

        if (texBG) {
            renderPass.setPipeline(texPipeline);
            renderPass.setBindGroup(0, texBG);
            renderPass.draw(4);
        }

        renderPass.setPipeline(curvePipeline);
        renderPass.setVertexBuffer(0, lineVB);
        renderPass.setBindGroup(0, curveBG);
        renderPass.draw(lineVerts.length / 6); // 6 floats per vertex

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    renderPartD();
}


main();