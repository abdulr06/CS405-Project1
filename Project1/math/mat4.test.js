import assert from 'node:assert';
import { Mat4 } from './mat4.js';

function approxEqual(a, b, eps = 1e-4) {
    assert.strictEqual(a.length, b.length, 'Length mismatch');
    for (let i = 0; i < a.length; i++) {
        assert.ok(Math.abs(a[i] - b[i]) < eps, `Index ${i}: expected ${b[i]}, got ${a[i]}`);
    }
}

// 1. Identity & Multiply
const I = Mat4.create();
const T = Mat4.translate(2, 3, 4);
approxEqual(Mat4.multiply(I, T), T);

// 2. Transpose
const T_transposed = Mat4.transpose(T);
assert.strictEqual(T_transposed[3], 2);
assert.strictEqual(T_transposed[7], 3);
assert.strictEqual(T_transposed[11], 4);
approxEqual(Mat4.transpose(T_transposed), T);

// 3. Rigid Inverse (R * T)^-1 * (R * T) = I
const R = Mat4.rotateY(0.75);
const RT = Mat4.multiply(T, R);
const RT_inv = Mat4.inverseRigid(RT);
approxEqual(Mat4.multiply(RT, RT_inv), I);

// 4. WebGPU Perspective Depth Mapping z in [0, 1]
const near = 1.0, far = 10.0;
const P = Mat4.perspective(Math.PI / 3, 1.0, near, far);
// Point on near plane in camera space: (0, 0, -near) -> NDC z should be 0
const zNear = Mat4.apply(P, [0, 0, -near])[2];
// Point on far plane in camera space: (0, 0, -far) -> NDC z should be 1
const zFar = Mat4.apply(P, [0, 0, -far])[2];
assert.ok(Math.abs(zNear - 0.0) < 1e-4, `Near plane z should map to 0, got ${zNear}`);
assert.ok(Math.abs(zFar - 1.0) < 1e-4, `Far plane z should map to 1, got ${zFar}`);

console.log('All Mat4 unit tests passed successfully!');