// mat4.js

export const Mat4 = {
    // Creates a new 4x4 identity matrix (column-major)
    create() {
        return new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]);
    },

    // Multiplies matrix A by matrix B: C = A * B
    multiply(a, b) {
        const out = new Float32Array(16);
        for (let c = 0; c < 4; c++) {
            for (let r = 0; r < 4; r++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    // a[k*4 + r] gets row r, col k of A
                    // b[c*4 + k] gets row k, col c of B
                    sum += a[k * 4 + r] * b[c * 4 + k];
                }
                out[c * 4 + r] = sum;
            }
        }
        return out;
    },

    // Generates a translation matrix
    translate(tx, ty, tz) {
        const out = this.create();
        out[12] = tx;
        out[13] = ty;
        out[14] = tz;
        return out;
    },

    // Generates a scaling matrix
    scale(sx, sy, sz) {
        const out = this.create();
        out[0] = sx;
        out[5] = sy;
        out[10] = sz;
        return out;
    },

    // Generates a rotation matrix around the X axis
    rotateX(angle) {
        const out = this.create();
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        out[5] = c;
        out[6] = s;
        out[9] = -s;
        out[10] = c;
        return out;
    },

    // Generates a rotation matrix around the Y axis
    rotateY(angle) {
        const out = this.create();
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        out[0] = c;
        out[2] = -s;
        out[8] = s;
        out[10] = c;
        return out;
    },

    // Generates a rotation matrix around the Z axis
    rotateZ(angle) {
        const out = this.create();
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        out[0] = c;
        out[1] = s;
        out[4] = -s;
        out[5] = c;
        return out;
    },

    // Transposes a matrix
    transpose(m) {
        const out = new Float32Array(16);
        for (let c = 0; c < 4; c++) {
            for (let r = 0; r < 4; r++) {
                out[r * 4 + c] = m[c * 4 + r];
            }
        }
        return out;
    },

    // Computes the inverse of a rigid transform (Rotation + Translation only)
    // M = T * R -> M^-1 = R^T * T^-1
    inverseRigid(m) {
        const out = this.create();
        
        // 1. Transpose the 3x3 rotation portion
        out[0] = m[0]; out[1] = m[4]; out[2] = m[8];
        out[4] = m[1]; out[5] = m[5]; out[6] = m[9];
        out[8] = m[2]; out[9] = m[6]; out[10] = m[10];

        // 2. Negate the translation vector
        const tx = -m[12];
        const ty = -m[13];
        const tz = -m[14];

        // 3. Multiply the transposed rotation by the negated translation
        out[12] = out[0] * tx + out[4] * ty + out[8] * tz;
        out[13] = out[1] * tx + out[5] * ty + out[9] * tz;
        out[14] = out[2] * tx + out[6] * ty + out[10] * tz;

        return out;
    },

    // Helper to format matrix for the UI readout
    format(m) {
        let str = '';
        for (let r = 0; r < 4; r++) {
            for (let c = 0; c < 4; c++) {
                const val = m[c * 4 + r];
                str += (Math.abs(val) < 1e-4 ? 0 : val).toFixed(2).padStart(7);
            }
            str += '\n';
        }
        return str;
    },

    // lookAt(eye, target, up) - WebGPU Right-Handed View Matrix
    lookAt(eye, target, up) {
        // Forward vector f (from eye to target)
        let fx = target[0] - eye[0];
        let fy = target[1] - eye[1];
        let fz = target[2] - eye[2];
        let rlen = 1.0 / (Math.hypot(fx, fy, fz) || 1);
        fx *= rlen; fy *= rlen; fz *= rlen;

        // View direction w = -f (camera looks down -Z in view space)
        const wx = -fx, wy = -fy, wz = -fz;

        // Check degenerate case: up vector parallel to view direction
        let ux = up[0], uy = up[1], uz = up[2];
        const dotUp = Math.abs(wx * ux + wy * uy + wz * uz);
        if (dotUp > 0.999) {
            // Pick fallback up vector
            ux = Math.abs(wz) < 0.999 ? 0 : 1;
            uy = 1;
            uz = 0;
        }

        // Right vector u = norm(cross(up, w))
        let rx = uy * wz - uz * wy;
        let ry = uz * wx - ux * wz;
        let rz = ux * wy - uy * wx;
        rlen = 1.0 / (Math.hypot(rx, ry, rz) || 1);
        rx *= rlen; ry *= rlen; rz *= rlen;

        // True Up vector v = cross(w, u)
        const vx = wy * rz - wz * ry;
        const vy = wz * rx - wx * rz;
        const vz = wx * ry - wy * rx;

        // Column-major View Matrix
        return new Float32Array([
            rx, vx, wx, 0,
            ry, vy, wy, 0,
            rz, vz, wz, 0,
            -(rx * eye[0] + ry * eye[1] + rz * eye[2]),
            -(vx * eye[0] + vy * eye[1] + vz * eye[2]),
            -(wx * eye[0] + wy * eye[1] + wz * eye[2]),
            1
        ]);
    },

    // WebGPU Perspective Matrix: z in [0, 1]
    perspective(fovY, aspect, near, far) {
        const f = 1.0 / Math.tan(fovY / 2);
        const rangeInv = 1.0 / (near - far);

        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, far * rangeInv, -1,
            0, 0, near * far * rangeInv, 0
        ]);
    },

    // WebGPU Orthographic Matrix: z in [0, 1]
    ortho(height, aspect, near, far) {
        const top = height / 2;
        const right = top * aspect;
        const rangeInv = 1.0 / (near - far);

        return new Float32Array([
            1 / right, 0, 0, 0,
            0, 1 / top, 0, 0,
            0, 0, rangeInv, 0,
            0, 0, near * rangeInv, 1
        ]);
    }
};