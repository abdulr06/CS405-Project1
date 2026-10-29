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
    }
};