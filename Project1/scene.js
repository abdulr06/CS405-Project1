// scene.js
import { Mat4 } from './mat4.js';

export class SceneNode {
    constructor(name) {
        this.name = name;
        this.localMatrix = Mat4.create();
        this.worldMatrix = Mat4.create();
        this.parent = null;
        this.children = [];
        
        // Optional rendering attributes
        this.color = [0.8, 0.8, 0.8]; // Default RGB
        this.scale = [1, 1, 1];       // Mesh relative scale
    }

    setParent(parent) {
        if (this.parent) {
            const ndx = this.parent.children.indexOf(this);
            if (ndx >= 0) {
                this.parent.children.splice(ndx, 1);
            }
        }
        if (parent) {
            parent.children.push(this);
        }
        this.parent = parent;
    }

    addChild(child) {
        child.setParent(this);
    }

    // Traverse the tree and update world matrices recursively
    updateWorldMatrix(parentWorldMatrix) {
        if (parentWorldMatrix) {
            this.worldMatrix = Mat4.multiply(parentWorldMatrix, this.localMatrix);
        } else {
            // Root node
            this.worldMatrix = new Float32Array(this.localMatrix);
        }

        // Propagate down to all child nodes
        for (const child of this.children) {
            child.updateWorldMatrix(this.worldMatrix);
        }
    }
}