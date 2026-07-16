import { d } from 'typegpu';

export type GridVertex = {
	position: d.v3f;
	uv: d.v2f;
};

// One face of the cloud's unit box, used as a spatial reference grid. `normal`
// points outward and is used to fade the face in based on view direction.
export type GridFace = {
	name: string;
	normal: d.v3f;
	vertices: GridVertex[];
};

const face = (name: string, normal: d.v3f, corner: d.v3f, right: d.v3f, up: d.v3f): GridFace => {
	const p = (u: number, v: number) => corner.add(right.mul(u)).add(up.mul(v));
	return {
		name,
		normal,
		vertices: [
			{ position: p(0, 0), uv: d.vec2f(0, 0) },
			{ position: p(1, 0), uv: d.vec2f(1, 0) },
			{ position: p(1, 1), uv: d.vec2f(1, 1) },
			{ position: p(0, 0), uv: d.vec2f(0, 0) },
			{ position: p(1, 1), uv: d.vec2f(1, 1) },
			{ position: p(0, 1), uv: d.vec2f(0, 1) }
		]
	};
};

// Triangle-list mesh of a unit sphere centered at the origin, so positions
// double as normals. The degenerate triangles at the poles are harmless.
export const sphereVertices = (rings = 12, segments = 16): d.v3f[] => {
	const p = (r: number, s: number) => {
		const phi = (r / rings) * Math.PI;
		const theta = (s / segments) * 2 * Math.PI;
		return d.vec3f(Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta));
	};
	const vertices: d.v3f[] = [];
	for (let r = 0; r < rings; r++) {
		for (let s = 0; s < segments; s++) {
			const a = p(r, s);
			const b = p(r + 1, s);
			const c = p(r + 1, s + 1);
			const e = p(r, s + 1);
			vertices.push(a, b, c, a, c, e);
		}
	}
	return vertices;
};

// The six faces of the cloud's unit box [0,1]^3, each spanning the full
// extent of the other two axes.
export const gridFaces: GridFace[] = [
	face('bottom', d.vec3f(0, -1, 0), d.vec3f(0, 0, 0), d.vec3f(1, 0, 0), d.vec3f(0, 0, 1)),
	face('top', d.vec3f(0, 1, 0), d.vec3f(0, 1, 0), d.vec3f(1, 0, 0), d.vec3f(0, 0, 1)),
	face('xMin', d.vec3f(-1, 0, 0), d.vec3f(0, 0, 0), d.vec3f(0, 0, 1), d.vec3f(0, 1, 0)),
	face('xMax', d.vec3f(1, 0, 0), d.vec3f(1, 0, 0), d.vec3f(0, 0, 1), d.vec3f(0, 1, 0)),
	face('zMin', d.vec3f(0, 0, -1), d.vec3f(0, 0, 0), d.vec3f(1, 0, 0), d.vec3f(0, 1, 0)),
	face('zMax', d.vec3f(0, 0, 1), d.vec3f(0, 0, 1), d.vec3f(1, 0, 0), d.vec3f(0, 1, 0))
];
