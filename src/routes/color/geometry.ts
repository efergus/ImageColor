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
