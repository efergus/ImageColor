import { d } from 'typegpu';

// A flat (non-indexed) triangle-list mesh with a single flat color, ready to
// hand to renderRasterScene.
export type Mesh = {
	vertices: d.v3f[];
	color: d.v4f;
};

const boxCenter = d.vec3f(0.5, 0.5, 0.5);

// The plane that has always bisected the cloud's unit box.
export const planeMesh = (): Mesh => ({
	vertices: [
		d.vec3f(0.0, 0.0, 0.5),
		d.vec3f(1.0, 0.0, 0.5),
		d.vec3f(1.0, 1.0, 0.5),
		d.vec3f(0.0, 0.0, 0.5),
		d.vec3f(1.0, 1.0, 0.5),
		d.vec3f(0.0, 1.0, 0.5)
	],
	color: d.vec4f(0.0, 0.0, 0.0, 1.0)
});

const axisFrames: Record<'x' | 'y' | 'z', { u: d.v3f; v: d.v3f; w: d.v3f }> = {
	x: { u: d.vec3f(1, 0, 0), v: d.vec3f(0, 1, 0), w: d.vec3f(0, 0, 1) },
	y: { u: d.vec3f(0, 1, 0), v: d.vec3f(1, 0, 0), w: d.vec3f(0, 0, 1) },
	z: { u: d.vec3f(0, 0, 1), v: d.vec3f(1, 0, 0), w: d.vec3f(0, 1, 0) }
};

// A capped cylinder centered on the cloud's unit-box center, aligned along
// one axis, expressed as a flat (non-indexed) triangle-list mesh.
export const cylinderMesh = (options: {
	axis: 'x' | 'y' | 'z';
	length: number;
	radius: number;
	color: d.v4f;
	segments?: number;
}): Mesh => {
	const { axis, length, radius, color, segments = 24 } = options;
	const { u, v, w } = axisFrames[axis];

	const half = length / 2;
	const bottom = boxCenter.sub(u.mul(half));
	const top = boxCenter.add(u.mul(half));

	const ring = (center: d.v3f, angle: number) =>
		center.add(v.mul(Math.cos(angle) * radius)).add(w.mul(Math.sin(angle) * radius));

	const vertices: d.v3f[] = [];
	for (let i = 0; i < segments; i++) {
		const a0 = (i / segments) * Math.PI * 2;
		const a1 = ((i + 1) / segments) * Math.PI * 2;

		const b0 = ring(bottom, a0);
		const b1 = ring(bottom, a1);
		const t0 = ring(top, a0);
		const t1 = ring(top, a1);

		vertices.push(b0, t0, t1, b0, t1, b1); // side wall
		vertices.push(bottom, b1, b0); // bottom cap
		vertices.push(top, t0, t1); // top cap
	}

	return { vertices, color };
};
