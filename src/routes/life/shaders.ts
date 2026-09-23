import tgpu, {
	d,
	std,
	type RenderFlag,
	type StorageFlag,
	type TgpuRoot,
	type TgpuTexture
} from 'typegpu';
import { textureLoad } from 'typegpu/std';
import { randf } from '@typegpu/noise';
import { once, onceBindGroup } from '$lib/gpu/gpu_utils';
import { quadVertex } from '../color/shaders';

const array2dSize = d.struct({
	width: d.u32,
	height: d.u32
});
const cellCoord = d.struct({
	x: d.u32,
	y: d.u32
});
const stepConstants = d.struct({
	discreteness: d.f32,
	// Region boundaries expressed as the average neighbor value in [0, 1].
	stay: d.f32,
	grow: d.f32,
	crowd: d.f32,
	linear: d.u32,
	window: d.u32,
	step: d.u32,
	seed: d.f32,
	radiation: d.f32,
});
const cameraConstants = d.struct({
	size: d.vec2f,
	center: d.vec2f,
});

export const lifeStateFormat = 'r32float';

// The cell-state textures as typegpu accepts them for storage-texture bind
// group entries.
export type LifeStateTexture = TgpuTexture<{
	size: readonly number[];
	format: typeof lifeStateFormat;
	dimension?: '2d';
}> &
	StorageFlag;

export const lifeStepLayout = tgpu.bindGroupLayout({
	size: { uniform: array2dSize },
	constants: { uniform: stepConstants },
	previous: { storageTexture: d.textureStorage2d(lifeStateFormat, 'read-only') },
	current: { storageTexture: d.textureStorage2d(lifeStateFormat, 'read-only') },
	next: { storageTexture: d.textureStorage2d(lifeStateFormat, 'write-only') }
});

const cellAtPrev = (x: number, y: number) => {
	'use gpu';
	return textureLoad(lifeStepLayout.$.previous, d.vec2u(x, y)).x;
};

const cellAt = (x: number, y: number) => {
	'use gpu';
	return textureLoad(lifeStepLayout.$.current, d.vec2u(x, y)).x;
};

export const applyLifeStep = (x: number, y: number) => {
	'use gpu';
	const width = lifeStepLayout.$.size.width;
	const height = lifeStepLayout.$.size.height;
	const window = lifeStepLayout.$.constants.window;
	const step = lifeStepLayout.$.constants.step;
	randf.seed3(d.vec3f(x, y, lifeStepLayout.$.constants.seed));
	randf.seed3(d.vec3f(randf.sample(), x * y % 1000, d.f32(step % 1000 + (step * 486653) % 1000)));

	// Sum every cell in the (2*window+1)^2 square around (x, y) with toroidal
	// wrap-around, then drop the center cell. The window*(size-1) offset is
	// a multiple of the size plus the -window shift, keeping the u32
	// arithmetic non-negative before the modulo.
	let total = d.f32(0.0);
	for (let dy = d.u32(0); dy <= 2 * window; dy++) {
		for (let dx = d.u32(0); dx <= 2 * window; dx++) {
			const nx = (x + dx + window * (width - 1)) % width;
			const ny = (y + dy + window * (height - 1)) % height;
			total = total + cellAt(nx, ny);
		}
	}

	total = total + cellAtPrev(x, y);

	const alive = cellAt(x, y);

	// The thresholds are the boundaries between the four regions of the
	// average neighbor value: starve | stay | grow | crowd. Comparing
	// averages rather than totals keeps the rule invariant to window size.
	const count = (2 * window + 1) * (2 * window + 1) - 1;
	const average = (total - alive) / d.f32(count);

	// Starve and crowd both die off; only the middle two regions differ.
	let target = d.f32(0.0);
	if (average >= lifeStepLayout.$.constants.grow && average < lifeStepLayout.$.constants.crowd) {
		// Birth zone: alive regardless of current state.
		target = 1.0;
	} else if (
		average >= lifeStepLayout.$.constants.stay &&
		average < lifeStepLayout.$.constants.grow
	) {
		// Stasis: keep the current state.
		target = alive;
	}

	// let blip2 = d.f32(1.0);
	// let radiation2 = 0.5;
	// while (radiation2 > d.f32(0.0) && radiation2 < (2 ** -8)) {
	// 	blip2 *= randf.bernoulli(2 ** -8);
	// 	radiation2 *= 2 ** 8;
	// }
	// blip2 *= std.select(randf.bernoulli(radiation2), 0, radiation2 === 0);
	// if (blip2) {
	// 	target = alive;
	// }

	// Exponential eases toward the target by a discreteness fraction of the
	// remaining distance each step; linear moves a fixed discreteness step,
	// clamped so it stops exactly at the target.
	const discreteness = lifeStepLayout.$.constants.discreteness;
	let updated = std.mix(alive, target, discreteness);
	if (lifeStepLayout.$.constants.linear !== 0) {
		updated = alive + std.clamp(target - alive, -discreteness, discreteness);
	}
	let blip = d.f32(1.0);
	let radiation = lifeStepLayout.$.constants.radiation;
	while (radiation > d.f32(0.0) && radiation < (2 ** -8)) {
		blip *= randf.bernoulli(2 ** -8);
		radiation *= 2 ** 8;
	}
	blip *= std.select(randf.bernoulli(radiation), 0, radiation === 0);

	updated = std.max(updated, blip);

	std.textureStore(lifeStepLayout.$.next, d.vec2u(x, y), d.vec4f(updated, 0.0, 0.0, 0.0));
};

export const gpuLifeStep = (
	root: TgpuRoot,
	previous: LifeStateTexture,
	current: LifeStateTexture,
	next: LifeStateTexture,
	options: {
		width: number;
		height: number;
		discreteness: number;
		stay: number;
		grow: number;
		crowd: number;
		radiation: number;
		linear: boolean;
		window: number;
		step: number;
		seed: number;
	}
) => {
	const { pipeline, sizeUniform, constantsUniform } = once(gpuLifeStep, () => {
		const pipeline = root.createGuardedComputePipeline(applyLifeStep);
		const sizeUniform = root.createBuffer(array2dSize).$usage('uniform');
		const constantsUniform = root.createBuffer(stepConstants).$usage('uniform');
		return { pipeline, sizeUniform, constantsUniform };
	});

	const bindGroup = onceBindGroup(root, lifeStepLayout, {
		size: sizeUniform,
		constants: constantsUniform,
		previous: previous,
		current: current,
		next: next
	});

	sizeUniform.write({
		width: options.width,
		height: options.height
	});

	constantsUniform.write({
		discreteness: options.discreteness,
		stay: options.stay,
		grow: options.grow,
		crowd: options.crowd,
		radiation: options.radiation,
		linear: options.linear ? 1 : 0,
		window: options.window,
		step: options.step,
		seed: options.seed
	});

	pipeline.with(bindGroup).dispatchThreads(options.width, options.height);
};

export const lifeToggleLayout = tgpu.bindGroupLayout({
	cell: { uniform: cellCoord },
	current: { storageTexture: d.textureStorage2d(lifeStateFormat, 'read-only') },
	next: { storageTexture: d.textureStorage2d(lifeStateFormat, 'write-only') }
});

// Copies the state unchanged except for the clicked cell, which is flipped.
export const applyToggle = (x: number, y: number) => {
	'use gpu';
	const value = textureLoad(lifeToggleLayout.$.current, d.vec2u(x, y)).x;
	let updated = value;
	if (x === lifeToggleLayout.$.cell.x && y === lifeToggleLayout.$.cell.y) {
		updated = 1.0 - value;
	}
	std.textureStore(lifeToggleLayout.$.next, d.vec2u(x, y), d.vec4f(updated, 0.0, 0.0, 0.0));
};

export const gpuToggleCell = (
	root: TgpuRoot,
	current: LifeStateTexture,
	next: LifeStateTexture,
	options: {
		x: number;
		y: number;
		width: number;
		height: number;
	}
) => {
	const { pipeline, cellUniform } = once(gpuToggleCell, () => {
		const pipeline = root.createGuardedComputePipeline(applyToggle);
		const cellUniform = root.createBuffer(cellCoord).$usage('uniform');
		return { pipeline, cellUniform };
	});

	const bindGroup = onceBindGroup(root, lifeToggleLayout, {
		cell: cellUniform,
		current: current,
		next: next
	});

	cellUniform.write({
		x: options.x,
		y: options.y
	});

	pipeline.with(bindGroup).dispatchThreads(options.width, options.height);
};

const drawSegment = d.struct({
	start: d.vec2f,
	end: d.vec2f
});

export const lifeDrawLayout = tgpu.bindGroupLayout({
	segment: { uniform: drawSegment },
	current: { storageTexture: d.textureStorage2d(lifeStateFormat, 'read-only') },
	next: { storageTexture: d.textureStorage2d(lifeStateFormat, 'write-only') }
});

// Copies the state, setting alive every cell whose center lies within half a
// cell of the dragged segment, so fast pointer movement leaves no gaps.
export const applyDraw = (x: number, y: number) => {
	'use gpu';
	const value = textureLoad(lifeDrawLayout.$.current, d.vec2u(x, y)).x;

	const p = d.vec2f(d.f32(x), d.f32(y));
	const a = lifeDrawLayout.$.segment.start;
	const b = lifeDrawLayout.$.segment.end;
	const pa = std.sub(p, a);
	const ba = std.sub(b, a);
	const t = std.clamp(std.dot(pa, ba) / std.max(std.dot(ba, ba), 0.0001), 0.0, 1.0);
	const dist = std.length(std.sub(pa, std.mul(ba, t)));

	let updated = value;
	if (dist < 0.5) {
		updated = 1.0;
	}
	std.textureStore(lifeDrawLayout.$.next, d.vec2u(x, y), d.vec4f(updated, 0.0, 0.0, 0.0));
};

export const gpuDrawSegment = (
	root: TgpuRoot,
	current: LifeStateTexture,
	next: LifeStateTexture,
	options: {
		x0: number;
		y0: number;
		x1: number;
		y1: number;
		width: number;
		height: number;
	}
) => {
	const { pipeline, segmentUniform } = once(gpuDrawSegment, () => {
		const pipeline = root.createGuardedComputePipeline(applyDraw);
		const segmentUniform = root.createBuffer(drawSegment).$usage('uniform');
		return { pipeline, segmentUniform };
	});

	const bindGroup = onceBindGroup(root, lifeDrawLayout, {
		segment: segmentUniform,
		current: current,
		next: next
	});

	segmentUniform.write({
		start: d.vec2f(options.x0, options.y0),
		end: d.vec2f(options.x1, options.y1)
	});

	pipeline.with(bindGroup).dispatchThreads(options.width, options.height);
};

export const lifeRenderLayout = tgpu.bindGroupLayout({
	size: { uniform: array2dSize },
	camera: { uniform: cameraConstants },
	cells: { storageTexture: d.textureStorage2d(lifeStateFormat, 'read-only') }
});

export const lifeFragmentOutput = d.struct({
	color: d.vec4f
});

export const lifeFragment = ({ uv }: { uv: d.v2f }): d.Infer<typeof lifeFragmentOutput> => {
	'use gpu';
	const width = lifeRenderLayout.$.size.width;
	const height = lifeRenderLayout.$.size.height;
	const cameraSize = lifeRenderLayout.$.camera.size;
	const cameraCenter = lifeRenderLayout.$.camera.center;
	const texturePosition = std.add(std.sub(std.mul(uv, cameraSize), std.mul(cameraSize, 0.5)), cameraCenter);
	let alive = 0;
	if (texturePosition.x >= 0 && texturePosition.x < width && texturePosition.y >= 0 && texturePosition.y < height) {
		alive = textureLoad(lifeRenderLayout.$.cells, d.vec2u(texturePosition)).x;
	}
	const brightness = std.clamp(alive, 0.0, 1.0);
	return {
		color: d.vec4f(brightness, brightness, brightness, 1.0)
	};
};

export const renderLife = (
	root: TgpuRoot,
	cells: LifeStateTexture,
	outputView: (TgpuTexture & RenderFlag) | GPUCanvasContext,
	options: {
		width: number;
		height: number;
		cameraSize: d.v2f;
		cameraCenter: d.v2f;
	}
) => {
	const { pipeline, sizeUniform, cameraUniform } = once(renderLife, () => {
		const pipeline = root.createRenderPipeline({
			primitive: { topology: 'triangle-list' },
			vertex: quadVertex,
			fragment: lifeFragment,
			targets: {
				color: { format: navigator.gpu.getPreferredCanvasFormat() }
			}
		});
		const sizeUniform = root.createBuffer(array2dSize).$usage('uniform');
		const cameraUniform = root.createBuffer(cameraConstants).$usage('uniform');
		return { pipeline, sizeUniform, cameraUniform };
	});

	const bindGroup = onceBindGroup(root, lifeRenderLayout, {
		size: sizeUniform,
		camera: cameraUniform,
		cells: cells
	});

	sizeUniform.write({
		width: options.width,
		height: options.height
	});

	cameraUniform.write({
		size: options.cameraSize,
		center: options.cameraCenter
	})

	pipeline
		.with(bindGroup)
		.withColorAttachment({
			color: { view: outputView }
		})
		.draw(6);
};
