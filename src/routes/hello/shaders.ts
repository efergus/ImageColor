import tgpu, { d, std } from 'typegpu';
import { textureSample, textureLoad, textureSampleLevel } from 'typegpu/std';
import { linear_rgb_to_oklab, linear_rgb_to_srgb, oklab_to_linear_rgb, srgb_to_linear_rgb } from './color_utils';

export const cameraUniform = d.struct({
	yaw: d.f32,
	pitch: d.f32,
	radius: d.f32,
	aspect: d.f32,
	steps: d.u32,
	sensitivity: d.f32,
});

export const cameraBindLayout = tgpu.bindGroupLayout({
	tex: { texture: d.texture2d() },
	samp: { sampler: 'filtering' },
	weightTexture: { texture: d.texture3d() },
	weightSampler: { sampler: 'filtering' },
	cameraUniform: { uniform: cameraUniform },
});

export const triangleVertex = ({ $vertexIndex: vid }: { $vertexIndex: number }) => {
	'use gpu';
	const positions = [
		d.vec2f(-1.0, 1.0),
		d.vec2f(-1.0, -1.0),
		d.vec2f(1.0, -1.0),
		d.vec2f(-1.0, 1.0),
		d.vec2f(1.0, -1.0),
		d.vec2f(1.0, 1.0),
	];
	const uvs = [
		d.vec2f(0.0, 0.0),
		d.vec2f(0.0, 1.0),
		d.vec2f(1.0, 1.0),
		d.vec2f(0.0, 0.0),
		d.vec2f(1.0, 1.0),
		d.vec2f(1.0, 0.0)
	];
	return {
		$position: d.vec4f(positions[vid], 0.0, 1.0),
		uv: uvs[vid]
	};
};

const textureSampleExample = (uv: d.v2f) => {
	'use gpu';
	// gotta figure out how to reference these in a more generic way?
	return textureSample(cameraBindLayout.$.tex, cameraBindLayout.$.samp, uv);
};

const rotX = (pitch: number) => {
	'use gpu';
	return d.mat3x3f(
		d.vec3f(1, 0, 0),
		d.vec3f(0, std.cos(pitch), -std.sin(pitch)),
		d.vec3f(0, std.sin(pitch), std.cos(pitch))
	);
};

const rotY = (yaw: number) => {
	'use gpu';
	return d.mat3x3f(
		d.vec3f(std.cos(yaw), 0, -std.sin(yaw)),
		d.vec3f(0, 1, 0),
		d.vec3f(std.sin(yaw), 0, std.cos(yaw))
	);
};

const boxIntersect = (origin: d.v3f, direction: d.v3f): d.v2f => {
	'use gpu';

	const invDir = std.div(1.0, direction);
	const t0 = std.mul(std.sub(d.vec3f(0.0, 0.0, 0.0), origin), invDir);
	const t1 = std.mul(std.sub(d.vec3f(1.0, 1.0, 1.0), origin), invDir);
	const tMin = std.min(t0, t1);
	const tMax = std.max(t0, t1);
	const tMinOverall = std.max(tMin.x, std.max(tMin.y, tMin.z));
	const tMaxOverall = std.min(tMax.x, std.min(tMax.y, tMax.z));
	return d.vec2f(tMinOverall, tMaxOverall);

}

export const cameraRotation = (yaw: number, pitch: number) => {
	'use gpu';
	const yawMatrix = rotY(yaw);
	const pitchMatrix = rotX(pitch);
	const transform = std.mul(yawMatrix, pitchMatrix);
	return transform;
}

export const RayStruct = d.struct({
	start: d.vec3f,
	direction: d.vec3f
})

export const eyeLocation = (transform: d.m3x3f, radius: number, center: d.v3f) => {
	'use gpu';
	const eye = std.add(center, std.mul(transform, d.vec3f(0.0, 0.0, radius)));
	return eye;
}

export const cameraRay = (uv: d.v2f, yaw: number, pitch: number, radius: number, aspect: number, center: d.v3f) => {
	'use gpu';
	const transform = cameraRotation(yaw, pitch);
	const eye = eyeLocation(transform, radius, center);
	const up = std.mul(transform, d.vec3f(0.0, 1.0, 0.0));
	const fwd = std.normalize(std.sub(center, eye));
	const right = std.normalize(std.cross(fwd, up));

	const screen = d.vec2f(
		2.0 * uv.x - 1.0,
		1.0 - 2.0 * uv.y
	);
	const rightOffset = std.mul(right, std.mul(screen.x, std.mul(aspect, d.f32(0.6))));
	const upOffset = std.mul(up, std.mul(screen.y, d.f32(0.6)));
	const direction = std.normalize(std.add(fwd, std.add(rightOffset, upOffset)));
	return RayStruct({ start: eye, direction });
}

export const triangleFragmentOutput = d.struct({
	color: d.vec4f,
	pick: d.vec4f,
})

export const triangleFragment = ({ uv }: { uv: d.v2f }): d.Infer<typeof triangleFragmentOutput> => {
	'use gpu';

	const bg = d.vec4f(1.0, 1.0, 1.0, 1.0);

	const center = d.vec3f(0.5, 0.5, 0.5);
	const ray = cameraRay(
		uv,
		cameraBindLayout.$.cameraUniform.yaw,
		cameraBindLayout.$.cameraUniform.pitch,
		cameraBindLayout.$.cameraUniform.radius,
		cameraBindLayout.$.cameraUniform.aspect,
		center
	);
	const eye = ray.start;
	const direction = ray.direction;

	const intersection = boxIntersect(eye, direction);
	const min = std.max(0.0, intersection.x);
	const max = intersection.y;
	if (min >= max) {
		return {
			color: bg,
			pick: d.vec4f(0.0, 0.0, 0.0, 0.0),
		};
	}

	const steps = d.i32(cameraBindLayout.$.cameraUniform.steps);
	const stepSize = std.div(max - min, d.f32(steps));
	const start = std.add(eye, std.mul(direction, min));
	let acc = d.vec4f(0.0, 0.0, 0.0, 0.0);
	const sensitivity = cameraBindLayout.$.cameraUniform.sensitivity;
	const wRange = std.max(sensitivity / 30.0, d.f32(0.0001));

	for (let i = 0; i < steps; i++) {
		const currentPoint = std.add(start, std.mul(direction, std.mul(stepSize, d.f32(i))));
		const sample = textureSampleLevel(cameraBindLayout.$.weightTexture, cameraBindLayout.$.weightSampler, currentPoint, 0);

		const remapped = std.min(sample.w / wRange, 1.0);
		const expTerm = std.exp(std.mul(-remapped * 100.0, stepSize));
		const alpha = std.max(std.sub(d.f32(1.0), expTerm), 0.0);

		// const sampleColor = d.vec3f(texCoords.x, texCoords.y, texCoords.z);
		// const sampleColor = linear_rgb_to_srgb(oklab_to_linear_rgb(d.vec3f(texCoords.y, (texCoords.x - 0.5), (texCoords.z - 0.5))));

		const invAccAlpha = 1.0 - acc.w;
		const colorAlpha = d.vec4f(std.mul(sample.xyz, alpha), alpha);
		acc = std.add(acc, std.mul(colorAlpha, invAccAlpha));
	}

	const color = std.add(acc.rgba, std.mul(bg.rgba, std.sub(d.f32(1.0), acc.a)));
	const pureColor = std.div(acc.xyz, std.max(d.f32(0.0001), acc.w));

	return {
		color: d.vec4f(color.xyz, 1.0),
		pick: d.vec4f(pureColor.xyz, 1.0),
	};
};

export const imageFragment = ({ uv }: { uv: d.v2f }) => {
	'use gpu';
	return textureSample(cameraBindLayout.$.tex, cameraBindLayout.$.samp, uv);
}

export const computeOptions = d.struct({
	tableSize: d.u32,
})
// TODO: need to check if that's the best way to be defining this bind group layout
const WeightArray = d.arrayOf(d.atomic(d.u32));

export const computeBindLayout = tgpu.bindGroupLayout({
	options: { uniform: computeOptions },
	image: { storageTexture: d.textureStorage2d('rgba8unorm', 'read-only') },
	weights: { storage: WeightArray, access: 'mutable' }
})


export const calculateWeights = (x: number, y: number) => {
	'use gpu';
	const val = textureLoad(computeBindLayout.$.image, d.vec2u(x, y));
	const tableSize = computeBindLayout.$.options.tableSize;
	// const oklab = linear_rgb_to_oklab(srgb_to_linear_rgb(val.xyz));
	// const x_ = d.u32(std.round((oklab.y + 0.5) * d.f32(tableSize - 1)));
	// const y_ = d.u32(std.round(oklab.x * d.f32(tableSize - 1)));
	// const z_ = d.u32(std.round((oklab.z + 0.5) * d.f32(tableSize - 1)));
	const x_ = d.u32(std.floor(val.x * d.f32(tableSize - 2)));
	const y_ = d.u32(std.floor(val.y * d.f32(tableSize - 2)));
	const z_ = d.u32(std.floor(val.z * d.f32(tableSize - 2)));
	const val_255 = std.ceil(std.mul(val, d.f32(255.0)));
	const idx = x_ + y_ * tableSize + z_ * tableSize * tableSize;
	const byteOffset = idx * 4;
	std.atomicAdd(computeBindLayout.$.weights[byteOffset + 0], d.u32(200)); // val_255.x
	std.atomicAdd(computeBindLayout.$.weights[byteOffset + 1], d.u32(200)); // val_255.y
	std.atomicAdd(computeBindLayout.$.weights[byteOffset + 2], d.u32(200)); // val_255.z
	std.atomicAdd(computeBindLayout.$.weights[byteOffset + 3], d.u32(1));
}

export const computePostProcessingLayout = tgpu.bindGroupLayout({
	options: { uniform: computeOptions },
	weights: { storage: d.arrayOf(d.u32) },
	outputTexture: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') },
});

export const processWeights = (x: number, y: number, z: number) => {
	'use gpu';

	const tableSize = computePostProcessingLayout.$.options.tableSize;

	const idx = x + y * tableSize + z * tableSize * tableSize;
	const byteOffset = idx * 4;

	const r = computePostProcessingLayout.$.weights[byteOffset + 0];
	const g = computePostProcessingLayout.$.weights[byteOffset + 1];
	const b = computePostProcessingLayout.$.weights[byteOffset + 2];
	const count = computePostProcessingLayout.$.weights[byteOffset + 3];

	const countf = std.max(d.f32(1.0), d.f32(count));
	const rgb = d.vec3f(
		d.f32(r) / countf / 255.0,
		d.f32(g) / countf / 255.0,
		d.f32(b) / countf / 255.0
	);

	if (count > 0) {
		std.textureStore(computePostProcessingLayout.$.outputTexture, d.vec3u(x, y, z), d.vec4f(rgb, 8));

	}
}


export const texture3dProcessingLayout = tgpu.bindGroupLayout({
	inputTexture: { texture: d.texture3d() },
	outputTexture: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') }
});

export const blurKernel = (x: number, y: number, z: number) => {
	'use gpu';
	// const coords = d.vec3u(d.u32(x), d.u32(y), d.u32(z));
	// const val = textureLoad(texture3dProcessingLayout.$.inputTexture, coords, 0);
	// return val;

	let accumulated = d.vec4f(0.0, 0.0, 0.0, 0.0);
	const weights = [1.0, 0.577, 0.333, 0.183];
	for (let i = -1; i <= 1; i++) {
		for (let j = -1; j <= 1; j++) {
			for (let k = -1; k <= 1; k++) {
				const cx = d.u32(d.i32(x) + i);
				const cy = d.u32(d.i32(y) + j);
				const cz = d.u32(d.i32(z) + k);
				const weight = weights[std.abs(i) + std.abs(j) + std.abs(k)];
				const coords = d.vec3u(cx, cy, cz);
				const val = textureLoad(texture3dProcessingLayout.$.inputTexture, coords, 0);
				accumulated = std.add(accumulated, std.mul(val, weight));
			}
		}
	}
	return std.div(accumulated, d.f32(weights[0] * 1 + weights[1] * 6 + weights[2] * 8 + weights[3] * 8));
}

export const blur = (x: number, y: number, z: number) => {
	'use gpu';
	const val = blurKernel(x, y, z);
	std.textureStore(texture3dProcessingLayout.$.outputTexture, d.vec3u(x, y, z), val);
}

export const rayMarch = (source: d.v3f, direction: d.v3f, texture: d.texture3d<d.F32>, sampler: d.sampler, samples: d.U32) => {
	'use gpu';
}