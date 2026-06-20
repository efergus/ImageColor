import tgpu, { d, std } from 'typegpu';
import { textureSample, textureLoad, textureSampleLevel, textureDimensions } from 'typegpu/std';
import { linear_rgb_to_oklab, linear_rgb_to_srgb, oklab_to_linear_rgb, oklab_to_srgb, srgb_to_linear_rgb, srgb_to_oklab } from './color_utils';
import { randf } from '@typegpu/noise';

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

const sphereIntersect = (origin: d.v3f, direction: d.v3f, radius: number): d.v2f => {
	'use gpu';
	const a = std.dot(direction, direction);
	const b = std.dot(origin, direction) * 2.0;
	const c = std.dot(origin, origin) - radius * radius;
	const discriminant = b * b - 4 * a * c;
	if (discriminant < 0.0) {
		return d.vec2f(-1.0, -1.0);
	}
	const t0 = (-b - std.sqrt(discriminant)) / (2.0 * a);
	const t1 = (-b + std.sqrt(discriminant)) / (2.0 * a);
	return d.vec2f(t0, t1);
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

export const weightCalculation = (x: number, y: number) => {
	'use gpu';
	const val = textureLoad(weightCalculationLayout.$.image, d.vec2u(x, y));
	const tableSize = weightCalculationLayout.$.options.tableSize;
	const oklab = srgb_to_oklab(val.xyz);
	const x_ = d.u32(std.round((oklab.y + 0.5) * d.f32(tableSize - 1)));
	const y_ = d.u32(std.round(oklab.x * d.f32(tableSize - 1)));
	const z_ = d.u32(std.round((oklab.z + 0.5) * d.f32(tableSize - 1)));
	// const x_ = d.u32(std.floor(val.x * d.f32(tableSize - 2)));
	// const y_ = d.u32(std.floor(val.y * d.f32(tableSize - 2)));
	// const z_ = d.u32(std.floor(val.z * d.f32(tableSize - 2)));
	const val_255 = std.ceil(std.mul(val, d.f32(255.0)));
	const idx = x_ + y_ * tableSize + z_ * tableSize * tableSize;
	const byteOffset = idx * 4;
	std.atomicAdd(weightCalculationLayout.$.weights[byteOffset + 0], d.u32(val_255.r)); // val_255.x
	std.atomicAdd(weightCalculationLayout.$.weights[byteOffset + 1], d.u32(val_255.g)); // val_255.y
	std.atomicAdd(weightCalculationLayout.$.weights[byteOffset + 2], d.u32(val_255.b)); // val_255.z
	std.atomicAdd(weightCalculationLayout.$.weights[byteOffset + 3], d.u32(1));
}

export const processWeights = (x: number, y: number, z: number) => {
	'use gpu';

	const tableSize = weightTransferLayout.$.options.tableSize;

	const idx = x + y * tableSize + z * tableSize * tableSize;
	const byteOffset = idx * 4;

	const r = weightTransferLayout.$.weights[byteOffset + 0];
	const g = weightTransferLayout.$.weights[byteOffset + 1];
	const b = weightTransferLayout.$.weights[byteOffset + 2];
	const count = weightTransferLayout.$.weights[byteOffset + 3];

	const countf = std.max(d.f32(1.0), d.f32(count));
	const rgb = d.vec3f(
		d.f32(r) / countf / 255.0,
		d.f32(g) / countf / 255.0,
		d.f32(b) / countf / 255.0
	);

	std.textureStore(weightTransferLayout.$.outputTexture, d.vec3u(x, y, z), d.vec4f(srgb_to_oklab(rgb), count));
}

export const blurKernel = (x: number, y: number, z: number) => {
	'use gpu';
	// let accumulated = d.vec4f(0.0, 0.0, 0.0, 0.0);
	// const weights = [1.0, 0.577, 0.333, 0.183];
	// let total = d.f32(0.0);
	// for (let i = -1; i <= 1; i++) {
	// 	for (let j = -1; j <= 1; j++) {
	// 		for (let k = -1; k <= 1; k++) {
	// 			const cx = d.u32(d.i32(x) + i);
	// 			const cy = d.u32(d.i32(y) + j);
	// 			const cz = d.u32(d.i32(z) + k);
	// 			const coords = d.vec3u(cx, cy, cz);

	// 			const val = textureLoad(weightProcessingLayout.$.inputTexture, coords, 0);
	// 			const populated = val.a;
	// 			const weight = weights[std.abs(i) + std.abs(j) + std.abs(k)] * d.f32(populated);
	// 			total += weight;
	// 			accumulated = std.add(accumulated, std.mul(val, weight));
	// 		}
	// 	}
	// }
	const textureSize = textureDimensions(weightProcessingLayout.$.inputTexture);
	const position = std.div(d.vec3f(x, y, z), d.vec3f(textureSize.x, textureSize.y, textureSize.z));
	const oklab = std.sub(position.yxz, d.vec3f(0.0, 0.5, 0.5));
	const originalColor = textureLoad(weightProcessingLayout.$.inputTexture, d.vec3u(x, y, z), 0);
	if (originalColor.a <= 0.0) {
		return d.vec4f(oklab, 0.0);
	}
	// return d.vec4f(color, originalColor.a);
	return originalColor;
	// return std.div(accumulated, std.max(total, 0.0001)); // d.f32(weights[0] * 1 + weights[1] * 6 + weights[2] * 8 + weights[3] * 8)
}

export const blur = (x: number, y: number, z: number) => {
	'use gpu';
	const val = blurKernel(x, y, z);
	std.textureStore(weightProcessingLayout.$.outputTexture, d.vec3u(x, y, z), val);
}

export const weightTextureFormat = 'rgba16float';

export const filterOptions = d.struct({
	textureSize: d.vec2u,
	selectedColor: d.vec4f,
	saturation: d.f32,
	contrast: d.f32,
});

export const textureRenderLayout = tgpu.bindGroupLayout({
	texture: { texture: d.texture2d() },
	sampler: { sampler: 'filtering' },
	options: { uniform: filterOptions }
});

export const computeOptions = d.struct({
	textureSize: d.vec2u,
	tableSize: d.u32,
})
const WeightArray = d.arrayOf(d.atomic(d.u32));

export const weightCalculationLayout = tgpu.bindGroupLayout({
	options: { uniform: computeOptions },
	image: { storageTexture: d.textureStorage2d('rgba16float', 'read-only') },
	weights: { storage: WeightArray, access: 'mutable' }
})

export const weightTransferLayout = tgpu.bindGroupLayout({
	options: { uniform: computeOptions },
	weights: { storage: d.arrayOf(d.u32) },
	outputTexture: { storageTexture: d.textureStorage3d(weightTextureFormat, 'write-only') },
})

export const weightProcessingLayout = tgpu.bindGroupLayout({
	options: { uniform: filterOptions },
	inputTexture: { texture: d.texture3d() },
	outputTexture: { storageTexture: d.textureStorage3d('rgba16float', 'write-only') }
});

export const cameraUniform = d.struct({
	yaw: d.f32,
	pitch: d.f32,
	radius: d.f32,
	aspect: d.f32,
	steps: d.u32,
	sensitivity: d.f32,
	bgColor: d.f32,
});

export const cameraBindLayout = tgpu.bindGroupLayout({
	cameraUniform: { uniform: cameraUniform },
	options: { uniform: filterOptions },
	weightTexture: { texture: d.texture3d() },
	weightSampler: { sampler: 'filtering' },
});

export const quadVertex = ({ $vertexIndex: vid }: { $vertexIndex: number }) => {
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

export const triangleFragmentOutput = d.struct({
	color: d.vec4f,
	pick: d.vec4f,
})

export const imageFragment = ({ uv }: { uv: d.v2f }) => {
	'use gpu';
	const original = textureSample(textureRenderLayout.$.texture, textureRenderLayout.$.sampler, uv);
	const targetDistance = textureRenderLayout.$.options.selectedColor.a;
	const targetColorOklab = srgb_to_oklab(textureRenderLayout.$.options.selectedColor.rgb);
	const originalOklab = srgb_to_oklab(original.rgb);

	const greyness = std.clamp((std.distance(targetColorOklab, originalOklab) - targetDistance) * 100.0, 0.0, 0.8);
	const grey = std.dot(original.rgb, d.vec3f(0.299, 0.587, 0.114));
	const outColor = std.mix(original.rgb, d.vec3f(0.4), d.f32(greyness));
	return d.vec4f(outColor, original.a);
}

export const triangleFragment = ({ uv }: { uv: d.v2f }): d.Infer<typeof triangleFragmentOutput> => {
	'use gpu';

	const bgLightness = cameraBindLayout.$.cameraUniform.bgColor;
	const bg = d.vec4f(bgLightness, bgLightness, bgLightness, 1.0);

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
	// const intersection = sphereIntersect(std.sub(eye, center), direction, 1.0);
	const min = std.max(0.0, intersection.x);
	const max = intersection.y;
	if (min >= max) {
		return {
			color: bg,
			pick: d.vec4f(0.0, 0.0, 0.0, 0.0),
		};
	}

	const steps = d.i32(cameraBindLayout.$.cameraUniform.steps);
	randf.seed2(uv);
	const stepSize = std.div(max - min, d.f32(steps) - randf.sample());
	const start = std.add(eye, std.mul(direction, min));
	let acc = d.vec4f(0.0, 0.0, 0.0, 0.0);
	const sensitivity = cameraBindLayout.$.cameraUniform.sensitivity;
	const wRange = std.max(sensitivity, d.f32(0.0001));
	const targetColorOklab = srgb_to_oklab(cameraBindLayout.$.options.selectedColor.rgb);
	const targetDistance = cameraBindLayout.$.options.selectedColor.a;

	for (let i = 0; i < steps; i++) {
		const currentPoint = std.add(start, std.mul(direction, std.mul(stepSize, d.f32(i))));
		const clampedPoint = std.clamp(currentPoint, d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0));
		let keep = d.f32(std.allEq(currentPoint, clampedPoint));
		const sample = textureSampleLevel(cameraBindLayout.$.weightTexture, cameraBindLayout.$.weightSampler, clampedPoint, 0);

		// This does not seem to have an effect
		// if (sample.a <= 0.0) {
		// 	continue;
		// }

		const outside = std.distance(targetColorOklab, sample.rgb) < targetDistance;
		keep = std.mul(keep, std.max(0.3, d.f32(outside)));


		const oklabColor = sample.rgb;
		const remapped = sample.a / wRange;
		const expTerm = std.exp(std.mul(-remapped, stepSize));
		const alpha = std.max(std.sub(d.f32(1.0), expTerm), 0.0);

		const invAccAlpha = 1.0 - acc.w;
		const colorAlpha = d.vec4f(std.mul(oklabColor, alpha), alpha);
		acc = std.add(acc, std.mul(keep, std.mul(colorAlpha, invAccAlpha)));
		if (acc.a > 0.99) {
			break;
		}
	}

	const keepPick = d.f32(acc.w > 0.0);
	const pureColorOklab = std.div(acc.xyz, std.max(d.f32(0.0001), acc.w));
	const pureColorRgb = oklab_to_srgb(pureColorOklab);
	const colorRgb = std.add(std.mul(pureColorRgb, acc.w), std.mul(bg.xyz, std.sub(d.f32(1.0), acc.w)));

	return {
		color: d.vec4f(colorRgb, 1.0),
		pick: std.mul(d.vec4f(pureColorRgb, 1.0), keepPick),
	};
};

export const filterSlot = tgpu.slot<(uv: d.v2f, color: d.v4f) => d.v4f>();

export const filterBindLayout = tgpu.bindGroupLayout({
	filterOptions: { uniform: filterOptions },
	texture: { texture: d.texture2d() },
	sampler: { sampler: 'filtering' }
});


export const filterFragmentOutput = d.struct({
	color: d.vec4f,
})

export const filterFragment = ({ uv }: { uv: d.v2f }): d.Infer<typeof filterFragmentOutput> => {
	'use gpu';
	const color = textureSample(filterBindLayout.$.texture, filterBindLayout.$.sampler, uv);
	// const processedColor = filterSlot.$(uv, color);
	const processedColor = contrastFilter(uv, saturationFilter(uv, color));
	return {
		color: processedColor,
	};
}

export const saturationFilter = (uv: d.v2f, color: d.v4f): d.v4f => {
	'use gpu';
	const saturation = filterBindLayout.$.filterOptions.saturation;
	const oklab = srgb_to_oklab(color.rgb);
	oklab.y = std.mul(oklab.y, saturation);
	oklab.z = std.mul(oklab.z, saturation);
	return d.vec4f(oklab_to_srgb(oklab), color.a);
}

export const contrastFilter = (uv: d.v2f, color: d.v4f): d.v4f => {
	'use gpu';
	const contrast = filterBindLayout.$.filterOptions.contrast;
	const oklab = srgb_to_oklab(color.rgb);
	oklab.x = std.mul((oklab.x - 0.5), contrast) + 0.5;
	return d.vec4f(oklab_to_srgb(oklab), color.a);
}