import tgpu, { d, std } from 'typegpu';
import { textureSample, textureLoad, textureSampleLevel, textureDimensions } from 'typegpu/std';
import { ColorSpace, hsl_to_srgb, hsv_to_srgb, linear_rgb_to_oklab, linear_rgb_to_srgb, oklab_to_linear_rgb, oklab_to_srgb, srgb_to_hsl, srgb_to_hsv, srgb_to_linear_rgb, srgb_to_oklab } from './color_utils';
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

export const colorSpaceSlot = tgpu.slot<(color: d.v3f) => d.v3f>();
export const colorSpaceInverseSlot = tgpu.slot<(color: d.v3f) => d.v3f>();

export const srgbColorSpace = (color: d.v3f) => {
	'use gpu';
	return d.vec3f(color);
}

export const linearRgbColorSpace = (color: d.v3f) => {
	'use gpu';
	return srgb_to_linear_rgb(color);
}

export const oklabColorSpace = (color: d.v3f) => {
	'use gpu';
	const raw = srgb_to_oklab(color);
	const transformed = d.vec3f(raw.y + 0.5, raw.x, raw.z + 0.5);
	return transformed;
}

export const srgbColorSpaceInverse = (color: d.v3f) => {
	'use gpu';
	return d.vec3f(color);
}

export const linearRgbColorSpaceInverse = (color: d.v3f) => {
	'use gpu';
	return linear_rgb_to_srgb(color);
}

export const oklabColorSpaceInverse = (color: d.v3f) => {
	'use gpu';
	const raw = d.vec3f(color.y, color.x - 0.5, color.z - 0.5);
	const transformed = oklab_to_srgb(raw);
	return transformed;
}

export const hsvColorSpace = (color: d.v3f) => {
	'use gpu';
	const raw = srgb_to_hsv(color);
	const theta = std.mul(d.f32(raw.x), d.f32(Math.PI / 180.0));
	
	// In HSV, the color space is a single cone.
	// Radius depends on both Saturation (raw.y) and Value (raw.z).
	// Value maps directly to the Y axis.
	return d.vec3f(
		std.cos(theta) * raw.y * raw.z * 0.5 + 0.5,
		raw.z,
		std.sin(theta) * raw.y * raw.z * 0.5 + 0.5
	);
}

export const hsvColorSpaceInverse = (color: d.v3f) => {
	'use gpu';
	const cx = (color.x - 0.5) * 2.0;
	const cz = (color.z - 0.5) * 2.0;
	
	const theta = std.atan2(cz, cx);
	let hue = theta * d.f32(180.0 / Math.PI);
	if (hue < d.f32(0.0)) {
		hue = hue + 360.0;
	}
	
	const value = color.y;
	const radius = std.length(d.vec2f(cx, cz));
	
	let saturation = d.f32(0.0);
	if (value > d.f32(0.0)) {
		saturation = std.min(radius / value, d.f32(1.0));
	}
	
	return hsv_to_srgb(d.vec3f(hue, saturation, value));
}

export const hslColorSpace = (color: d.v3f) => {
	'use gpu';
	const raw = srgb_to_hsl(color);
	const theta = std.mul(d.f32(raw.x), d.f32(Math.PI / 180.0));
	
	// In HSL, radius (chroma) goes to 0 at both ends of the Lightness axis.
	const chroma = std.mul(raw.y, std.sub(d.f32(1.0), std.abs(std.sub(std.mul(raw.z, d.f32(2.0)), d.f32(1.0)))));
	const radius = std.mul(chroma, d.f32(0.5)); // Scale to max radius 0.5
	
	return d.vec3f(
		std.cos(theta) * radius + 0.5,
		raw.z, // Lightness maps directly to the Y axis
		std.sin(theta) * radius + 0.5
	);
}

export const hslColorSpaceInverse = (color: d.v3f) => {
	'use gpu';
	const cx = (color.x - 0.5) * 2.0;
	const cz = (color.z - 0.5) * 2.0;
	
	const theta = std.atan2(cz, cx);
	let hue = theta * d.f32(180.0 / Math.PI);
	if (hue < d.f32(0.0)) {
		hue = hue + 360.0;
	}
	
	const lightness = color.y;
	const radius = std.length(d.vec2f(cx, cz)); // This is actually half the chroma
	
	const maxChroma = std.sub(d.f32(1.0), std.abs(std.sub(std.mul(lightness, d.f32(2.0)), d.f32(1.0))));
	let saturation = d.f32(0.0);
	if (maxChroma > d.f32(0.0)) {
		saturation = std.min(radius / maxChroma, d.f32(1.0)); // Because maxChroma here is already equivalent to max radius
	}
	
	return hsl_to_srgb(d.vec3f(hue, saturation, lightness));
}

export const weightCalculation = (x: number, y: number) => {
	'use gpu';
	const val = textureLoad(weightCalculationLayout.$.image, d.vec2u(x, y));
	const colorSpace = colorSpaceSlot.$(val.xyz);
	const tableSize = weightCalculationLayout.$.options.tableSize;
	const x_ = d.u32(std.round(colorSpace.x * d.f32(tableSize - 1)));
	const y_ = d.u32(std.round(colorSpace.y * d.f32(tableSize - 1)));
	const z_ = d.u32(std.round(colorSpace.z * d.f32(tableSize - 1)));
	const colorSpace_255 = std.ceil(std.mul(colorSpace, d.f32(255.0)));
	const idx = x_ + y_ * tableSize + z_ * tableSize * tableSize;
	const byteOffset = idx * 4;
	std.atomicAdd(weightCalculationLayout.$.weights[byteOffset + 0], d.u32(colorSpace_255.x));
	std.atomicAdd(weightCalculationLayout.$.weights[byteOffset + 1], d.u32(colorSpace_255.y));
	std.atomicAdd(weightCalculationLayout.$.weights[byteOffset + 2], d.u32(colorSpace_255.z));
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
	const rgb = d.vec4f(
		d.f32(r) / countf / 255.0,
		d.f32(g) / countf / 255.0,
		d.f32(b) / countf / 255.0,
		count
	);

	const tableSizef = d.f32(tableSize - 1);
	const alt = d.vec4f(d.f32(x) / tableSizef, d.f32(y) / tableSizef, d.f32(z) / tableSizef, 0.0);
	const finalColor = std.mix(alt, rgb, std.min(d.f32(count), 1.0));

	std.textureStore(weightTransferLayout.$.outputTexture, d.vec3u(x, y, z), finalColor);
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
	const originalColor = textureLoad(weightProcessingLayout.$.inputTexture, d.vec3u(x, y, z), 0);
	return originalColor;
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
});

export const cameraBindLayout = tgpu.bindGroupLayout({
	cameraUniform: { uniform: cameraUniform },
	weightTexture: { texture: d.texture3d() },
	weightSampler: { sampler: 'filtering' },
	rasterDepth: { texture: d.textureDepth2d() },
});

// Near/far planes for the rasterized scene's depth buffer. Shared by the
// raster pass (encoding) and the cloud raymarch (decoding back to a distance).
export const rasterNear = 0.05;
export const rasterFar = 8.0;

export const rasterLayout = tgpu.bindGroupLayout({
	cameraUniform: { uniform: cameraUniform },
});

// Projects a world-space point with the exact same camera geometry as
// cameraRay (tan(halfFov) = 0.6 vertically and 0.6 * aspect horizontally),
// so rasterized geometry and raymarched volume line up pixel-for-pixel.
// Depth uses the standard perspective mapping
// depth = (far / (far - near)) * (1 - near / viewZ), which interpolates
// correctly across triangles (linear in 1/viewZ).
export const worldToClip = (p: d.v3f, yaw: number, pitch: number, radius: number, aspect: number) => {
	'use gpu';
	const center = d.vec3f(0.5, 0.5, 0.5);
	const transform = cameraRotation(yaw, pitch);
	const eye = eyeLocation(transform, radius, center);
	const up = std.mul(transform, d.vec3f(0.0, 1.0, 0.0));
	const fwd = std.normalize(std.sub(center, eye));
	const right = std.normalize(std.cross(fwd, up));
	const v = std.sub(p, eye);
	const viewX = std.dot(v, right);
	const viewY = std.dot(v, up);
	const viewZ = std.dot(v, fwd);
	const zClip = (viewZ - rasterNear) * (rasterFar / (rasterFar - rasterNear));
	return d.vec4f(viewX / (aspect * 0.6), viewY / 0.6, zClip, viewZ);
};

export const rasterQuadVertex = ({ $vertexIndex: vid }: { $vertexIndex: number }) => {
	'use gpu';
	// A 1x1 vertical plane through the middle of the cloud's unit box.
	const positions = [
		d.vec3f(0.0, 0.0, 0.5),
		d.vec3f(1.0, 0.0, 0.5),
		d.vec3f(1.0, 1.0, 0.5),
		d.vec3f(0.0, 0.0, 0.5),
		d.vec3f(1.0, 1.0, 0.5),
		d.vec3f(0.0, 1.0, 0.5),
	];
	const camera = rasterLayout.$.cameraUniform;
	const worldPosition = positions[vid];
	return {
		$position: worldToClip(worldPosition, camera.yaw, camera.pitch, camera.radius, camera.aspect),
		uv: d.vec2f(worldPosition.x, worldPosition.y)
	};
};

export const rasterFragmentOutput = d.struct({
	color: d.vec4f,
});

// uv is unused for now (the quad is solid black), but it is the hook for
// texturing the plane with actual rasterized content later.
export const rasterFragment = ({ uv: _uv }: { uv: d.v2f }): d.Infer<typeof rasterFragmentOutput> => {
	'use gpu';
	return {
		color: d.vec4f(0.0, 0.0, 0.0, 1.0)
	};
};

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

export const imageFragment = ({ uv }: { uv: d.v2f }) => {
	'use gpu';
	const original = textureSample(textureRenderLayout.$.texture, textureRenderLayout.$.sampler, uv);
	const targetDistance = textureRenderLayout.$.options.selectedColor.a;
	const targetColorOklab = srgb_to_oklab(textureRenderLayout.$.options.selectedColor.rgb);
	const originalOklab = srgb_to_oklab(original.rgb);

	const greyness = std.clamp((std.distance(targetColorOklab, originalOklab) - targetDistance) * 100.0, 0.0, 0.8);
	const outColor = std.mix(original.rgb, d.vec3f(0.4), d.f32(greyness));
	return d.vec4f(outColor, original.a);
}

export const triangleFragmentOutput = d.struct({
	color: d.vec4f,
	pick: d.vec4f,
});

export const triangleFragment = ({ uv, $position: fragCoord }: { uv: d.v2f; $position: d.v4f }): d.Infer<typeof triangleFragmentOutput> => {
	'use gpu';

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
	let max = intersection.y;

	// The rasterized scene occludes the cloud: decode this pixel's depth-buffer
	// value back into a distance along the ray and stop marching there.
	// fragCoord is in framebuffer pixels, which matches the raster pass's
	// depth texture texel-for-texel since both render at the same size.
	const rasterDepth = textureLoad(
		cameraBindLayout.$.rasterDepth,
		d.vec2u(d.u32(fragCoord.x), d.u32(fragCoord.y)),
		0
	);
	if (rasterDepth < 1.0) {
		const viewZ = (rasterNear * rasterFar) / (rasterFar - rasterDepth * (rasterFar - rasterNear));
		const fwd = std.normalize(std.sub(center, eye));
		max = std.min(max, viewZ / std.dot(direction, fwd));
	}

	if (min >= max) {
		return {
			color: d.vec4f(0.0, 0.0, 0.0, 0.0),
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

	for (let i = 0; i < steps; i++) {
		const currentPoint = std.add(start, std.mul(direction, std.mul(stepSize, d.f32(i))));
		const clampedPoint = std.clamp(currentPoint, d.vec3f(0.0, 0.0, 0.0), d.vec3f(1.0, 1.0, 1.0));
		const keep = d.f32(std.allEq(currentPoint, clampedPoint));
		const sample = textureSampleLevel(cameraBindLayout.$.weightTexture, cameraBindLayout.$.weightSampler, clampedPoint, 0);

		// This does not seem to have an effect
		// if (sample.a <= 0.0) {
		// 	continue;
		// }

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
	const pureColorSpaceColor = std.div(acc.xyz, std.max(d.f32(0.0001), acc.w));
	const pureColor = colorSpaceInverseSlot.$(pureColorSpaceColor);

	return {
		color: std.mul(d.vec4f(pureColor, acc.w), keepPick),
		pick: std.mul(d.vec4f(pureColor, 1.0), keepPick),
	};
};

export const cloudCompositeOptions = d.struct({
	selectedColor: d.vec4f,
	bgColor: d.f32,
});

export const cloudCompositeLayout = tgpu.bindGroupLayout({
	cloudTexture: { texture: d.texture2d() },
	pickTexture: { texture: d.texture2d() },
	rasterTexture: { texture: d.texture2d() },
	sampler: { sampler: 'filtering' },
	options: { uniform: cloudCompositeOptions }
});

export const cloudCompositeFragment = ({ uv }: { uv: d.v2f }) => {
	'use gpu';
	const cloud = textureSample(cloudCompositeLayout.$.cloudTexture, cloudCompositeLayout.$.sampler, uv);
	const pick = textureSample(cloudCompositeLayout.$.pickTexture, cloudCompositeLayout.$.sampler, uv);
	const raster = textureSample(cloudCompositeLayout.$.rasterTexture, cloudCompositeLayout.$.sampler, uv);

	const targetDistance = cloudCompositeLayout.$.options.selectedColor.a;
	const targetColorOklab = srgb_to_oklab(cloudCompositeLayout.$.options.selectedColor.rgb);
	const pickOklab = srgb_to_oklab(pick.rgb);

	const fade = std.clamp((std.distance(targetColorOklab, pickOklab) - targetDistance) * 100.0, 0.0, 0.7);
	const alpha = cloud.a * (1.0 - fade);

	const bgLightness = cloudCompositeLayout.$.options.bgColor;
	const bg = d.vec3f(bgLightness, bgLightness, bgLightness);
	// The ray was clamped at the rasterized surface, so the cloud only holds
	// what lies in front of it: composite cloud over raster over background.
	const base = std.mix(bg, raster.rgb, raster.a);
	const outColor = std.mix(base, cloud.rgb, d.f32(alpha));
	return d.vec4f(outColor, 1.0);
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