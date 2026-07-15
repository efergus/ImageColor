import {
	d,
	type RenderFlag,
	type SampledFlag,
	type StorageFlag,
	type TgpuBuffer,
	type TgpuFixedSampler,
	type TgpuQuerySet,
	type TgpuRoot,
	type TgpuTexture
} from 'typegpu';
import { once, onceBindGroup } from '../../lib/gpu/gpu_utils';
import {
	computeOptions,
	filterBindLayout,
	filterFragment,
	filterOptions,
	quadVertex,
	textureRenderLayout,
	weightCalculation,
	weightCalculationLayout,
	processWeights,
	weightTransferLayout,
	weightTextureFormat,
	blur,
	weightProcessingLayout,
	imageFragment,
	triangleFragment,
	cameraBindLayout,
	cameraUniform,
	cloudCompositeFragment,
	cloudCompositeLayout,
	cloudCompositeOptions,
	colorSpaceSlot,
	linearRgbColorSpace,
	srgbColorSpace,
	oklabColorSpace,
	colorSpaceInverseSlot,
	linearRgbColorSpaceInverse,
	srgbColorSpaceInverse,
	oklabColorSpaceInverse,
	hsvColorSpace,
	hsvColorSpaceInverse,
	hslColorSpace,
	hslColorSpaceInverse,
	meshPositionLayout,
	meshBindLayout,
	meshVertex,
	meshFragment,
	meshOptions
} from './shaders';
import { ColorSpace } from './color_utils';
import type { Mesh } from './geometry';

// A render pass output: either a texture to draw into or a canvas context.
type RenderTarget = (TgpuTexture & RenderFlag) | GPUCanvasContext;

// TgpuTexture's default props assume a 2d texture, so 3d textures need their
// dimension spelled out for typegpu to accept 3d views of them.
type Texture3d = TgpuTexture<{
	size: readonly number[];
	format: GPUTextureFormat;
	dimension: '3d';
	sampleCount?: 1 | undefined;
}>;

let querySet: TgpuQuerySet<'timestamp'> | null = null;
const querySetNames: Map<string, number> = new Map();

const timestampOptions = (root: TgpuRoot, name: string) => {
	if (querySet === null) {
		querySet = root.createQuerySet('timestamp', 64);
	}

	let index = querySetNames.get(name);
	if (index === undefined) {
		index = querySetNames.size;
		if (index >= 32) {
			console.error(`Too many pipelines! Could not time ${name}`);
		}
		querySetNames.set(name, index);
	}
	return {
		querySet,
		beginningOfPassWriteIndex: index * 2,
		endOfPassWriteIndex: index * 2 + 1
	};
};

const querySetTimings: Map<string, number[]> = new Map();

export const readTimings = async () => {
	if (!querySet) {
		console.warn('Query set not available');
		return;
	}
	if (querySet.available) {
		querySet.resolve();
		const values = await querySet.read();
		for (const name of querySetNames.keys()) {
			const index = querySetNames.get(name)! * 2;

			const start = values[index];
			const end = values[index + 1];
			const time = Number(end - start);
			const past = querySetTimings.get(name) ?? [];
			if (past.length && past[past.length - 1] === time) {
				continue;
			}

			past.push(time);
			querySetTimings.set(name, past);
			console.log(`Pass ${name}: ${time / 1e6} ms`);
		}
	} else {
		console.warn('querySet not available');
	}
};

export const filterTexture = (
	root: TgpuRoot,
	inputTexture: TgpuTexture & SampledFlag,
	inputSampler: TgpuFixedSampler,
	options: d.Infer<typeof filterOptions>
) => {
	const format = 'rgba16float';

	const { optionsBuffer, pipeline } = once(filterTexture, () => {
		const pipeline = root
			.createRenderPipeline({
				primitive: { topology: 'triangle-list' },
				vertex: quadVertex,
				fragment: filterFragment,
				targets: {
					color: { format }
				}
			})
			.withTimestampWrites(timestampOptions(root, 'filterTexture'));
		const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
		return { optionsBuffer, pipeline };
	});

	const { outputTexture, outputView } = once(
		[filterTexture, options.textureSize.x, options.textureSize.y],
		() => {
			const outputTexture = root
				.createTexture({
					size: [options.textureSize.x, options.textureSize.y],
					format
				})
				.$usage('render', 'sampled', 'storage');
			const outputView = outputTexture.createView('render');
			return { outputTexture, outputView };
		}
	);

	optionsBuffer.write(options);

	const inputTextureView = once([filterTexture, inputTexture], () => {
		return inputTexture.createView(d.texture2d());
	});

	const filterBindGroup = onceBindGroup(root, filterBindLayout, {
		texture: inputTextureView,
		sampler: inputSampler,
		filterOptions: optionsBuffer
	});

	pipeline
		.with(filterBindGroup)
		.withColorAttachment({
			color: { view: outputView }
		})
		.draw(6);

	return outputTexture;
};

export const calculateWeights = (
	root: TgpuRoot,
	inputTexture: TgpuTexture & StorageFlag,
	colorSpace: ColorSpace,
	options: d.Infer<typeof computeOptions>
) => {
	const { pipeline, optionsBuffer } = once([calculateWeights, colorSpace], () => {
		const pipeline = root
			.with(colorSpaceSlot, colorSpacesConfig[colorSpace].forward)
			.createGuardedComputePipeline(weightCalculation)
			.withTimestampWrites(timestampOptions(root, 'calculateWeights'));
		const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
		return { pipeline, optionsBuffer };
	});

	const { outputBuffer } = once([calculateWeights, options.tableSize], () => {
		const outputBuffer = root
			.createBuffer(d.arrayOf(d.u32, options.tableSize * options.tableSize * options.tableSize * 4))
			.$usage('storage');
		return { outputBuffer };
	});

	optionsBuffer.write(options);
	outputBuffer.clear();

	const inputTextureView = once([calculateWeights, inputTexture], () => {
		return inputTexture.createView(d.textureStorage2d(weightTextureFormat, 'read-only'));
	});

	const bindGroup = onceBindGroup(root, weightCalculationLayout, {
		image: inputTextureView,
		weights: outputBuffer,
		options: optionsBuffer
	});

	pipeline.with(bindGroup).dispatchThreads(options.textureSize.x, options.textureSize.y);

	return outputBuffer;
};

export const processWeightTexture = (
	root: TgpuRoot,
	weightsBuffer: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag,
	options: d.Infer<typeof computeOptions>
) => {
	const { pipeline, optionsBuffer } = once(processWeightTexture, () => {
		const pipeline = root
			.createGuardedComputePipeline(processWeights)
			.withTimestampWrites(timestampOptions(root, 'processWeightTexture'));
		const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
		return { pipeline, optionsBuffer };
	});

	const { outputTexture } = once([processWeightTexture, options.tableSize], () => {
		const outputTexture = root
			.createTexture({
				size: [options.tableSize, options.tableSize, options.tableSize],
				format: weightTextureFormat,
				dimension: '3d'
			})
			.$usage('render', 'sampled', 'storage');
		return { outputTexture };
	});

	optionsBuffer.write(options);

	const bindGroup = onceBindGroup(root, weightTransferLayout, {
		options: optionsBuffer,
		weights: weightsBuffer,
		outputTexture: outputTexture
	});

	pipeline.with(bindGroup).dispatchThreads(options.tableSize, options.tableSize, options.tableSize);

	return outputTexture;
};

export const blurWeightTexture = (
	root: TgpuRoot,
	inputTexture: Texture3d & SampledFlag,
	options: d.Infer<typeof filterOptions>,
	tableSize: number
) => {
	const { pipeline, optionsBuffer } = once(blurWeightTexture, () => {
		const pipeline = root
			.createGuardedComputePipeline(blur)
			.withTimestampWrites(timestampOptions(root, 'blurWeightTexture'));
		const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
		return { pipeline, optionsBuffer };
	});

	const { outputTexture } = once([blurWeightTexture, tableSize], () => {
		const outputTexture = root
			.createTexture({
				size: [tableSize, tableSize, tableSize],
				format: weightTextureFormat,
				dimension: '3d'
			})
			.$usage('sampled', 'storage');
		return { outputTexture };
	});

	optionsBuffer.write(options);

	const inputTextureView = once([blurWeightTexture, inputTexture], () => {
		return inputTexture.createView(d.texture3d());
	});

	const bindGroup = onceBindGroup(root, weightProcessingLayout, {
		options: optionsBuffer,
		inputTexture: inputTextureView,
		outputTexture: outputTexture.createView(d.textureStorage3d(weightTextureFormat, 'write-only'))
	});

	pipeline.with(bindGroup).dispatchThreads(tableSize, tableSize, tableSize);

	return outputTexture;
};

export const renderImage = (
	root: TgpuRoot,
	inputTexture: TgpuTexture & SampledFlag,
	inputSampler: TgpuFixedSampler,
	outputView: RenderTarget,
	options: d.Infer<typeof filterOptions>
) => {
	const { pipeline, optionsBuffer } = once(renderImage, () => {
		const pipeline = root
			.createRenderPipeline({
				primitive: { topology: 'triangle-list' },
				vertex: quadVertex,
				fragment: imageFragment
			})
			.withTimestampWrites(timestampOptions(root, 'renderImage'));
		const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
		return { pipeline, optionsBuffer };
	});

	const inputTextureView = once([renderImage, inputTexture], () => {
		return inputTexture.createView(d.texture2d());
	});

	const bindGroup = onceBindGroup(root, textureRenderLayout, {
		texture: inputTextureView,
		sampler: inputSampler,
		options: optionsBuffer
	});

	optionsBuffer.write(options);

	pipeline
		.with(bindGroup)
		.withColorAttachment({
			view: outputView
		})
		.draw(6);
};

export const colorSpacesConfig = {
	[ColorSpace.oklab]: { label: 'Oklab', forward: oklabColorSpace, inverse: oklabColorSpaceInverse },
	[ColorSpace.hsv]: { label: 'HSV', forward: hsvColorSpace, inverse: hsvColorSpaceInverse },
	[ColorSpace.hsl]: { label: 'HSL', forward: hslColorSpace, inverse: hslColorSpaceInverse },
	[ColorSpace.srgb]: { label: 'sRGB', forward: srgbColorSpace, inverse: srgbColorSpaceInverse },
	[ColorSpace.linear_rgb]: {
		label: 'Linear RGB',
		forward: linearRgbColorSpace,
		inverse: linearRgbColorSpaceInverse
	}
};

// Rasterizes an arbitrary set of scene meshes (each a flat-colored,
// non-indexed triangle list) into a color texture and a depth buffer, using
// the same camera as the cloud raymarch. The plane that used to be
// hardcoded here is now just one more mesh, provided by the caller.
export const renderRasterScene = (
	root: TgpuRoot,
	outputTexture: TgpuTexture & RenderFlag,
	depthTexture: TgpuTexture & RenderFlag & SampledFlag,
	camera: d.Infer<typeof cameraUniform>,
	meshes: Mesh[]
) => {
	const { pipeline, cameraBuffer } = once(renderRasterScene, () => {
		const pipeline = root.createRenderPipeline({
			vertex: meshVertex,
			fragment: meshFragment,
			attribs: { position: meshPositionLayout.attrib },
			targets: { format: 'rgba8unorm' },
			depthStencil: {
				format: 'depth24plus',
				depthWriteEnabled: true,
				depthCompare: 'less'
			}
		});
		const cameraBuffer = root.createBuffer(cameraUniform).$usage('uniform');
		return { pipeline, cameraBuffer };
	});

	cameraBuffer.write(camera);

	// Share one command encoder (and one queue.submit) across every mesh's
	// pass instead of letting each .draw() open its own encoder. A query
	// index can only be timestamp-written once per encoder, so timing is
	// only attached to the first mesh's pass.
	const encoder = root.device.createCommandEncoder();

	meshes.forEach((mesh, i) => {
		const { vertexBuffer, bindGroup } = once([renderRasterScene, mesh], () => {
			const vertexBuffer = root
				.createBuffer(meshPositionLayout.schemaForCount(mesh.vertices.length), mesh.vertices)
				.$usage('vertex');
			const colorBuffer = root.createBuffer(meshOptions, { color: mesh.color }).$usage('uniform');
			const bindGroup = root.createBindGroup(meshBindLayout, {
				cameraUniform: cameraBuffer,
				meshOptions: colorBuffer
			});
			return { vertexBuffer, bindGroup };
		});

		const loadOp = i === 0 ? 'clear' : 'load';
		let meshPipeline = pipeline.with(encoder);
		if (i === 0) {
			meshPipeline = meshPipeline.withTimestampWrites(timestampOptions(root, 'renderRasterScene'));
		}

		meshPipeline
			.withColorAttachment({ view: outputTexture, loadOp })
			.withDepthStencilAttachment({
				view: depthTexture,
				depthClearValue: 1.0,
				depthLoadOp: loadOp,
				depthStoreOp: 'store'
			})
			.with(meshPositionLayout, vertexBuffer)
			.with(bindGroup)
			.draw(mesh.vertices.length);
	});

	root.device.queue.submit([encoder.finish()]);
};

export const renderColorCloud = (
	root: TgpuRoot,
	inputTexture: Texture3d & SampledFlag,
	inputSampler: TgpuFixedSampler,
	rasterDepthTexture: TgpuTexture & SampledFlag,
	outputTexture: TgpuTexture & RenderFlag,
	pickTexture: TgpuTexture & RenderFlag,
	colorSpace: ColorSpace,
	camera: d.Infer<typeof cameraUniform>
) => {
	const { pipeline, cameraBuffer } = once([renderColorCloud, colorSpace], () => {
		const pipeline = root
			.with(colorSpaceSlot, colorSpacesConfig[colorSpace].forward)
			.with(colorSpaceInverseSlot, colorSpacesConfig[colorSpace].inverse)
			.createRenderPipeline({
				primitive: { topology: 'triangle-list' },
				vertex: quadVertex,
				fragment: triangleFragment,
				targets: {
					color: { format: 'rgba8unorm' },
					pick: { format: 'rgba8unorm' }
				}
			})
			.withTimestampWrites(timestampOptions(root, 'renderColorCloud'));
		const cameraBuffer = root.createBuffer(cameraUniform).$usage('uniform');
		return { pipeline, cameraBuffer };
	});

	const inputTextureView = once([renderColorCloud, inputTexture], () => {
		return inputTexture.createView(d.texture3d());
	});

	const rasterDepthView = once([renderColorCloud, rasterDepthTexture], () => {
		return rasterDepthTexture.createView(d.textureDepth2d());
	});

	const cameraBindGroup = onceBindGroup(root, cameraBindLayout, {
		cameraUniform: cameraBuffer,
		weightTexture: inputTextureView,
		weightSampler: inputSampler,
		rasterDepth: rasterDepthView
	});

	cameraBuffer.write(camera);

	pipeline
		.with(cameraBindGroup)
		.withColorAttachment({
			color: { view: outputTexture },
			pick: { view: pickTexture }
		})
		.draw(6);
};

export const compositeColorCloud = (
	root: TgpuRoot,
	cloudTexture: TgpuTexture & SampledFlag,
	pickTexture: TgpuTexture & SampledFlag,
	rasterTexture: TgpuTexture & SampledFlag,
	inputSampler: TgpuFixedSampler,
	outputView: RenderTarget,
	options: d.Infer<typeof cloudCompositeOptions>
) => {
	const { pipeline, optionsBuffer } = once(compositeColorCloud, () => {
		const pipeline = root
			.createRenderPipeline({
				primitive: { topology: 'triangle-list' },
				vertex: quadVertex,
				fragment: cloudCompositeFragment
			})
			.withTimestampWrites(timestampOptions(root, 'compositeColorCloud'));
		const optionsBuffer = root.createBuffer(cloudCompositeOptions).$usage('uniform');
		return { pipeline, optionsBuffer };
	});

	const cloudTextureView = once([compositeColorCloud, cloudTexture], () => {
		return cloudTexture.createView(d.texture2d());
	});

	const pickTextureView = once([compositeColorCloud, pickTexture], () => {
		return pickTexture.createView(d.texture2d());
	});

	const rasterTextureView = once([compositeColorCloud, rasterTexture], () => {
		return rasterTexture.createView(d.texture2d());
	});

	const bindGroup = onceBindGroup(root, cloudCompositeLayout, {
		cloudTexture: cloudTextureView,
		pickTexture: pickTextureView,
		rasterTexture: rasterTextureView,
		sampler: inputSampler,
		options: optionsBuffer
	});

	optionsBuffer.write(options);

	pipeline
		.with(bindGroup)
		.withColorAttachment({
			view: outputView
		})
		.draw(6);
};
