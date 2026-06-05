<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import tgpu, {
		NotUniformError,
		type RenderFlag,
		type StorageFlag,
		type TgpuBuffer,
		type TgpuGuardedComputePipeline,
		type TgpuRenderPipeline,
		type TgpuRoot,
		type TgpuTexture,
		type UniformFlag
	} from 'typegpu';
	import * as d from 'typegpu/data';
	import beeCloseImg from '$lib/assets/bee_close.jpg';

	import {
		triangleVertex,
		triangleFragment,
		cameraBindLayout,
		cameraUniform,
		calculateWeights,
		computeOptions,
		computeBindLayout,
		processWeights,
		computePostProcessingLayout,
		imageFragment,
		cameraRay,
		blur,
		texture3dProcessingLayout
	} from './shaders';
	import { sampler } from 'typegpu/data';
	import { linear_rgb_to_oklab, oklab_to_linear_rgb } from './color_utils';

	let colorCanvas: HTMLCanvasElement;
	let imageCanvas: HTMLCanvasElement;
	let animationFrameId: number;
	let updated = $state(0.0);
	let yaw = $state(0.0);
	let pitch = $state(0.2);
	let radius = $state(2);
	let steps = $state(16);
	let sensitivity = $state(50.0);
	let tableSize = $state(64);
	let color = $state('rgba(0, 0, 0, 1)');

	let gpuState: {
		root: TgpuRoot;
		weightCalculationPipeline: TgpuGuardedComputePipeline;
		weightProcessingPipeline: TgpuGuardedComputePipeline;
		blurPipeline: TgpuGuardedComputePipeline;
		renderPipeline: TgpuRenderPipeline;
		imageRenderPipeline: TgpuRenderPipeline;
		context: GPUCanvasContext;
		imageContext: GPUCanvasContext;
		optionsBuffer: TgpuBuffer<typeof computeOptions> & UniformFlag;
		cameraUniformBuffer: TgpuBuffer<typeof cameraUniform> & UniformFlag;
		imageBitmap: ImageBitmap;
		imageTexture: TgpuTexture & StorageFlag;
		pickTexture: TgpuTexture & RenderFlag;
		pickStagingBuffer: TgpuBuffer<d.WgslArray<d.U32>>;
		linearSampler: GPUSampler;
	} | null = null;

	let isReadingBack = false;

	console.log(oklab_to_linear_rgb(linear_rgb_to_oklab(d.vec3f(0.5, 0.2, 0.1))));
	console.log(linear_rgb_to_oklab(oklab_to_linear_rgb(d.vec3f(0.5, 0.2, 0.1))));

	async function readColorAtPixel(px: number, py: number) {
		if (isReadingBack || !gpuState || !colorCanvas) return;
		const { root, pickTexture, pickStagingBuffer } = gpuState;

		px = Math.max(0, Math.min(colorCanvas.width - 1, Math.floor(px)));
		py = Math.max(0, Math.min(colorCanvas.height - 1, Math.floor(py)));

		isReadingBack = true;
		const enc = root.device.createCommandEncoder();
		enc.copyTextureToBuffer(
			{ texture: root.unwrap(pickTexture), origin: [px, py, 0] },
			{ buffer: pickStagingBuffer.buffer, bytesPerRow: 256 },
			[1, 1, 1]
		);
		root.device.queue.submit([enc.finish()]);

		try {
			const data = await pickStagingBuffer.read();
			const pixel = data[0] ?? 0;
			const r = pixel & 0xff;
			const g = (pixel >>> 8) & 0xff;
			const b = (pixel >>> 16) & 0xff;
			const a = (pixel >>> 24) & 0xff;

			console.log(`Hovered pick texture value: rgba(${r}, ${g}, ${b}, ${a})`);
			color = `rgba(${r}, ${g}, ${b}, ${a})`;
		} catch (e) {
			console.error(e);
		} finally {
			isReadingBack = false;
		}
	}

	let weightBuffers: Map<number, TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag> = new Map();
	let weightTextures: Map<
		number,
		TgpuTexture<
			{
				size: [number, number, number];
				format: 'rgba16float';
				dimension: '3d';
			} & StorageFlag
		>
	> = new Map();

	let blurredWeightTextures: Map<
		number,
		TgpuTexture<
			{
				size: [number, number, number];
				format: 'rgba16float';
				dimension: '3d';
			} & StorageFlag
		>
	> = new Map();

	const imageUrlToBitmap = async (url: string) => {
		const img = new Image();
		img.src = url;
		await img.decode();
		return await createImageBitmap(img);
	};

	const bitmapToTexture = async (root: TgpuRoot, bitmap: ImageBitmap) => {
		const texture = root
			.createTexture({
				size: [bitmap.width, bitmap.height],
				format: 'rgba8unorm'
			})
			.$usage('render', 'sampled', 'storage');
		texture.write(bitmap);
		return texture;
	};

	const getWeightBuffer = (root: TgpuRoot, tableSize: number) => {
		if (weightBuffers.has(tableSize)) {
			return weightBuffers.get(tableSize)!;
		}
		const buffer = root
			.createBuffer(d.arrayOf(d.u32, tableSize * tableSize * tableSize * 4))
			.$usage('storage');
		weightBuffers.set(tableSize, buffer);
		return buffer;
	};

	const getWeightTexture = (root: TgpuRoot, tableSize: number) => {
		if (weightTextures.has(tableSize)) {
			return weightTextures.get(tableSize)!;
		}
		const texture = root
			.createTexture({
				size: [tableSize, tableSize, tableSize],
				format: 'rgba16float',
				dimension: '3d'
			})
			.$usage('render', 'sampled', 'storage');
		weightTextures.set(tableSize, texture);
		return texture;
	};

	const getBlurredWeightTexture = (root: TgpuRoot, tableSize: number) => {
		if (blurredWeightTextures.has(tableSize)) {
			return blurredWeightTextures.get(tableSize)!;
		}
		const texture = root
			.createTexture({
				size: [tableSize, tableSize, tableSize],
				format: 'rgba16float',
				dimension: '3d'
			})
			.$usage('render', 'sampled', 'storage');
		blurredWeightTextures.set(tableSize, texture);
		return texture;
	};

	const computeWeightTexture = (tableSize: number) => {
		if (!gpuState) {
			console.log('gpuState is null');
			return;
		}

		const {
			root,
			renderPipeline,
			weightCalculationPipeline,
			weightProcessingPipeline,
			blurPipeline,
			optionsBuffer,
			imageBitmap,
			imageTexture,
			cameraUniformBuffer,
			linearSampler
		} = gpuState;
		const weightsBuffer = getWeightBuffer(root, tableSize);
		const weightTexture = getWeightTexture(root, tableSize);
		const blurredWeightTexture = getBlurredWeightTexture(root, tableSize);

		const computeBindGroup = root.createBindGroup(computeBindLayout, {
			options: optionsBuffer,
			image: imageTexture,
			weights: weightsBuffer
		});

		weightCalculationPipeline
			.with(computeBindGroup)
			.dispatchThreads(imageBitmap.width, imageBitmap.height);

		const weightProcessingBindGroup = root.createBindGroup(computePostProcessingLayout, {
			options: optionsBuffer,
			weights: weightsBuffer,
			outputTexture: weightTexture
		});

		weightProcessingPipeline
			.with(weightProcessingBindGroup)
			.dispatchThreads(tableSize, tableSize, tableSize);

		const blurBindGroup = root.createBindGroup(texture3dProcessingLayout, {
			options: optionsBuffer,
			inputTexture: weightTexture.createView('sampled'),
			outputTexture: blurredWeightTexture
		});

		blurPipeline.with(blurBindGroup).dispatchThreads(tableSize, tableSize, tableSize);
	};

	const renderScene = () => {
		if (!gpuState) {
			console.log('gpuState is null');
			return;
		}
		requestAnimationFrame(renderScene);

		const now = Date.now();
		if (now - updated > 100) {
			return;
		}

		const {
			root,
			renderPipeline,
			imageRenderPipeline,
			cameraUniformBuffer,
			imageTexture,
			pickTexture,
			linearSampler,
			context,
			imageContext
		} = gpuState;
		const blurredWeightTexture = getBlurredWeightTexture(root, tableSize);

		cameraUniformBuffer.write({
			yaw,
			pitch,
			radius,
			aspect: colorCanvas.width / colorCanvas.height,
			steps,
			sensitivity
		});

		const bindGroup = root.createBindGroup(cameraBindLayout, {
			tex: imageTexture.createView('render'),
			samp: linearSampler,
			cameraUniform: cameraUniformBuffer,
			weightTexture: blurredWeightTexture.createView('render'),
			weightSampler: linearSampler
		});

		renderPipeline
			.with(bindGroup)
			.withColorAttachment({
				color: { view: context },
				pick: { view: pickTexture.createView('render') }
			})
			.draw(6);

		imageRenderPipeline.with(bindGroup).withColorAttachment({ view: imageContext }).draw(6);
	};

	onMount(async () => {
		if (!colorCanvas) return;

		try {
			// Initialize TypeGPU root
			const root = await tgpu.init({
				device: {
					optionalFeatures: ['timestamp-query']
				}
			});

			// Configure canvas context
			const context = root.configureContext({
				canvas: colorCanvas,
				alphaMode: 'premultiplied'
			});

			const imageContext = root.configureContext({
				canvas: imageCanvas,
				alphaMode: 'premultiplied'
			});

			const bitmap = await imageUrlToBitmap(beeCloseImg);
			const texture = await bitmapToTexture(root, bitmap);

			const pickTexture = root
				.createTexture({
					size: [colorCanvas.width, colorCanvas.height],
					format: 'rgba8unorm'
				})
				.$usage('render', 'sampled');

			const pickStagingBuffer = root
				.createBuffer(d.arrayOf(d.u32, 64))
				.$addFlags(GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

			const sampler = root.device.createSampler({
				magFilter: 'linear',
				minFilter: 'linear'
			});

			const weightCalculationPipeline = root
				.createGuardedComputePipeline(calculateWeights)
				.withPerformanceCallback((start, end) => {
					console.log(`calculateWeights: ${Number(end - start) / 1e6} ms`);
				});

			const weightProcessingPipeline = root
				.createGuardedComputePipeline(processWeights)
				.withPerformanceCallback((start, end) => {
					console.log(`processWeights: ${Number(end - start) / 1e6} ms`);
				});

			const blurPipeline = root
				.createGuardedComputePipeline(blur)
				.withPerformanceCallback((start, end) => {
					console.log(`blur: ${Number(end - start) / 1e6} ms`);
				});

			const renderPipeline = root
				.createRenderPipeline({
					primitive: { topology: 'triangle-list' },
					vertex: triangleVertex,
					fragment: triangleFragment,
					targets: {
						color: { format: navigator.gpu.getPreferredCanvasFormat() },
						pick: { format: 'rgba8unorm' }
					}
				})
				.withPerformanceCallback((start, end) => {
					console.log(`renderScene: ${Number(end - start) / 1e6} ms`);
				});

			const imageRenderPipeline = root.createRenderPipeline({
				primitive: { topology: 'triangle-list' },
				vertex: triangleVertex,
				fragment: imageFragment
			});

			const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
			optionsBuffer.write({ tableSize });

			const cameraUniformBuffer = root.createBuffer(cameraUniform).$usage('uniform');
			cameraUniformBuffer.write({
				yaw: 0.5,
				pitch: 0.2,
				radius: 2,
				aspect: colorCanvas.width / colorCanvas.height,
				steps,
				sensitivity
			});

			gpuState = {
				root,
				weightCalculationPipeline,
				weightProcessingPipeline,
				blurPipeline,
				renderPipeline,
				imageRenderPipeline,
				optionsBuffer,
				cameraUniformBuffer,
				context,
				imageContext,
				imageBitmap: bitmap,
				imageTexture: texture,
				pickTexture,
				pickStagingBuffer,
				linearSampler: sampler
			};

			computeWeightTexture(tableSize);
			updated = Date.now();
			renderScene();
		} catch (e) {
			console.error('Failed to initialize WebGPU:', e);
		}
	});

	// onDestroy(() => {
	// 	if (animationFrameId) {
	// 		cancelAnimationFrame(animationFrameId);
	// 	}
	// });

	$effect(() => {});

	const onUpdate = () => {
		updated = Date.now();
	};

	const onTableSizeChange = () => {
		if (gpuState) {
			gpuState.optionsBuffer.write({ tableSize });
			computeWeightTexture(tableSize);
			updated = Date.now();
		}
	};
</script>

<svelte:head>
	<title>Hello Triangle</title>
</svelte:head>

<div class="page-container">
	<header>
		<h1>Hello World Triangle</h1>
		<p>A simple triangle rendered with TypeGPU</p>
	</header>

	<main>
		<div class="canvases">
			<div
				role="img"
				class="canvas-container"
				onmousemove={(event) => {
					if (event.buttons === 0) {
						const canvasRect = colorCanvas.getBoundingClientRect();
						const uv = d.vec2f(
							(event.clientX - canvasRect.left) / colorCanvas.width,
							(event.clientY - canvasRect.top) / colorCanvas.height
						);
						const ray = cameraRay(
							uv,
							yaw,
							pitch,
							radius,
							colorCanvas.width / colorCanvas.height,
							d.vec3f(0.5, 0.5, 0.5)
						);
						console.log(ray.start, ray.direction);

						const px = (event.clientX - canvasRect.left) * (colorCanvas.width / canvasRect.width);
						const py = (event.clientY - canvasRect.top) * (colorCanvas.height / canvasRect.height);
						readColorAtPixel(px, py);

						return;
					}
					const deltaX = event.movementX;
					const deltaY = event.movementY;

					yaw -= deltaX / 100;
					pitch += deltaY / 100;
					updated = Date.now();
				}}
				onwheel={(event) => {
					event.preventDefault();
					radius += event.deltaY * 0.005;
					radius = Math.max(0.5, Math.min(3.0, radius));
					onUpdate();
				}}
			>
				<canvas bind:this={colorCanvas} width="800" height="600"></canvas>
				<div class="color-display" style="background-color: {color};"></div>
			</div>
			<div class="canvas-container">
				<canvas bind:this={imageCanvas} width="800" height="600"></canvas>
			</div>
		</div>

		<input type="range" min="1" max="10000" step="1" bind:value={sensitivity} oninput={onUpdate} />

		<div class="radio-group">
			<span>Table Size:</span>
			{#each [16, 32, 64, 128] as size}
				<label>
					<input
						type="radio"
						name="tableSize"
						value={size}
						bind:group={tableSize}
						onchange={onTableSizeChange}
					/>
					{size}
				</label>
			{/each}
		</div>
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		font-family:
			'Inter',
			system-ui,
			-apple-system,
			sans-serif;
		background-color: #0f1115;
		color: #e2e8f0;
	}

	.page-container {
		display: flex;
		flex-direction: column;
		align-items: center;
		min-height: 100vh;
		padding: 2rem;
	}

	header {
		text-align: center;
		margin-bottom: 2rem;
	}

	h1 {
		font-size: 2.5rem;
		font-weight: 700;
		margin: 0;
		background: linear-gradient(135deg, #ff0080, #7928ca);
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
	}

	p {
		color: #94a3b8;
		font-size: 1.1rem;
		margin-top: 0.5rem;
	}

	.canvases {
		display: flex;
		flex-direction: row;
		align-items: center;
	}

	.canvas-container {
		background: rgba(255, 255, 255, 0.03);
		padding: 1rem;
		border-radius: 16px;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
		border: 1px solid rgba(255, 255, 255, 0.1);
		position: relative;
	}

	.color-display {
		position: absolute;
		top: 1rem;
		right: 1rem;
		width: 40px;
		height: 40px;
		border-radius: 8px;
		background-color: #000;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
		border: 2px solid rgba(255, 255, 255, 0.8);
	}

	canvas {
		display: block;
		border-radius: 8px;
		background-color: #000;
		width: 100%;
		max-width: 800px;
		height: auto;
		aspect-ratio: 4/3;
	}

	.radio-group {
		display: flex;
		gap: 1rem;
		align-items: center;
		margin-top: 1rem;
	}

	.radio-group label {
		display: flex;
		align-items: center;
		gap: 0.25rem;
		cursor: pointer;
	}
</style>
