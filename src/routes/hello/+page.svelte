<script lang="ts">
	import { onMount } from 'svelte';
	import tgpu, {
		type RenderFlag,
		type SampledFlag,
		type StorageFlag,
		type TgpuBuffer,
		type TgpuGuardedComputePipeline,
		type TgpuQuerySet,
		type TgpuRenderPipeline,
		type TgpuRoot,
		type TgpuTexture,
		type UniformFlag
	} from 'typegpu';
	import * as d from 'typegpu/data';
	import beeCloseImg from '$lib/assets/bee_close.jpg';
	import flowerImg from '$lib/assets/flower.jpg';
	import pastelsImg from '$lib/assets/pastels.jpg';

	const presets = [
		{ name: 'Bee', src: beeCloseImg },
		{ name: 'Flower', src: flowerImg },
		{ name: 'Pastels', src: pastelsImg }
	];

	import {
		quadVertex,
		triangleFragment,
		cameraBindLayout,
		cameraUniform,
		calculateWeights,
		computeOptions,
		weightCalculationLayout,
		processWeights,
		weightTransferLayout,
		imageFragment,
		cameraRay,
		blur,
		weightProcessingLayout,
		filterFragment,
		saturationFilter,
		filterSlot,
		filterBindLayout,
		filterOptions,
		textureRenderLayout
	} from './shaders';
	import { once } from './gpu_utils';
	import { select } from 'typegpu/std';

	let colorCanvas: HTMLCanvasElement;
	let imageCanvas: HTMLCanvasElement;
	let animationFrameId: number;
	let updated = $state(0.0);
	let yaw = $state(0.0);
	let pitch = $state(0.2);
	let radius = $state(2);
	let steps = $state(42);
	let sensitivity = $state(5.0);
	let saturation = $state(1.0);
	let contrast = $state(1.0);
	let tableSize = $state(64);
	let color = $state('rgba(0, 0, 0, 1)');
	let isHovering = $state(false);
	let imageName = $state('Bee');
	let hoveredRGB = d.vec3f(0.0, 0.0, 0.0);

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
		filterOptionsBuffer: TgpuBuffer<typeof filterOptions> & UniformFlag;
		imageBitmap: ImageBitmap;
		imageTexture: TgpuTexture & StorageFlag & SampledFlag;
		filteredTexture: TgpuTexture & RenderFlag & SampledFlag;
		pickTexture: TgpuTexture & RenderFlag;
		pickStagingBuffer: TgpuBuffer<d.WgslArray<d.U32>>;
		linearSampler: GPUSampler;
		querySet: TgpuQuerySet<'timestamp'>;
	} | null = null;

	let isReadingBack = false;
	let pickTextureCache: { data: Uint8Array; width: number; height: number; bytesPerRow: number } | null = null;
	let filteredTextureCache: { data: Uint8Array; width: number; height: number; bytesPerRow: number } | null = null;

	const invalidateCaches = () => {
		pickTextureCache = null;
		filteredTextureCache = null;
	};

	async function cacheTexture(texture: TgpuTexture) {
		if (!gpuState) return null;
		const { root } = gpuState;
		const rawTexture = root.unwrap(texture);
		const width = rawTexture.width;
		const height = rawTexture.height;

		const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
		const bufferSize = bytesPerRow * height;

		const stagingBuffer = root.device.createBuffer({
			size: bufferSize,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});

		const enc = root.device.createCommandEncoder();
		enc.copyTextureToBuffer(
			{ texture: rawTexture },
			{ buffer: stagingBuffer, bytesPerRow },
			[width, height]
		);
		root.device.queue.submit([enc.finish()]);

		const startTime = performance.now();
		await stagingBuffer.mapAsync(GPUMapMode.READ);
		const data = new Uint8Array(stagingBuffer.getMappedRange().slice(0));
		stagingBuffer.unmap();
		stagingBuffer.destroy();
		const endTime = performance.now();
		console.log(`Texture read took ${(endTime - startTime).toFixed(2)} ms`);

		return { data, width, height, bytesPerRow };
	}

	async function readColorAtPixel(texture: TgpuTexture, px: number, py: number) {
		if (isReadingBack || !gpuState || !colorCanvas) return;

		let cache: { data: Uint8Array; width: number; height: number; bytesPerRow: number } | null = null;

		if (texture === gpuState.pickTexture) {
			if (!pickTextureCache) {
				isReadingBack = true;
				try {
					pickTextureCache = await cacheTexture(texture);
				} finally {
					isReadingBack = false;
				}
			}
			cache = pickTextureCache;
		} else if (texture === gpuState.filteredTexture) {
			if (!filteredTextureCache) {
				isReadingBack = true;
				try {
					filteredTextureCache = await cacheTexture(texture);
				} finally {
					isReadingBack = false;
				}
			}
			cache = filteredTextureCache;
		}

		if (!cache) return;

		px = Math.max(0, Math.min(cache.width - 1, Math.floor(px)));
		py = Math.max(0, Math.min(cache.height - 1, Math.floor(py)));

		const offset = py * cache.bytesPerRow + px * 4;
		const r = cache.data[offset];
		const g = cache.data[offset + 1];
		const b = cache.data[offset + 2];
		const a = cache.data[offset + 3];

		return d.vec4i(r, g, b, a);
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

	const loadNewImage = async (url: string, name: string) => {
		if (!gpuState) return;
		try {
			const bitmap = await imageUrlToBitmap(url);
			const texture = await bitmapToTexture(gpuState.root, bitmap);
			gpuState.imageBitmap = bitmap;
			gpuState.imageTexture = texture;
			gpuState.filteredTexture = texture;
			imageName = name;

			invalidateCaches();
			const encoder = gpuState.root.device.createCommandEncoder();
			computeWeightTexture(tableSize, encoder);
			updated = Date.now();
			gpuState.root.device.queue.submit([encoder.finish()]);
		} catch (e) {
			console.error('Failed to load new image', e);
		}
	};

	const handleImageUpload = (e: Event) => {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		const url = URL.createObjectURL(file);
		loadNewImage(url, file.name).finally(() => {
			URL.revokeObjectURL(url);
		});
	};

	const loadPreset = (preset: { name: string; src: string }) => {
		loadNewImage(preset.src, preset.name);
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

	const computeWeightTexture = (tableSize: number, encoder?: GPUCommandEncoder) => {
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
			linearSampler,
			filteredTexture
		} = gpuState;

		const _encoder = encoder ?? root.device.createCommandEncoder();

		applyFilters();

		const weightsBuffer = getWeightBuffer(root, tableSize);
		const weightTexture = getWeightTexture(root, tableSize);
		const blurredWeightTexture = getBlurredWeightTexture(root, tableSize);

		_encoder.clearBuffer(weightsBuffer.buffer, 0);

		const computeBindGroup = root.createBindGroup(weightCalculationLayout, {
			options: optionsBuffer,
			image: filteredTexture,
			weights: weightsBuffer
		});

		weightCalculationPipeline
			.with(computeBindGroup)
			.with(_encoder)
			.dispatchThreads(imageBitmap.width, imageBitmap.height);

		const weightProcessingBindGroup = root.createBindGroup(weightTransferLayout, {
			options: optionsBuffer,
			weights: weightsBuffer,
			outputTexture: weightTexture
		});

		weightProcessingPipeline
			.with(weightProcessingBindGroup)
			.with(_encoder)
			.dispatchThreads(tableSize, tableSize, tableSize);

		const blurBindGroup = root.createBindGroup(weightProcessingLayout, {
			options: gpuState.filterOptionsBuffer,
			inputTexture: weightTexture.createView('sampled'),
			outputTexture: blurredWeightTexture
		});

		blurPipeline
			.with(blurBindGroup)
			.with(_encoder)
			.dispatchThreads(tableSize, tableSize, tableSize);

		if (!encoder) {
			root.device.queue.submit([_encoder.finish()]);
		}
	};

	const renderScene = async (encoder?: GPUCommandEncoder) => {
		if (!gpuState) {
			console.log('gpuState is null');
			return;
		}
		requestAnimationFrame(() => renderScene());

		const now = Date.now();
		if (now - updated > 100) {
			return;
		}

		const {
			root,
			renderPipeline,
			imageRenderPipeline,
			cameraUniformBuffer,
			pickTexture,
			linearSampler,
			context,
			imageContext,
			filteredTexture,
			querySet
		} = gpuState;
		const _encoder = encoder ?? root.device.createCommandEncoder();
		const blurredWeightTexture = getBlurredWeightTexture(root, tableSize);
		const weightTexture = getWeightTexture(root, tableSize);

		gpuState.filterOptionsBuffer.write({
			selectedColor: isHovering
				? d.vec4f(hoveredRGB.x, hoveredRGB.y, hoveredRGB.z, 0.05)
				: d.vec4f(0.0, 0.0, 0.0, 1000.0),
			saturation: saturation,
			contrast: contrast
		});

		cameraUniformBuffer.write({
			yaw,
			pitch,
			radius,
			aspect: colorCanvas.width / colorCanvas.height,
			steps,
			sensitivity
		});

		const textureBindGroup = root.createBindGroup(textureRenderLayout, {
			texture: filteredTexture.createView('render'),
			sampler: linearSampler,
			options: gpuState.filterOptionsBuffer
		});

		const bindGroup = root.createBindGroup(cameraBindLayout, {
			cameraUniform: cameraUniformBuffer,
			options: gpuState.filterOptionsBuffer,
			weightTexture: blurredWeightTexture.createView('render'),
			weightSampler: linearSampler
		});

		renderPipeline
			.with(bindGroup)
			.with(_encoder)
			.withColorAttachment({
				color: { view: context },
				pick: { view: pickTexture.createView('render') }
			})
			.draw(6);

		imageRenderPipeline.with(textureBindGroup).withColorAttachment({ view: imageContext }).draw(6);

		if (!encoder) {
			root.device.queue.submit([_encoder.finish()]);
		}
		await root.device.queue.onSubmittedWorkDone();

		if (querySet.available) {
			querySet.resolve();
			const values = await querySet.read();
			for (let i = 0; i < 16; i += 2) {
				const start = values[i];
				const end = values[i + 1];
				const time = Number(end - start);
				console.log(`Pass ${i / 2}: ${time / 1e6} ms`);
			}
		} else {
			console.warn('querySet not available');
		}
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

			const querySet = root.createQuerySet('timestamp', 16);

			const weightCalculationPipeline = root
				.createGuardedComputePipeline(calculateWeights)
				.withTimestampWrites({
					querySet,
					beginningOfPassWriteIndex: 0,
					endOfPassWriteIndex: 1
				});

			const weightProcessingPipeline = root
				.createGuardedComputePipeline(processWeights)
				.withTimestampWrites({
					querySet,
					beginningOfPassWriteIndex: 2,
					endOfPassWriteIndex: 3
				});

			const blurPipeline = root.createGuardedComputePipeline(blur).withTimestampWrites({
				querySet,
				beginningOfPassWriteIndex: 4,
				endOfPassWriteIndex: 5
			});

			const renderPipeline = root
				.createRenderPipeline({
					primitive: { topology: 'triangle-list' },
					vertex: quadVertex,
					fragment: triangleFragment,
					targets: {
						color: { format: navigator.gpu.getPreferredCanvasFormat() },
						pick: { format: 'rgba8unorm' }
					}
				})
				.withTimestampWrites({
					querySet,
					beginningOfPassWriteIndex: 6,
					endOfPassWriteIndex: 7
				});

			const imageRenderPipeline = root
				.createRenderPipeline({
					primitive: { topology: 'triangle-list' },
					vertex: quadVertex,
					fragment: imageFragment
				})
				.withTimestampWrites({
					querySet,
					beginningOfPassWriteIndex: 8,
					endOfPassWriteIndex: 9
				});

			const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
			optionsBuffer.write({ tableSize });

			const filterOptionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
			filterOptionsBuffer.write({
				selectedColor: d.vec4f(0.0, 0.0, 0.0, 1000.0),
				saturation: saturation,
				contrast: contrast
			});

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
				filterOptionsBuffer,
				context,
				imageContext,
				imageBitmap: bitmap,
				imageTexture: texture,
				pickTexture,
				pickStagingBuffer,
				linearSampler: sampler,
				filteredTexture: texture,
				querySet
			};

			const encoder = root.device.createCommandEncoder();
			computeWeightTexture(tableSize, encoder);
			updated = Date.now();
			renderScene(encoder);
			root.device.queue.submit([encoder.finish()]);
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
			const encoder = gpuState.root.device.createCommandEncoder();

			gpuState.optionsBuffer.write({ tableSize });
			invalidateCaches();
			computeWeightTexture(tableSize, encoder);
			updated = Date.now();
			gpuState.root.device.queue.submit([encoder.finish()]);
		}
	};

	const applyFilters = () => {
		if (!gpuState) return;
		const root = gpuState.root;
		const width = gpuState.imageBitmap.width;
		const height = gpuState.imageBitmap.height;
		const imageTexture = gpuState.imageTexture;

		const { textures, pipeline, sampler } = once(`filters-${width}-${height}`, () => {
			const textures = [
				root
					.createTexture({
						size: [width, height],
						format: 'rgba8unorm'
					})
					.$usage('render', 'sampled', 'storage'),
				root
					.createTexture({
						size: [width, height],
						format: 'rgba8unorm'
					})
					.$usage('render', 'sampled', 'storage')
			];

			const pipeline = root.with(filterSlot, saturationFilter).createRenderPipeline({
				primitive: { topology: 'triangle-list' },
				vertex: quadVertex,
				fragment: filterFragment,
				targets: {
					color: { format: 'rgba8unorm' }
				}
			});

			const sampler = root.createSampler({
				magFilter: 'linear',
				minFilter: 'linear'
			});

			return {
				textures,
				pipeline,
				sampler
			};
		});

		const bindGroup = root.createBindGroup(filterBindLayout, {
			texture: imageTexture,
			sampler: sampler,
			filterOptions: gpuState.filterOptionsBuffer
		});

		pipeline
			.with(bindGroup)
			.withColorAttachment({ color: { view: textures[1].createView('render') } })
			.draw(6);

		gpuState.filteredTexture = textures[1];
		return textures[1];
	};
	// const calculateWeights = () => {};
	// const renderScene = () => {};
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
				onmousemove={async (event) => {
					if (event.buttons === 0) {
						const canvasRect = colorCanvas.getBoundingClientRect();

						const px = (event.clientX - canvasRect.left) * (colorCanvas.width / canvasRect.width);
						const py = (event.clientY - canvasRect.top) * (colorCanvas.height / canvasRect.height);
						if (gpuState) {
							const v = await readColorAtPixel(gpuState?.pickTexture, px, py);

							if (v && v.a > 0) {
								isHovering = true;
								color = `rgba(${v.r} ${v.g} ${v.b} / ${v.a / 255})`;
								hoveredRGB = d.vec3f(v.r / 255.0, v.g / 255.0, v.b / 255.0);
								onUpdate();
							} else if (isHovering && v) {
								isHovering = false;
								onUpdate();
							}
						}

						return;
					}
					const deltaX = event.movementX;
					const deltaY = event.movementY;

					yaw -= deltaX / 100;
					pitch += deltaY / 100;
					updated = Date.now();
					invalidateCaches();
				}}
				onmouseenter={() => {
					isHovering = true;
					onUpdate();
				}}
				onmouseleave={() => {
					isHovering = false;
					onUpdate();
				}}
				onwheel={(event) => {
					event.preventDefault();
					radius += event.deltaY * 0.005;
					radius = Math.max(0.5, Math.min(3.0, radius));
					onUpdate();
					invalidateCaches();
				}}
			>
				<canvas bind:this={colorCanvas} width="800" height="600"></canvas>
				{#if isHovering}
					<div class="color-display" style="background-color: {color};"></div>
				{/if}
			</div>
			<div
				class="canvas-container"
				role="img"
				onmousemove={async (event) => {
					if (!gpuState) return;
					const canvasRect = imageCanvas.getBoundingClientRect();
					const rawTexture = gpuState.root.unwrap(gpuState.filteredTexture);
					const px = (event.clientX - canvasRect.left) * (rawTexture.width / canvasRect.width);
					const py = (event.clientY - canvasRect.top) * (rawTexture.height / canvasRect.height);

					const v = await readColorAtPixel(gpuState.filteredTexture, px, py);

					if (isHovering && v && v.a > 0) {
						color = `rgba(${v.r} ${v.g} ${v.b} / ${v.a / 255})`;
						hoveredRGB = d.vec3f(v.r / 255.0, v.g / 255.0, v.b / 255.0);
						onUpdate();
					} else if (isHovering && v) {
						onUpdate();
					}
				}}
				onmouseenter={() => {
					isHovering = true;
					onUpdate();
				}}
				onmouseleave={() => {
					isHovering = false;
					onUpdate();
				}}
			>
				<canvas bind:this={imageCanvas} width="800" height="600"></canvas>
				{#if isHovering}
					<div class="color-display" style="background-color: {color};"></div>
				{/if}
			</div>
		</div>

		<div class="controls">
			<label class="file-label">
				<input type="file" accept="image/*" onchange={handleImageUpload} />
				<span class="btn">📂 Load Image</span>
			</label>
			<div class="presets">
				<span class="preset-label">Presets:</span>
				{#each presets as preset}
					<button class="btn preset-btn" onclick={() => loadPreset(preset)}>{preset.name}</button>
				{/each}
			</div>
			{#if imageName}
				<span class="img-name">{imageName}</span>
			{/if}
		</div>

		<input type="range" min="0" max="10" step="0.01" bind:value={sensitivity} oninput={() => { onUpdate(); invalidateCaches(); }} />
		<input
			type="range"
			min="0"
			max="2"
			step="0.01"
			bind:value={saturation}
			oninput={() => {
				onUpdate();
				invalidateCaches();
				computeWeightTexture(tableSize);
			}}
		/>
		<input
			type="range"
			min="0"
			max="2"
			step="0.01"
			bind:value={contrast}
			oninput={() => {
				onUpdate();
				invalidateCaches();
				computeWeightTexture(tableSize);
			}}
		/>

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

	.controls {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
		justify-content: center;
		margin-top: 1rem;
		margin-bottom: 1rem;
	}
	.file-label input {
		display: none;
	}
	.btn {
		display: inline-block;
		padding: 0.45rem 1.1rem;
		background: #2a6be8;
		border-radius: 6px;
		cursor: pointer;
		font-size: 0.9rem;
		font-weight: 600;
		transition: background 0.15s;
		user-select: none;
		border: none;
		color: white;
	}
	.btn:hover {
		background: #3a7ef8;
	}
	.img-name {
		font-size: 0.8rem;
		color: #aaa;
	}

	.presets {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-left: 0.5rem;
	}
	.preset-label {
		font-size: 0.85rem;
		color: #888;
	}
	.preset-btn {
		background: #333;
		padding: 0.4rem 0.8rem;
		font-size: 0.85rem;
		border: none;
		color: #eee;
	}
	.preset-btn:hover {
		background: #444;
	}
</style>
