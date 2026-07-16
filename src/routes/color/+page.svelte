<script lang="ts">
	import { onMount } from 'svelte';
	import { Select, Slider } from 'bits-ui';
	import {
		DropHalfIcon,
		CircleHalfIcon,
		CaretUpDownIcon,
		CaretDoubleUpIcon,
		CaretDoubleDownIcon,
		CheckIcon,
		GaugeIcon,
		SquareHalfIcon
	} from 'phosphor-svelte';
	import tgpu, {
		type RenderFlag,
		type SampledFlag,
		type StorageFlag,
		type TgpuBuffer,
		type TgpuFixedSampler,
		type TgpuRoot,
		type TgpuTexture
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
		calculateWeights,
		filterTexture,
		processWeightTexture,
		renderImage,
		renderRasterScene,
		renderColorCloud,
		compositeColorCloud,
		readTimings,
		colorSpacesConfig
	} from './orchestration';
	import { once } from '$lib/gpu/gpu_utils';
	import { ColorSpace, srgb_to_oklab } from './color_utils';

	const colorSpaces = Object.entries(colorSpacesConfig).map(([value, config]) => ({
		value: value as ColorSpace,
		label: config.label
	}));

	let colorCanvas: HTMLCanvasElement;
	let imageCanvas: HTMLCanvasElement;
	let updated = $state(0.0);
	let filterUpdated = $state(0.0);
	let filterCalculated = $state(0.0);
	let cloudUpdated = $state(0.0);
	let cloudRendered = $state(0.0);
	let cloudRenderedFast = $state(false);
	let yaw = $state(0.4);
	let pitch = $state(0.2);
	let radius = $state(1);
	let sensitivitySlider = $state(3.0);
	let sensitivity = $derived(sensitivitySlider === 6 ? 0 : Math.pow(10, 3 - sensitivitySlider));
	let bgColor = $state(0.2);
	let saturation = $state(1.0);
	let contrast = $state(1.0);
	let tableSize = $state(128);
	let colorSpace = $state(ColorSpace.oklab);
	let color = $state('rgba(0, 0, 0, 1)');
	let isHovering = $state(false);
	let imageName = $state('Bee');
	let hoveredRGB = $state(d.vec3f(0.0, 0.0, 0.0));
	let contrastRGB = $state(d.vec3f(0.0, 0.0, 0.0));
	let textureSize = $state(d.vec2u(128, 128));
	let startTime = $state(0);
	let savedRGB: d.v3f | null = $state(null);

	let gpuState: {
		root: TgpuRoot;
		context: GPUCanvasContext;
		imageContext: GPUCanvasContext;
		imageBitmap: ImageBitmap;
		imageTexture: TgpuTexture & RenderFlag & StorageFlag & SampledFlag;
		filteredTexture: TgpuTexture & RenderFlag & StorageFlag & SampledFlag;
		pickStagingBuffer: TgpuBuffer<d.WgslArray<d.U32>>;
		linearSampler: TgpuFixedSampler;
		blurredWeightTexture: ReturnType<typeof processWeightTexture> | null;
	} | null = null;

	let isReadingBack = false;
	let pickTextureCache: {
		data: Uint8Array;
		width: number;
		height: number;
		bytesPerRow: number;
	} | null = null;
	let filteredTextureCache: {
		data: Uint8Array;
		width: number;
		height: number;
		bytesPerRow: number;
	} | null = null;

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

		const isFloat16 = rawTexture.format === 'rgba16float';
		const bytesPerPixel = isFloat16 ? 8 : 4;
		const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
		const bufferSize = bytesPerRow * height;

		const stagingBuffer = root.device.createBuffer({
			size: bufferSize,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});

		const enc = root.device.createCommandEncoder();
		enc.copyTextureToBuffer({ texture: rawTexture }, { buffer: stagingBuffer, bytesPerRow }, [
			width,
			height
		]);
		root.device.queue.submit([enc.finish()]);

		await stagingBuffer.mapAsync(GPUMapMode.READ);
		const data = new Uint8Array(stagingBuffer.getMappedRange().slice(0));
		stagingBuffer.unmap();
		stagingBuffer.destroy();

		return { data, width, height, bytesPerRow };
	}

	function rgbToCssColor(rgba: { r: number; g: number; b: number; a?: number }) {
		return `rgba(${rgba.r * 255} ${rgba.g * 255} ${rgba.b * 255} / ${rgba.a ?? 1})`;
	}

	function rgbToHexColor(rgba: { r: number; g: number; b: number; a?: number }) {
		return `#${Math.round(rgba.r * 255)
			.toString(16)
			.padStart(2, '0')}${Math.round(rgba.g * 255)
			.toString(16)
			.padStart(2, '0')}${Math.round(rgba.b * 255)
			.toString(16)
			.padStart(2, '0')}`;
	}

	async function readColorAtPixel(texture: TgpuTexture, px: number, py: number) {
		if (isReadingBack || !gpuState || !colorCanvas) return;

		let cache: { data: Uint8Array; width: number; height: number; bytesPerRow: number } | null =
			null;

		const pickTexture = getTexture(
			gpuState.root,
			'pickTexture',
			colorCanvas.width,
			colorCanvas.height
		);

		if (texture === pickTexture) {
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

		const isFloat16 = texture === gpuState.filteredTexture;
		const bytesPerPixel = isFloat16 ? 8 : 4;
		const offset = py * cache.bytesPerRow + px * bytesPerPixel;

		if (isFloat16) {
			const view = new DataView(cache.data.buffer, cache.data.byteOffset + offset, 8);
			const r = view.getFloat16(0, true) * 255;
			const g = view.getFloat16(2, true) * 255;
			const b = view.getFloat16(4, true) * 255;
			const a = view.getFloat16(6, true) * 255;
			return d.vec4i(r, g, b, a);
		}

		const r = cache.data[offset];
		const g = cache.data[offset + 1];
		const b = cache.data[offset + 2];
		const a = cache.data[offset + 3];

		return d.vec4i(r, g, b, a);
	}

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
			// texture.generateMipmaps();
			gpuState.imageBitmap = bitmap;
			gpuState.imageTexture = texture;
			gpuState.filteredTexture = texture;
			imageName = name;

			invalidateCaches();
			const encoder = gpuState.root.device.createCommandEncoder();
			computeWeightTexture(tableSize);
			onCloudUpdate();
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

	const getTexture = (root: TgpuRoot, uniqueName: string, width: number, height: number) =>
		once([getTexture, uniqueName, width, height], () =>
			root
				.createTexture({
					size: [width, height],
					format: 'rgba8unorm'
				})
				.$usage('render', 'sampled')
		);

	const getDepthTexture = (root: TgpuRoot, uniqueName: string, width: number, height: number) =>
		once([getDepthTexture, uniqueName, width, height], () =>
			root
				.createTexture({
					size: [width, height],
					format: 'depth24plus'
				})
				.$usage('render', 'sampled')
		);

	const computeWeightTexture = (tableSize: number) => {
		if (!gpuState) {
			console.warn('gpuState is null');
			return;
		}

		const { root } = gpuState;

		const imageTexture = gpuState.imageTexture;

		const filteredImageTexture = filterTexture(root, imageTexture, gpuState.linearSampler, {
			textureSize: textureSize,
			selectedColor: d.vec4f(0.0, 0.0, 0.0, 1000.0),
			saturation: saturation,
			contrast: contrast
		});

		gpuState.filteredTexture = filteredImageTexture;

		const weightsBuffer = calculateWeights(root, gpuState.filteredTexture, colorSpace, {
			tableSize: tableSize,
			textureSize: textureSize
		});

		const weightTexture = processWeightTexture(root, weightsBuffer, {
			tableSize: tableSize,
			textureSize: textureSize
		});

		// gpuState.blurredWeightTexture = blurWeightTexture(
		// 	root,
		// 	weightTexture,
		// 	{
		// 		textureSize: textureSize,
		// 		selectedColor: d.vec4f(0.0, 0.0, 0.0, 1000.0),
		// 		saturation: saturation,
		// 		contrast: contrast
		// 	},
		// 	tableSize
		// );
		gpuState.blurredWeightTexture = weightTexture;

		filterCalculated = filterUpdated;
	};

	const renderScene = async () => {
		if (!gpuState) {
			console.warn('gpuState is null');
			return;
		}

		if (filterUpdated > filterCalculated) {
			computeWeightTexture(tableSize);
		}

		const now = Date.now();
		if (now - updated > 500) {
			requestAnimationFrame(() => renderScene());
			return;
		}

		const { root, linearSampler, context, imageContext, filteredTexture, blurredWeightTexture } =
			gpuState;

		if (!blurredWeightTexture) {
			requestAnimationFrame(() => renderScene());
			return;
		}

		const selectedColor = isHovering
			? d.vec4f(hoveredRGB.x, hoveredRGB.y, hoveredRGB.z, 0.05)
			: d.vec4f(0.0, 0.0, 0.0, 1000.0);
		const colorOklab = srgb_to_oklab(selectedColor.xyz);
		contrastRGB = colorOklab.x > 0.45 ? d.vec3f(0, 0, 0) : d.vec3f(1, 1, 1);

		const fast = now - cloudUpdated < 100 && now > startTime + 1000;
		const renderSize = fast
			? [colorCanvas.width >> 1, colorCanvas.height >> 1]
			: [colorCanvas.width, colorCanvas.height];
		const pickTexture = getTexture(root, 'pickTexture', renderSize[0], renderSize[1]);
		const cloudTexture = getTexture(root, 'cloudTexture', renderSize[0], renderSize[1]);
		const rasterTexture = getTexture(root, 'rasterTexture', colorCanvas.width, colorCanvas.height);
		const rasterDepthTexture = getDepthTexture(
			root,
			'rasterDepthTexture',
			colorCanvas.width,
			colorCanvas.height
		);

		const cloudDirty = cloudUpdated > cloudRendered || fast !== cloudRenderedFast;
		if (cloudDirty) {
			const steps = fast ? 20 : 64;
			const camera = {
				yaw,
				pitch,
				radius,
				aspect: colorCanvas.width / colorCanvas.height,
				steps,
				sensitivity
			};
			renderRasterScene(root, rasterTexture, rasterDepthTexture, camera, bgColor);
			renderColorCloud(
				root,
				blurredWeightTexture,
				linearSampler,
				rasterDepthTexture,
				cloudTexture,
				pickTexture,
				colorSpace,
				camera
			);
			cloudRendered = now;
			cloudRenderedFast = fast;
		}

		renderImage(root, filteredTexture, linearSampler, imageContext, {
			textureSize,
			saturation,
			contrast,
			selectedColor
		});

		compositeColorCloud(root, cloudTexture, pickTexture, rasterTexture, linearSampler, context, {
			selectedColor,
			bgColor
		});

		await root.device.queue.onSubmittedWorkDone();
		if (!fast && cloudDirty) {
			invalidateCaches();
		}
		readTimings();
		requestAnimationFrame(() => renderScene());
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

			const pickStagingBuffer = root
				.createBuffer(d.arrayOf(d.u32, 64))
				.$addFlags(GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

			const sampler = root.createSampler({
				magFilter: 'linear',
				minFilter: 'linear'
			});

			// const querySet = root.createQuerySet('timestamp', 16);

			textureSize = d.vec2u(bitmap.width, bitmap.height);

			gpuState = {
				root,
				context,
				imageContext,
				imageBitmap: bitmap,
				imageTexture: texture,
				pickStagingBuffer,
				linearSampler: sampler,
				filteredTexture: texture,
				blurredWeightTexture: null
			};

			computeWeightTexture(tableSize);
			onCloudUpdate();
			startTime = Date.now();
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

	const onCloudUpdate = () => {
		const now = Date.now();
		cloudUpdated = now;
		updated = now;
	};

	const onFilterUpdate = () => {
		const now = Date.now();
		filterUpdated = now;
		cloudUpdated = now;
		updated = now;
	};
</script>

<svelte:head>
	<title>Color Cloud</title>
</svelte:head>

<div class="page-container">
	<header>
		<h1>Color Cloud</h1>
		<p>A simple triangle rendered with TypeGPU</p>
	</header>

	<main>
		<div class="canvases">
			<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
			<!-- svelte-ignore a11y_click_events_have_key_events -->
			<div
				role="img"
				class="canvas-container"
				onmousemove={async (event) => {
					if (event.buttons === 0) {
						const canvasRect = colorCanvas.getBoundingClientRect();

						const px = (event.clientX - canvasRect.left) * (colorCanvas.width / canvasRect.width);
						const py = (event.clientY - canvasRect.top) * (colorCanvas.height / canvasRect.height);
						if (gpuState) {
							const pickTexture = getTexture(
								gpuState.root,
								'pickTexture',
								colorCanvas.width,
								colorCanvas.height
							);
							const v = await readColorAtPixel(pickTexture, px, py);

							if (v && v.a > 0) {
								isHovering = true;
								hoveredRGB = d.vec3f(v.r / 255.0, v.g / 255.0, v.b / 255.0);
								color = rgbToCssColor(hoveredRGB);
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
					onCloudUpdate();
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
					onCloudUpdate();
					invalidateCaches();
				}}
				onclick={() => {
					if (isHovering) {
						savedRGB = hoveredRGB;
						onUpdate();
					}
				}}
			>
				<canvas bind:this={colorCanvas} width="400" height="300"></canvas>
				{#if isHovering}
					<div class="color-display absolute" style="background-color: {color};"></div>
				{/if}
			</div>
			<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
			<!-- svelte-ignore a11y_click_events_have_key_events -->
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
						hoveredRGB = d.vec3f(v.r / 255.0, v.g / 255.0, v.b / 255.0);
						color = rgbToCssColor(hoveredRGB);
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
				onclick={() => {
					if (isHovering) {
						savedRGB = hoveredRGB;
						onUpdate();
					}
				}}
			>
				<canvas bind:this={imageCanvas} width="400" height="300"></canvas>
				{#if isHovering}
					<div class="color-display absolute" style="background-color: {color};"></div>
				{/if}
			</div>
		</div>

		<div class="flex justify-between">
			<div>
				<div
					class="color-display relative"
					style="background-color: {rgbToHexColor(
						isHovering ? hoveredRGB : (savedRGB ?? d.vec3f(0, 0, 0))
					)}; border-color: {rgbToHexColor(contrastRGB)}; color: {rgbToHexColor(contrastRGB)}"
				>
					{rgbToHexColor(isHovering ? hoveredRGB : (savedRGB ?? d.vec3f(0, 0, 0)))}
				</div>
			</div>
			<div class="controls">
				<label class="file-label">
					<input type="file" accept="image/*,image/png" onchange={handleImageUpload} />
					<span class="btn">📂 Load Image</span>
				</label>
				<div class="presets">
					<span class="preset-label">Presets:</span>
					{#each presets as preset (preset.name)}
						<button class="btn preset-btn" onclick={() => loadPreset(preset)}>{preset.name}</button>
					{/each}
				</div>
				{#if imageName}
					<span class="img-name">{imageName}</span>
				{/if}
			</div>
		</div>

		<div class="mt-4 grid w-full max-w-[1000px] grid-cols-2 gap-12 px-4">
			<!-- Left Column: Color Cloud Settings -->
			<div class="flex flex-col gap-6">
				<h3 class="border-b border-white/10 pb-2 text-lg font-semibold text-slate-200">
					Color Cloud Settings
				</h3>

				<div class="flex w-full flex-col gap-1">
					<div class="pl-8 text-sm text-slate-400">
						<span>Sensitivity</span>
					</div>
					<div class="flex items-center gap-2">
						<GaugeIcon size={24} class="text-slate-400" />
						<Slider.Root
							type="single"
							value={sensitivitySlider}
							max={6}
							step={0.01}
							class="relative flex w-full touch-none items-center select-none"
							onValueChange={(v) => {
								sensitivitySlider = v;
								onCloudUpdate();
								invalidateCaches();
							}}
						>
							<span
								class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20"
							>
								<Slider.Range class="absolute h-full bg-blue-600" />
							</span>
							<Slider.Thumb
								index={0}
								class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
							/>
						</Slider.Root>
					</div>
				</div>

				<div class="flex w-full flex-col gap-1">
					<div class="pl-8 text-sm text-slate-400">
						<span>Background Color</span>
					</div>
					<div class="flex items-center gap-2">
						<SquareHalfIcon size={24} class="text-slate-400" />
						<Slider.Root
							type="single"
							value={bgColor}
							max={1}
							step={0.01}
							class="relative flex w-full touch-none items-center select-none"
							onValueChange={(v) => {
								bgColor = v;
								onCloudUpdate();
								invalidateCaches();
							}}
						>
							<span
								class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20"
							>
								<Slider.Range class="absolute h-full bg-blue-600" />
							</span>
							<Slider.Thumb
								index={0}
								class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
							/>
						</Slider.Root>
					</div>
				</div>

				<!-- <div class="flex items-center gap-2">
					<span class="text-slate-200">Table Size:</span>
					<Select.Root
						type="single"
						value={tableSizeStr}
						onValueChange={onTableSizeChange}
						items={tableSizes}
						allowDeselect={false}
					>
						<Select.Trigger
							class="inline-flex h-10 w-[200px] touch-none items-center rounded-md border border-white/10 bg-slate-800 px-3 text-sm transition-colors select-none"
							aria-label="Select a table size"
						>
							<Select.Value placeholder="Select a table size" />
							<CaretUpDownIcon class="ml-auto size-5 text-slate-400" />
						</Select.Trigger>
						<Select.Portal>
							<Select.Content
								class="data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=open]:animate-in z-50 h-auto max-h-96 w-[var(--bits-select-anchor-width)] min-w-[var(--bits-select-anchor-width)] rounded-xl border border-slate-700 bg-slate-800 px-1 py-1 shadow-md outline-hidden select-none"
								sideOffset={10}
							>
								<Select.ScrollUpButton class="flex w-full items-center justify-center">
									<CaretDoubleUpIcon class="size-3 text-slate-400" />
								</Select.ScrollUpButton>
								<Select.Viewport class="p-1">
									{#each tableSizes as size, i (i + size.value)}
										<Select.Item
											class="flex h-9 w-full cursor-pointer items-center rounded-md py-2 pr-1.5 pl-3 text-sm text-white outline-hidden select-none data-highlighted:bg-blue-600"
											value={size.value}
											label={size.label}
										>
											{#snippet children({ selected })}
												{size.label}
												{#if selected}
													<div class="ml-auto">
														<CheckIcon aria-label="check" class="size-4" />
													</div>
												{/if}
											{/snippet}
										</Select.Item>
									{/each}
								</Select.Viewport>
								<Select.ScrollDownButton class="flex w-full items-center justify-center">
									<CaretDoubleDownIcon class="size-3 text-slate-400" />
								</Select.ScrollDownButton>
							</Select.Content>
						</Select.Portal>
					</Select.Root>
				</div> -->

				<div class="flex items-center gap-2">
					<span class="text-slate-200">Color Space:</span>
					<Select.Root
						type="single"
						value={colorSpace}
						onValueChange={(v) => {
							colorSpace = v as ColorSpace;
							if (gpuState) {
								const encoder = gpuState.root.device.createCommandEncoder();
								invalidateCaches();
								computeWeightTexture(tableSize);
								onCloudUpdate();
								gpuState.root.device.queue.submit([encoder.finish()]);
							}
						}}
						items={colorSpaces}
						allowDeselect={false}
					>
						<Select.Trigger
							class="inline-flex h-10 w-[200px] touch-none items-center rounded-md border border-white/10 bg-slate-800 px-3 text-sm transition-colors select-none"
							aria-label="Select a color space"
						>
							<Select.Value placeholder="Select a color space" />
							<CaretUpDownIcon class="ml-auto size-5 text-slate-400" />
						</Select.Trigger>
						<Select.Portal>
							<Select.Content
								class="data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=open]:animate-in z-50 h-auto max-h-96 w-[var(--bits-select-anchor-width)] min-w-[var(--bits-select-anchor-width)] rounded-xl border border-slate-700 bg-slate-800 px-1 py-1 shadow-md outline-hidden select-none"
								sideOffset={10}
							>
								<Select.ScrollUpButton class="flex w-full items-center justify-center">
									<CaretDoubleUpIcon class="size-3 text-slate-400" />
								</Select.ScrollUpButton>
								<Select.Viewport class="p-1">
									{#each colorSpaces as space, i (i + space.value)}
										<Select.Item
											class="flex h-9 w-full cursor-pointer items-center rounded-md py-2 pr-1.5 pl-3 text-sm text-white outline-hidden select-none data-highlighted:bg-blue-600"
											value={space.value}
											label={space.label}
										>
											{#snippet children({ selected })}
												{space.label}
												{#if selected}
													<div class="ml-auto">
														<CheckIcon aria-label="check" class="size-4" />
													</div>
												{/if}
											{/snippet}
										</Select.Item>
									{/each}
								</Select.Viewport>
								<Select.ScrollDownButton class="flex w-full items-center justify-center">
									<CaretDoubleDownIcon class="size-3 text-slate-400" />
								</Select.ScrollDownButton>
							</Select.Content>
						</Select.Portal>
					</Select.Root>
				</div>
			</div>

			<!-- Right Column: Image Filters -->
			<div class="flex flex-col gap-6">
				<h3 class="border-b border-white/10 pb-2 text-lg font-semibold text-slate-200">
					Image Filter Settings
				</h3>

				<div class="flex w-full flex-col gap-1">
					<div class="pl-8 text-sm text-slate-400">
						<span>Saturation</span>
					</div>
					<div class="flex items-center gap-2">
						<DropHalfIcon size={24} class="text-slate-400" />
						<Slider.Root
							type="single"
							value={saturation}
							max={2}
							step={0.01}
							class="relative flex w-full touch-none items-center select-none"
							onValueChange={(v) => {
								saturation = v;
								onFilterUpdate();
								invalidateCaches();
							}}
						>
							<span
								class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20"
							>
								<div
									class="absolute h-full bg-blue-600"
									style="left: {Math.min(saturation, 1.0) * 50}%; right: {(2.0 -
										Math.max(saturation, 1.0)) *
										50}%;"
								></div>
							</span>
							<Slider.Thumb
								index={0}
								class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
							/>
						</Slider.Root>
					</div>
				</div>

				<div class="flex w-full flex-col gap-1">
					<div class="pl-8 text-sm text-slate-400">
						<span>Contrast</span>
					</div>
					<div class="flex items-center gap-2">
						<CircleHalfIcon size={24} class="text-slate-400" />
						<Slider.Root
							type="single"
							value={contrast}
							max={2}
							step={0.01}
							class="relative flex w-full touch-none items-center select-none"
							onValueChange={(v) => {
								contrast = v;
								onFilterUpdate();
								invalidateCaches();
							}}
						>
							<span
								class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20"
							>
								<div
									class="absolute h-full bg-blue-600"
									style="left: {Math.min(contrast, 1.0) * 50}%; right: {(2.0 -
										Math.max(contrast, 1.0)) *
										50}%;"
								></div>
							</span>
							<Slider.Thumb
								index={0}
								class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
							/>
						</Slider.Root>
					</div>
				</div>
			</div>
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
		text-align: left;
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
		top: 1rem;
		right: 1rem;
		min-width: 40px;
		height: 40px;
		border-radius: 8px;
		background-color: #000;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
		border: 2px solid rgba(255, 255, 255, 0.8);
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 8px 12px 8px 12px;
	}

	canvas {
		display: block;
		border-radius: 8px;
		background-color: #000;
		width: 100%;
		max-width: 800px;
		height: 300px;
		aspect-ratio: 4/3;
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
