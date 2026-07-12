<script lang="ts">
	import { Slider, Switch } from 'bits-ui';
	import {
		ArrowsCounterClockwiseIcon,
		PauseIcon,
		PlayIcon,
		SkipForwardIcon
	} from 'phosphor-svelte';
	import { onDestroy, onMount } from 'svelte';
	import tgpu, { type TgpuRoot, type TgpuTexture } from 'typegpu';
	import {
		gpuDrawSegment,
		gpuLifeStep,
		gpuToggleCell,
		lifeStateFormat,
		renderLife
	} from './shaders';

	let fps = $state(1);
	let paused = $state(false);
	let cellSize = $state(8);
	let discreteness = $state(1);
	let linearChange = $state(false);

	// Neighbor-sum boundaries between the starve | stay | grow | crowd
	// regions; the maximum possible sum is 8 live neighbors.
	let thresholds = $state([1.5, 2.5, 3.5]);
	const thresholdMax = 8;
	// Tick marks at every half neighbor count, excluding the track ends.
	const thresholdTicks = Array.from({ length: thresholdMax * 2 - 1 }, (_, i) => (i + 1) / 2);
	const thresholdSections = $derived([
		{ color: 'bg-red-500/70', from: 0, to: thresholds[0] },
		{ color: 'bg-blue-600/70', from: thresholds[0], to: thresholds[1] },
		{ color: 'bg-green-500/70', from: thresholds[1], to: thresholds[2] },
		{ color: 'bg-amber-500/70', from: thresholds[2], to: thresholdMax }
	]);

	let canvas: HTMLCanvasElement;

	const seedAliveChance = 0.6;
	const seedVariance = 20;

	// Grid dimensions follow the viewport so cells stay roughly square:
	// both axes are the viewport extent divided by the same cell size.
	let width = 0;
	let height = 0;

	let root: TgpuRoot | null = null;
	let textures: TgpuTexture[] | null = null;
	let needsRender = false;

	// Each cell is seeded alive with a probability that falls off as a Gaussian
	// from the grid center: seedAliveChance at the peak, variance in cells^2.
	const buildSeedState = () => {
		const data = new Float32Array(width * height);
		const cx = (width - 1) / 2;
		const cy = (height - 1) / 2;
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				const distSq = (x - cx) * (x - cx) + (y - cy) * (y - cy);
				const chance = seedAliveChance * Math.exp(-distSq / (2 * seedVariance));
				data[x + y * width] = Math.random() < chance ? 1 : 0;
			}
		}
		return data;
	};

	// Resizes the canvas to the viewport, recreates the state textures at the
	// grid resolution implied by cellSize, and reseeds.
	const rebuild = () => {
		if (!root) return;
		const dpr = window.devicePixelRatio || 1;
		canvas.width = Math.round(window.innerWidth * dpr);
		canvas.height = Math.round(window.innerHeight * dpr);
		width = Math.max(1, Math.round(window.innerWidth / cellSize));
		height = Math.max(1, Math.round(window.innerHeight / cellSize));

		const old = textures;
		const r = root;
		textures = [0, 1].map(() =>
			r.createTexture({ size: [width, height], format: lifeStateFormat }).$usage('storage')
		);
		textures[0].write(buildSeedState());
		old?.forEach((t) => t.destroy());
		needsRender = true;
	};

	let resizeTimer: ReturnType<typeof setTimeout> | null = null;
	const onResize = () => {
		if (resizeTimer) clearTimeout(resizeTimer);
		resizeTimer = setTimeout(rebuild, 150);
	};

	// The UI panel fades out unless the cursor moved recently or is near it.
	let uiVisible = $state(true);
	let uiPanel: HTMLDivElement | null = null;
	let hideTimer: ReturnType<typeof setTimeout> | null = null;
	let lastPointer = { x: 0, y: 0 };

	const nearUi = () => {
		if (!uiPanel) return false;
		const rect = uiPanel.getBoundingClientRect();
		const margin = 100;
		return (
			lastPointer.x > rect.left - margin &&
			lastPointer.x < rect.right + margin &&
			lastPointer.y > rect.top - margin &&
			lastPointer.y < rect.bottom + margin
		);
	};

	const showUi = () => {
		uiVisible = true;
		if (hideTimer) clearTimeout(hideTimer);
		hideTimer = setTimeout(() => {
			if (nearUi()) {
				showUi();
			} else {
				uiVisible = false;
			}
		}, 2500);
	};

	const onWindowPointerMove = (e: PointerEvent) => {
		lastPointer = { x: e.clientX, y: e.clientY };
		showUi();
	};

	const cellFromEvent = (e: PointerEvent) => {
		const rect = canvas.getBoundingClientRect();
		const x = Math.floor(((e.clientX - rect.left) / rect.width) * width);
		const y = Math.floor(((e.clientY - rect.top) / rect.height) * height);
		return {
			x: Math.min(Math.max(x, 0), width - 1),
			y: Math.min(Math.max(y, 0), height - 1)
		};
	};

	let downPos: { x: number; y: number } | null = null;
	let dragging = false;
	let lastCell: { x: number; y: number } | null = null;

	const onPointerDown = (e: PointerEvent) => {
		canvas.setPointerCapture(e.pointerId);
		downPos = { x: e.clientX, y: e.clientY };
		lastCell = cellFromEvent(e);
		dragging = false;
	};

	const onPointerMove = (e: PointerEvent) => {
		if (!downPos || !lastCell || !root || !textures) return;
		if (!dragging) {
			const dx = e.clientX - downPos.x;
			const dy = e.clientY - downPos.y;
			if (dx * dx + dy * dy < 9) return;
			dragging = true;
		}
		const cell = cellFromEvent(e);
		gpuDrawSegment(root, textures[0], textures[1], {
			x0: lastCell.x,
			y0: lastCell.y,
			x1: cell.x,
			y1: cell.y,
			width,
			height
		});
		textures = [textures[1], textures[0]];
		lastCell = cell;
		needsRender = true;
	};

	const stepOnce = () => {
		if (!root || !textures) return;
		gpuLifeStep(root, textures[0], textures[1], {
			width,
			height,
			discreteness,
			stay: thresholds[0],
			grow: thresholds[1],
			crowd: thresholds[2],
			linear: linearChange
		});
		textures = [textures[1], textures[0]];
		needsRender = true;
	};

	const reset = () => {
		if (!textures) return;
		textures[0].write(buildSeedState());
		needsRender = true;
	};

	const onPointerUp = (e: PointerEvent) => {
		if (downPos && !dragging && root && textures) {
			const { x, y } = cellFromEvent(e);
			gpuToggleCell(root, textures[0], textures[1], { x, y, width, height });
			textures = [textures[1], textures[0]];
			needsRender = true;
		}
		downPos = null;
		dragging = false;
		lastCell = null;
	};

	onMount(async () => {
		root = await tgpu.init();
		const context = root.configureContext({
			canvas,
			alphaMode: 'premultiplied'
		});

		rebuild();
		showUi();

		let lastStep = performance.now();
		const frame = (now: number) => {
			if (!root || !textures) return;
			const interval = 1000 / fps;
			if (paused) {
				lastStep = now;
			} else if (now - lastStep >= interval) {
				stepOnce();
				lastStep = now;
			}
			if (needsRender) {
				renderLife(root, textures[0], context, { width, height });
				needsRender = false;
			}
			requestAnimationFrame(frame);
		};
		requestAnimationFrame(frame);
	});

	onDestroy(() => {
		if (hideTimer) clearTimeout(hideTimer);
		if (resizeTimer) clearTimeout(resizeTimer);
	});
</script>

<svelte:window onresize={onResize} onpointermove={onWindowPointerMove} />

<canvas
	bind:this={canvas}
	class="fixed inset-0 h-full w-full"
	onpointerdown={onPointerDown}
	onpointermove={onPointerMove}
	onpointerup={onPointerUp}
></canvas>

<div
	bind:this={uiPanel}
	class="fixed bottom-6 left-1/2 flex w-[min(90vw,26rem)] -translate-x-1/2 flex-col gap-3 rounded-xl bg-black/60 p-4 text-white backdrop-blur transition-opacity duration-500 {uiVisible
		? 'opacity-100'
		: 'pointer-events-none opacity-0'}"
>
	<div class="flex justify-center gap-4">
		<button
			class="cursor-pointer rounded-full bg-blue-600 p-3 transition-colors hover:bg-blue-500"
			aria-label={paused ? 'Play' : 'Pause'}
			onclick={() => {
				paused = !paused;
			}}
		>
			{#if paused}
				<PlayIcon size={22} weight="fill" />
			{:else}
				<PauseIcon size={22} weight="fill" />
			{/if}
		</button>
		<button
			class="cursor-pointer rounded-full bg-blue-600 p-3 transition-colors hover:bg-blue-500"
			aria-label="Next step"
			onclick={stepOnce}
		>
			<SkipForwardIcon size={22} weight="fill" />
		</button>
		<button
			class="cursor-pointer rounded-full bg-blue-600 p-3 transition-colors hover:bg-blue-500"
			aria-label="Reset"
			onclick={reset}
		>
			<ArrowsCounterClockwiseIcon size={22} weight="fill" />
		</button>
	</div>

	<span>Framerate: {fps} fps</span>
	<Slider.Root
		type="single"
		value={fps}
		min={0.25}
		max={60}
		step={0.25}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={(v) => {
			fps = v;
		}}
	>
		<span class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20">
			<Slider.Range class="absolute h-full bg-blue-600" />
		</span>
		<Slider.Thumb
			index={0}
			class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
		/>
	</Slider.Root>

	<span>Discreteness: {discreteness}</span>
	<Slider.Root
		type="single"
		value={discreteness}
		min={0}
		max={1}
		step={0.01}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={(v) => {
			discreteness = v;
		}}
	>
		<span class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20">
			<Slider.Range class="absolute h-full bg-blue-600" />
		</span>
		<Slider.Thumb
			index={0}
			class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
		/>
	</Slider.Root>

	<div class="flex items-center justify-between">
		<span>Change: {linearChange ? 'linear' : 'exponential'}</span>
		<Switch.Root
			bind:checked={linearChange}
			class="relative h-6 w-11 shrink-0 cursor-pointer rounded-full bg-white/20 transition-colors focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden data-[state=checked]:bg-blue-600"
		>
			<Switch.Thumb
				class="block size-5 translate-x-0.5 rounded-full bg-white shadow-sm transition-transform data-[state=checked]:translate-x-[22px]"
			/>
		</Switch.Root>
	</div>

	<span>
		Starve {thresholds[0].toFixed(1)} stay {thresholds[1].toFixed(1)} grow {thresholds[2].toFixed(
			1
		)} crowd
	</span>
	<Slider.Root
		type="multiple"
		value={thresholds}
		min={0}
		max={thresholdMax}
		step={0.1}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={(v) => {
			thresholds = v;
		}}
	>
		<span class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20">
			{#each thresholdSections as section}
				<span
					class="absolute h-full {section.color}"
					style="left: {(section.from / thresholdMax) * 100}%; width: {((section.to -
						section.from) /
						thresholdMax) *
						100}%"
				></span>
			{/each}
			{#each thresholdTicks as tick}
				<span
					class="absolute w-px -translate-x-1/2 {Number.isInteger(tick)
						? 'top-0 h-full bg-white/70'
						: 'top-1/4 h-1/2 bg-white/35'}"
					style="left: {(tick / thresholdMax) * 100}%"
				></span>
			{/each}
		</span>
		{#each thresholds as _, i (i)}
			<Slider.Thumb
				index={i}
				class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
			/>
		{/each}
	</Slider.Root>

	<span>Cell size: {cellSize} px</span>
	<Slider.Root
		type="single"
		value={cellSize}
		min={2}
		max={50}
		step={1}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={(v) => {
			cellSize = v;
			rebuild();
		}}
	>
		<span class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20">
			<Slider.Range class="absolute h-full bg-blue-600" />
		</span>
		<Slider.Thumb
			index={0}
			class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
		/>
	</Slider.Root>
</div>
