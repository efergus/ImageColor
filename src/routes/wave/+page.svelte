<script lang="ts">
	import { Slider } from 'bits-ui';
	import { onMount } from 'svelte';
	import tgpu, { d, type StorageFlag, type TgpuBuffer } from 'typegpu';
	import { gpuWaveFunction, renderWave } from './shaders';

	let speed = $state(100);
	let substeps = $state(4);
	let refractionIndex = $state(1.5);

	let canvas: HTMLCanvasElement;

	const width = 800;
	const height = 600;
	const deltat = 1 / 60;
	const brushRadius = 10;

	// Sponge boundary layer: sigma ramps quadratically from 0 at the interior
	// edge of the band up to sigmaMax at the wall, so waves lose most of their
	// energy before ever reaching the (still fully reflective) grid edge.
	const spongeBandWidth = 60;
	const spongeSigmaTopBottom = 400;
	const spongeSigmaLeftRight = 1;

	const buildSpongeField = () => {
		const data = new Float32Array(width * height);
		for (let y = 0; y < height; y++) {
			for (let x = 0; x < width; x++) {
				let sigma = 0;
				const distTop = y;
				const distBottom = height - 1 - y;
				const distLeft = x;
				const distRight = width - 1 - x;
				if (distTop < spongeBandWidth) {
					const t = (spongeBandWidth - distTop) / spongeBandWidth;
					sigma += spongeSigmaTopBottom * t * t;
				}
				if (distBottom < spongeBandWidth) {
					const t = (spongeBandWidth - distBottom) / spongeBandWidth;
					sigma += spongeSigmaTopBottom * t * t;
				}
				if (distLeft < spongeBandWidth) {
					const t = (spongeBandWidth - distLeft) / spongeBandWidth;
					sigma += spongeSigmaLeftRight * t * t;
				}
				if (distRight < spongeBandWidth) {
					const t = (spongeBandWidth - distRight) / spongeBandWidth;
					sigma += spongeSigmaLeftRight * t * t;
				}
				data[x + y * width] = sigma;
			}
		}
		return data;
	};

	type GPUArray = TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag;
	let buffers: GPUArray[] | null = null;
	let refraction: GPUArray | null = null;
	let damping: GPUArray | null = null;

	let downPos: { x: number; y: number } | null = null;
	let dragging = false;

	const cellFromEvent = (e: PointerEvent) => {
		const rect = canvas.getBoundingClientRect();
		const x = Math.floor(((e.clientX - rect.left) / rect.width) * width);
		const y = Math.floor(((e.clientY - rect.top) / rect.height) * height);
		return { x, y };
	};

	const drawRefraction = (e: PointerEvent) => {
		if (!refraction) return;
		const { x, y } = cellFromEvent(e);
		const values: Record<number, number> = {};
		for (let dy = -brushRadius; dy <= brushRadius; dy++) {
			for (let dx = -brushRadius; dx <= brushRadius; dx++) {
				if (dx * dx + dy * dy > brushRadius * brushRadius) continue;
				const cx = x + dx;
				const cy = y + dy;
				if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
				values[cx + cy * width] = refractionIndex;
			}
		}
		refraction.patch(values);
	};

	const onPointerDown = (e: PointerEvent) => {
		canvas.setPointerCapture(e.pointerId);
		downPos = { x: e.clientX, y: e.clientY };
		dragging = false;
	};

	const onPointerMove = (e: PointerEvent) => {
		if (!downPos) return;
		if (!dragging) {
			const dx = e.clientX - downPos.x;
			const dy = e.clientY - downPos.y;
			if (dx * dx + dy * dy < 9) return;
			dragging = true;
		}
		drawRefraction(e);
	};

	const onPointerUp = (e: PointerEvent) => {
		if (downPos && !dragging && buffers) {
			const { x, y } = cellFromEvent(e);
			buffers[1].patch({ [x + y * width]: 1 });
		}
		downPos = null;
		dragging = false;
	};

	onMount(async () => {
		const root = await tgpu.init();
		const context = root.configureContext({
			canvas,
			alphaMode: 'premultiplied'
		});

		buffers = [0, 1, 2].map(() =>
			root.createBuffer(d.arrayOf(d.f32, width * height)).$usage('storage')
		);
		refraction = root.createBuffer(d.arrayOf(d.f32, width * height)).$usage('storage');
		refraction.write(new Float32Array(width * height).fill(1).buffer);

		damping = root.createBuffer(d.arrayOf(d.f32, width * height)).$usage('storage');
		damping.write(buildSpongeField().buffer);

		const frame = () => {
			if (!buffers || !refraction || !damping) return;
			const dt = deltat / substeps;
			let bufs: GPUArray[] = buffers;
			for (let i = 0; i < substeps; i++) {
				gpuWaveFunction(root, bufs[0], bufs[1], bufs[2], refraction, damping, {
					width,
					height,
					speed,
					deltat: dt
				});
				bufs = [bufs[1], bufs[2], bufs[0]];
			}
			buffers = bufs;
			renderWave(root, bufs[1], refraction, context, { width, height });
			requestAnimationFrame(frame);
		};
		frame();
	});
</script>

<div class="page-container">
	<div role="img" class="canvas-container">
		<canvas
			bind:this={canvas}
			width="800"
			height="600"
			onpointerdown={onPointerDown}
			onpointermove={onPointerMove}
			onpointerup={onPointerUp}
		></canvas>
	</div>

	<span>Speed: {speed}</span>
	<Slider.Root
		type="single"
		value={speed}
		min={10}
		max={1000}
		step={0.01}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={(v) => {
			speed = v;
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

	<span>Substeps: {substeps}</span>
	<Slider.Root
		type="single"
		value={substeps}
		min={1}
		max={32}
		step={1}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={(v) => {
			substeps = v;
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

	<span>Refraction index: {refractionIndex}</span>
	<Slider.Root
		type="single"
		value={refractionIndex}
		min={1}
		max={3}
		step={0.01}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={(v) => {
			refractionIndex = v;
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
