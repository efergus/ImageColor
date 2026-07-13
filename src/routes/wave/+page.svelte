<script lang="ts">
	import { Slider } from 'bits-ui';
	import { onMount } from 'svelte';
	import tgpu, { d, type StorageFlag, type TgpuBuffer } from 'typegpu';
	import { emitterData, gpuWaveFunction, renderWave, type EmitterArray } from './shaders';

	type Tool = 'emitter' | 'refractor' | 'eraser';
	const tools: Tool[] = ['emitter', 'refractor', 'eraser'];

	let tool: Tool = $state('emitter');
	let penSize = $state(10);
	let frequency = $state(2);
	let amplitude = $state(1);
	let phase = $state(0);
	let refractionIndex = $state(1.5);
	let speed = $state(100);
	let substeps = $state(4);

	let canvas: HTMLCanvasElement;

	const width = 800;
	const height = 600;
	const deltat = 1 / 60;

	// Sponge boundary layer: sigma ramps quadratically from 0 at the interior
	// edge of the band up to sigmaMax at the wall, so waves lose most of their
	// energy before ever reaching the (still fully reflective) grid edge.
	const spongeBandWidth = 20;
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
	let emitters: EmitterArray | null = null;

	let downPos: { x: number; y: number } | null = null;
	let dragging = false;

	const cellFromEvent = (e: PointerEvent) => {
		const rect = canvas.getBoundingClientRect();
		const x = Math.floor(((e.clientX - rect.left) / rect.width) * width);
		const y = Math.floor(((e.clientY - rect.top) / rect.height) * height);
		return { x, y };
	};

	const forEachPenCell = (e: PointerEvent, fn: (idx: number) => void) => {
		const { x, y } = cellFromEvent(e);
		for (let dy = -penSize; dy <= penSize; dy++) {
			for (let dx = -penSize; dx <= penSize; dx++) {
				if (dx * dx + dy * dy > penSize * penSize) continue;
				const cx = x + dx;
				const cy = y + dy;
				if (cx < 0 || cx >= width || cy < 0 || cy >= height) continue;
				fn(cx + cy * width);
			}
		}
	};

	type EmitterValue = { frequency: number; phase: number; amplitude: number };

	const paint = (e: PointerEvent) => {
		if (!refraction || !emitters) return;
		if (tool === 'emitter') {
			const values: Record<number, EmitterValue> = {};
			forEachPenCell(e, (idx) => {
				values[idx] = { frequency, phase, amplitude };
			});
			emitters.patch(values);
		} else if (tool === 'refractor') {
			const values: Record<number, number> = {};
			forEachPenCell(e, (idx) => {
				values[idx] = refractionIndex;
			});
			refraction.patch(values);
		} else {
			const emitterValues: Record<number, EmitterValue> = {};
			const refractionValues: Record<number, number> = {};
			forEachPenCell(e, (idx) => {
				emitterValues[idx] = { frequency: 0, phase: 0, amplitude: 0 };
				refractionValues[idx] = 1;
			});
			emitters.patch(emitterValues);
			refraction.patch(refractionValues);
		}
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
		paint(e);
	};

	const onPointerUp = (e: PointerEvent) => {
		if (downPos && !dragging) {
			paint(e);
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

		emitters = root.createBuffer(d.arrayOf(emitterData, width * height)).$usage('storage');

		let simTime = 0;
		const frame = () => {
			if (!buffers || !refraction || !damping || !emitters) return;
			const dt = deltat / substeps;
			let bufs: GPUArray[] = buffers;
			for (let i = 0; i < substeps; i++) {
				gpuWaveFunction(root, bufs[0], bufs[1], bufs[2], refraction, damping, emitters, {
					width,
					height,
					speed,
					deltat: dt,
					time: simTime
				});
				simTime += dt;
				bufs = [bufs[1], bufs[2], bufs[0]];
			}
			buffers = bufs;
			renderWave(root, bufs[1], refraction, emitters, context, { width, height });
			requestAnimationFrame(frame);
		};
		frame();
	});
</script>

{#snippet labeledSlider(
	label: string,
	value: number,
	min: number,
	max: number,
	step: number,
	set: (v: number) => void
)}
	<span>{label}: {value}</span>
	<Slider.Root
		type="single"
		{value}
		{min}
		{max}
		{step}
		class="relative flex w-full touch-none items-center select-none"
		onValueChange={set}
	>
		<span class="relative h-2 w-full grow cursor-pointer overflow-hidden rounded-full bg-white/20">
			<Slider.Range class="absolute h-full bg-blue-600" />
		</span>
		<Slider.Thumb
			index={0}
			class="block size-[20px] cursor-pointer rounded-full border-2 border-blue-600 bg-white shadow-sm transition-colors hover:border-white/30 focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2 focus-visible:outline-hidden disabled:pointer-events-none disabled:opacity-50 data-active:scale-[0.98] data-active:border-white/30"
		/>
	</Slider.Root>
{/snippet}

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

	<div class="my-2 flex gap-2">
		{#each tools as t (t)}
			<button
				class="cursor-pointer rounded px-4 py-1 capitalize transition-colors {tool === t
					? 'bg-blue-600 text-white'
					: 'bg-white/20 hover:bg-white/30'}"
				onclick={() => (tool = t)}
			>
				{t}
			</button>
		{/each}
	</div>

	{@render labeledSlider('Pen size', penSize, 1, 50, 1, (v) => (penSize = v))}

	{#if tool === 'emitter'}
		{@render labeledSlider('Frequency', frequency, 0.2, 10, 0.01, (v) => (frequency = v))}
		{@render labeledSlider('Amplitude', amplitude, 0.1, 3, 0.01, (v) => (amplitude = v))}
		{@render labeledSlider('Phase', phase, 0, 6.28, 0.01, (v) => (phase = v))}
	{:else if tool === 'refractor'}
		{@render labeledSlider(
			'Refraction index',
			refractionIndex,
			1,
			3,
			0.01,
			(v) => (refractionIndex = v)
		)}
	{/if}

	{@render labeledSlider('Speed', speed, 10, 1000, 0.01, (v) => (speed = v))}
	{@render labeledSlider('Substeps', substeps, 1, 32, 1, (v) => (substeps = v))}
</div>
