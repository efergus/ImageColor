<script lang="ts">
	import { Slider } from 'bits-ui';
	import { onMount } from 'svelte';
	import tgpu, { d, type StorageFlag, type TgpuBuffer } from 'typegpu';
	import {
		emitterData,
		screenData,
		gpuWaveFunction,
		renderWave,
		type EmitterArray,
		type ScreenArray
	} from './shaders';

	type Tool = 'poke' | 'emitter' | 'refractor' | 'damper' | 'screen' | 'eraser';
	const tools: Tool[] = ['poke', 'emitter', 'refractor', 'damper', 'screen', 'eraser'];

	type Boundary = 'absorbing' | 'reflective' | 'periodic';
	const boundaries: Boundary[] = ['absorbing', 'reflective', 'periodic'];

	type Preset =
		| 'doubleSlit'
		| 'singleSlit'
		| 'subwavelength'
		| 'doubleEmitter'
		| 'huygensFresnel'
		| 'empty';
	const presets: { id: Preset; label: string }[] = [
		{ id: 'doubleSlit', label: 'Double slit' },
		{ id: 'singleSlit', label: 'Single slit' },
		{ id: 'subwavelength', label: 'Subwavelength aperture' },
		{ id: 'doubleEmitter', label: 'Double emitter' },
		{ id: 'huygensFresnel', label: 'Huygens-Fresnel' },
		{ id: 'empty', label: 'Empty' }
	];

	let tool: Tool = $state('poke');
	let boundary: Boundary = $state('absorbing');
	let preset: Preset = $state('doubleSlit');
	let penSize = $state(4);
	let frequency = $state(2);
	let amplitude = $state(10);
	let phase = $state(0);
	let refractionIndex = $state(1.5);
	let dampingIntensity = $state(100);
	let screenAttenuation = $state(0);
	let speed = $state(50);
	let substeps = $state(4);

	let canvas: HTMLCanvasElement;

	const width = 800;
	const height = 600;
	const deltat = 1 / 60;

	// Sponge boundary layer: sigma ramps quadratically from 0 at the interior
	// edge of the band up to sigmaMax at the wall, so waves lose most of their
	// energy before ever reaching the (still fully reflective) grid edge.
	const spongeBandWidth = 60;
	const spongeSigmaTopBottom = 40;
	const spongeSigmaLeftRight = 40;

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
	let screen: ScreenArray | null = null;

	// The damping buffer holds the boundary sponge (absorbing mode only) plus
	// whatever the user painted with the damper tool, so switching boundary
	// modes rebuilds the sponge without losing painted damping.
	const spongeField = buildSpongeField();
	const paintedDamping = new Float32Array(width * height);

	const baseDamping = (idx: number) => (boundary === 'absorbing' ? spongeField[idx] : 0);

	const writeDamping = () => {
		if (!damping) return;
		const data = new Float32Array(width * height);
		for (let i = 0; i < data.length; i++) {
			data[i] = baseDamping(i) + paintedDamping[i];
		}
		damping.write(data.buffer);
	};

	const setBoundary = (b: Boundary) => {
		boundary = b;
		writeDamping();
	};

	let simTime = 0;

	const forEachInRect = (
		xStart: number,
		xEnd: number,
		yStart: number,
		yEnd: number,
		fn: (idx: number) => void
	) => {
		for (let y = Math.max(0, yStart); y < Math.min(height, yEnd); y++) {
			for (let x = Math.max(0, xStart); x < Math.min(width, xEnd); x++) {
				fn(x + y * width);
			}
		}
	};

	// Preset geometry. All presets share the same emitter x-position and
	// damping-band x-position; only the frequency and slit/hole layout differ.
	const presetEmitterX = 100;
	const presetEmitterY = Math.floor(height / 2);
	const presetEmitterRadius = 2; // 5x5 emitter block, "small" per spec

	const presetDampingX = 300; // middle-left
	const presetDampingWidth = 12;
	const presetDampingValue = 400;

	const presetScreenX = 600; // near the right, full height
	const presetScreenWidth = 20;

	// Symmetric quadratic bump across a wall's thickness: 0 at both faces,
	// ramping up to presetDampingValue at the center — the same t*t shape as
	// the border sponge in buildSpongeField, just mirrored on both sides
	// instead of ramping into a hard wall on one side. A flat-topped block of
	// damping is a hard impedance step the wave partially reflects off;
	// easing into it over many pixels (this needs real thickness, not a
	// couple of pixels, since the ramp has to span a few wavelengths to work)
	// lets the wave pass through without bouncing off the front face.
	const wallDampingRamp = (offsetIntoWall: number, thickness: number) => {
		const center = thickness / 2;
		const dist = Math.abs(offsetIntoWall + 0.5 - center);
		const t = Math.max(0, (center - dist) / center);
		return 50 * t * t;
	};

	const buildPresetEmitters = (
		points: { x: number; y: number; frequency: number; amplitude: number }[]
	) => {
		const data: EmitterValue[] = new Array(width * height)
			.fill(null)
			.map(() => ({ frequency: 0, phase: 0, amplitude: 0 }));
		for (const { x, y, frequency, amplitude } of points) {
			forEachInRect(
				x - presetEmitterRadius,
				x + presetEmitterRadius + 1,
				y - presetEmitterRadius,
				y + presetEmitterRadius + 1,
				(idx) => {
					data[idx] = { frequency, phase: 0, amplitude };
				}
			);
		}
		return data;
	};

	// A damping wall across the full height at presetDampingX, with `gaps`
	// (each gapHeight tall, centered at the given y positions) left undamped.
	const buildPresetDamping = (gapCenters: number[], gapHeight: number) => {
		const data = new Float32Array(width * height);
		const half = gapHeight / 2;
		for (let y = 0; y < height; y++) {
			const inGap = gapCenters.some((c) => y >= c - half && y < c + half);
			if (inGap) continue;
			for (let x = presetDampingX; x < presetDampingX + presetDampingWidth; x++) {
				data[x + y * width] = presetDampingValue;
			}
		}
		return data;
	};

	const buildPresetScreen = (includeScreen: boolean) => {
		const data: { isScreen: number; accum: number }[] = new Array(width * height)
			.fill(null)
			.map(() => ({ isScreen: 0, accum: 0 }));
		if (includeScreen) {
			forEachInRect(presetScreenX, presetScreenX + presetScreenWidth, 0, height, (idx) => {
				data[idx] = { isScreen: 1, accum: 0 };
			});
		}
		return data;
	};

	const applyPreset = (name: Preset) => {
		preset = name;
		if (!refraction || !emitters || !buffers || !damping || !screen) return;

		refraction.write(new Float32Array(width * height).fill(1).buffer);
		const zero = new Float32Array(width * height);
		for (const buf of buffers) buf.write(zero.buffer);
		simTime = 0;

		if (name === 'doubleSlit') {
			emitters.write(
				buildPresetEmitters([{ x: presetEmitterX, y: presetEmitterY, frequency: 2, amplitude: 10 }])
			);
			const slitSeparation = 80;
			paintedDamping.set(
				buildPresetDamping(
					[presetEmitterY - slitSeparation / 2, presetEmitterY + slitSeparation / 2],
					20
				)
			);
			screen.write(buildPresetScreen(true));
		} else if (name === 'singleSlit') {
			emitters.write(
				buildPresetEmitters([{ x: presetEmitterX, y: presetEmitterY, frequency: 2, amplitude: 10 }])
			);
			paintedDamping.set(buildPresetDamping([presetEmitterY], 20));
			screen.write(buildPresetScreen(true));
		} else if (name === 'subwavelength') {
			emitters.write(
				buildPresetEmitters([{ x: presetEmitterX, y: presetEmitterY, frequency: 1, amplitude: 10 }])
			);
			const holeCount = 7;
			const holeSpacing = 80;
			const firstCenter = presetEmitterY - ((holeCount - 1) * holeSpacing) / 2;
			const holeCenters = new Array(holeCount).fill(0).map((_, i) => firstCenter + i * holeSpacing);
			paintedDamping.set(buildPresetDamping(holeCenters, 4));
			screen.write(buildPresetScreen(true));
		} else if (name === 'doubleEmitter') {
			// Stands in for the double-slit setup by emitting from the slit
			// locations directly, so amplitude is much lower than the emitter's
			// normal 10 since the wave no longer has to pass through a barrier.
			const slitSeparation = 80;
			const emitterX = presetDampingX + presetDampingWidth / 2;
			emitters.write(
				buildPresetEmitters([
					{ x: emitterX, y: presetEmitterY - slitSeparation / 2, frequency: 2, amplitude: 1 },
					{ x: emitterX, y: presetEmitterY + slitSeparation / 2, frequency: 2, amplitude: 1 }
				])
			);
			paintedDamping.fill(0);
			screen.write(buildPresetScreen(true));
		} else if (name === 'huygensFresnel') {
			// Two mirrored, top-to-bottom experiments side by side, split by a
			// vertical damping wall down the middle: the left half runs a single
			// emitter through a double slit, the right half emits directly from
			// the slit locations (Huygens' construction) at much lower amplitude
			// since there's no barrier to pass through.
			const halfWidth = width / 2;
			const wallY = presetEmitterY; // 300, same row used by the other presets' slit wall
			const slitWallThickness = presetDampingWidth;
			// Much thicker than the slit wall: the ramp needs room to ease the
			// wave in over several wavelengths, not just a couple of pixels.
			const separatorThickness = 80;
			const screenY = 500;
			const screenThickness = presetScreenWidth;
			const slitSeparation = 80;
			const slitWidth = 20;
			const leftSlitCenter1 = halfWidth / 2 - slitSeparation / 2;
			const leftSlitCenter2 = halfWidth / 2 + slitSeparation / 2;

			const huygensDamping = new Float32Array(width * height);
			// Separator: waves travel along x through it, so ramp along x.
			for (let y = 0; y < height; y++) {
				for (let x = 0; x < separatorThickness; x++) {
					const px = halfWidth - separatorThickness / 2 + x;
					huygensDamping[px + y * width] = wallDampingRamp(x, separatorThickness);
				}
			}
			// Double-slit wall: kept as a flat block (unlike the separator, some
			// reflection off the front face here is fine).
			for (let y = wallY - slitWallThickness / 2; y < wallY + slitWallThickness / 2; y++) {
				for (let x = 0; x < halfWidth; x++) {
					const inSlit1 =
						x >= leftSlitCenter1 - slitWidth / 2 && x < leftSlitCenter1 + slitWidth / 2;
					const inSlit2 =
						x >= leftSlitCenter2 - slitWidth / 2 && x < leftSlitCenter2 + slitWidth / 2;
					if (inSlit1 || inSlit2) continue;
					huygensDamping[x + y * width] = presetDampingValue;
				}
			}
			paintedDamping.set(huygensDamping);

			emitters.write(
				buildPresetEmitters([
					{ x: halfWidth / 2, y: 100, frequency: 2, amplitude: 5 },
					{ x: halfWidth + leftSlitCenter1, y: wallY, frequency: 2, amplitude: 1 },
					{ x: halfWidth + leftSlitCenter2, y: wallY, frequency: 2, amplitude: 1 }
				])
			);

			const huygensScreen: { isScreen: number; accum: number }[] = new Array(width * height)
				.fill(null)
				.map(() => ({ isScreen: 0, accum: 0 }));
			forEachInRect(
				0,
				width,
				screenY - screenThickness / 2,
				screenY + screenThickness / 2,
				(idx) => {
					huygensScreen[idx] = { isScreen: 1, accum: 0 };
				}
			);
			screen.write(huygensScreen);
		} else {
			emitters.write(buildPresetEmitters([]));
			paintedDamping.fill(0);
			screen.write(buildPresetScreen(false));
		}
		writeDamping();
	};

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
		if (!refraction || !emitters || !buffers || !damping || !screen) return;
		if (tool === 'poke') {
			const values: Record<number, number> = {};
			forEachPenCell(e, (idx) => {
				values[idx] = 1;
			});
			buffers[1].patch(values);
		} else if (tool === 'emitter') {
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
		} else if (tool === 'damper') {
			const values: Record<number, number> = {};
			forEachPenCell(e, (idx) => {
				paintedDamping[idx] = dampingIntensity;
				values[idx] = baseDamping(idx) + dampingIntensity;
			});
			damping.patch(values);
		} else if (tool === 'screen') {
			const values: Record<number, { isScreen: number }> = {};
			forEachPenCell(e, (idx) => {
				values[idx] = { isScreen: 1 };
			});
			screen.patch(values);
		} else {
			const emitterValues: Record<number, EmitterValue> = {};
			const refractionValues: Record<number, number> = {};
			const dampingValues: Record<number, number> = {};
			const screenValues: Record<number, { isScreen: number; accum: number }> = {};
			forEachPenCell(e, (idx) => {
				emitterValues[idx] = { frequency: 0, phase: 0, amplitude: 0 };
				refractionValues[idx] = 1;
				paintedDamping[idx] = 0;
				dampingValues[idx] = baseDamping(idx);
				screenValues[idx] = { isScreen: 0, accum: 0 };
			});
			emitters.patch(emitterValues);
			refraction.patch(refractionValues);
			damping.patch(dampingValues);
			screen.patch(screenValues);
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
		damping = root.createBuffer(d.arrayOf(d.f32, width * height)).$usage('storage');
		emitters = root.createBuffer(d.arrayOf(emitterData, width * height)).$usage('storage');
		screen = root.createBuffer(d.arrayOf(screenData, width * height)).$usage('storage');

		applyPreset(preset);

		const frame = () => {
			if (!buffers || !refraction || !damping || !emitters || !screen) return;
			const dt = deltat / substeps;
			let bufs: GPUArray[] = buffers;
			for (let i = 0; i < substeps; i++) {
				gpuWaveFunction(root, bufs[0], bufs[1], bufs[2], refraction, damping, emitters, screen, {
					width,
					height,
					speed,
					deltat: dt,
					time: simTime,
					boundary: boundary === 'periodic' ? 1 : 0,
					screenAttenuation
				});
				simTime += dt;
				bufs = [bufs[1], bufs[2], bufs[0]];
			}
			buffers = bufs;
			renderWave(root, bufs[1], refraction, emitters, damping, screen, context, {
				width,
				height
			});
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

	<div class="my-2 flex items-center gap-2">
		<span>Preset:</span>
		{#each presets as p (p.id)}
			<button
				class="cursor-pointer rounded px-4 py-1 transition-colors {preset === p.id
					? 'bg-blue-600 text-white'
					: 'bg-white/20 hover:bg-white/30'}"
				onclick={() => applyPreset(p.id)}
			>
				{p.label}
			</button>
		{/each}
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

	<div class="my-2 flex items-center gap-2">
		<span>Boundary:</span>
		{#each boundaries as b (b)}
			<button
				class="cursor-pointer rounded px-4 py-1 capitalize transition-colors {boundary === b
					? 'bg-blue-600 text-white'
					: 'bg-white/20 hover:bg-white/30'}"
				onclick={() => setBoundary(b)}
			>
				{b}
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
	{:else if tool === 'damper'}
		{@render labeledSlider(
			'Damping intensity',
			dampingIntensity,
			1,
			400,
			1,
			(v) => (dampingIntensity = v)
		)}
	{:else if tool === 'screen'}
		{@render labeledSlider(
			'Attenuation',
			screenAttenuation,
			0,
			5,
			0.01,
			(v) => (screenAttenuation = v)
		)}
	{/if}

	{@render labeledSlider('Speed', speed, 10, 200, 0.01, (v) => (speed = v))}
	{@render labeledSlider('Substeps', substeps, 1, 32, 1, (v) => (substeps = v))}
</div>
