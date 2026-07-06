<script lang="ts">
	import { Slider } from 'bits-ui';
	import { onMount } from 'svelte';

	let speed = $state(1);

	let canvas: HTMLCanvasElement;
</script>

<div class="page-container">
	<div role="img" class="canvas-container">
		<canvas bind:this={canvas} width="800" height="600"></canvas>
	</div>

	<Slider.Root
		type="single"
		value={speed}
		min={0}
		max={10}
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
</div>
