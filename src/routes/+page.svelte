<script lang="ts">
	import { resolve } from '$app/paths';
</script>

<svelte:head><title>Image Color</title></svelte:head>

<div class="wrapper">
	<h1>Image Color</h1>
	<nav>
		<a class="btn" href={resolve('/color')}>Color Cloud</a>
		<a class="btn" href={resolve('/wave')}>Wave</a>
		<a class="btn" href={resolve('/life')}>Game of Life</a>
	</nav>
</div>

<style>
	.wrapper {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 100vh;
		gap: 1.5rem;
		color: #eee;
		font-family: system-ui, sans-serif;
	}

	nav {
		display: flex;
		gap: 1rem;
	}

	.btn {
		padding: 0.6rem 1.4rem;
		background: #2a6be8;
		border-radius: 6px;
		text-decoration: none;
		color: #fff;
		font-weight: 600;
		transition: background 0.15s;
	}

	.btn:hover {
		background: #3a7ef8;
	}
</style>
