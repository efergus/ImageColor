# Working in this repo

- Do not try to start the dev server, launch a browser, or otherwise verify changes visually (including via the `run` skill or Playwright). The user verifies in the browser themselves. Type-checking (`svelte-check`) and manual/algebraic review are sufficient before handing off a change.

## Finding TypeGPU examples

`node_modules/typegpu` only ships `.d.ts`/`.js` — no usage examples. For real usage patterns (e.g. `tgpu.vertexLayout`, `tgpu.vertexFn`/`tgpu.fragmentFn`, multi-object draws, bind groups), shallow-clone the TypeGPU repo's docs app and grep its examples:

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/software-mansion/TypeGPU.git <scratch-dir>
cd <scratch-dir>
git sparse-checkout set apps/typegpu-docs
grep -rl "<api name>" apps/typegpu-docs/src/examples
```

Examples live under `apps/typegpu-docs/src/examples/<category>/<name>/index.ts` (categories: `simple`, `rendering`, `simulation`, `algorithms`, `react`). `rendering/two-boxes` is a good reference for drawing multiple distinct objects (own vertex buffer + bind group each) in one frame with a shared pipeline. Clone into the scratchpad dir, not into this repo.
