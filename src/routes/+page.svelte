<script lang="ts">
	// TODO notes:
	// When hovering the image, I want to show a cloud of only colors in the hover zone

	import { run } from 'svelte/legacy';

	import { onMount, onDestroy } from 'svelte';
	import tgpu from 'typegpu';
	import * as d from 'typegpu/data';
	import beeCloseImg from '$lib/assets/bee_close.jpg';
	import flowerImg from '$lib/assets/flower.jpg';
	import pastelsImg from '$lib/assets/pastels.jpg';

	// ── TypeGPU typed uniform schemas ─────────────────────────────────────────
	// d.struct() mirrors the WGSL uniform structs; TypeGPU enforces vec3f 16-byte
	// alignment automatically, matching WGSL layout rules exactly.
	const CameraStruct = d.struct({
		yaw: d.f32,
		pitch: d.f32,
		radius: d.f32,
		aspect: d.f32,
		weightMin: d.f32,
		weightMax: d.f32,
		hoverEnabled: d.f32,
		hoverThreshold: d.f32,
		hoveredColor: d.vec3f,
		colorSpace: d.f32,
		lutSize: d.f32,
		steps: d.f32,
		_pad1: d.f32,
		_pad2: d.f32
	});

	// 96-byte struct matching HighlightUniforms in imgPreviewFragWGSL.
	const HighlightStruct = d.struct({
		hoveredColor: d.vec3f,
		threshold: d.f32,
		enabled: d.u32,
		colorSpace: d.u32,
		flattenWeight: d.f32,
		contrast: d.f32,
		cloudCenter: d.vec3f,
		saturation: d.f32,
		cameraNormal: d.vec3f,
		avgL: d.f32,
		rotation: d.f32,
		targetHue: d.f32,
		hueWeight: d.f32,
		targetChroma: d.f32,
		chromaWeight: d.f32,
		_pad3: d.f32,
		_pad4: d.f32,
		_pad5: d.f32
	});

	const presets = [
		{ name: 'Bee', src: beeCloseImg },
		{ name: 'Flower', src: flowerImg },
		{ name: 'Pastels', src: pastelsImg }
	];

	// ── Constants ─────────────────────────────────────────────────────────────
	const WORKGROUP_SIZE = 8;
	const CANVAS_W = 400;
	const CANVAS_H = 300;

	// ── Reactive state ────────────────────────────────────────────────────────
	let webgpuSupported = $state(true);
	let webgpuError = $state('');
	let imageLoaded = $state(false);
	let imageName = $state('');
	let isBuilding = $state(false);

	// Camera
	let yaw = $state(0.4);
	let pitch = $state(-0.25);
	let radius = $state(2.2);

	// Weight-space opacity window.
	//   weightMin: voxels with fewer pixels than this fraction of max are hidden (reveals interior)
	//   weightMax: voxels at or above this fraction of max are treated as fully opaque (amplifies faint)
	// Both are in [0, 1] relative to the max raw weight in the table.
	let sliderWeightMin = $state(0.0);
	let sliderWeightMax = $state(0.3); // auto-set to 1.0 on load (= the actual maximum)
	let weightMin = $derived(sliderWeightMin === 0 ? 0 : Math.pow(10, sliderWeightMin * 3 - 3));
	let weightMax = $derived(sliderWeightMax === 0 ? 0 : Math.pow(10, sliderWeightMax * 3 - 3));

	let lutSize = $state(42);
	let pendingLutSize = $state(42);
	let colorSpace = $state('oklab');
	let flattenWeight = $state(1.0);
	let enableFlattening = $state(true);
	let contrast = $state(1.0);
	let saturation = $state(1.0);
	let rotation = $state(0.0);

	let enableHuePull = $state(false);
	let targetHue = $state(0.0);
	let hueWeight = $state(0.0);

	let enableChromaPull = $state(false);
	let targetChroma = $state(0.1);
	let chromaWeight = $state(0.0);

	let lastBuiltYaw = null,
		lastBuiltPitch = null,
		lastBuiltFlatten = 1.0,
		lastBuiltEnableFlattening = true;
	let lastBuiltContrast = 1.0,
		lastBuiltSaturation = 1.0,
		lastBuiltRotation = 0.0;
	let lastBuiltTargetHue = 0.0,
		lastBuiltHueWeight = 0.0,
		lastBuiltEnableHuePull = false;
	let lastBuiltTargetChroma = 0.1,
		lastBuiltChromaWeight = 0.0,
		lastBuiltEnableChromaPull = false;

	// Cloud Center for Flattening
	let cloudCenterRGB = [0.5, 0.5, 0.5];
	let cloudCenterOKLab = [0.5, 0.5, 0.5];

	// Interaction
	let isDragging = false;
	let lastX = 0,
		lastY = 0;

	// Hover / pick
	let hoveredColor = $state(null); // {r,g,b} in [0,1] or null
	let colorThreshold = $state(30); // Euclidean RGB-255 distance
	let isReadingBack = false;

	// Image preview sizing
	let previewW = $state(0),
		previewH = $state(0);
	let imgPreviewCanvas = $state();

	// ── GPU handles ───────────────────────────────────────────────────────────
	// `root` is the TypeGPU root — it wraps GPUDevice and manages typed resources.
	let root: any; // TgpuRoot
	let device: GPUDevice; // root.device shortcut set in init()
	let canvasCtx: GPUCanvasContext, canvasFormat: GPUTextureFormat;
	let animationId: number;
	let renderPipeline: any, sampler: GPUSampler;
	let renderBindGroup: any;
	let computePipeline: any;
	let checkerTexture: GPUTexture; // raw GPUTexture (writeTexture needs raw handle)
	let colorTex3D: GPUTexture; // raw GPUTexture (3D LUT — rebuilt on LUT size change)

	// TypeGPU typed uniform buffers — write() replaces manual Float32Array indexing.
	let cameraUniform: any; // TgpuUniform<CameraStruct>
	let hlUniform: any; // TgpuUniform<HighlightStruct>

	// MRT: color-pick render target + staging buffer (raw — needs RENDER_ATTACHMENT)
	let colorPickTexture: GPUTexture;
	let colorPickStagingBuf: GPUBuffer;

	// Image preview GPU handles
	let imgTexture: GPUTexture;
	let imgPreviewCtx: GPUCanvasContext, imgPreviewFormat: GPUTextureFormat;
	let imgPreviewPipeline: any, imgPreviewBindGroup: any;
	// hlUniform doubles as the image-preview highlight uniform buffer (same struct)

	// Keep the last loaded imageData so we can rebuild on LUT change
	let lastImageData: ImageData | null = null;

	// Dynamic resolution
	let renderSteps = 128;
	let motionTimeout = null;

	function triggerMotion() {
		renderSteps = 32;
		isDirty = true;
		if (motionTimeout) clearTimeout(motionTimeout);
		motionTimeout = setTimeout(() => {
			renderSteps = 128;
			isDirty = true;
			motionTimeout = null;
			if (imageLoaded) {
				const currentEffectiveFlatten = enableFlattening ? flattenWeight : 1.0;
				const lastEffectiveFlatten = lastBuiltEnableFlattening ? lastBuiltFlatten : 1.0;
				const needsRebuild =
					(currentEffectiveFlatten !== 1.0 &&
						(lastBuiltYaw !== yaw ||
							lastBuiltPitch !== pitch ||
							currentEffectiveFlatten !== lastEffectiveFlatten)) ||
					(currentEffectiveFlatten === 1.0 && lastEffectiveFlatten !== 1.0) ||
					contrast !== lastBuiltContrast ||
					saturation !== lastBuiltSaturation ||
					rotation !== lastBuiltRotation ||
					targetHue !== lastBuiltTargetHue ||
					hueWeight !== lastBuiltHueWeight ||
					enableHuePull !== lastBuiltEnableHuePull ||
					targetChroma !== lastBuiltTargetChroma ||
					chromaWeight !== lastBuiltChromaWeight ||
					enableChromaPull !== lastBuiltEnableChromaPull;
				if (needsRebuild) {
					lastBuiltYaw = yaw;
					lastBuiltPitch = pitch;
					lastBuiltFlatten = flattenWeight;
					lastBuiltEnableFlattening = enableFlattening;
					lastBuiltContrast = contrast;
					lastBuiltSaturation = saturation;
					lastBuiltRotation = rotation;
					lastBuiltTargetHue = targetHue;
					lastBuiltHueWeight = hueWeight;
					lastBuiltEnableHuePull = enableHuePull;
					lastBuiltTargetChroma = targetChroma;
					lastBuiltChromaWeight = chromaWeight;
					lastBuiltEnableChromaPull = enableChromaPull;
					if (!isBuilding && lastImageData) {
						isBuilding = true;
						buildVolumeFromImage(lastImageData, lutSize, colorSpace).then(() => {
							isBuilding = false;
							isDirty = true;
						});
					} else if (isBuilding) {
						// try again soon
						motionTimeout = setTimeout(triggerMotion, 150);
					}
				}
			}
		}, 150);
	}

	// Dirty flag to prevent excessive rendering
	let isDirty = true;
	run(() => {
		yaw;
		pitch;
		radius;
		weightMin;
		weightMax;
		colorThreshold;
		hoveredColor;
		lutSize;
		colorSpace;
		imageLoaded;
		flattenWeight;
		enableFlattening;
		contrast;
		saturation;
		rotation;
		targetHue;
		hueWeight;
		enableHuePull;
		targetChroma;
		chromaWeight;
		enableChromaPull;
		triggerMotion();
	});

	// ── Utils ─────────────────────────────────────────────────────────────────
	const wgslColorSpaceUtils = `
        fn linear_srgb(c: vec3<f32>) -> vec3<f32> {
            let b = c <= vec3<f32>(0.04045);
            let low = c / 12.92;
            let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
            return select(high, low, b);
        }
        fn rgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
            let l_c = linear_srgb(c);
            let l = 0.4122214708 * l_c.r + 0.5363325363 * l_c.g + 0.0514459929 * l_c.b;
            let m = 0.2119034982 * l_c.r + 0.6806995451 * l_c.g + 0.1073969566 * l_c.b;
            let s = 0.0883024619 * l_c.r + 0.2817188376 * l_c.g + 0.6299787005 * l_c.b;
            let l_ = pow(max(l, 0.0), 1.0 / 3.0);
            let m_ = pow(max(m, 0.0), 1.0 / 3.0);
            let s_ = pow(max(s, 0.0), 1.0 / 3.0);
            return vec3<f32>(
                0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
            );
        }
        fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
            let b = c <= vec3<f32>(0.0031308);
            let low = c * 12.92;
            let high = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
            return select(high, low, b);
        }
        fn oklab_to_rgb(c: vec3<f32>) -> vec3<f32> {
            let l_ = c.x + 0.3963377774 * c.y + 0.2158037573 * c.z;
            let m_ = c.x - 0.1055613458 * c.y - 0.0638541728 * c.z;
            let s_ = c.x - 0.0894841775 * c.y - 1.2914855480 * c.z;
            let l = l_ * l_ * l_;
            let m = m_ * m_ * m_;
            let s = s_ * s_ * s_;
            let r =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
            let g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
            let b_val = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
            return clamp(linear_to_srgb(vec3<f32>(r, g, b_val)), vec3<f32>(0.0), vec3<f32>(1.0));
        }
        fn color_dist(c1: vec3<f32>, c2: vec3<f32>, space: u32) -> f32 {
            if (space == 1u) {
                let o1 = rgb_to_oklab(c1);
                let o2 = rgb_to_oklab(c2);
                return length(o1 - o2) * 255.0;
            }
            return length((c1 - c2) * 255.0);
        }
    `;

	function js_oklab_to_linear_srgb(L, a, b) {
		let l_ = L + 0.3963377774 * a + 0.2158037573 * b;
		let m_ = L - 0.1055613458 * a - 0.0638541728 * b;
		let s_ = L - 0.0894841775 * a - 1.291485548 * b;
		let l = l_ * l_ * l_;
		let m = m_ * m_ * m_;
		let s = s_ * s_ * s_;
		return [
			4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
			-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
			-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s
		];
	}
	function js_linear_to_srgb(c) {
		return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(Math.max(0, c), 1.0 / 2.4) - 0.055;
	}
	function js_oklab_to_srgb(L, a, b) {
		let lin = js_oklab_to_linear_srgb(L, a, b);
		return [js_linear_to_srgb(lin[0]), js_linear_to_srgb(lin[1]), js_linear_to_srgb(lin[2])];
	}

	// ── WGSL: vertex (static) ─────────────────────────────────────────────────
	const vertWGSL = /* wgsl */ `
        struct VSOut {
            @builtin(position) pos : vec4<f32>,
            @location(0)       uv  : vec2<f32>,
        }
        @vertex
        fn main(@builtin(vertex_index) vi : u32) -> VSOut {
            var positions = array<vec2<f32>, 6>(
                vec2(-1., -1.), vec2( 1., -1.), vec2(-1.,  1.),
                vec2(-1.,  1.), vec2( 1., -1.), vec2( 1.,  1.)
            );
            var uvs = array<vec2<f32>, 6>(
                vec2(0., 1.), vec2(1., 1.), vec2(0., 0.),
                vec2(0., 0.), vec2(1., 1.), vec2(1., 0.)
            );
            var out : VSOut;
            out.pos = vec4<f32>(positions[vi], 0., 1.);
            out.uv  = uvs[vi];
            return out;
        }
    `;

	// ── WGSL: fragment (ray marcher) ──────────────────────────────────────────
	const fragWGSL = `
        ${wgslColorSpaceUtils}
        @group(0) @binding(0) var checkerTex : texture_2d<f32>;
        @group(0) @binding(1) var texSampler  : sampler;

        @group(1) @binding(0) var colorVol   : texture_3d<f32>;
        @group(1) @binding(1) var<uniform> cam : CameraUniforms;

        struct CameraUniforms {
            yaw            : f32,
            pitch          : f32,
            radius         : f32,
            aspect         : f32,
            weightMin      : f32,
            weightMax      : f32,
            hoverEnabled   : f32,   // >0 when hovering
            hoverThreshold : f32,
            hoveredColor   : vec3<f32>,
            colorSpace     : f32,   // 0: rgb, 1: oklab
            lutSize        : f32,
            steps          : f32,
            _pad1          : f32,
            _pad2          : f32,
        }

        fn rotY(a : f32) -> mat3x3<f32> {
            let c = cos(a); let s = sin(a);
            return mat3x3<f32>(
                vec3( c, 0., s),
                vec3( 0., 1., 0.),
                vec3(-s, 0., c)
            );
        }
        fn rotX(a : f32) -> mat3x3<f32> {
            let c = cos(a); let s = sin(a);
            return mat3x3<f32>(
                vec3(1.,  0.,  0.),
                vec3(0.,  c,  -s),
                vec3(0.,  s,   c)
            );
        }

        fn boxIntersect(ro : vec3<f32>, rd : vec3<f32>) -> vec2<f32> {
            let invD = 1.0 / rd;
            let t1   = (vec3(0.) - ro) * invD;
            let t2   = (vec3(1.) - ro) * invD;
            let tMin = min(t1, t2);
            let tMax = max(t1, t2);
            return vec2(
                max(max(tMin.x, tMin.y), tMin.z),
                min(min(tMax.x, tMax.y), tMax.z)
            );
        }

        struct FragOut {
            @location(0) color : vec4<f32>,
            @location(1) pick  : vec4<f32>,
        }

        @fragment
        fn main(@location(0) uv : vec2<f32>) -> FragOut {
            let bg = textureSample(checkerTex, texSampler, uv);

            let center = vec3<f32>(0.5);
            let eye    = center + rotY(cam.yaw) * (rotX(-cam.pitch) * vec3(0., 0., cam.radius));
            let fwd    = normalize(center - eye);
            let right  = normalize(cross(fwd, vec3(0., 1., 0.)));
            let up     = cross(right, fwd);

            let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
            let rd  = normalize(fwd + right * ndc.x * cam.aspect * 0.6
                                     + up    * ndc.y * 0.6);

            let hit = boxIntersect(eye, rd);
            if (hit.x > hit.y) {
                var out : FragOut;
                out.color = bg;
                out.pick  = vec4(0.0);
                return out;
            }

            let tStart = max(hit.x, 0.);
            let tEnd   = hit.y;
            let fSteps = max(1.0, cam.steps);
            let STEPS  = i32(fSteps);
            let step   = (tEnd - tStart) / fSteps;

            let dims   = vec3<f32>(cam.lutSize);
            let wRange = max(cam.weightMax - cam.weightMin, 0.0001);

            var acc = vec4<f32>(0.);
            var firstHit = vec4<f32>(0.);

            for (var i = 0; i < STEPS; i++) {
                let pos = eye + rd * (tStart + (f32(i) + 0.5) * step);
                let uvw = clamp(pos, vec3(0.001), vec3(0.999));

                let texel = textureSampleLevel(colorVol, texSampler, uvw, 0.0);

                let w = texel.a;
                if (w < cam.weightMin) { continue; }

                let remapped = (w - cam.weightMin) / wRange;
                var alpha    = 1.0 - exp(-remapped * step * 14.0);

                // When hovering, halve opacity for voxels whose colour
                // is NOT close to the picked colour (display output only;
                // firstHit / pick is recorded below, unaffected).
                if (cam.hoverEnabled > 0.0) {
                    if (cam.colorSpace > 0.5) {
                        let o1 = rgb_to_oklab(texel.rgb);
                        let o2 = rgb_to_oklab(cam.hoveredColor);
                        let dist = length(o1 - o2) * 255.0;
                        if (dist >= cam.hoverThreshold) {
                            alpha *= 0.1;
                        }
                    } else {
                        let cdiff = (texel.rgb - cam.hoveredColor) * 255.0;
                        if (length(cdiff) >= cam.hoverThreshold) {
                            alpha *= 0.1;
                        }
                    }
                }

                acc += vec4(texel.rgb * alpha, alpha) * (1.0 - acc.a);

                // Record the voxel colour once overall opacity reaches 20%
                if (firstHit.a == 0.0 && acc.a >= 0.2) {
                    firstHit = vec4(texel.rgb, 1.0);
                }

                if (acc.a > 0.99) { break; }
            }

            var out : FragOut;
            out.color = vec4(acc.rgb + bg.rgb * (1.0 - acc.a), 1.0);
            out.pick  = firstHit;
            return out;
        }
    `;

	// ── WGSL: compute (weight accumulation) ───────────────────────────────────
	const computeWGSL = `
        ${wgslColorSpaceUtils}
        @group(0) @binding(0) var<storage, read>       pixels  : array<u32>;
        @group(0) @binding(1) var<storage, read_write> weights : array<atomic<u32>>;
        
        struct ComputeSettings {
            imgSize        : vec2<u32>,
            lutSize        : u32,
            colorSpace     : u32,
            flattenWeight  : f32,
            contrast       : f32,
            saturation     : f32,
            avgL           : f32,
            cloudCenter    : vec3<f32>,
            rotation       : f32,
            cameraNormal   : vec3<f32>,
            _pad4          : f32,
            targetHue      : f32,
            hueWeight      : f32,
            targetChroma   : f32,
            chromaWeight   : f32,
        }
        @group(0) @binding(2) var<uniform> settings : ComputeSettings;

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
            let x = gid.x; let y = gid.y;
            if (x >= settings.imgSize.x || y >= settings.imgSize.y) { return; }

            let rgba = pixels[y * settings.imgSize.x + x];
            let r8   = (rgba >>  0u) & 0xFFu;
            let g8   = (rgba >>  8u) & 0xFFu;
            let b8   = (rgba >> 16u) & 0xFFu;

            var ri : u32;
            var gi : u32;
            var bi : u32;

            var cPos : vec3<f32>;
            let cOrig = vec3<f32>(f32(r8)/255.0, f32(g8)/255.0, f32(b8)/255.0);

            if (abs(settings.contrast - 1.0) > 0.001 || abs(settings.saturation - 1.0) > 0.001 || abs(settings.rotation) > 0.001 || settings.hueWeight > 0.0 || settings.chromaWeight > 0.0) {
                var o = rgb_to_oklab(cOrig);
                if (abs(settings.contrast - 1.0) > 0.001) {
                    o.x = settings.avgL + (o.x - settings.avgL) * settings.contrast;
                }
                if (abs(settings.saturation - 1.0) > 0.001) {
                    o.y = o.y * settings.saturation;
                    o.z = o.z * settings.saturation;
                }
                if (abs(settings.rotation) > 0.001) {
                    let rad = radians(settings.rotation);
                    let s = sin(rad);
                    let c = cos(rad);
                    let a = o.y;
                    let b = o.z;
                    o.y = a * c - b * s;
                    o.z = a * s + b * c;
                }
                if (settings.hueWeight > 0.0 || settings.chromaWeight > 0.0) {
                    var C = sqrt(o.y * o.y + o.z * o.z);
                    var H = atan2(o.z, o.y);
                    if (settings.hueWeight > 0.0) {
                        let targetH_rad = radians(settings.targetHue);
                        var deltaH = targetH_rad - H;
                        let PI = 3.14159265359;
                        deltaH = deltaH - 2.0 * PI * floor((deltaH + PI) / (2.0 * PI));
                        H = H + deltaH * settings.hueWeight;
                    }
                    if (settings.chromaWeight > 0.0) {
                        C = mix(C, settings.targetChroma, settings.chromaWeight);
                    }
                    o.y = C * cos(H);
                    o.z = C * sin(H);
                }
                if (settings.colorSpace > 0u) {
                    cPos = vec3<f32>((o.y + 0.4)/0.8, o.x, (o.z + 0.4)/0.8);
                } else {
                    cPos = oklab_to_rgb(o);
                }
            } else {
                if (settings.colorSpace > 0u) {
                    let o = rgb_to_oklab(cOrig);
                    cPos = vec3<f32>((o.y + 0.4)/0.8, o.x, (o.z + 0.4)/0.8);
                } else {
                    cPos = cOrig;
                }
            }

            if (abs(settings.flattenWeight - 1.0) > 0.001) {
                let N = normalize(settings.cameraNormal);
                let distToPlane = dot(cPos - settings.cloudCenter, N);
                let projPos = cPos - distToPlane * N;
                cPos = mix(projPos, cPos, settings.flattenWeight);
            }

            if (settings.colorSpace > 0u) {
                let a_norm = clamp(cPos.x, 0.0, 1.0);
                let L_norm = clamp(cPos.y, 0.0, 1.0);
                let b_norm = clamp(cPos.z, 0.0, 1.0);

                ri = u32(round(a_norm * f32(settings.lutSize - 1u)));
                gi = u32(round(L_norm * f32(settings.lutSize - 1u)));
                bi = u32(round(b_norm * f32(settings.lutSize - 1u)));
            } else {
                let cClamp = clamp(cPos, vec3<f32>(0.0), vec3<f32>(1.0));
                ri = u32(round(cClamp.x * f32(settings.lutSize - 1u)));
                gi = u32(round(cClamp.y * f32(settings.lutSize - 1u)));
                bi = u32(round(cClamp.z * f32(settings.lutSize - 1u)));
            }

            if (ri < settings.lutSize && gi < settings.lutSize && bi < settings.lutSize) {
                atomicAdd(&weights[ri + gi * settings.lutSize + bi * settings.lutSize * settings.lutSize], 1u);
            }
        }
    `;

	// ── WGSL: image preview fragment (highlight matching colours) ────────────
	const imgPreviewFragWGSL =
		`
        ${wgslColorSpaceUtils}` +
		/* wgsl */ `
        @group(0) @binding(0) var imgTex  : texture_2d<f32>;
        @group(0) @binding(1) var imgSamp : sampler;
        @group(0) @binding(2) var<uniform> hl : HighlightUniforms;

        struct HighlightUniforms {
            hoveredColor   : vec3<f32>,
            threshold      : f32,
            enabled        : u32,
            colorSpace     : u32,
            flattenWeight  : f32,
            contrast       : f32,
            cloudCenter    : vec3<f32>,
            saturation     : f32,
            cameraNormal   : vec3<f32>,
            avgL           : f32,
            rotation       : f32,
            targetHue      : f32,
            hueWeight      : f32,
            targetChroma   : f32,
            chromaWeight   : f32,
            _pad3          : f32,
            _pad4          : f32,
            _pad5          : f32,
        }

        struct VSOut {
            @builtin(position) pos : vec4<f32>,
            @location(0)       uv  : vec2<f32>,
        }
        @vertex
        fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
            var positions = array<vec2<f32>, 6>(
                vec2(-1., -1.), vec2( 1., -1.), vec2(-1.,  1.),
                vec2(-1.,  1.), vec2( 1., -1.), vec2( 1.,  1.)
            );
            var uvs = array<vec2<f32>, 6>(
                vec2(0., 1.), vec2(1., 1.), vec2(0., 0.),
                vec2(0., 0.), vec2(1., 1.), vec2(1., 0.)
            );
            var out : VSOut;
            out.pos = vec4<f32>(positions[vi], 0., 1.);
            out.uv  = uvs[vi];
            return out;
        }

        fn getProcessedColor(cOrig : vec3<f32>) -> vec3<f32> {
            var cPos : vec3<f32>;
            
            if (abs(hl.contrast - 1.0) > 0.001 || abs(hl.saturation - 1.0) > 0.001 || abs(hl.rotation) > 0.001 || hl.hueWeight > 0.0 || hl.chromaWeight > 0.0) {
                var o = rgb_to_oklab(cOrig);
                if (abs(hl.contrast - 1.0) > 0.001) {
                    o.x = hl.avgL + (o.x - hl.avgL) * hl.contrast;
                }
                if (abs(hl.saturation - 1.0) > 0.001) {
                    o.y = o.y * hl.saturation;
                    o.z = o.z * hl.saturation;
                }
                if (abs(hl.rotation) > 0.001) {
                    let rad = radians(hl.rotation);
                    let s = sin(rad);
                    let c = cos(rad);
                    let a = o.y;
                    let b = o.z;
                    o.y = a * c - b * s;
                    o.z = a * s + b * c;
                }
                if (hl.hueWeight > 0.0 || hl.chromaWeight > 0.0) {
                    var C = sqrt(o.y * o.y + o.z * o.z);
                    var H = atan2(o.z, o.y);
                    if (hl.hueWeight > 0.0) {
                        let targetH_rad = radians(hl.targetHue);
                        var deltaH = targetH_rad - H;
                        let PI = 3.14159265359;
                        deltaH = deltaH - 2.0 * PI * floor((deltaH + PI) / (2.0 * PI));
                        H = H + deltaH * hl.hueWeight;
                    }
                    if (hl.chromaWeight > 0.0) {
                        C = mix(C, hl.targetChroma, hl.chromaWeight);
                    }
                    o.y = C * cos(H);
                    o.z = C * sin(H);
                }
                if (hl.colorSpace > 0u) {
                    cPos = vec3<f32>((o.y + 0.4)/0.8, o.x, (o.z + 0.4)/0.8);
                } else {
                    cPos = oklab_to_rgb(o);
                }
            } else {
                if (hl.colorSpace > 0u) {
                    let o = rgb_to_oklab(cOrig);
                    cPos = vec3<f32>((o.y + 0.4)/0.8, o.x, (o.z + 0.4)/0.8);
                } else {
                    cPos = cOrig;
                }
            }
            
            let N = normalize(hl.cameraNormal);
            let distToPlane = dot(cPos - hl.cloudCenter, N);
            let projPos = cPos - distToPlane * N;
            cPos = mix(projPos, cPos, hl.flattenWeight);

            if (hl.colorSpace > 0u) {
                let a = cPos.x * 0.8 - 0.4;
                let L = cPos.y;
                let b_val = cPos.z * 0.8 - 0.4;
                return oklab_to_rgb(vec3<f32>(L, a, b_val));
            } else {
                return clamp(cPos, vec3<f32>(0.0), vec3<f32>(1.0));
            }
        }

        @fragment
        fn fs(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
            let pixel = textureSample(imgTex, imgSamp, uv);
            let finalColor = vec4<f32>(getProcessedColor(pixel.rgb), pixel.a);

            if (hl.enabled == 0u) { return finalColor; }

            let dist = color_dist(finalColor.rgb, hl.hoveredColor, hl.colorSpace);

            if (dist < hl.threshold) {
                return finalColor;
            }

            // Not highlighted. Check neighbors for outline
            let texDim = vec2<f32>(textureDimensions(imgTex));
            let invDim = 1.0 / texDim;

            let cRight = getProcessedColor(textureSampleLevel(imgTex, imgSamp, uv + vec2(invDim.x, 0.0), 0.0).rgb);
            let cLeft  = getProcessedColor(textureSampleLevel(imgTex, imgSamp, uv + vec2(-invDim.x, 0.0), 0.0).rgb);
            let cUp    = getProcessedColor(textureSampleLevel(imgTex, imgSamp, uv + vec2(0.0, invDim.y), 0.0).rgb);
            let cDown  = getProcessedColor(textureSampleLevel(imgTex, imgSamp, uv + vec2(0.0, -invDim.y), 0.0).rgb);

            let dRight = color_dist(cRight, hl.hoveredColor, hl.colorSpace);
            let dLeft  = color_dist(cLeft, hl.hoveredColor, hl.colorSpace);
            let dUp    = color_dist(cUp, hl.hoveredColor, hl.colorSpace);
            let dDown  = color_dist(cDown, hl.hoveredColor, hl.colorSpace);

            if (dRight < hl.threshold || dLeft < hl.threshold || 
                dUp < hl.threshold || dDown < hl.threshold) {
                return vec4(1.0, 1.0, 1.0, 1.0); // Bright white outline
            }

            return vec4(mix(finalColor.rgb, vec3(0.25), 0.45), pixel.a);
        }
    `;

	// ── Checkerboard (CPU, baked once) ────────────────────────────────────────
	function makeCheckerData(w, h, sq = 32) {
		const data = new Uint8Array(w * h * 4);
		for (let y = 0; y < h; y++)
			for (let x = 0; x < w; x++) {
				const v = (Math.floor(x / sq) + Math.floor(y / sq)) % 2 === 0 ? 240 : 180;
				const i = (y * w + x) * 4;
				data[i] = data[i + 1] = data[i + 2] = v;
				data[i + 3] = 255;
			}
		return data;
	}

	// ── Create / replace the 3-D texture for the current lutSize ─────────────
	function createColorTex3D(lut) {
		if (colorTex3D) colorTex3D.destroy();
		colorTex3D = device.createTexture({
			size: [lut, lut, lut],
			format: 'rgba8unorm',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
			dimension: '3d'
		});
		// Zero-fill so it's valid before any image is loaded
		device.queue.writeTexture(
			{ texture: colorTex3D },
			new Uint8Array(lut * lut * lut * 4),
			{ bytesPerRow: lut * 4, rowsPerImage: lut },
			[lut, lut, lut]
		);
	}

	// ── Build render and compute pipelines ONCE ───────────────────────────────
	function buildPipelines() {
		const vertModule = device.createShaderModule({ code: vertWGSL });
		const fragModule = device.createShaderModule({ code: fragWGSL });
		renderPipeline = device.createRenderPipeline({
			layout: 'auto',
			vertex: { module: vertModule, entryPoint: 'main' },
			fragment: {
				module: fragModule,
				entryPoint: 'main',
				targets: [{ format: canvasFormat }, { format: 'rgba8unorm' }]
			},
			primitive: { topology: 'triangle-list' }
		});

		computePipeline = device.createComputePipeline({
			layout: 'auto',
			compute: {
				module: device.createShaderModule({ code: computeWGSL }),
				entryPoint: 'main'
			}
		});
	}

	function buildBindGroups() {
		// Group 0: checker texture + sampler (static, no TypeGPU uniform here)
		// Group 1: 3D LUT + camera uniform — cameraUniform is a TypeGPU TgpuUniform;
		//          root.unwrap() extracts the underlying GPUBuffer for the raw bind group.
		renderBindGroup = [
			device.createBindGroup({
				layout: renderPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: checkerTexture.createView() },
					{ binding: 1, resource: sampler }
				]
			}),
			device.createBindGroup({
				layout: renderPipeline.getBindGroupLayout(1),
				entries: [
					{ binding: 0, resource: colorTex3D.createView({ dimension: '3d' }) },
					// .buffer extracts the TgpuBuffer from the TgpuUniform shorthand;
					// root.unwrap() then converts it to the raw GPUBuffer.
					{ binding: 1, resource: { buffer: root.unwrap(cameraUniform.buffer) } }
				]
			})
		];
	}

	// ── WebGPU init ───────────────────────────────────────────────────────────
	async function init() {
		if (!navigator.gpu) {
			webgpuSupported = false;
			webgpuError = 'WebGPU is not supported. Please use Chrome 113+ or Edge 113+.';
			return;
		}

		// tgpu.init() requests an adapter + device internally and returns a TgpuRoot.
		// root.device exposes the raw GPUDevice for operations not yet abstracted by TypeGPU.
		try {
			root = await tgpu.init();
		} catch (e) {
			webgpuSupported = false;
			webgpuError = String(e);
			return;
		}
		device = root.device;

		const canvas = document.getElementById('main-canvas');
		canvas.width = CANVAS_W;
		canvas.height = CANVAS_H;
		canvasCtx = canvas.getContext('webgpu');
		canvasFormat = navigator.gpu.getPreferredCanvasFormat();
		canvasCtx.configure({ device, format: canvasFormat });

		// Checkerboard texture (raw — we need writeTexture which works on GPUTexture)
		checkerTexture = device.createTexture({
			size: [CANVAS_W, CANVAS_H],
			format: 'rgba8unorm',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
		});
		device.queue.writeTexture(
			{ texture: checkerTexture },
			makeCheckerData(CANVAS_W, CANVAS_H),
			{ bytesPerRow: CANVAS_W * 4 },
			[CANVAS_W, CANVAS_H]
		);

		// TypeGPU typed uniform for camera — .write() uses the CameraStruct schema,
		// so field names are checked at compile time instead of raw float indices.
		cameraUniform = root.createUniform(CameraStruct);

		// TypeGPU typed uniform for image-preview highlight parameters.
		hlUniform = root.createUniform(HighlightStruct);

		sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

		// MRT: color-pick render target (raw — needs RENDER_ATTACHMENT)
		colorPickTexture = device.createTexture({
			size: [CANVAS_W, CANVAS_H],
			format: 'rgba8unorm',
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
		});

		// Staging buffer for 1-pixel readback (256-byte aligned row)
		colorPickStagingBuf = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});

		buildPipelines();
		createColorTex3D(lutSize);
		buildBindGroups();

		requestAnimationFrame(renderLoop);
		loadPreset(presets[0]); // Load Bee image by default
	}

	// ── Build weight volume from imageData at the given lut resolution ────────
	async function buildVolumeFromImage(imageData, lut, space) {
		const { width: iw, height: ih, data: pixels } = imageData;

		const pixelBuf = device.createBuffer({
			size: iw * ih * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(pixelBuf, 0, pixels.buffer);

		const voxelCount = lut ** 3;
		const weightBuf = device.createBuffer({
			size: voxelCount * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		});

		// Settings struct: imgSize (vec2<u32>), lutSize (u32), colorSpace (u32), flattenWeight (f32) ...
		const computeSettingsBuf = device.createBuffer({
			size: 80,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});
		const csData = new ArrayBuffer(80);
		const csU32 = new Uint32Array(csData);
		const csF32 = new Float32Array(csData);
		csU32[0] = iw;
		csU32[1] = ih;
		csU32[2] = lut;
		csU32[3] = space === 'oklab' ? 1 : 0;
		let effectiveFlatten = enableFlattening ? flattenWeight : 1.0;
		csF32[4] = effectiveFlatten;
		csF32[5] = contrast;
		csF32[6] = saturation;
		csF32[7] = cloudCenterOKLab[1]; // avgL

		let center = space === 'oklab' ? cloudCenterOKLab : cloudCenterRGB;
		csF32[8] = center[0];
		csF32[9] = center[1];
		csF32[10] = center[2];
		csF32[11] = rotation;

		let nx = -Math.sin(yaw) * Math.cos(pitch);
		let ny = -Math.sin(pitch);
		let nz = Math.cos(yaw) * Math.cos(pitch);
		csF32[12] = nx;
		csF32[13] = ny;
		csF32[14] = nz;

		csF32[16] = targetHue;
		csF32[17] = enableHuePull ? hueWeight : 0.0;
		csF32[18] = targetChroma;
		csF32[19] = enableChromaPull ? chromaWeight : 0.0;

		device.queue.writeBuffer(computeSettingsBuf, 0, csData);

		const computeBG = device.createBindGroup({
			layout: computePipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: pixelBuf } },
				{ binding: 1, resource: { buffer: weightBuf } },
				{ binding: 2, resource: { buffer: computeSettingsBuf } }
			]
		});

		const enc = device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(computePipeline);
		pass.setBindGroup(0, computeBG);
		pass.dispatchWorkgroups(Math.ceil(iw / WORKGROUP_SIZE), Math.ceil(ih / WORKGROUP_SIZE));
		pass.end();
		device.queue.submit([enc.finish()]);

		// Readback
		const readBuf = device.createBuffer({
			size: voxelCount * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc2 = device.createCommandEncoder();
		enc2.copyBufferToBuffer(weightBuf, 0, readBuf, 0, voxelCount * 4);
		device.queue.submit([enc2.finish()]);

		await readBuf.mapAsync(GPUMapMode.READ);
		const rawWeights = new Uint32Array(readBuf.getMappedRange().slice());
		readBuf.unmap();

		// Clean up temporary buffers to prevent memory leak
		pixelBuf.destroy();
		weightBuf.destroy();
		computeSettingsBuf.destroy();
		readBuf.destroy();

		// Find max raw weight
		let maxW = 1;
		for (let i = 0; i < rawWeights.length; i++) if (rawWeights[i] > maxW) maxW = rawWeights[i];

		// Build RGBA8 volume.
		// The alpha stored is the raw weight normalised to [0,1] (linear, no sqrt).
		// The opacity window (weightMin/weightMax) is applied in the shader against
		// these raw-weight fractions, so "max" in the slider truly means "the most
		// frequent colour in the image".
		const volData = new Uint8Array(voxelCount * 4);
		for (let bi = 0; bi < lut; bi++) {
			for (let gi = 0; gi < lut; gi++) {
				for (let ri = 0; ri < lut; ri++) {
					const voxel = ri + gi * lut + bi * lut * lut;
					const w = rawWeights[voxel];
					const off = voxel * 4;
					if (space === 'oklab') {
						let a = (ri / (lut - 1)) * 0.8 - 0.4;
						let L = gi / (lut - 1);
						let b_val = (bi / (lut - 1)) * 0.8 - 0.4;
						let rgb = js_oklab_to_srgb(L, a, b_val);
						volData[off] = Math.max(0, Math.min(255, Math.round(rgb[0] * 255)));
						volData[off + 1] = Math.max(0, Math.min(255, Math.round(rgb[1] * 255)));
						volData[off + 2] = Math.max(0, Math.min(255, Math.round(rgb[2] * 255)));
					} else {
						volData[off] = Math.round((ri * 255) / (lut - 1));
						volData[off + 1] = Math.round((gi * 255) / (lut - 1));
						volData[off + 2] = Math.round((bi * 255) / (lut - 1));
					}
					volData[off + 3] = w === 0 ? 0 : Math.round((w / maxW) * 255);
				}
			}
		}

		// Auto-set weightMax to 1.0 (= the actual maximum in the table) so the
		// cloud is fully visible by default.  weightMin stays at 0.
		sliderWeightMin = 0.0;
		sliderWeightMax = 0.3;

		device.queue.writeTexture(
			{ texture: colorTex3D },
			volData,
			{ bytesPerRow: lut * 4, rowsPerImage: lut },
			[lut, lut, lut]
		);

		buildBindGroups();
	}

	// ── Apply the LUT slider: rebuild texture + volume ────────────────────────
	async function applyLutSize() {
		if (!device || isBuilding) return;
		isBuilding = true;
		lutSize = pendingLutSize;
		createColorTex3D(lutSize);
		buildBindGroups();
		if (lastImageData) {
			await buildVolumeFromImage(lastImageData, lutSize, colorSpace);
			lastBuiltYaw = yaw;
			lastBuiltPitch = pitch;
			lastBuiltFlatten = flattenWeight;
			lastBuiltEnableFlattening = enableFlattening;
			lastBuiltContrast = contrast;
			lastBuiltSaturation = saturation;
			lastBuiltRotation = rotation;
			lastBuiltTargetHue = targetHue;
			lastBuiltHueWeight = hueWeight;
			lastBuiltEnableHuePull = enableHuePull;
			lastBuiltTargetChroma = targetChroma;
			lastBuiltChromaWeight = chromaWeight;
			lastBuiltEnableChromaPull = enableChromaPull;
		}
		isBuilding = false;
		isDirty = true;
	}

	// ── Render loop ───────────────────────────────────────────────────────────
	function renderLoop() {
		if (!device) return;

		if (!isDirty) {
			animationId = requestAnimationFrame(renderLoop);
			return;
		}
		isDirty = false;

		const hc = hoveredColor || { r: 0, g: 0, b: 0 };

		// TypeGPU typed write — field names match the CameraStruct schema,
		// eliminating error-prone float array index arithmetic.
		cameraUniform.write({
			yaw,
			pitch,
			radius,
			aspect: CANVAS_W / CANVAS_H,
			weightMin,
			weightMax,
			hoverEnabled: hoveredColor ? 1.0 : 0.0,
			hoverThreshold: colorThreshold,
			hoveredColor: { x: hc.r, y: hc.g, z: hc.b },
			colorSpace: colorSpace === 'oklab' ? 1.0 : 0.0,
			lutSize,
			steps: renderSteps,
			_pad1: 0,
			_pad2: 0
		});

		// Update image-preview highlight uniforms via typed write
		if (hlUniform && imageLoaded) {
			const effectiveFlatten = enableFlattening ? flattenWeight : 1.0;
			const center = colorSpace === 'oklab' ? cloudCenterOKLab : cloudCenterRGB;
			const nx = -Math.sin(yaw) * Math.cos(pitch);
			const ny = -Math.sin(pitch);
			const nz = Math.cos(yaw) * Math.cos(pitch);

			// TypeGPU typed write — matches HighlightStruct schema exactly.
			// vec3f fields accept { x, y, z } objects.
			hlUniform.write({
				hoveredColor: { x: hc.r, y: hc.g, z: hc.b },
				threshold: colorThreshold,
				enabled: hoveredColor ? 1 : 0,
				colorSpace: colorSpace === 'oklab' ? 1 : 0,
				flattenWeight: effectiveFlatten,
				contrast,
				cloudCenter: { x: center[0], y: center[1], z: center[2] },
				saturation,
				cameraNormal: { x: nx, y: ny, z: nz },
				avgL: cloudCenterOKLab[1],
				rotation,
				targetHue,
				hueWeight: enableHuePull ? hueWeight : 0.0,
				targetChroma,
				chromaWeight: enableChromaPull ? chromaWeight : 0.0,
				_pad3: 0,
				_pad4: 0,
				_pad5: 0
			});
		}

		const enc = device.createCommandEncoder();

		// Main cloud pass (MRT: canvas + pick texture)
		const pass = enc.beginRenderPass({
			colorAttachments: [
				{
					view: canvasCtx.getCurrentTexture().createView(),
					clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 },
					loadOp: 'clear',
					storeOp: 'store'
				},
				{
					view: colorPickTexture.createView(),
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});
		pass.setPipeline(renderPipeline);
		pass.setBindGroup(0, renderBindGroup[0]);
		pass.setBindGroup(1, renderBindGroup[1]);
		pass.draw(6);
		pass.end();

		// Image preview pass
		if (imgPreviewCtx && imgPreviewPipeline && imgPreviewBindGroup) {
			const pass2 = enc.beginRenderPass({
				colorAttachments: [
					{
						view: imgPreviewCtx.getCurrentTexture().createView(),
						clearValue: { r: 0.07, g: 0.07, b: 0.07, a: 1 },
						loadOp: 'clear',
						storeOp: 'store'
					}
				]
			});
			pass2.setPipeline(imgPreviewPipeline);
			pass2.setBindGroup(0, imgPreviewBindGroup);
			pass2.draw(6);
			pass2.end();
		}

		device.queue.submit([enc.finish()]);

		animationId = requestAnimationFrame(renderLoop);
	}

	// ── Image processing ──────────────────────────────────────────────────────
	function processImage(img, name) {
		imageName = name;
		// Compute preview dimensions: max 400px wide, 250px tall
		let pw = img.width,
			ph = img.height;
		if (pw > 400) {
			ph = (ph * 400) / pw;
			pw = 400;
		}
		if (ph > 250) {
			pw = (pw * 250) / ph;
			ph = 250;
		}
		previewW = Math.round(pw);
		previewH = Math.round(ph);

		// Rescale to preview size – this is the image "as rendered on the page"
		const offscreen = new OffscreenCanvas(previewW, previewH);
		const ctx2d = offscreen.getContext('2d');
		ctx2d.drawImage(img, 0, 0, previewW, previewH);
		lastImageData = ctx2d.getImageData(0, 0, previewW, previewH);

		computeCloudCenters(lastImageData);
		setupImagePreview(lastImageData);
		buildVolumeFromImage(lastImageData, lutSize, colorSpace).then(() => {
			imageLoaded = true;
			lastBuiltYaw = yaw;
			lastBuiltPitch = pitch;
			lastBuiltFlatten = flattenWeight;
			lastBuiltEnableFlattening = enableFlattening;
			lastBuiltContrast = contrast;
			lastBuiltSaturation = saturation;
			lastBuiltRotation = rotation;
			lastBuiltTargetHue = targetHue;
			lastBuiltHueWeight = hueWeight;
			lastBuiltEnableHuePull = enableHuePull;
			lastBuiltTargetChroma = targetChroma;
			lastBuiltChromaWeight = chromaWeight;
			lastBuiltEnableChromaPull = enableChromaPull;
			isDirty = true;
		});
	}

	function computeCloudCenters(imageData) {
		const data = imageData.data;
		let sumR = 0,
			sumG = 0,
			sumB = 0;
		let sumL = 0,
			sumA = 0,
			sumB_ok = 0;
		let count = 0;
		for (let i = 0; i < data.length; i += 4) {
			if (data[i + 3] < 10) continue;
			let r = data[i] / 255,
				g = data[i + 1] / 255,
				b = data[i + 2] / 255;
			sumR += r;
			sumG += g;
			sumB += b;

			let lin = js_srgb_to_linear(r, g, b);
			let lr = lin[0],
				lg = lin[1],
				lb = lin[2];

			let l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
			let m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
			let s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

			let l_ = Math.cbrt(Math.max(l, 0.0));
			let m_ = Math.cbrt(Math.max(m, 0.0));
			let s_ = Math.cbrt(Math.max(s, 0.0));

			let L = 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_;
			let a = 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_;
			let b_val = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_;

			let nx = (a + 0.4) / 0.8;
			let ny = L;
			let nz = (b_val + 0.4) / 0.8;

			sumA += nx;
			sumL += ny;
			sumB_ok += nz;
			count++;
		}
		if (count > 0) {
			cloudCenterRGB = [sumR / count, sumG / count, sumB / count];
			cloudCenterOKLab = [sumA / count, sumL / count, sumB_ok / count];
		} else {
			cloudCenterRGB = [0.5, 0.5, 0.5];
			cloudCenterOKLab = [0.5, 0.5, 0.5];
		}
	}

	function js_srgb_to_linear(r, g, b) {
		let lr = r <= 0.04045 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
		let lg = g <= 0.04045 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
		let lb = b <= 0.04045 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);
		return [lr, lg, lb];
	}

	// ── Image upload & Presets ────────────────────────────────────────────────
	function handleImageUpload(e) {
		const file = e.target.files[0];
		if (!file) return;
		const url = URL.createObjectURL(file);
		const img = new Image();
		img.onload = () => {
			processImage(img, file.name);
			URL.revokeObjectURL(url);
		};
		img.src = url;
	}

	function loadPreset(preset) {
		const img = new Image();
		img.onload = () => {
			processImage(img, preset.name);
		};
		img.src = preset.src;
	}

	// ── Image preview setup ──────────────────────────────────────────────────
	function setupImagePreview(imageData) {
		const { width, height } = imageData;

		// Configure canvas (dimensions already match the rescaled image)
		imgPreviewCanvas.width = width;
		imgPreviewCanvas.height = height;
		imgPreviewCtx = imgPreviewCanvas.getContext('webgpu');
		imgPreviewFormat = navigator.gpu.getPreferredCanvasFormat();
		imgPreviewCtx.configure({ device, format: imgPreviewFormat });

		// Upload rescaled image to 2D GPU texture
		if (imgTexture) imgTexture.destroy();
		imgTexture = device.createTexture({
			size: [width, height],
			format: 'rgba8unorm',
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
		});
		device.queue.writeTexture({ texture: imgTexture }, imageData.data, { bytesPerRow: width * 4 }, [
			width,
			height
		]);

		// Build image preview pipeline only once
		if (!imgPreviewPipeline) {
			const shaderModule = device.createShaderModule({
				code: imgPreviewFragWGSL
			});
			imgPreviewPipeline = device.createRenderPipeline({
				layout: 'auto',
				vertex: { module: shaderModule, entryPoint: 'vs' },
				fragment: {
					module: shaderModule,
					entryPoint: 'fs',
					targets: [{ format: imgPreviewFormat }]
				},
				primitive: { topology: 'triangle-list' }
			});
		}

		imgPreviewBindGroup = device.createBindGroup({
			layout: imgPreviewPipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: imgTexture.createView() },
				{ binding: 1, resource: sampler },
				// .buffer extracts the TgpuBuffer from the TgpuUniform shorthand;
				// root.unwrap() then converts it to the raw GPUBuffer.
				{ binding: 2, resource: { buffer: root.unwrap(hlUniform.buffer) } }
			]
		});
	}

	// ── Hover readback ───────────────────────────────────────────────────────
	async function readColorAtPixel(px, py) {
		if (isReadingBack || !device || !colorPickTexture) return;
		px = Math.max(0, Math.min(CANVAS_W - 1, Math.floor(px)));
		py = Math.max(0, Math.min(CANVAS_H - 1, Math.floor(py)));

		isReadingBack = true;
		const enc = device.createCommandEncoder();
		enc.copyTextureToBuffer(
			{ texture: colorPickTexture, origin: [px, py, 0] },
			{ buffer: colorPickStagingBuf, bytesPerRow: 256 },
			[1, 1, 1]
		);
		device.queue.submit([enc.finish()]);

		try {
			await colorPickStagingBuf.mapAsync(GPUMapMode.READ);
			const data = new Uint8Array(colorPickStagingBuf.getMappedRange(0, 4));
			const r = data[0],
				g = data[1],
				b = data[2],
				a = data[3];
			colorPickStagingBuf.unmap();
			if (a > 0) {
				const nr = r / 255,
					ng = g / 255,
					nb = b / 255;
				if (
					!hoveredColor ||
					hoveredColor.r !== nr ||
					hoveredColor.g !== ng ||
					hoveredColor.b !== nb
				) {
					hoveredColor = { r: nr, g: ng, b: nb };
				}
			} else {
				if (hoveredColor !== null) {
					hoveredColor = null;
				}
			}
		} catch {
			hoveredColor = null;
		}
		isReadingBack = false;
	}

	// ── Camera interaction ────────────────────────────────────────────────────
	function onMouseDown(e) {
		isDragging = true;
		lastX = e.clientX;
		lastY = e.clientY;
	}
	function onMouseMove(e) {
		if (isDragging) {
			yaw -= (e.clientX - lastX) * 0.005;
			pitch = Math.max(-1.4, Math.min(1.4, pitch - (e.clientY - lastY) * 0.005));
			lastX = e.clientX;
			lastY = e.clientY;
		} else if (imageLoaded) {
			const canvas = document.getElementById('main-canvas');
			const rect = canvas.getBoundingClientRect();
			const x = (e.clientX - rect.left) * (CANVAS_W / rect.width);
			const y = (e.clientY - rect.top) * (CANVAS_H / rect.height);
			readColorAtPixel(x, y);
		}
	}
	function onMouseUp() {
		isDragging = false;
	}
	function onMouseLeave() {
		isDragging = false;
		hoveredColor = null;
	}
	function onWheel(e) {
		e.preventDefault();
		radius = Math.max(0.5, Math.min(6.0, radius + e.deltaY * 0.003));
	}

	// ── Lifecycle ─────────────────────────────────────────────────────────────
	onMount(() =>
		init().catch((err) => {
			webgpuSupported = false;
			webgpuError = String(err);
		})
	);
	onDestroy(() => {
		if (animationId) cancelAnimationFrame(animationId);
	});
</script>

<svelte:head><title>Image Colour Cloud</title></svelte:head>

<div class="wrapper">
	{#if !webgpuSupported}
		<div class="error">
			<h2>WebGPU not available</h2>
			<p>{webgpuError}</p>
			<p>Please use Chrome 113+, Edge 113+, or another WebGPU-capable browser.</p>
		</div>
	{:else}
		<header class="main-header">
			<h1>Image Colour Cloud</h1>
			<p class="subtitle">Load an image to visualise its colour distribution as a 3-D cloud</p>
		</header>

		<div class="content-row">
			<div class="canvas-col">
				<div
					class="canvas-wrap"
					onmousedown={onMouseDown}
					onmousemove={onMouseMove}
					onmouseup={onMouseUp}
					onmouseleave={onMouseLeave}
					onwheel={onWheel}
				>
					<canvas id="main-canvas"></canvas>
					{#if !imageLoaded}
						<div class="overlay-hint">
							<span>Checkerboard background · empty colour space</span>
							<span>Load an image below to fill the cloud</span>
						</div>
					{:else}
						<div class="overlay-hint loaded">
							<span>🖱 Drag to orbit · Scroll to zoom · Hover to pick colour</span>
						</div>
					{/if}
				</div>

				<div class="legend">
					{#if colorSpace === 'rgb'}
						<div class="legend-item">
							<span class="dot" style="background:#ff5555"></span>Red axis
						</div>
						<div class="legend-item">
							<span class="dot" style="background:#55ff55"></span>Green axis
						</div>
						<div class="legend-item">
							<span class="dot" style="background:#5555ff"></span>Blue axis
						</div>
					{:else}
						<div class="legend-item">
							<span class="dot" style="background:#ff55ff"></span>a (Green-Red) axis
						</div>
						<div class="legend-item">
							<span class="dot" style="background:#fff"></span>L (Lightness) axis
						</div>
						<div class="legend-item">
							<span class="dot" style="background:#ffff55"></span>b (Blue-Yellow) axis
						</div>
					{/if}
					<div class="legend-item">
						<span class="dot" style="background:#aaa;opacity:0.4"></span>Alpha = frequency
					</div>
				</div>
			</div>

			<div class="side-col">
				<div
					class="preview-wrap"
					class:visible={imageLoaded}
					style="width:{previewW}px;height:{previewH}px"
				>
					<canvas bind:this={imgPreviewCanvas}></canvas>
					{#if imageLoaded && hoveredColor}
						<div
							class="preview-swatch"
							style="background:rgb({Math.round(hoveredColor.r * 255)},{Math.round(
								hoveredColor.g * 255
							)},{Math.round(hoveredColor.b * 255)})"
						></div>
					{/if}
				</div>

				<div class="controls">
					<label class="file-label">
						<input type="file" accept="image/*" onchange={handleImageUpload} />
						<span class="btn">📂 Load Image</span>
					</label>
					<div class="presets">
						<span class="preset-label">Presets:</span>
						{#each presets as preset}
							<button class="btn preset-btn" onclick={() => loadPreset(preset)}
								>{preset.name}</button
							>
						{/each}
					</div>
					{#if imageName}
						<span class="img-name">{imageName}</span>
					{/if}
				</div>

				<div class="sliders">
					<!-- Weight window -->
					<div class="slider-group-label">Weight window</div>

					<div class="slider-row">
						<span class="slider-label">
							Min weight
							<span class="hint">hides low-frequency voxels</span>
						</span>
						<input
							type="range"
							min="0"
							max="1"
							step="0.001"
							bind:value={sliderWeightMin}
							oninput={() => {
								if (sliderWeightMin > sliderWeightMax) sliderWeightMax = sliderWeightMin;
							}}
						/>
						<span class="slider-val">{weightMin.toFixed(3)}</span>
					</div>

					<div class="slider-row">
						<span class="slider-label">
							Max weight
							<span class="hint">amplifies faint clouds</span>
						</span>
						<input
							type="range"
							min="0"
							max="1"
							step="0.001"
							bind:value={sliderWeightMax}
							oninput={() => {
								if (sliderWeightMax < sliderWeightMin) sliderWeightMin = sliderWeightMax;
							}}
						/>
						<span class="slider-val">{weightMax.toFixed(3)}</span>
					</div>

					<!-- Colour similarity -->
					<div class="slider-group-label" style="margin-top:0.5rem">Colour similarity</div>

					<div class="slider-row">
						<span class="slider-label">
							Threshold
							<span class="hint">Euclidean RGB distance</span>
						</span>
						<input type="range" min="1" max="80" step="1" bind:value={colorThreshold} />
						<span class="slider-val">{colorThreshold}</span>
					</div>

					<!-- Color space -->
					<div class="slider-group-label" style="margin-top:0.5rem">Color Space</div>
					<div class="slider-row">
						<label class="radio-label">
							<input type="radio" bind:group={colorSpace} value="rgb" onchange={applyLutSize} />
							RGB
						</label>
						<label class="radio-label" style="margin-left:1rem">
							<input type="radio" bind:group={colorSpace} value="oklab" onchange={applyLutSize} />
							OKLab
						</label>
					</div>

					<!-- Color Processing -->
					<div class="slider-group-label" style="margin-top:0.5rem">Color Processing</div>

					<div class="slider-row">
						<span class="slider-label">
							Contrast
							<span class="hint">scales OKLab Lightness</span>
						</span>
						<input type="range" min="0" max="3" step="0.01" bind:value={contrast} />
						<span class="slider-val">{contrast.toFixed(2)}</span>
					</div>

					<div class="slider-row">
						<span class="slider-label">
							Saturation
							<span class="hint">scales distance from L axis</span>
						</span>
						<input type="range" min="0" max="3" step="0.01" bind:value={saturation} />
						<span class="slider-val">{saturation.toFixed(2)}</span>
					</div>

					<div class="slider-row">
						<span class="slider-label">
							Rotation
							<span class="hint">hue shift in degrees</span>
						</span>
						<input type="range" min="-180" max="180" step="1" bind:value={rotation} />
						<span class="slider-val">{rotation}°</span>
					</div>

					<div class="slider-group-label" style="margin-top:0.5rem">Target Pulls</div>
					<div class="slider-row">
						<label class="radio-label">
							<input type="checkbox" bind:checked={enableHuePull} />
							Hue Pull
						</label>
					</div>
					<div
						class="slider-row"
						style="opacity: {enableHuePull ? 1 : 0.5}; pointer-events: {enableHuePull
							? 'auto'
							: 'none'}"
					>
						<span class="slider-label"> Target Hue </span>
						<input
							type="range"
							min="-180"
							max="180"
							step="1"
							bind:value={targetHue}
							disabled={!enableHuePull}
						/>
						<span class="slider-val">{targetHue}°</span>
					</div>
					<div
						class="slider-row"
						style="opacity: {enableHuePull ? 1 : 0.5}; pointer-events: {enableHuePull
							? 'auto'
							: 'none'}"
					>
						<span class="slider-label">
							Hue Weight
							<span class="hint">strength to pull colors</span>
						</span>
						<input
							type="range"
							min="0"
							max="1"
							step="0.01"
							bind:value={hueWeight}
							disabled={!enableHuePull}
						/>
						<span class="slider-val">{hueWeight.toFixed(2)}</span>
					</div>

					<div class="slider-row" style="margin-top: 0.25rem">
						<label class="radio-label">
							<input type="checkbox" bind:checked={enableChromaPull} />
							Chroma Pull
						</label>
					</div>
					<div
						class="slider-row"
						style="opacity: {enableChromaPull ? 1 : 0.5}; pointer-events: {enableChromaPull
							? 'auto'
							: 'none'}"
					>
						<span class="slider-label"> Target Chroma </span>
						<input
							type="range"
							min="0"
							max="0.5"
							step="0.01"
							bind:value={targetChroma}
							disabled={!enableChromaPull}
						/>
						<span class="slider-val">{targetChroma.toFixed(2)}</span>
					</div>
					<div
						class="slider-row"
						style="opacity: {enableChromaPull ? 1 : 0.5}; pointer-events: {enableChromaPull
							? 'auto'
							: 'none'}"
					>
						<span class="slider-label">
							Chroma Weight
							<span class="hint">strength to pull colors</span>
						</span>
						<input
							type="range"
							min="0"
							max="1"
							step="0.01"
							bind:value={chromaWeight}
							disabled={!enableChromaPull}
						/>
						<span class="slider-val">{chromaWeight.toFixed(2)}</span>
					</div>

					<div class="slider-group-label" style="margin-top:0.5rem">Flattening</div>
					<div class="slider-row">
						<label class="radio-label">
							<input type="checkbox" bind:checked={enableFlattening} />
							Apply Flattening
						</label>
					</div>
					<div
						class="slider-row"
						style="opacity: {enableFlattening ? 1 : 0.5}; pointer-events: {enableFlattening
							? 'auto'
							: 'none'}"
					>
						<span class="slider-label">
							Flatten weight
							<span class="hint">pull to camera plane</span>
						</span>
						<input
							type="range"
							min="0"
							max="2"
							step="0.01"
							bind:value={flattenWeight}
							disabled={!enableFlattening}
						/>
						<span class="slider-val">{flattenWeight.toFixed(2)}</span>
					</div>

					<!-- LUT resolution -->
					<div class="slider-group-label" style="margin-top:0.5rem">
						LUT resolution
						{#if isBuilding}<span class="building">rebuilding…</span>{/if}
					</div>

					<div class="slider-row">
						<span class="slider-label">
							Grid size
							<span class="hint">applied on release</span>
						</span>
						<input
							type="range"
							min="10"
							max="256"
							step="1"
							bind:value={pendingLutSize}
							onchange={applyLutSize}
						/>
						<span class="slider-val">{pendingLutSize}³</span>
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	:global(body) {
		margin: 0;
		background: #111;
		color: #eee;
		font-family: system-ui, sans-serif;
	}
	.wrapper {
		display: flex;
		flex-direction: column;
		align-items: center;
		min-height: 100vh;
		padding: 1.5rem 1rem 3rem;
		gap: 1rem;
	}
	.main-header {
		text-align: left;
		width: 100%;
		max-width: 1250px;
		margin-bottom: 0.5rem;
	}
	h1 {
		margin: 0 0 0.25rem;
		font-size: 1.7rem;
		font-weight: 700;
		letter-spacing: 0.03em;
	}
	.subtitle {
		margin: 0;
		font-size: 0.85rem;
		color: #888;
	}

	.content-row {
		display: flex;
		align-items: flex-start;
		gap: 1.5rem;
		max-width: 1250px;
		width: 100%;
		justify-content: center;
		flex-wrap: wrap;
	}

	.canvas-col {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		flex-shrink: 0;
	}

	.side-col {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
		min-width: 420px;
		flex: 1;
	}
	.preview-wrap {
		position: relative;
		border-radius: 6px;
		overflow: hidden;
		box-shadow: 0 0 16px #000a;
		background: #181818;
		flex-shrink: 0;
		opacity: 0;
		transition: opacity 0.3s;
	}
	.preview-wrap.visible {
		opacity: 1;
	}
	.preview-wrap canvas {
		display: block;
		width: 100%;
		height: 100%;
	}
	.preview-swatch {
		position: absolute;
		bottom: 4px;
		right: 4px;
		width: 18px;
		height: 18px;
		border-radius: 4px;
		border: 2px solid #fff;
		box-shadow: 0 0 4px #000a;
	}

	.canvas-wrap {
		position: relative;
		cursor: grab;
		border-radius: 8px;
		overflow: hidden;
		box-shadow: 0 0 32px #000a;
	}
	.canvas-wrap:active {
		cursor: grabbing;
	}
	canvas {
		display: block;
	}

	.overlay-hint {
		position: absolute;
		bottom: 0.75rem;
		left: 50%;
		transform: translateX(-50%);
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.2rem;
		font-size: 0.75rem;
		color: #ffffffaa;
		pointer-events: none;
		text-shadow: 0 1px 4px #000;
		white-space: nowrap;
	}
	.overlay-hint.loaded {
		flex-direction: row;
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 1rem;
		flex-wrap: wrap;
		justify-content: flex-start;
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

	.sliders {
		display: flex;
		flex-direction: column;
		gap: 0.45rem;
		background: #1a1a1a;
		border: 1px solid #333;
		border-radius: 8px;
		padding: 0.85rem 1.25rem;
		min-width: 420px;
	}
	.slider-group-label {
		font-size: 0.72rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: #666;
	}
	.building {
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
		color: #f0a040;
		margin-left: 0.5rem;
	}
	.slider-row {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}
	.slider-label {
		font-size: 0.8rem;
		color: #bbb;
		width: 8.5rem;
		flex-shrink: 0;
		display: flex;
		flex-direction: column;
		gap: 0.1rem;
	}
	.hint {
		font-size: 0.68rem;
		color: #555;
		font-style: italic;
	}
	.radio-label {
		font-size: 0.8rem;
		color: #bbb;
		display: flex;
		align-items: center;
		gap: 0.4rem;
		cursor: pointer;
	}
	.slider-row input[type='range'] {
		flex: 1;
		accent-color: #2a6be8;
	}
	.slider-val {
		font-size: 0.78rem;
		color: #ccc;
		width: 3.5rem;
		text-align: right;
		font-variant-numeric: tabular-nums;
	}

	.legend {
		display: flex;
		gap: 1.2rem;
		flex-wrap: wrap;
		justify-content: center;
		font-size: 0.78rem;
		color: #aaa;
	}
	.legend-item {
		display: flex;
		align-items: center;
		gap: 0.35rem;
	}
	.dot {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
	}

	.error {
		max-width: 480px;
		background: #2a1010;
		border: 1px solid #7a2020;
		border-radius: 8px;
		padding: 1.5rem 2rem;
		text-align: center;
	}
	.error h2 {
		margin-top: 0;
		color: #ff8080;
	}
</style>
