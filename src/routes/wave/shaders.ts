
import tgpu, { d, std, type StorageFlag, type TgpuBuffer, type TgpuBufferReadonly, type TgpuRoot } from 'typegpu';
import { once, onceBindGroup } from "$lib/gpu/gpu_utils";
import { quadVertex } from "../color/shaders";

const array2dSize = d.struct({
    width: d.u32,
    height: d.u32
});
const waveFunctionConstants = d.struct({
    speed: d.f32,
    deltat: d.f32,
    time: d.f32
});
const scalarArray = d.arrayOf(d.f32);

export const emitterData = d.struct({
    frequency: d.f32,
    phase: d.f32,
    amplitude: d.f32
});
const emitterArray = d.arrayOf(emitterData);

export const waveFunctionLayout = tgpu.bindGroupLayout({
    size: { uniform: array2dSize },
    constants: { uniform: waveFunctionConstants },
    refraction: { storage: scalarArray, access: 'readonly' },
    damping: { storage: scalarArray, access: 'readonly' },
    emitters: { storage: emitterArray, access: 'readonly' },
    prevX: { storage: scalarArray, access: 'readonly' },
    currX: { storage: scalarArray, access: 'readonly' },
    nextX: { storage: scalarArray, access: 'mutable' },
});

export const arrayIndex = (x: number, y: number, width: number) => {
    'use gpu';
    return x + y * width;
}

export const applyWaveFunction = (x: number, y: number) => {
    'use gpu';

    const width = waveFunctionLayout.$.size.width;
    const height = waveFunctionLayout.$.size.height;
    const speed = waveFunctionLayout.$.constants.speed;
    const deltat = waveFunctionLayout.$.constants.deltat;

    const idx = arrayIndex(x, y, width);
    const up = arrayIndex(x, y - 1, width);
    const down = arrayIndex(x, y + 1, width);
    const left = arrayIndex(x - 1, y, width);
    const right = arrayIndex(x + 1, y, width);

    const curr = waveFunctionLayout.$.currX;
    const prev = waveFunctionLayout.$.prevX;

    const effectiveSpeed = speed / waveFunctionLayout.$.refraction[idx];
    const c2dt2 = effectiveSpeed * effectiveSpeed * deltat * deltat;
    const laplacian = curr[left] + curr[right] + curr[up] + curr[down] - 4 * curr[idx];

    // Sponge boundary: a per-cell damping coefficient sigma turns the wave
    // equation into d2u/dt2 + sigma*du/dt = c^2*lap(u), discretized with a
    // semi-implicit central difference so sigma=0 reduces to the plain scheme.
    const sigma = waveFunctionLayout.$.damping[idx];
    const halfSigmaDt = sigma * deltat * 0.5;
    let updated = (2 * curr[idx] - prev[idx] * (1 - halfSigmaDt) + c2dt2 * laplacian) / (1 + halfSigmaDt);

    // Hard source: emitter cells are clamped to their own oscillation instead
    // of evolving under the wave equation.
    const amplitude = waveFunctionLayout.$.emitters[idx].amplitude;
    if (amplitude > 0) {
        const frequency = waveFunctionLayout.$.emitters[idx].frequency;
        const phase = waveFunctionLayout.$.emitters[idx].phase;
        const time = waveFunctionLayout.$.constants.time;
        updated = amplitude * std.sin(6.28318530718 * frequency * time + phase);
    }

    waveFunctionLayout.$.nextX[idx] = updated;
}

export const waveFunctionRunner = tgpu.computeFn({
    in: {
        globalIndex: d.builtin.globalInvocationId
    },
    workgroupSize: [8, 8],
})((val) => {
    const x = val.globalIndex.x;
    const y = val.globalIndex.y;

    if (x >= waveFunctionLayout.$.size.width) {
        return;
    }
    if (y >= waveFunctionLayout.$.size.height) {
        return;
    }
    applyWaveFunction(x, y);
});

type GPUArray = TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag;
export type EmitterArray = TgpuBuffer<d.WgslArray<typeof emitterData>> & StorageFlag;
export const gpuWaveFunction = (root: TgpuRoot, prevX: GPUArray, currX: GPUArray, nextX: GPUArray, refraction: GPUArray, damping: GPUArray, emitters: EmitterArray, options: {
    width: number,
    height: number,
    speed: number,
    deltat: number,
    time: number
}) => {
    const {
        pipeline,
        sizeUniform,
        constantsUniform,
    } = once(gpuWaveFunction, () => {
        const pipeline = root.createComputePipeline({
            compute: waveFunctionRunner
        });
        const sizeUniform = root.createBuffer(array2dSize).$usage('uniform');
        const constantsUniform = root.createBuffer(waveFunctionConstants).$usage('uniform');
        return {
            pipeline,
            sizeUniform,
            constantsUniform
        };
    });

    const bindGroup = onceBindGroup(root, waveFunctionLayout, {
        size: sizeUniform,
        constants: constantsUniform,
        refraction,
        damping,
        emitters,
        prevX,
        currX,
        nextX
    });

    sizeUniform.write({
        width: options.width,
        height: options.height
    });

    constantsUniform.write({
        speed: options.speed,
        deltat: options.deltat,
        time: options.time
    });

    pipeline.with(bindGroup).dispatchWorkgroups(Math.ceil(options.width / 8), Math.ceil(options.height / 8));
}

export const waveRenderLayout = tgpu.bindGroupLayout({
    size: { uniform: array2dSize },
    field: { storage: scalarArray, access: 'readonly' },
    refraction: { storage: scalarArray, access: 'readonly' },
    emitters: { storage: emitterArray, access: 'readonly' },
});

export const waveFragmentOutput = d.struct({
    color: d.vec4f,
});

export const waveFragment = ({ uv }: { uv: d.v2f }): d.Infer<typeof waveFragmentOutput> => {
    'use gpu';
    const width = waveRenderLayout.$.size.width;
    const height = waveRenderLayout.$.size.height;
    const x = std.min(d.u32(uv.x * d.f32(width)), width - 1);
    const y = std.min(d.u32(uv.y * d.f32(height)), height - 1);
    const idx = arrayIndex(x, y, width);
    const value = waveRenderLayout.$.field[idx];
    const brightness = std.clamp(0.5 + 0.5 * value, 0.0, 1.0);
    const tint = std.clamp((waveRenderLayout.$.refraction[idx] - 1.0) * 0.4, 0.0, 0.5);
    const base = d.vec3f(brightness * (1.0 - tint), brightness * (1.0 - tint), std.min(brightness + tint, 1.0));
    const emitterMark = std.select(0.0, 0.35, waveRenderLayout.$.emitters[idx].amplitude > 0);
    const color = std.mix(base, d.vec3f(1.0, 0.55, 0.1), emitterMark);
    return {
        color: d.vec4f(color, 1.0)
    };
}

export const renderWave = (root: TgpuRoot, field: GPUArray, refraction: GPUArray, emitters: EmitterArray, outputView: any, options: {
    width: number,
    height: number
}) => {
    const {
        pipeline,
        sizeUniform
    } = once(renderWave, () => {
        const pipeline = root.createRenderPipeline({
            primitive: { topology: 'triangle-list' },
            vertex: quadVertex,
            fragment: waveFragment,
            targets: {
                color: { format: navigator.gpu.getPreferredCanvasFormat() }
            }
        });
        const sizeUniform = root.createBuffer(array2dSize).$usage('uniform');
        return { pipeline, sizeUniform };
    });

    const bindGroup = onceBindGroup(root, waveRenderLayout, {
        size: sizeUniform,
        field,
        refraction,
        emitters
    });

    sizeUniform.write({
        width: options.width,
        height: options.height
    });

    pipeline.with(bindGroup).withColorAttachment({
        color: { view: outputView }
    }).draw(6);
}