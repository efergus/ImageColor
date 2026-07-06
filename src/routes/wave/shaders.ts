
import tgpu, { d, std, type StorageFlag, type TgpuBuffer, type TgpuBufferReadonly, type TgpuRoot } from 'typegpu';
import { once, onceBindGroup } from "$lib/gpu/gpu_utils";
import { quadVertex } from "../hello/shaders";

const array2dSize = d.struct({
    width: d.u32,
    height: d.u32
});
const waveFunctionConstants = d.struct({
    speed: d.f32,
    deltat: d.f32
});
const scalarArray = d.arrayOf(d.f32);

export const waveFunctionLayout = tgpu.bindGroupLayout({
    size: { uniform: array2dSize },
    constants: { uniform: waveFunctionConstants },
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
    const speed = waveFunctionLayout.$.constants.speed;
    const deltat = waveFunctionLayout.$.constants.deltat;

    const idx = arrayIndex(x, y, width);
    const up = arrayIndex(x, y - 1, width);
    const down = arrayIndex(x, y + 1, width);
    const left = arrayIndex(x - 1, y, width);
    const right = arrayIndex(x + 1, y, width);

    const curr = waveFunctionLayout.$.currX;
    const prev = waveFunctionLayout.$.prevX;

    const updated = 2 * curr[idx] - prev[idx] + speed * speed * deltat * deltat * (
        curr[left] + curr[right] + curr[up] + curr[down] - 4 * curr[idx]
    );

    waveFunctionLayout.$.nextX[idx] = updated;
}

export const waveFunctionRunner = tgpu.computeFn({
    in: {
        globalIndex: d.builtin.globalInvocationId
    },
    workgroupSize: [8, 8],
})((val) => {
    const inX = val.globalIndex.x + 1;
    const inY = val.globalIndex.y + 1;

    if (inX >= waveFunctionLayout.$.size.width - 1) {
        return;
    }
    if (inY >= waveFunctionLayout.$.size.height - 1) {
        return;
    }
    applyWaveFunction(inX, inY);
});

type GPUArray = TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag;
export const gpuWaveFunction = (root: TgpuRoot, prevX: GPUArray, currX: GPUArray, nextX: GPUArray, options: {
    width: number,
    height: number,
    speed: number,
    deltat: number
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
        deltat: options.deltat
    });

    pipeline.with(bindGroup).dispatchWorkgroups(Math.ceil(options.width / 8), Math.ceil(options.height / 8));
}

export const waveRenderLayout = tgpu.bindGroupLayout({
    size: { uniform: array2dSize },
    field: { storage: scalarArray, access: 'readonly' },
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
    const value = waveRenderLayout.$.field[arrayIndex(x, y, width)];
    const brightness = std.clamp(0.5 + 0.5 * value, 0.0, 1.0);
    return {
        color: d.vec4f(brightness, brightness, brightness, 1.0)
    };
}

export const renderWave = (root: TgpuRoot, field: GPUArray, outputView: any, options: {
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
        field
    });

    sizeUniform.write({
        width: options.width,
        height: options.height
    });

    pipeline.with(bindGroup).withColorAttachment({
        color: { view: outputView }
    }).draw(6);
}