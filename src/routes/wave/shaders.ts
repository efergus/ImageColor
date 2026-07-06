
import tgpu, { d, std, type StorageFlag, type TgpuBuffer, type TgpuBufferReadonly, type TgpuRoot } from 'typegpu';
import { once, onceBindGroup } from "$lib/gpu/gpu_utils";

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

    // 
    const updated = 2 * curr[idx] - prev[idx] + speed * speed * deltat * deltat * (
        curr[left] + curr[right] + curr[up] + curr[down] - 4 * prev[idx]
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