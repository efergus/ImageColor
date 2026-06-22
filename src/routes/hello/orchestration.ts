import { d, type StorageFlag, type TgpuBuffer, type TgpuFixedSampler, type TgpuQuerySet, type TgpuRenderPipeline, type TgpuRoot, type TgpuTexture } from "typegpu";
import { once, onceBindGroup } from "./gpu_utils";
import { computeOptions, filterBindLayout, filterFragment, filterOptions, quadVertex, textureRenderLayout, weightCalculation, weightCalculationLayout, processWeights, weightTransferLayout, weightTextureFormat, blur, weightProcessingLayout, imageFragment, triangleFragment, cameraBindLayout, cameraUniform } from "./shaders";
import { textureDimensions } from "typegpu/std";

let querySet: TgpuQuerySet<'timestamp'> | null = null;
let querySetNames: Map<string, number> = new Map()

const timestampOptions = (root: TgpuRoot, name: string) => {
    if (querySet === null) {
        querySet = root.createQuerySet('timestamp', 64);
    }

    let index = querySetNames.get(name);
    if (index === undefined) {
        index = querySetNames.size;
        if (index >= 32) {
            console.error(`Too many pipelines! Could not time ${name}`);
        }
        querySetNames.set(name, index);
    }
    return {
        querySet,
        beginningOfPassWriteIndex: index * 2,
        endOfPassWriteIndex: index * 2 + 1
    }
}

export const filterTexture = (root: TgpuRoot, inputTexture: TgpuTexture, inputSampler: TgpuFixedSampler, options: d.Infer<typeof filterOptions>) => {
    const {
        optionsBuffer,
        pipeline
    } = once(filterTexture, () => {
        const pipeline = root.createRenderPipeline({
            primitive: { topology: 'triangle-list' },
            vertex: quadVertex,
            fragment: filterFragment,
            targets: {
                color: { format: 'rgba16float' }
            }
        }).withTimestampWrites(timestampOptions(root, 'filterTexture'));
        const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
        return { optionsBuffer, pipeline }
    })

    const { outputTexture, outputView } = once([filterTexture, options.textureSize.x, options.textureSize.y], () => {
        const outputTexture = root.createTexture({
            size: [options.textureSize.x, options.textureSize.y],
            format: 'rgba16float',
        }).$usage('render', 'sampled', 'storage');
        const outputView = outputTexture.createView('render');
        return { outputTexture, outputView }
    })

    optionsBuffer.write(options);

    const filterBindGroup = onceBindGroup(root, filterBindLayout, {
        texture: inputTexture as any,
        sampler: inputSampler,
        filterOptions: optionsBuffer
    })

    pipeline.with(filterBindGroup).withColorAttachment({
        color: { view: outputView }
    }).draw(6);

    return outputTexture;
}

export const calculateWeights = (root: TgpuRoot, inputTexture: TgpuTexture, options: d.Infer<typeof computeOptions>) => {
    const {
        pipeline,
        optionsBuffer,
    } = once(calculateWeights, () => {
        const pipeline = root.createGuardedComputePipeline(weightCalculation).withTimestampWrites(timestampOptions(root, 'calculateWeights'))
        const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const {
        outputBuffer
    } = once([calculateWeights, options.tableSize], () => {
        const outputBuffer = root
            .createBuffer(d.arrayOf(d.u32, options.tableSize * options.tableSize * options.tableSize * 4))
            .$usage('storage');
        return { outputBuffer }
    })

    optionsBuffer.write(options);
    outputBuffer.clear();

    const bindGroup = onceBindGroup(root, weightCalculationLayout, {
        image: (inputTexture as any),
        weights: outputBuffer,
        options: optionsBuffer
    });

    pipeline.with(bindGroup).dispatchThreads(options.textureSize.x, options.textureSize.y);

    return outputBuffer;
}

export const processWeightTexture = (root: TgpuRoot, weightsBuffer: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag, options: d.Infer<typeof computeOptions>) => {
    const {
        pipeline,
        optionsBuffer,
    } = once(processWeightTexture, () => {
        const pipeline = root.createGuardedComputePipeline(processWeights).withTimestampWrites(timestampOptions(root, 'processWeightTexture'));
        const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const {
        outputTexture
    } = once([processWeightTexture, options.tableSize], () => {
        const outputTexture = root.createTexture({
            size: [options.tableSize, options.tableSize, options.tableSize],
            format: weightTextureFormat,
            dimension: '3d'
        }).$usage('render', 'sampled', 'storage');
        return { outputTexture }
    });

    optionsBuffer.write(options);

    const bindGroup = onceBindGroup(root, weightTransferLayout, {
        options: optionsBuffer,
        weights: weightsBuffer,
        outputTexture: outputTexture
    });

    pipeline.with(bindGroup).dispatchThreads(options.tableSize, options.tableSize, options.tableSize);

    return outputTexture;
}

export const blurWeightTexture = (root: TgpuRoot, inputTexture: TgpuTexture, options: d.Infer<typeof filterOptions>, tableSize: number) => {
    const {
        pipeline,
        optionsBuffer,
    } = once(blurWeightTexture, () => {
        const pipeline = root.createGuardedComputePipeline(blur).withTimestampWrites(timestampOptions(root, 'blurWeightTexture'));
        const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const {
        outputTexture
    } = once([blurWeightTexture, tableSize], () => {
        const outputTexture = root.createTexture({
            size: [tableSize, tableSize, tableSize],
            format: weightTextureFormat,
            dimension: '3d'
        }).$usage('sampled', 'storage');
        return { outputTexture }
    });

    optionsBuffer.write(options);

    const bindGroup = onceBindGroup(root, weightProcessingLayout, {
        options: optionsBuffer,
        inputTexture: (inputTexture as any).createView('sampled'),
        outputTexture: outputTexture as any
    });

    pipeline.with(bindGroup).dispatchThreads(tableSize, tableSize, tableSize);

    return outputTexture;
}

export const renderImage = (root: TgpuRoot, inputTexture: TgpuTexture, inputSampler: TgpuFixedSampler, outputView: any, options: d.Infer<typeof filterOptions>) => {
    const {
        pipeline,
        optionsBuffer
    } = once(renderImage, () => {
        const pipeline = root
            .createRenderPipeline({
                primitive: { topology: 'triangle-list' },
                vertex: quadVertex,
                fragment: imageFragment
            }).withTimestampWrites(timestampOptions(root, 'renderImage'));
        const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const bindGroup = onceBindGroup(root, textureRenderLayout, {
        texture: (inputTexture as any).createView('sampled'),
        sampler: inputSampler,
        options: optionsBuffer
    });

    optionsBuffer.write(options);

    pipeline.with(bindGroup).withColorAttachment({
        view: outputView
    }).draw(6);
}

export const renderColorCloud = (root: TgpuRoot, inputTexture: TgpuTexture, inputSampler: TgpuFixedSampler, outputView: any, pickView: any, options: d.Infer<typeof filterOptions>, camera: d.Infer<typeof cameraUniform>) => {
    const {
        pipeline,
        optionsBuffer,
        cameraBuffer
    } = once(renderColorCloud, () => {
        const pipeline = root
            .createRenderPipeline({
                primitive: { topology: 'triangle-list' },
                vertex: quadVertex,
                fragment: triangleFragment,
                targets: {
                    color: { format: navigator.gpu.getPreferredCanvasFormat() },
                    pick: { format: 'rgba8unorm' }
                }
            }).withTimestampWrites(timestampOptions(root, 'renderColorCloud'));
        const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
        const cameraBuffer = root.createBuffer(cameraUniform).$usage('uniform');
        return { pipeline, optionsBuffer, cameraBuffer };
    });

    const bindGroup = onceBindGroup(root, textureRenderLayout, {
        texture: (inputTexture as any).createView('sampled'),
        sampler: inputSampler,
        options: optionsBuffer
    });

    const cameraBindGroup = onceBindGroup(root, cameraBindLayout, {
        options: optionsBuffer,
        cameraUniform: cameraBuffer,
        weightTexture: (inputTexture as any).createView('sampled'),
        weightSampler: inputSampler
    })

    optionsBuffer.write(options);
    cameraBuffer.write(camera);

    pipeline.with(bindGroup).with(cameraBindGroup)
        .withColorAttachment({
            color: { view: outputView },
            pick: { view: pickView }
        }).draw(6);
}