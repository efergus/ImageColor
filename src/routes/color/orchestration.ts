import {
    d,
    std,
    type RenderFlag,
    type SampledFlag,
    type StorageFlag,
    type TgpuBuffer,
    type TgpuFixedSampler,
    type TgpuQuerySet,
    type TgpuRoot,
    type TgpuTexture
} from 'typegpu';
import { once, onceBindGroup } from '../../lib/gpu/gpu_utils';
import {
    computeOptions,
    filterBindLayout,
    filterFragment,
    filterOptions,
    quadVertex,
    textureRenderLayout,
    weightCalculation,
    weightCalculationLayout,
    processWeights,
    weightTransferLayout,
    weightTextureFormat,
    blur,
    weightProcessingLayout,
    imageFragment,
    triangleFragment,
    cameraBindLayout,
    cameraUniform,
    cloudCompositeFragment,
    cloudCompositeLayout,
    cloudCompositeOptions,
    colorSpaceSlot,
    linearRgbColorSpace,
    srgbColorSpace,
    oklabColorSpace,
    colorSpaceInverseSlot,
    linearRgbColorSpaceInverse,
    srgbColorSpaceInverse,
    oklabColorSpaceInverse,
    hsvColorSpace,
    hsvColorSpaceInverse,
    hslColorSpace,
    hslColorSpaceInverse,
    gridVertexLayout,
    gridBindLayout,
    gridVertex,
    gridFragment,
    gridFaceOptions,
    sphereVertexLayout,
    sphereBindLayout,
    sphereVertex,
    sphereFragment,
    sphereOptions
} from './shaders';
import { ColorSpace } from './color_utils';
import { gridFaces, sphereVertices } from './geometry';

// A render pass output: either a texture to draw into or a canvas context.
type RenderTarget = (TgpuTexture & RenderFlag) | GPUCanvasContext;

// TgpuTexture's default props assume a 2d texture, so 3d textures need their
// dimension spelled out for typegpu to accept 3d views of them.
type Texture3d = TgpuTexture<{
    size: readonly number[];
    format: GPUTextureFormat;
    dimension: '3d';
    sampleCount?: 1 | undefined;
}>;

let querySet: TgpuQuerySet<'timestamp'> | null = null;
const querySetNames: Map<string, number> = new Map();

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
    };
};

const querySetTimings: Map<string, number[]> = new Map();

export const readTimings = async () => {
    if (!querySet) {
        console.warn('Query set not available');
        return;
    }
    if (querySet.available) {
        querySet.resolve();
        const values = await querySet.read();
        for (const name of querySetNames.keys()) {
            const index = querySetNames.get(name)! * 2;

            const start = values[index];
            const end = values[index + 1];
            const time = Number(end - start);
            const past = querySetTimings.get(name) ?? [];
            if (past.length && past[past.length - 1] === time) {
                continue;
            }

            past.push(time);
            querySetTimings.set(name, past);
            console.log(`Pass ${name}: ${time / 1e6} ms`);
        }
    } else {
        console.warn('querySet not available');
    }
};

export const filterTexture = (
    root: TgpuRoot,
    inputTexture: TgpuTexture & SampledFlag,
    inputSampler: TgpuFixedSampler,
    options: d.Infer<typeof filterOptions>
) => {
    const format = 'rgba16float';

    const { optionsBuffer, pipeline } = once(filterTexture, () => {
        const pipeline = root
            .createRenderPipeline({
                primitive: { topology: 'triangle-list' },
                vertex: quadVertex,
                fragment: filterFragment,
                targets: {
                    color: { format }
                }
            })
            .withTimestampWrites(timestampOptions(root, 'filterTexture'));
        const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
        return { optionsBuffer, pipeline };
    });

    const { outputTexture, outputView } = once(
        [filterTexture, options.textureSize.x, options.textureSize.y],
        () => {
            const outputTexture = root
                .createTexture({
                    size: [options.textureSize.x, options.textureSize.y],
                    format
                })
                .$usage('render', 'sampled', 'storage');
            const outputView = outputTexture.createView('render');
            return { outputTexture, outputView };
        }
    );

    optionsBuffer.write(options);

    const inputTextureView = once([filterTexture, inputTexture], () => {
        return inputTexture.createView(d.texture2d());
    });

    const filterBindGroup = onceBindGroup(root, filterBindLayout, {
        texture: inputTextureView,
        sampler: inputSampler,
        filterOptions: optionsBuffer
    });

    pipeline
        .with(filterBindGroup)
        .withColorAttachment({
            color: { view: outputView }
        })
        .draw(6);

    return outputTexture;
};

export const calculateWeights = (
    root: TgpuRoot,
    inputTexture: TgpuTexture & StorageFlag,
    colorSpace: ColorSpace,
    options: d.Infer<typeof computeOptions>
) => {
    const { pipeline, optionsBuffer } = once([calculateWeights, colorSpace], () => {
        const pipeline = root
            .with(colorSpaceSlot, colorSpacesConfig[colorSpace].forward)
            .createGuardedComputePipeline(weightCalculation)
            .withTimestampWrites(timestampOptions(root, 'calculateWeights'));
        const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const { outputBuffer } = once([calculateWeights, options.tableSize], () => {
        const outputBuffer = root
            .createBuffer(d.arrayOf(d.u32, options.tableSize * options.tableSize * options.tableSize * 4))
            .$usage('storage');
        return { outputBuffer };
    });

    optionsBuffer.write(options);
    outputBuffer.clear();

    const inputTextureView = once([calculateWeights, inputTexture], () => {
        return inputTexture.createView(d.textureStorage2d(weightTextureFormat, 'read-only'));
    });

    const bindGroup = onceBindGroup(root, weightCalculationLayout, {
        image: inputTextureView,
        weights: outputBuffer,
        options: optionsBuffer
    });

    pipeline.with(bindGroup).dispatchThreads(options.textureSize.x, options.textureSize.y);

    return outputBuffer;
};

export const processWeightTexture = (
    root: TgpuRoot,
    weightsBuffer: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag,
    options: d.Infer<typeof computeOptions>
) => {
    const { pipeline, optionsBuffer } = once(processWeightTexture, () => {
        const pipeline = root
            .createGuardedComputePipeline(processWeights)
            .withTimestampWrites(timestampOptions(root, 'processWeightTexture'));
        const optionsBuffer = root.createBuffer(computeOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const { outputTexture } = once([processWeightTexture, options.tableSize], () => {
        const outputTexture = root
            .createTexture({
                size: [options.tableSize, options.tableSize, options.tableSize],
                format: weightTextureFormat,
                dimension: '3d'
            })
            .$usage('render', 'sampled', 'storage');
        return { outputTexture };
    });

    optionsBuffer.write(options);

    const bindGroup = onceBindGroup(root, weightTransferLayout, {
        options: optionsBuffer,
        weights: weightsBuffer,
        outputTexture: outputTexture
    });

    pipeline.with(bindGroup).dispatchThreads(options.tableSize, options.tableSize, options.tableSize);

    return outputTexture;
};

export const blurWeightTexture = (
    root: TgpuRoot,
    inputTexture: Texture3d & SampledFlag,
    options: d.Infer<typeof filterOptions>,
    tableSize: number
) => {
    const { pipeline, optionsBuffer } = once(blurWeightTexture, () => {
        const pipeline = root
            .createGuardedComputePipeline(blur)
            .withTimestampWrites(timestampOptions(root, 'blurWeightTexture'));
        const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const { outputTexture } = once([blurWeightTexture, tableSize], () => {
        const outputTexture = root
            .createTexture({
                size: [tableSize, tableSize, tableSize],
                format: weightTextureFormat,
                dimension: '3d'
            })
            .$usage('sampled', 'storage');
        return { outputTexture };
    });

    optionsBuffer.write(options);

    const inputTextureView = once([blurWeightTexture, inputTexture], () => {
        return inputTexture.createView(d.texture3d());
    });

    const bindGroup = onceBindGroup(root, weightProcessingLayout, {
        options: optionsBuffer,
        inputTexture: inputTextureView,
        outputTexture: outputTexture.createView(d.textureStorage3d(weightTextureFormat, 'write-only'))
    });

    pipeline.with(bindGroup).dispatchThreads(tableSize, tableSize, tableSize);

    return outputTexture;
};

export const renderImage = (
    root: TgpuRoot,
    inputTexture: TgpuTexture & SampledFlag,
    inputSampler: TgpuFixedSampler,
    outputView: RenderTarget,
    options: d.Infer<typeof filterOptions>
) => {
    const { pipeline, optionsBuffer } = once(renderImage, () => {
        const pipeline = root
            .createRenderPipeline({
                primitive: { topology: 'triangle-list' },
                vertex: quadVertex,
                fragment: imageFragment
            })
            .withTimestampWrites(timestampOptions(root, 'renderImage'));
        const optionsBuffer = root.createBuffer(filterOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const inputTextureView = once([renderImage, inputTexture], () => {
        return inputTexture.createView(d.texture2d());
    });

    const bindGroup = onceBindGroup(root, textureRenderLayout, {
        texture: inputTextureView,
        sampler: inputSampler,
        options: optionsBuffer
    });

    optionsBuffer.write(options);

    pipeline
        .with(bindGroup)
        .withColorAttachment({
            view: outputView
        })
        .draw(6);
};

export const colorSpacesConfig = {
    [ColorSpace.oklab]: { label: 'Oklab', forward: oklabColorSpace, inverse: oklabColorSpaceInverse },
    [ColorSpace.hsv]: { label: 'HSV', forward: hsvColorSpace, inverse: hsvColorSpaceInverse },
    [ColorSpace.hsl]: { label: 'HSL', forward: hslColorSpace, inverse: hslColorSpaceInverse },
    [ColorSpace.srgb]: { label: 'sRGB', forward: srgbColorSpace, inverse: srgbColorSpaceInverse },
    [ColorSpace.linear_rgb]: {
        label: 'Linear RGB',
        forward: linearRgbColorSpace,
        inverse: linearRgbColorSpaceInverse
    }
};

// Rasterizes the cloud box's reference grid faces (bottom/top and the walls)
// into a color texture and a depth buffer, using the same camera as the
// cloud raymarch. Each face fades in only once it lies on the far side of
// the camera's view direction, so the visible faces change smoothly as the
// camera orbits instead of popping in/out.
export const renderRasterScene = (
    root: TgpuRoot,
    outputTexture: TgpuTexture & RenderFlag,
    depthTexture: TgpuTexture & RenderFlag & SampledFlag,
    camera: d.Infer<typeof cameraUniform>,
    bgColor: number
) => {
    const { pipeline, cameraBuffer, vertexBuffer, faces } = once(renderRasterScene, () => {
        const pipeline = root.createRenderPipeline({
            vertex: gridVertex,
            fragment: gridFragment,
            attribs: gridVertexLayout.attrib,
            targets: { format: 'rgba8unorm' },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less'
            }
        });
        const cameraBuffer = root.createBuffer(cameraUniform).$usage('uniform');

        const allVertices = gridFaces.flatMap((face) => face.vertices);
        const vertexBuffer = root
            .createBuffer(gridVertexLayout.schemaForCount(allVertices.length), allVertices)
            .$usage('vertex');

        const faces = gridFaces.map((face, i) => {
            const faceOptionsBuffer = root.createBuffer(gridFaceOptions).$usage('uniform');
            const bindGroup = root.createBindGroup(gridBindLayout, {
                cameraUniform: cameraBuffer,
                faceOptions: faceOptionsBuffer
            });
            return { normal: face.normal, firstVertex: i * 6, faceOptionsBuffer, bindGroup };
        });

        return { pipeline, cameraBuffer, vertexBuffer, faces };
    });

    cameraBuffer.write(camera);

    // The camera's forward (view) direction, derived directly from yaw/pitch
    // with the same basis as worldToClip/cameraRay.
    const fwd = d.vec3f(
        -Math.sin(camera.yaw) * Math.cos(camera.pitch),
        -Math.sin(camera.pitch),
        -Math.cos(camera.yaw) * Math.cos(camera.pitch)
    );

    // Contrast the grid lines against the background rather than always
    // drawing them white.
    const lineColor = bgColor > 0.6 ? bgColor - 0.3 : bgColor + 0.3;

    const encoder = root.device.createCommandEncoder();

    faces.forEach(({ normal, firstVertex, faceOptionsBuffer, bindGroup }, i) => {
        const facing = std.dot(fwd, normal);
        faceOptionsBuffer.write({ alpha: Math.max(0, std.clamp((facing - 0.05) * 6, 0.0, 1.0)), color: lineColor });

        const loadOp = i === 0 ? 'clear' : 'load';
        let facePipeline = pipeline.with(encoder);
        if (i === 0) {
            facePipeline = facePipeline.withTimestampWrites(timestampOptions(root, 'renderRasterScene'));
        }

        facePipeline
            .withColorAttachment({ view: outputTexture, loadOp })
            .withDepthStencilAttachment({
                view: depthTexture,
                depthClearValue: 1.0,
                depthLoadOp: loadOp,
                depthStoreOp: 'store'
            })
            .with(gridVertexLayout, vertexBuffer)
            .with(bindGroup)
            .draw(6, 1, firstVertex);
    });

    root.device.queue.submit([encoder.finish()]);
};

// Rasterizes an opaque marker sphere for each saved color, positioned at the
// color's location in the current color space. Draws into the same
// color/depth targets as renderRasterScene (which must run first, since it
// clears them), so the cloud raymarch stops at the spheres.
export const renderColorSpheres = (
    root: TgpuRoot,
    outputTexture: TgpuTexture & RenderFlag,
    depthTexture: TgpuTexture & RenderFlag & SampledFlag,
    camera: d.Infer<typeof cameraUniform>,
    colorSpace: ColorSpace,
    colors: d.v3f[],
    lighting: boolean
) => {
    if (colors.length === 0) {
        return;
    }

    const pipeline = once([renderColorSpheres, colorSpace], () =>
        root.with(colorSpaceSlot, colorSpacesConfig[colorSpace].forward).createRenderPipeline({
            vertex: sphereVertex,
            fragment: sphereFragment,
            attribs: sphereVertexLayout.attrib,
            targets: { format: 'rgba8unorm' },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less'
            }
        })
    );

    const { cameraBuffer, vertexBuffer, vertexCount } = once(renderColorSpheres, () => {
        const vertices = sphereVertices().map((position) => ({ position }));
        const vertexBuffer = root
            .createBuffer(sphereVertexLayout.schemaForCount(vertices.length), vertices)
            .$usage('vertex');
        const cameraBuffer = root.createBuffer(cameraUniform).$usage('uniform');
        return { cameraBuffer, vertexBuffer, vertexCount: vertices.length };
    });

    cameraBuffer.write(camera);

    const encoder = root.device.createCommandEncoder();

    colors.forEach((color, i) => {
        // Buffers and bind groups are pooled by sphere index and rewritten
        // every pass, so the color list can grow and shrink freely.
        const { optionsBuffer, bindGroup } = once([renderColorSpheres, 'sphere', i], () => {
            const optionsBuffer = root.createBuffer(sphereOptions).$usage('uniform');
            const bindGroup = root.createBindGroup(sphereBindLayout, {
                cameraUniform: cameraBuffer,
                sphereOptions: optionsBuffer
            });
            return { optionsBuffer, bindGroup };
        });
        optionsBuffer.write({
            color: d.vec4f(color.x, color.y, color.z, 1.0),
            radius: 0.02,
            lighting: lighting ? 1.0 : 0.0
        });

        pipeline
            .with(encoder)
            .withColorAttachment({ view: outputTexture, loadOp: 'load' })
            .withDepthStencilAttachment({
                view: depthTexture,
                depthClearValue: 1.0,
                depthLoadOp: 'load',
                depthStoreOp: 'store'
            })
            .with(sphereVertexLayout, vertexBuffer)
            .with(bindGroup)
            .draw(vertexCount);
    });

    root.device.queue.submit([encoder.finish()]);
};

export const renderColorCloud = (
    root: TgpuRoot,
    inputTexture: Texture3d & SampledFlag,
    inputSampler: TgpuFixedSampler,
    rasterDepthTexture: TgpuTexture & SampledFlag,
    outputTexture: TgpuTexture & RenderFlag,
    pickTexture: TgpuTexture & RenderFlag,
    colorSpace: ColorSpace,
    camera: d.Infer<typeof cameraUniform>
) => {
    const { pipeline, cameraBuffer, rasterDepthSizeBuffer } = once(
        [renderColorCloud, colorSpace],
        () => {
            const pipeline = root
                .with(colorSpaceSlot, colorSpacesConfig[colorSpace].forward)
                .with(colorSpaceInverseSlot, colorSpacesConfig[colorSpace].inverse)
                .createRenderPipeline({
                    primitive: { topology: 'triangle-list' },
                    vertex: quadVertex,
                    fragment: triangleFragment,
                    targets: {
                        color: { format: 'rgba8unorm' },
                        pick: { format: 'rgba8unorm' }
                    }
                })
                .withTimestampWrites(timestampOptions(root, 'renderColorCloud'));
            const cameraBuffer = root.createBuffer(cameraUniform).$usage('uniform');
            const rasterDepthSizeBuffer = root.createBuffer(d.vec2u).$usage('uniform');
            return { pipeline, cameraBuffer, rasterDepthSizeBuffer };
        }
    );

    const inputTextureView = once([renderColorCloud, inputTexture], () => {
        return inputTexture.createView(d.texture3d());
    });

    const rasterDepthView = once([renderColorCloud, rasterDepthTexture], () => {
        return rasterDepthTexture.createView(d.textureDepth2d());
    });

    const cameraBindGroup = onceBindGroup(root, cameraBindLayout, {
        cameraUniform: cameraBuffer,
        weightTexture: inputTextureView,
        weightSampler: inputSampler,
        rasterDepth: rasterDepthView,
        rasterDepthSize: rasterDepthSizeBuffer
    });

    cameraBuffer.write(camera);
    const rawDepth = root.unwrap(rasterDepthTexture);
    rasterDepthSizeBuffer.write(d.vec2u(rawDepth.width, rawDepth.height));

    pipeline
        .with(cameraBindGroup)
        .withColorAttachment({
            color: { view: outputTexture },
            pick: { view: pickTexture }
        })
        .draw(6);
};

export const compositeColorCloud = (
    root: TgpuRoot,
    cloudTexture: TgpuTexture & SampledFlag,
    pickTexture: TgpuTexture & SampledFlag,
    rasterTexture: TgpuTexture & SampledFlag,
    inputSampler: TgpuFixedSampler,
    outputView: RenderTarget,
    options: d.Infer<typeof cloudCompositeOptions>
) => {
    const { pipeline, optionsBuffer } = once(compositeColorCloud, () => {
        const pipeline = root
            .createRenderPipeline({
                primitive: { topology: 'triangle-list' },
                vertex: quadVertex,
                fragment: cloudCompositeFragment
            })
            .withTimestampWrites(timestampOptions(root, 'compositeColorCloud'));
        const optionsBuffer = root.createBuffer(cloudCompositeOptions).$usage('uniform');
        return { pipeline, optionsBuffer };
    });

    const cloudTextureView = once([compositeColorCloud, cloudTexture], () => {
        return cloudTexture.createView(d.texture2d());
    });

    const pickTextureView = once([compositeColorCloud, pickTexture], () => {
        return pickTexture.createView(d.texture2d());
    });

    const rasterTextureView = once([compositeColorCloud, rasterTexture], () => {
        return rasterTexture.createView(d.texture2d());
    });

    const bindGroup = onceBindGroup(root, cloudCompositeLayout, {
        cloudTexture: cloudTextureView,
        pickTexture: pickTextureView,
        rasterTexture: rasterTextureView,
        sampler: inputSampler,
        options: optionsBuffer
    });

    optionsBuffer.write(options);

    pipeline
        .with(bindGroup)
        .withColorAttachment({
            view: outputView
        })
        .draw(6);
};
