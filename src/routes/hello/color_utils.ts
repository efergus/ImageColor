import tgpu, { d, std } from 'typegpu';

export const linear_rgb_to_srgb = (color: d.v3f) => {
    'use gpu';
    const b = std.le(color, d.vec3f(0.04045));
    const low = std.div(color, d.f32(12.92));
    const high = std.pow(std.div(std.add(color, d.f32(0.055)), d.f32(1.055)), d.vec3f(2.4, 2.4, 2.4));
    return std.select(high, low, b);
}

export const srgb_to_linear_rgb = (color: d.v3f) => {
    'use gpu';
    const b = std.le(color, d.vec3f(0.0031308));
    const low = std.mul(color, d.f32(12.92));
    const high = std.sub(std.mul(d.f32(1.055), std.pow(std.max(color, d.vec3f(0.0)), d.vec3f(2.4, 2.4, 2.4))), d.f32(0.055));
    return std.select(high, low, b);
}

export const linear_rgb_to_oklab = (color: d.v3f) => {
    'use gpu';
    const l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
    const m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
    const s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;
    const l_ = std.pow(std.max(l, d.f32(0.0)), d.f32(1.0 / 3.0));
    const m_ = std.pow(std.max(m, d.f32(0.0)), d.f32(1.0 / 3.0));
    const s_ = std.pow(std.max(s, d.f32(0.0)), d.f32(1.0 / 3.0));
    return d.vec3f(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

export const oklab_to_linear_rgb = (color: d.v3f) => {
    'use gpu';
    const l_ = color.x + 0.3963377774 * color.y + 0.2158037573 * color.z;
    const m_ = color.x - 0.1055613458 * color.y - 0.0638541728 * color.z;
    const s_ = color.x - 0.0894841775 * color.y - 1.2914855480 * color.z;
    const l = std.pow(l_, d.f32(3.0));
    const m = std.pow(m_, d.f32(3.0));
    const s = std.pow(s_, d.f32(3.0));
    const r = std.add(std.add(std.mul(d.f32(4.0767416621), l), std.mul(d.f32(-3.3077115913), m)), std.mul(d.f32(0.2309699292), s));
    const g = std.add(std.add(std.mul(d.f32(-1.2684380046), l), std.mul(d.f32(2.6097574011), m)), std.mul(d.f32(-0.3413193965), s));
    const b = std.add(std.add(std.mul(d.f32(-0.0041960863), l), std.mul(d.f32(-0.7034186147), m)), std.mul(d.f32(1.7076147010), s));
    return std.clamp(d.vec3f(r, g, b), d.vec3f(0.0), d.vec3f(1.0));
}