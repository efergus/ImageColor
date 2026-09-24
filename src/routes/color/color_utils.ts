import { d, std } from 'typegpu';

export enum ColorSpace {
	srgb = 'srgb',
	linear_rgb = 'linear_rgb',
	oklab = 'oklab',
	hsv = 'hsv',
	hsl = 'hsl'
}

export const srgb_to_linear_rgb = (color: d.v3f) => {
	'use gpu';
	const b = std.le(color, d.vec3f(0.04045));
	const low = std.div(color, d.f32(12.92));
	const high = std.pow(std.div(std.add(color, d.f32(0.055)), d.f32(1.055)), d.vec3f(2.4, 2.4, 2.4));
	return std.select(high, low, b);
};

export const linear_rgb_to_srgb = (color: d.v3f) => {
	'use gpu';
	const b = std.le(color, d.vec3f(0.0031308));
	const low = std.mul(color, d.f32(12.92));
	const high = std.sub(
		std.mul(
			d.f32(1.055),
			std.pow(std.max(color, d.vec3f(0.0)), d.vec3f(1.0 / 2.4, 1.0 / 2.4, 1.0 / 2.4))
		),
		d.f32(0.055)
	);
	return std.select(high, low, b);
};

export const linear_rgb_to_oklab = (color: d.v3f) => {
	'use gpu';
	const l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
	const m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
	const s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;
	const power = d.f32(1.0 / 3.0);
	const l_ = std.pow(std.max(l, d.f32(0.0)), power);
	const m_ = std.pow(std.max(m, d.f32(0.0)), power);
	const s_ = std.pow(std.max(s, d.f32(0.0)), power);
	return d.vec3f(
		0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
		1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
		0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_
	);
};

export const oklab_to_linear_rgb = (color: d.v3f) => {
	'use gpu';
	const l_ = color.x + 0.3963377774 * color.y + 0.2158037573 * color.z;
	const m_ = color.x - 0.1055613458 * color.y - 0.0638541728 * color.z;
	const s_ = color.x - 0.0894841775 * color.y - 1.291485548 * color.z;
	const l = l_ * l_ * l_;
	const m = m_ * m_ * m_;
	const s = s_ * s_ * s_;
	const r = std.add(
		std.add(std.mul(d.f32(4.0767416621), l), std.mul(d.f32(-3.3077115913), m)),
		std.mul(d.f32(0.2309699292), s)
	);
	const g = std.add(
		std.add(std.mul(d.f32(-1.2684380046), l), std.mul(d.f32(2.6097574011), m)),
		std.mul(d.f32(-0.3413193965), s)
	);
	const b = std.add(
		std.add(std.mul(d.f32(-0.0041960863), l), std.mul(d.f32(-0.7034186147), m)),
		std.mul(d.f32(1.707614701), s)
	);
	return d.vec3f(r, g, b);
};

export const srgb_to_oklab = (color: d.v3f) => {
	'use gpu';
	return linear_rgb_to_oklab(srgb_to_linear_rgb(color));
};

export const oklab_to_srgb = (color: d.v3f) => {
	'use gpu';
	return linear_rgb_to_srgb(oklab_to_linear_rgb(color));
};

export const srgb_to_hsv = (color: d.v3f) => {
	'use gpu';
	// Converts a standard RGB color to HSV (Hue, Saturation, Value) space.
	// Hue is calculated in degrees [0, 360). Saturation and Value are in [0, 1].
	// This color space is mapped as a single cone where Value is the vertical axis.
	const r = color.r;
	const g = color.g;
	const b = color.b;
	const max = std.max(r, std.max(g, b));
	const min = std.min(r, std.min(g, b));
	const delta = std.sub(max, min);
	let h = d.f32(0);
	let s = d.f32(0);
	const v = max;
	if (delta !== d.f32(0)) {
		if (max === r) {
			h = std.div(std.sub(g, b), delta);
		} else if (max === g) {
			h = std.add(std.div(std.sub(b, r), delta), d.f32(2.0));
		} else {
			h = std.add(std.div(std.sub(r, g), delta), d.f32(4.0));
		}
		h = std.mod(std.add(h, d.f32(6.0)), d.f32(6.0));
		h = std.mul(h, d.f32(60.0));
		s = std.div(delta, max);
	}
	return d.vec3f(h, s, v);
};

export const hsv_to_srgb = (color: d.v3f) => {
	'use gpu';
	// Converts an HSV color back to the standard RGB color space.
	// Reverses the single cone mapping to return RGB components in [0, 1].
	const h = color.r;
	const s = color.g;
	const v = color.b;
	const i = std.floor(std.mul(std.div(h, d.f32(60.0)), d.f32(1.0)));
	const f = std.sub(std.div(h, d.f32(60.0)), i);
	const p = std.sub(v, std.mul(v, std.mul(s, d.f32(1.0))));
	const q = std.sub(v, std.mul(v, std.mul(s, std.mul(f, d.f32(1.0)))));
	const t = std.sub(v, std.mul(v, std.mul(s, std.sub(d.f32(1.0), f))));
	// Initialized to the i == 5 sector's values; the branches cover the rest.
	let r = v;
	let g = p;
	let b = q;
	if (i === d.f32(0)) {
		r = v;
		g = t;
		b = p;
	} else if (i === d.f32(1)) {
		r = q;
		g = v;
		b = p;
	} else if (i === d.f32(2)) {
		r = p;
		g = v;
		b = t;
	} else if (i === d.f32(3)) {
		r = p;
		g = q;
		b = v;
	} else if (i === d.f32(4)) {
		r = t;
		g = p;
		b = v;
	}
	return d.vec3f(r, g, b);
};

export const srgb_to_hsl = (color: d.v3f) => {
	'use gpu';
	// Converts a standard RGB color to HSL (Hue, Saturation, Lightness) space.
	// Hue is calculated in degrees [0, 360). Saturation and Lightness are in [0, 1].
	// This color space is typically mapped as a double hexcone.
	const r = color.r;
	const g = color.g;
	const b = color.b;
	const max = std.max(r, std.max(g, b));
	const min = std.min(r, std.min(g, b));
	const delta = std.sub(max, min);

	let h = d.f32(0);
	let s = d.f32(0);
	const l = std.div(std.add(max, min), d.f32(2.0));

	if (delta !== d.f32(0)) {
		if (max === r) {
			h = std.div(std.sub(g, b), delta);
		} else if (max === g) {
			h = std.add(std.div(std.sub(b, r), delta), d.f32(2.0));
		} else {
			h = std.add(std.div(std.sub(r, g), delta), d.f32(4.0));
		}
		h = std.mod(std.add(h, d.f32(6.0)), d.f32(6.0));
		h = std.mul(h, d.f32(60.0));

		// Saturation calculation for HSL
		const l_step = std.abs(std.sub(std.mul(l, d.f32(2.0)), d.f32(1.0)));
		s = std.div(delta, std.sub(d.f32(1.0), l_step));
	}
	return d.vec3f(h, s, l);
};

export const hsl_to_srgb = (color: d.v3f) => {
	'use gpu';
	// Converts an HSL color back to the standard RGB color space.
	// Reverses the double hexcone mapping to return RGB components in [0, 1].
	const h = color.r;
	const s = color.g;
	const l = color.b;

	const l_step = std.abs(std.sub(std.mul(l, d.f32(2.0)), d.f32(1.0)));
	const c = std.mul(std.sub(d.f32(1.0), l_step), s);
	const h_prime = std.div(h, d.f32(60.0));
	const x = std.mul(
		c,
		std.sub(d.f32(1.0), std.abs(std.sub(std.mod(h_prime, d.f32(2.0)), d.f32(1.0))))
	);
	const m = std.sub(l, std.div(c, d.f32(2.0)));

	// Initialized to the i == 5 sector's values; the branches cover the rest.
	let r = c;
	let g = d.f32(0);
	let b = x;

	const i = std.floor(h_prime);

	if (i === d.f32(0)) {
		r = c;
		g = x;
		b = d.f32(0);
	} else if (i === d.f32(1)) {
		r = x;
		g = c;
		b = d.f32(0);
	} else if (i === d.f32(2)) {
		r = d.f32(0);
		g = c;
		b = x;
	} else if (i === d.f32(3)) {
		r = d.f32(0);
		g = x;
		b = c;
	} else if (i === d.f32(4)) {
		r = x;
		g = d.f32(0);
		b = c;
	}

	return d.vec3f(std.add(r, m), std.add(g, m), std.add(b, m));
};
