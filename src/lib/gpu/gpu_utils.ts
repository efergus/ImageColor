import type {
	TgpuBindGroup,
	TgpuBindGroupLayout,
	TgpuRoot,
	ExtractBindGroupInputFromLayout,
	TgpuLayoutEntry
} from 'typegpu';

type OnceLeaf<V> = {
	child: OnceMap<V>;
	value: V | null;
};

type OnceMap<V = unknown> = Map<unknown, OnceLeaf<V>>;

// One cache shared by every call site, each keyed by its own unique path
// (e.g. [fnReference, ...args]), so distinct call sites never collide.
// Type safety is enforced by convention rather than by the map's declared
// type: each `once`/`onceBindGroup` caller casts the shared cache to the
// value type it alone reads and writes at its key path.
const globalOnceMap: OnceMap<unknown> = new Map();

export const once = <T>(keys: unknown | unknown[], fn: () => T): T => {
	const keyArray: unknown[] = Array.isArray(keys) ? keys : [keys];
	return once2(globalOnceMap as OnceMap<T>, keyArray, fn);
};

const once2 = <V>(context: OnceMap<V>, keys: unknown[], fn: () => V): V => {
	let result: OnceLeaf<V> = {
		child: context,
		value: null
	};
	for (const key of keys) {
		let next = result.child.get(key);
		if (next === undefined) {
			next = {
				child: new Map(),
				value: null
			};
			result.child.set(key, next);
		}
		result = next;
	}
	if (result.value === null) {
		result.value = fn();
	}
	return result.value;
};

const bindGroupMap: OnceMap<unknown> = new Map();

export const onceBindGroup = <Entries extends Record<string, TgpuLayoutEntry | null>>(
	root: TgpuRoot,
	bindGroupLayout: TgpuBindGroupLayout<Entries>,
	entries: ExtractBindGroupInputFromLayout<Entries>
) => {
	const entryKeys = Object.keys(entries) as (keyof typeof entries)[];
	entryKeys.sort();
	const key = [bindGroupLayout, ...entryKeys.map((key) => entries[key])];
	return once2(bindGroupMap as OnceMap<TgpuBindGroup<Entries>>, key, () => {
		return root.createBindGroup(bindGroupLayout, entries);
	});
};
