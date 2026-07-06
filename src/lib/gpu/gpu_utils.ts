import type { TgpuBindGroup, TgpuBindGroupLayout, TgpuRoot, ExtractBindGroupInputFromLayout, TgpuLayoutEntry } from "typegpu";

const globalOnceMap = new Map<(...args: any[]) => void, any>();

export const once = <T>(keys: any | any[], fn: () => T): T => {
    if (!Array.isArray(keys)) {
        keys = [keys];
    }
    return once2(globalOnceMap, keys, fn);
}
type OnceLeaf<V> = {
    child: OnceMap<V>;
    value: V | null
}


type OnceMap<V = any> = Map<any, OnceLeaf<V>>;

const once2 = <V>(context: OnceMap<V>, keys: any[], fn: () => V): V => {
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
            }
            result.child.set(key, next);
        }
        result = next;
    }
    if (result.value === null) {
        result.value = fn();
    }
    return result.value;
}

const bindGroupMap: OnceMap<TgpuBindGroup<any>> = new Map();

export const onceBindGroup = <Entries extends Record<string, TgpuLayoutEntry | null>>(root: TgpuRoot, bindGroupLayout: TgpuBindGroupLayout<Entries>, entries: ExtractBindGroupInputFromLayout<Entries>) => {
    const entryKeys = Object.keys(entries) as (keyof typeof entries)[];
    entryKeys.sort();
    const key = [bindGroupLayout, ...entryKeys.map((key) => entries[key])];
    return once2<TgpuBindGroup<Entries>>(bindGroupMap, key, () => {
        return root.createBindGroup(
            bindGroupLayout,
            entries
        )
    });
}