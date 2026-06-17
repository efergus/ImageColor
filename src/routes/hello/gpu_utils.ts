
const gpuMap = new Map<string, any>();

export const once = <T>(key: string, fn: () => T): T => {
    if (gpuMap.has(key)) {
        return gpuMap.get(key)!;
    }
    const value = fn();
    gpuMap.set(key, value);
    return value;
}