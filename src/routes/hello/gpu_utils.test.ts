import { onceObjEntries, type ObjMap } from './gpu_utils';
import { expect, test } from "vitest";

test('gpu_utils.onceObjEntries', () => {
    const map: ObjMap = new Map();
    const factory = (entries: Record<string, string>) => {
        return {
            value: 2,
            entries: entries
        }
    };
    const result = onceObjEntries<string, {
        value: number,
        entries: Record<string, string>
    }>(map, { a: '1', b: '2' }, factory);
    expect(result.value).toBe(2);

    const result2 = onceObjEntries<string, {
        value: number
    }>(map, { a: '1', c: '2' }, factory);
    expect(result2).toBe(result);

    const result3 = onceObjEntries<string, {
        value: number
    }>(map, { a: '1', b: '3' }, factory);
    expect(result3).not.toBe(result);
});