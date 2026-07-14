import { once } from '../../lib/gpu/gpu_utils';
import { expect, test } from "vitest";

test('gpu_utils.once', () => {
    const factory = () => {
        return {
            value: 2
        }
    };
    const result = once(['a', '1'], factory);
    expect(result.value).toBe(2);

    const result2 = once(['a', '1'], factory);
    expect(result2).toBe(result);

    const result3 = once(['a', '2'], factory);
    expect(result3).not.toBe(result);
});
