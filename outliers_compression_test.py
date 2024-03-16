import unittest

import torch
import numpy as np

from src.aq import MaskCompressor, ValuesCompressor, QuantizedOutliers


class MyTestCase(unittest.TestCase):

    def test_compress_diff(self):
        idx_diff = torch.from_numpy(np.array([1, 2, 255, 256, 1024, 2390, 23901, 256 ** 2 + 12345, 2]))
        idx_diff_compressed = MaskCompressor.compress_diff(idx_diff)
        idx_diff_decompressed = MaskCompressor.decompress_diff(idx_diff_compressed)
        print(idx_diff_decompressed)
        torch.testing.assert_close(idx_diff, idx_diff_decompressed)

    def test_compress_mask(self):
        h, w = 1024, 512
        mask = (torch.rand((h, w)) < 0.01)
        mask_compressed = MaskCompressor.compress_mask(mask)
        mask_decompressed = MaskCompressor.decompress_mask(*mask_compressed)
        torch.testing.assert_close(mask, mask_decompressed)

    def test_compress_values(self):
        values = torch.Tensor([0, 15, 5, 3, 0, 15, 15, 0])

        for block_size in (4, 8, 16, 32, 64, 128):
            length, values_compressed, min_values, max_values = ValuesCompressor.compress_values(values, block_size=block_size)
            values_decompressed = ValuesCompressor.decompress_values(length, values_compressed, min_values, max_values, block_size=block_size)
            torch.testing.assert_close(values, values_decompressed)

    def test_compress_outliers(self):
        outliers = torch.tensor([0, 4, 2, 17], dtype=torch.float64).reshape(2, 2)
        quantized_outliers = QuantizedOutliers(outliers=outliers)
        torch.testing.assert_close(outliers, quantized_outliers())

    def test_compress_outliers_2(self):
        outliers = torch.tensor([0, 4, 5, 3, 2, 17], dtype=torch.float64).reshape(3, 2)
        quantized_outliers = QuantizedOutliers(outliers=outliers)
        torch.testing.assert_close(outliers, quantized_outliers())

    def test_compress_outliers_3(self):
        outliers = torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.float64).reshape(2, 3)
        quantized_outliers = QuantizedOutliers(outliers=outliers)
        torch.testing.assert_close(outliers, quantized_outliers())


if __name__ == '__main__':
    unittest.main()
