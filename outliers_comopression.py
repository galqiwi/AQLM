import torch
import numpy as np
import torch.nn as nn


class MaskCompressor:
    @classmethod
    def compress_mask(cls, mask: torch.Tensor):
        assert mask.dtype == torch.bool
        h, w = mask.shape

        idx = cls.get_idx(mask)
        assert len(idx) > 2

        idx_diff = idx[1:] - idx[:-1]
        idx_diff_compressed = cls.compress_diff(idx_diff)
        assert idx_diff_compressed.dtype == torch.int8

        return torch.LongTensor([idx[0], h, w]), idx_diff_compressed

    @classmethod
    def decompress_mask(cls, first_idx: torch.Tensor, idx_diff_compressed: torch.Tensor):
        assert first_idx.dtype == torch.int64
        assert idx_diff_compressed.dtype == torch.int8

        idx_diff = cls.decompress_diff(idx_diff_compressed)
        idx0, h, w = first_idx

        idx = torch.cumsum(torch.cat((torch.LongTensor([idx0.item()]), idx_diff), dim=0), dim=0)
        assert idx.dtype == torch.int64

        output = cls.get_mask(idx, h.item(), w.item())
        assert output.dtype == torch.bool

        return output

    @staticmethod
    def get_idx(mask: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(mask.reshape(-1))[:, 0]

    @staticmethod
    def get_mask(idx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        output = torch.zeros((h * w,)).to(torch.bool)
        output[idx] = True
        return output.reshape(h, w)

    @staticmethod
    def compress_diff(idx_diff: torch.Tensor):
        idx_diff = idx_diff.cpu().numpy()

        output = []
        for delta in idx_diff:
            if delta > 255:
                assert (delta // 256) <= 255
                output.extend((0, delta // 256, delta % 256))
                continue
            assert delta > 0
            output.append(delta)

        output = torch.LongTensor(output)

        assert (output > 255).sum() == 0
        assert (output < 0).sum() == 0

        return (output + np.iinfo(np.int8).min).to(torch.int8)

    @staticmethod
    def decompress_diff(idx_diff_compressed: torch.Tensor):
        idx_diff_compressed = idx_diff_compressed.cpu()

        output = []

        idx = 0
        while idx < len(idx_diff_compressed):
            if idx_diff_compressed[idx].item() != np.iinfo(np.int8).min:
                output.append(idx_diff_compressed[idx].item() - np.iinfo(np.int8).min)
                idx += 1
                continue
            output.append(idx_diff_compressed[idx + 1].item() * 256 + idx_diff_compressed[idx + 2].item() - np.iinfo(np.int8).min * 257)
            idx += 3

        return torch.LongTensor(output)


class ValuesCompressor:
    @staticmethod
    def compress_values(values: torch.Tensor, block_size: int = 64):
        assert block_size % 2 == 0
        length, = values.shape

        if length % block_size != 0:
            append_value = torch.mean(values[(length // block_size) * block_size:])
            values = torch.cat([
                values,
                torch.full(
                    size=(block_size - length % block_size,),
                    fill_value=append_value,
                ),
            ])

        assert len(values) % block_size == 0
        values = values.reshape(-1, block_size)

        min_values = values.min(dim=1).values
        max_values = values.max(dim=1).values

        max_values = max_values + (max_values == min_values).float()

        values = (values - min_values[:, None]) / (max_values - min_values)[:, None]

        values = values.reshape(-1)

        values = torch.round(values * 15).to(torch.int64)
        assert len(values) % 2 == 0

        values = values.reshape(-1, 2)
        values = 16 * values[:, 0] + values[:, 1] + np.iinfo(np.int8).min
        values = values.to(torch.int8)

        return torch.LongTensor([length]), values, min_values, max_values

    @staticmethod
    def decompress_values(length: torch.LongTensor, values: torch.Tensor, min_values: torch.Tensor, max_values: torch.Tensor, block_size: int = 64):
        length = length.item()
        values = values.to(torch.int64) - np.iinfo(np.int8).min
        values = torch.cat((
            (values // 16)[:, None],
            (values % 16)[:, None],
        ), dim=1)
        values = values.reshape(-1, block_size)
        values = min_values[:, None] + (max_values - min_values)[:, None] * (values.float() / 15)
        values = values.reshape(-1)
        return values[:length]


class QuantizedOutliers(nn.Module):
    def __init__(self, outliers: torch.Tensor):
        super().__init__()
        outliers_sparse = outliers.to_sparse_coo()
        self.outliers_values = ValuesCompressor.compress_values(outliers_sparse.values())
        self.outliers_matrix = MaskCompressor.compress_mask(outliers != 0.)

    def forward(self) -> torch.Tensor:
        indices = MaskCompressor.decompress_mask(*self.outliers_matrix).to_sparse_coo().indices()
        return torch.sparse_coo_tensor(
            indices=indices,
            values=ValuesCompressor.decompress_values(*self.outliers_values),
        ).to_dense()
