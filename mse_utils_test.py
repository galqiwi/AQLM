import torch
import unittest
from mse_utils import get_c


class MyTestCase(unittest.TestCase):
    def test_C(self):
        n_codes = 2
        codes = torch.tensor([0, 1, 1], dtype=torch.int64)
        C = get_c(codes, n_codes)
        torch.testing.assert_close(
            C,
            torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.float32).reshape(3, 2),
        )

    def test_just_works(self):
        n_codes = 2
        codes = torch.tensor([0, 1, 1], dtype=torch.int64)
        XTX = torch.eye(3, dtype=torch.float32)
        target = torch.tensor([-1, 7, 7], dtype=torch.float32)
        C = get_c(codes, n_codes)
        P = C.T @ XTX @ C
        Q = target @ XTX @ C
        solution = torch.linalg.solve(P, Q)
        torch.testing.assert_close(
            solution,
            torch.tensor([-1, 7], dtype=torch.float32),
        )


if __name__ == '__main__':
    unittest.main()
