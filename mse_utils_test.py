import tqdm
import torch
import unittest
# from mse_utils import get_c
from numba import njit
import torch.nn.functional as F


def get_c(codes, n_codes):
    return F.one_hot(codes, num_classes=n_codes).to(torch.float32)


@torch.no_grad()
def calculate_P_batch(XTX, codes, n_codes):
    out_size, in_size = codes.shape
    P = torch.zeros((out_size, n_codes * n_codes), device=XTX.device)
    XTX_flat = XTX.reshape(1, -1).repeat(out_size, 1)
    indexes = (
            codes.reshape(out_size, in_size, 1) * n_codes +
            codes.reshape(out_size, 1, in_size)
    ).reshape(out_size, in_size * in_size)
    P = P.scatter_add(dim=1, src=XTX_flat, index=indexes)
    P = P.sum(0).reshape(n_codes, n_codes)
    return P


@torch.no_grad()
def get_codebooks(codes_slice, target, XTX, nbits_per_codebook, in_group_size=8, out_group_size=1):
    assert out_group_size == 1

    n_codes = 2 ** nbits_per_codebook * in_group_size
    out_size, in_size = target.shape
    assert codes_slice.shape == (out_size, in_size // in_group_size)
    assert XTX.shape == (in_size, in_size)

    print(codes_slice, in_group_size)

    codes_slice = codes_slice[:, :, None].repeat(1, 1, in_group_size) * in_group_size
    codes_slice += torch.arange(in_group_size, device=codes_slice.device)[None, None, :]
    codes_slice = codes_slice.reshape(out_size, in_size)

    assert codes_slice.shape == (out_size, in_size)

    print(codes_slice, in_group_size)

    U = target @ XTX
    Q = torch.zeros((out_size, n_codes), dtype=torch.float32, device=U.device)
    Q = Q.scatter_add(dim=1, index=codes_slice, src=U).sum(dim=0)

    P = torch.zeros((n_codes, n_codes), device=XTX.device)

    batch_size = 8 if out_size % 8 == 0 else 1
    for start in tqdm.tqdm(range(0, out_size, batch_size)):
        P += calculate_P_batch(XTX, codes_slice[start:start + batch_size], n_codes)

    output = torch.linalg.solve(P, Q)

    # return P, Q, output
    assert output.shape == (n_codes,)

    return output.reshape(n_codes // in_group_size, out_group_size, in_group_size)


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
        n_codes = 4
        codes = torch.tensor([3, 0, 1, 2], dtype=torch.int64)

        XTX = torch.randn((4, 4))
        XTX = XTX.T @ XTX
        torch.testing.assert_close(XTX, XTX.T)

        target = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        C = get_c(codes, n_codes)
        print(C)
        P = C.T @ XTX @ C

        Q = (target[None, :] @ XTX @ C).T[:, 0]
        Q = C.T @ XTX @ target

        print(P.shape, Q.shape)
        solution = torch.linalg.solve(P, Q)
        # solution = torch.linalg.solve(C.T @ XTX @ C, C.T @ XTX @ target)
        # solution = torch.linalg.solve(XTX, target.T @ XTX)
        torch.testing.assert_close(
            solution,
            torch.tensor([2, 3, 4, 1], dtype=torch.float32),
            atol=1e-3,
            rtol=1,
        )

    def get_q_p_ref(self, XTX, target, codes, n_codes):
        out_size, in_size = codes.shape
        Q = None
        P = None
        for out_idx in range(out_size):
            C = get_c(codes[out_idx], n_codes)
            # DQ = target[out_idx] @ XTX @ C
            DQ = C.T @ XTX @ target[out_idx]
            DP = C.T @ XTX @ C
            if Q is None:
                Q = DQ
                P = DP
            else:
                Q += DQ
                P += DP
        return P, Q

    def test_Q_base(self):
        n_codes = 4
        codes = torch.tensor([[3, 0, 1, 2]], dtype=torch.int64)
        out_size, in_size = codes.shape

        XTX = torch.randn((4, 4))
        XTX = XTX.T @ XTX
        torch.testing.assert_close(XTX, XTX.T)

        target = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

        P_ref, Q_ref = self.get_q_p_ref(XTX, target, codes, n_codes)
        solution = torch.linalg.solve(P_ref, Q_ref)
        torch.testing.assert_close(
            solution,
            torch.tensor([2, 3, 4, 1], dtype=torch.float32),
            atol=1e-3,
            rtol=1,
        )

    def get_x_times_c(self, x, codes, n_codes):
        result_len, in_size = x.shape
        out_size, _ = codes.shape
        assert codes.shape == (out_size, in_size)

        output = torch.zeros((result_len, n_codes), dtype=torch.float32, device=x.device)
        output = output.scatter_add(dim=1, index=codes, src=x)
        return output


    def test_Q_tensor(self):
        n_codes = 4
        codes = torch.tensor([[3, 0, 1, 2]], dtype=torch.int64)
        out_size, in_size = codes.shape

        XTX = torch.randn((4, 4))
        XTX = XTX.T @ XTX
        # target = torch.randn((2, 3), dtype=torch.float32)

        target = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

        P_ref, Q_ref = self.get_q_p_ref(XTX, target, codes, n_codes)

        U = target @ XTX
        assert U.shape == (out_size, in_size)

        Q = self.get_x_times_c(U, codes, n_codes)
        assert Q.shape == (out_size, n_codes)

        Q = Q.sum(dim=0)

        torch.testing.assert_close(
            Q,
            Q_ref,
        )

    @staticmethod
    @njit
    def calculate_P(P, XTX, codes, in_size, out_size):
        for alpha in range(out_size):
            for a in range(in_size):
                for b in range(in_size):
                    P[codes[alpha][a]][codes[alpha][b]] += XTX[a][b]

    @staticmethod
    def calculate_P_batch(XTX, codes, n_size, out_size, n_codes):
        out_size, in_size = codes.shape
        P = torch.zeros((out_size, n_codes * n_codes), device=XTX.device)
        XTX_flat = XTX.reshape(1, -1).repeat(out_size, 1)
        indexes = (
                codes.reshape(out_size, in_size, 1) * n_codes +
                codes.reshape(out_size, 1, in_size)
        ).reshape(out_size, in_size * in_size)
        P = P.scatter_add(dim=1, src=XTX_flat, index=indexes)
        P = P.sum(0).reshape(n_codes, n_codes)
        return P

    @torch.no_grad()
    def test_P_tensor(self):
        n_codes = 2
        codes = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64)
        out_size, in_size = codes.shape

        XTX = torch.randn((3, 3))
        XTX = XTX.T @ XTX
        # target = torch.randn((2, 3), dtype=torch.float32)
        target = torch.tensor([[-1, 7, 7], [7, -1, 7]], dtype=torch.float32)

        P_ref, Q_ref = self.get_q_p_ref(XTX, target, codes, n_codes)

        P = torch.zeros((out_size, n_codes * n_codes))
        XTX_flat = XTX.reshape(1, -1).repeat(out_size, 1)

        indexes = (
            codes.reshape(out_size, in_size, 1) * n_codes +
            codes.reshape(out_size, 1, in_size)
        ).reshape(out_size, in_size * in_size)

        P = P.scatter_add(dim=1, src=XTX_flat, index=indexes)
        P = P.sum(0).reshape(n_codes, n_codes)

        P = self.calculate_P_batch(XTX, codes, in_size, out_size, n_codes)

        torch.testing.assert_close(P, P_ref)


    def test_Q(self):
        n_codes = 2
        codes = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64)
        out_size, in_size = codes.shape

        XTX = torch.randn((3, 3))
        XTX = XTX.T @ XTX

        target = torch.tensor([[-1, 7, 7], [7, -1, 7]], dtype=torch.float32)
        C = get_c(codes, n_codes)
        Q_ref = target @ XTX @ C

        Q = torch.zeros((n_codes, out_size))
        U = target @ XTX
        print(U)
        print(Q_ref, Q)


    def test_get_codebooks(self):
        n_codes = 4
        codes = torch.tensor([[3, 0, 1, 2], [0, 1, 2, 3]], dtype=torch.int64)
        out_size, in_size = codes.shape

        XTX = torch.randn((4, 4))
        XTX = XTX.T @ XTX
        # target = torch.randn((2, 3), dtype=torch.float32)

        target = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1]], dtype=torch.float32)

        solution = get_codebooks(codes, target, XTX, nbits_per_codebook=2, in_group_size=1)
        torch.testing.assert_close(
            solution.reshape(-1),
            torch.tensor([2, 3, 4, 1], dtype=torch.float32),
            atol=1e-3,
            rtol=1,
        )


if __name__ == '__main__':
    unittest.main()
