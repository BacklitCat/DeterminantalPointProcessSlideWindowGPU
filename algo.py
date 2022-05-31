import torch
from timer import time_loger


class DeterminantalPointProcessSlideWindowGPU:
    time_dpp = [0.]

    def __init__(self, max_length=20, window_size=8, device='cuda'):
        self.max_length = max_length
        self.window_size = window_size
        self.device = device

    @time_loger(time_dpp)
    def __call__(self, batch_kernel_matrix):
        max_length = self.max_length
        window_size = self.window_size
        device = self.device
        kernel_matrix = batch_kernel_matrix
        epsilon = 1e-6
        batch_size = kernel_matrix.shape[0]
        item_num = kernel_matrix.shape[1]
        v = torch.zeros((batch_size, max_length, max_length), device=device)
        cis = torch.zeros((batch_size, max_length, item_num), device=device)
        di2s = kernel_matrix.diagonal(dim1=1, dim2=2).clone()
        selected_items = torch.zeros((batch_size, max_length), device=device)
        selected_item = di2s.argmax(dim=1)
        selected_items[:, 0] = selected_item
        selected_nums = 1
        window_left_index = 0
        while selected_nums < max_length:
            k = selected_nums - 1
            ci_optimal = torch.stack([cis[kk, window_left_index:k, selected_item][:, kk] for kk in range(cis.shape[0])])
            di_optimal = di2s.max(dim=1).values.sqrt()
            v[:, k, window_left_index:k] = ci_optimal
            v[:, k, k] = di_optimal
            elements = kernel_matrix[:, selected_item, :].diagonal(dim1=0, dim2=1).T
            eis = (elements - torch.matmul(ci_optimal.unsqueeze(1), cis[:, window_left_index:k, :]).squeeze(
                1)) / di_optimal.unsqueeze(1)
            cis[:, k, :] = eis
            di2s -= eis.square()
            if selected_nums >= window_size:
                window_left_index += 1
                for ind in range(window_left_index, k + 1):
                    t = (v[:, ind, ind] ** 2 + v[:, ind, window_left_index - 1] ** 2).sqrt()
                    c = t / v[:, ind, ind]
                    s = v[:, ind, window_left_index - 1] / v[:, ind, ind]
                    v[:, ind, ind] = t
                    v[:, ind + 1:k + 1, ind] += s.unsqueeze(1) * v[:, ind + 1:k + 1, window_left_index - 1]
                    v[:, ind + 1:k + 1, ind] /= c.unsqueeze(1)
                    v[:, ind + 1:k + 1, window_left_index - 1] *= c.unsqueeze(1)
                    v[:, ind + 1:k + 1, window_left_index - 1] -= s.unsqueeze(1) * v[:, ind + 1:k + 1, ind]
                    cis[:, ind, :] += s.unsqueeze(1) * cis[:, window_left_index - 1, :]
                    cis[:, ind, :] /= c.unsqueeze(1)
                    cis[:, window_left_index - 1, :] *= c.unsqueeze(1)
                    cis[:, window_left_index - 1, :] -= s.unsqueeze(1) * cis[:, ind, :]
                di2s += (cis[:, window_left_index - 1, :]).square()
            di2s[:, selected_item] = di2s[:, selected_item].fill_diagonal_(torch.tensor(-float('inf')))
            selected_item = di2s.argmax(dim=1)
            if (di2s[:, selected_item].diag() < epsilon).sum() > 0 < epsilon:
                print('Warning, di^2 <0, mat cause error')
                # break
            selected_items[:, selected_nums] = selected_item
            selected_nums += 1
        return selected_items.int().tolist()
