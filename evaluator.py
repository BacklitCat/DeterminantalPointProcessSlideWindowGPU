from torch import Tensor, empty, zeros
from timer import time_loger
from dataset import Recall
from torch.multiprocessing import Pool


class Evaluator:
    """
    这个类负责生产batch数据、评估batch数据（打散算法的成功率）。
    """
    time_gen, time_eval = [0.], [0.]

    def __init__(self, feat_names=None, feat_nums=None, recall_num=20, rule=None, use_sparse=True, device='cuda'):
        self.feat_names = feat_names if feat_names else ['author', 'category', 'music']
        self.feat_nums = feat_nums if feat_nums else [1000, 30, 100]
        self.recall_num = recall_num
        self.rule = rule if rule else [2, 3, 1]
        self.use_sparse = use_sparse
        self.device = device

    @time_loger(time_gen)
    def gen_batch_data(self, batch_size=512) -> tuple:
        """
        生成一批数据。

        :param batch_size: int. 指定一个batch的大小
        :return: tuple(list, list). 两个list，第一个list由数据类组成，第二个由对应的核矩阵组成
        """
        r = Recall(self.feat_names, self.feat_nums, self.device)
        batch_dataset = []
        batch_kernel_matrix = empty((batch_size, self.recall_num, self.recall_num), device=self.device)
        for i in range(batch_size):
            d = r.recall(self.recall_num, use_sparse=self.use_sparse)
            batch_dataset.append(d)
            batch_kernel_matrix[i, :, :] = d.kernel_matrix.to_dense() if self.use_sparse else d.kernel_matrix
        return batch_dataset, batch_kernel_matrix

    @time_loger(time_eval)
    def eval_batch_data(self, batch_dataset, batch_index, reduction='sum') -> Tensor:
        """
        在一批数据上，计算碰撞失败率。

        :param batch_dataset: list. 一个batch的数据。
        :param batch_index: list. 一个batch的数据，对应的打散排序后的顺序。
        :param reduction: str. 碰撞失败率是加和还是求平均。
        :return: Tensor. 碰撞失败率.
        """
        batch_size = len(batch_dataset)
        res = zeros(batch_dataset[0].n_keys, device=self.device)
        for i in range(batch_size):
            res += batch_dataset[i].slide_collide(batch_index[i])
        del batch_dataset, batch_index
        if reduction in ['sum', 'SUM', 'Sum']:
            return res
        elif reduction in ['mean', 'MEAN', 'Mean']:
            return res / batch_size
        else:
            raise ValueError

    def eval_batch_data_mp(self, batch_dataset, batch_index, reduction='sum', n_workers=8, n_cut=8):
        """
        调用多进程，在一批数据上，计算碰撞失败率。

        :param batch_dataset: list. 一个batch的数据。
        :param batch_index: list. 一个batch的数据，对应的打散排序后的顺序。
        :param reduction: str. 碰撞失败率是加和还是求平均。
        :param n_workers: int. 进程池最大进程数量。
        :param n_cut: int. 切片个数。
        :return: Tensor. 碰撞失败率.
        """
        pool = Pool(n_workers)
        obj_list = []
        batch_size = len(batch_dataset)
        res = zeros(batch_dataset[0].n_keys, device=self.device)
        cut_size = round(batch_size/n_cut)
        cut_last = batch_size - cut_size*(n_cut-1)
        left = 0
        for i in range(n_cut):
            thread_batch_size = cut_size if i < n_cut - 1 else cut_last
            obj = pool.apply_async(self.eval_batch_data,
                                   args=(batch_dataset[left:left + thread_batch_size],
                                         batch_index[left:left + thread_batch_size],
                                         'sum'),
                                   error_callback=lambda err: print(f'error：{str(err)}')
                                   )
            left += thread_batch_size
            obj_list.append(obj)
        pool.close()
        pool.join()
        for obj in obj_list:
            res += obj.get()
        if reduction in ['sum', 'SUM', 'Sum']:
            return res
        elif reduction in ['mean', 'MEAN', 'Mean']:
            return res / batch_size
        else:
            raise ValueError
