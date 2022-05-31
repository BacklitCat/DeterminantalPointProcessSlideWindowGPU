from abc import abstractmethod
from typing import Union
from torch import Tensor, tensor, randint, stack, cat, sparse, mm, where, empty, int
from torch.nn.functional import one_hot


class RecallData:
    def __init__(self, **tensor_dict):
        """
        data成员负责存储每一次召回的视频数据。结构：{feat_name: feat_vector(video_num,emb_dim)}
        """
        self.data = tensor_dict
        self.device = tensor_dict.values().__iter__().__next__().device
        self.keys = list(tensor_dict.keys())
        self.n_keys = len(self.keys)
        self.n_data = len(self.data[self.keys.__iter__().__next__()])
        self.kernel_matrix = self.build_kernel_matrix()

    def __len__(self) -> int:
        """
        返回recall数量。
        """
        return self.n_data

    def __getitem__(self, i) -> dict:
        """
        对于recall后的第i个视频，返回其特征域的dict。
        :param i: 下标。
        :return: Tensor
        """
        return {k: v[i] for k, v in self.data.items()}

    @abstractmethod
    def window_collide(self, index: list) -> dict:
        pass

    def slide_collide(self, index: list, window_size: int = 8, rule=None) -> Tensor:
        """
        该函数滑动窗口，计算碰撞规则成功率。

        :param index: list. 排好序的视频下标。
        :param window_size: int. 滑动窗口大小
        :param rule: list. 打散规则。
        :return: Tensor. 每个特征的规则碰撞(打散失败)率
        """
        video_num = len(index)
        rule = tensor(rule) if rule else tensor([2, 3, 1])
        slide_num = video_num - window_size + 1
        feat_num = len(self.data)
        collide_num = empty((feat_num, slide_num), dtype=int, device=self.device)
        for j in range(slide_num):
            running_collide = self.window_collide(index[j:j + window_size])
            for i, feat in enumerate(self.keys):
                collide_num[i][j] = (running_collide[feat] > rule[i] - 1)
        return collide_num.sum(dim=1) / slide_num

    @abstractmethod
    def build_kernel_matrix(self):
        pass


class RecallDataSparse(RecallData):
    """
    该类负责实现数据结构，以one hot sparse形式存储。
    """

    def __init__(self, **tensor_dict):
        """
        data成员负责存储每一次召回的视频数据。结构：{feat_name: feat_vector(video_num,emb_dim)}
        """
        super(RecallDataSparse, self).__init__(**tensor_dict)

    def feat_stack(self, feat, index) -> Tensor:
        """
        仅限存储格式为稀疏矩阵使用，解决稀疏矩阵切片访问问题。\n
        给定具体特征域和下标index，返回在0轴堆叠好的稀疏矩阵。

        :param feat: str. 特征。
        :param index: index，指定访问下标。
        :return: sparse Tensor. 稀疏张量。
        """
        return stack([self.data[feat][i] for i in index])

    def feats_stack(self, index) -> dict:
        """
        仅限存储格式为稀疏矩阵使用，解决稀疏矩阵切片访问问题。 \n
        给定下标index，返回所有特征域在0轴堆叠好的稀疏矩阵词典。

        :param index: 可迭代对象，指定访问下标。
        :return: sparse Tensor dict. 稀疏张量词典。
        """
        return {feat: self.feat_stack(feat, index) for feat in self.keys}

    def i(self, *index) -> dict:
        """
        简化feats_stack方法名，提供一个友善的语法糖。

        Example
        ---------
        d.i(0)         d.i([0]) \n
        d.i(0,1,3)     d.i([0,1,3])


        :param index: list或多个int参数，指定访问下标。
        :return: sparse Tensor dict. 稀疏张量词典。
        """
        return self.feats_stack(index[0]) if type(index[0]) is list else self.feats_stack(index)

    @staticmethod
    def sparse_feat_mm(query: Tensor, window: Tensor) -> Tensor:
        """
        Example
        ----------
        query = d.feat_stack('music', [0, 1]) \n
        window = d.feat_stack('music', [2, 3, 4]) \n
        sparse_feat_mm(query, window) \n
        output: tensor([0., 0.], device='cuda:0')

        :param query: Tensor. 查询向量。
        :param window: Tensor. 窗口内向量
        :return: Tensor. 每个查询向量碰撞的次数。
        """
        return sparse.mm(query, window.transpose(0, 1)).to_dense().sum(dim=1)

    def sparse_feats_mm(self, query_dict: dict, window_dict: dict) -> dict:
        """
        Example
        -------
        sparse_feats_mm(d.i([0,2]),d.i([1,2,3])) \n
        output: \n
        {'author': tensor([0., 1.], device='cuda:0'), \n
         'category': tensor([0., 1.], device='cuda:0'), \n
         'music': tensor([0., 1.], device='cuda:0')}

        :param query_dict: dict. 查询向量词典。
        :param window_dict: dict. 窗口向量词典。
        :return: dict. 所有特征下每个查询向量碰撞的次数。
        """
        return {k: self.sparse_feat_mm(query_dict[k], window_dict[k]) for k in self.keys}

    def query_collide(self, query_list: list, window_list: list) -> dict:
        """
        简化sparse_feats_mm方法名，提供一个友善的语法糖。

        Example
        -----------
        print(d.query_collide([2, 2, 3], [1, 2, 2, 3])) \n
        output: \n
        {'author': tensor([2., 2., 1.], device='cuda:0'), \n
        'category': tensor([2., 2., 1.], device='cuda:0'), \n
        'music': tensor([4., 4., 4.], device='cuda:0')} \n

        :param query_list: list. 查询向量的id list。
        :param window_list: dict. 窗口向量的的id list。
        :return: dict. 所有特征下每个查询向量碰撞的次数。
        """
        return self.sparse_feats_mm(self.i(query_list), self.i(window_list))

    def window_collide(self, index: list) -> dict:
        """
        不同于(query,key)型的collide，该函数计算self-collide，用来计算窗口内的collide数量。

        Example
        --------
        print(d.window_collide([1,1,2,1])) \n
        output: \n
        {'author': tensor(2., device='cuda:0'), \n
        'category': tensor(2., device='cuda:0'), \n
        'music': tensor(2., device='cuda:0')}

        :param index: list. 指定窗口内视频id。
        :return: dict. 所有特征的collide数量词典。
        """
        res = {k: 0 for k in self.keys}
        for k in self.data.keys():
            t = self.i(index)[k].to_dense().sum(dim=0)
            res[k] = where(t > 1, t - 1, tensor(0., device=t.device)).sum()
        return res

    def build_kernel_matrix(self):
        """
        创建DPP核矩阵。
        """
        feat_matrix = cat([self.data[k] for k in self.keys], dim=1) / len(self.keys)
        kernel_matrix = sparse.mm(feat_matrix, feat_matrix.transpose(0, 1))
        return kernel_matrix


class RecallDataDense(RecallData):
    """
    该类负责实现数据结构，以one hot dense形式存储。
    """

    def __init__(self, **tensor_dict):
        """
        data成员负责存储每一次召回的视频数据。结构：{feat_name: feat_vector(video_num,emb_dim)}
        """
        super(RecallDataDense, self).__init__(**tensor_dict)

    def window_collide(self, index: list) -> dict:
        """
        不同于(query,key)型的collide，该函数计算self-collide，用来计算窗口内的collide数量。

        Example
        --------
        print(d.window_collide([1,1,2,1])) \n
        output: \n
        {'author': tensor(2., device='cuda:0'), \n
        'category': tensor(2., device='cuda:0'), \n
        'music': tensor(2., device='cuda:0')}

        :param index: list. 指定窗口内视频id。
        :return: dict. 所有特征的collide数量词典。
        """
        res = {k: 0 for k in self.keys}
        for k in self.keys:
            t = self.__getitem__(index)[k].sum(dim=0)
            res[k] = where(t > 1, t - 1, tensor(0., device=t.device)).sum()
        return res

    def build_kernel_matrix(self) -> Tensor:
        """
        创建DPP核矩阵。
        """
        feat_matrix = cat([self.data[k] for k in self.keys], dim=1) / len(self.keys)
        kernel_matrix = mm(feat_matrix, feat_matrix.transpose(0, 1))
        return kernel_matrix


class Recall:
    """
    该类模拟召回行为。每调用一次recall方法，生成一次recall视频数据。
    """

    def __init__(self, feat_names: list = None, feat_nums: list = None, device: str = 'cpu'):
        """
        :param feat_names: list. 特征名。默认: ['author', 'category', 'music']
        :param feat_nums:  list. 对应的特征数量。默认: [1000, 30, 100]
        :param device: str. 计算设备。默认：'cuda'
        """
        self.feat_names = feat_names if feat_names else ['author', 'category', 'music']
        self.feat_nums = feat_nums if feat_nums else [1000, 30, 100]
        self.device = device

    def recall(self, recall_num: int = 20, use_sparse=True) -> Union[RecallDataSparse, RecallDataDense]:
        """
        调用recall方法，返回一个RecallData类。
        :param recall_num: int. 每批次召回视频数量。默认：20
        :param use_sparse: bool. 是否存储为稀疏矩阵。默认：True
        :return: RecallData类。
        """
        tensor_dict = {name: one_hot(randint(0, nums, (recall_num,), device=self.device), nums).float()
                       for name, nums in zip(self.feat_names, self.feat_nums)}

        if use_sparse:
            tensor_dict = {k: v.to_sparse() for k, v in tensor_dict.items()}
            return RecallDataSparse(**tensor_dict)
        else:
            return RecallDataDense(**tensor_dict)


if __name__ == '__main__':
    pass
