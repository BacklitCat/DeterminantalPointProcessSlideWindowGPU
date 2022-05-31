import math
from tqdm import trange
from torch import zeros
from evaluator import Evaluator
from algo import DeterminantalPointProcessSlideWindowGPU

"""
TEST_NUM: 10240, BATCH_SIZE: 512
res_monkey: tensor([0.0004, 0.0300, 0.2443]), res_dpp: tensor([0.0000e+00, 0.0000e+00, 3.7560e-05])
time_gen: [1.8111028671264648], time_eval: [131.89220452308655], time_dpp: [3.1590394973754883]

TEST_NUM: 10240, BATCH_SIZE: 1024
res_monkey: tensor([0.0003, 0.0290, 0.2431]), res_dpp: tensor([0.0000e+00, 0.0000e+00, 2.2536e-05])
time_gen: [1.9000129699707031], time_eval: [132.89892864227295], time_dpp: [2.998013734817505]
"""

if __name__ == '__main__':
    """参数"""
    TEST_NUM = 10240
    BATCH_SIZE = 1024
    FEAT_NAMES = ['author', 'category', 'music']
    FEAT_NUMS = [1000, 30, 100]
    RECALL_NUM = 20
    RULE = [2, 3, 1]
    DEVICE = 'cuda'
    USE_SPARSE = False

    """谈个对象"""
    evaluator = Evaluator(FEAT_NAMES, FEAT_NUMS, RECALL_NUM, RULE, USE_SPARSE, DEVICE)
    dpp_slide_window = DeterminantalPointProcessSlideWindowGPU(max_length=20, window_size=8, device='cuda')

    """run"""
    iter_num = math.ceil(TEST_NUM / BATCH_SIZE)
    batch_last = TEST_NUM - BATCH_SIZE * (iter_num-1)
    res_monkey, res_dpp = zeros(len(FEAT_NAMES), device=DEVICE), zeros(len(FEAT_NAMES), device=DEVICE)
    for i in trange(iter_num):
        batch_size = BATCH_SIZE if i < iter_num-1 else batch_last
        # 生成数据
        batch_dataset, batch_kernel_matrix = evaluator.gen_batch_data(batch_size=batch_size)
        # 猴子打散算法
        res_monkey += evaluator.eval_batch_data(batch_dataset, [list(range(20)) for _ in range(batch_size)], reduction='sum')
        # DPP滑窗算法
        index = dpp_slide_window(batch_kernel_matrix)
        res_dpp += evaluator.eval_batch_data(batch_dataset, index, reduction='sum')

    """result"""
    res_monkey /= TEST_NUM
    res_dpp /= TEST_NUM
    print(f'TEST_NUM: {TEST_NUM}, BATCH_SIZE: {BATCH_SIZE}')
    print(f'res_monkey: {res_monkey}, res_dpp: {res_dpp}')
    print(f'time_gen: {evaluator.time_gen}, time_eval: {evaluator.time_eval}, time_dpp: {dpp_slide_window.time_dpp}')
