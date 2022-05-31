from tqdm import trange
from torch import zeros
from evaluator import Evaluator
from algo import DeterminantalPointProcessSlideWindowGPU

"""
[GPU] GP102 P40 = 1080Ti
TEST_NUM: 10240, BATCH_SIZE: 1024
res_monkey: tensor([0.0003, 0.0290, 0.2431]), res_dpp: tensor([0.0000e+00, 0.0000e+00, 2.2536e-05])
time_gen: [1.9000129699707031], time_eval: [132.89892864227295], time_dpp: [2.998013734817505]

[CPU] i7-11700 4.9Ghz 
TEST_NUM: 10240, BATCH_SIZE: 1
res_dpp: tensor([4.5072e-05, 4.6875e-03, 3.4901e-02])
time_gen: [1.8594985008239746], time_eval: [34.645976305007935], time_dpp: [179.23463892936707]

[CPU] i5-8250U 3.4Ghz
TEST_NUM: 10240
res_dpp: tensor([7.5120e-06, 4.5448e-03, 3.4766e-02])
time_gen: [8.80226993560791], time_eval: [764.6965029239655], time_dpp: [338.82056999206543]
"""

if __name__ == '__main__':
    """参数"""
    TEST_NUM = 10240
    FEAT_NAMES = ['author', 'category', 'music']
    FEAT_NUMS = [1000, 30, 100]
    RECALL_NUM = 20
    RULE = [2, 3, 1]
    DEVICE = 'cpu'
    USE_SPARSE = True

    """谈个对象"""
    evaluator = Evaluator(FEAT_NAMES, FEAT_NUMS, RECALL_NUM, RULE, USE_SPARSE, DEVICE)
    dpp_slide_window = DeterminantalPointProcessSlideWindowGPU(max_length=20, window_size=8, device='cpu')

    """run"""
    res_dpp = zeros(len(FEAT_NAMES), device=DEVICE)

    for i in trange(TEST_NUM):
        # 生成数据
        batch_dataset, batch_kernel_matrix = evaluator.gen_batch_data(batch_size=1)
        # DPP滑窗算法 CPU运行
        index = dpp_slide_window(batch_kernel_matrix)
        res_dpp += evaluator.eval_batch_data(batch_dataset, index, reduction='sum')

    """result"""
    res_dpp /= TEST_NUM
    print(f'TEST_NUM: {TEST_NUM}')
    print(f'res_dpp: {res_dpp}')
    print(f'time_gen: {evaluator.time_gen}, time_eval: {evaluator.time_eval}, time_dpp: {dpp_slide_window.time_dpp}')
