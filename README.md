# Determinantal Point Process Slide Window GPU
行列式点过程(DPP)算法滑窗版本的GPU实现

## Requirements
```
tqdm
pytorch
```
因为涉及到大量的矩阵运算，并且为了使用Cuda加速，所以选择PyTorch，希望利用其Tensor类，作为一个调用Cuda的高级接口，所以没有选用C++/numpy。其实，算子都是拿C/C++写的，这样兼顾了开发效率和运行效率。

## How to use
```bash
$ python run_dpp_cpu.py
TEST_NUM: 10240
res_dpp: tensor([4.5072e-05, 4.6875e-03, 3.4901e-02])
time_gen: [1.8594985008239746], time_eval: [34.645976305007935], time_dpp: [179.23463892936707]

$ python run_monkey_dpp_gpu.py
TEST_NUM: 10240, BATCH_SIZE: 1024
res_monkey: tensor([0.0003, 0.0290, 0.2431]), res_dpp: tensor([0.0000e+00, 0.0000e+00, 2.2536e-05])
time_gen: [1.9000129699707031], time_eval: [132.89892864227295], time_dpp: [2.998013734817505]
```


## Architecture
![Architecture](./pic/Architecture.svg)

### dataset.py
1. RecallData 数据类的基类。所有特征域构成一个dict。
   1. \_\_init__ 初始化
   2. \_\_len__ 返回recall数量
   3. \_\_getitem__ 对于recall后的第i个视频，返回其特征域的dict。
   4. window_collide 抽象方法，子类必须根据自己的存储格式，具体计算窗口内collide数量。
   5. slide_collide 根据window_collide，滑窗计算所有特征域的规则碰撞率。
2. RecallDataSparse 基于稀疏矩阵存储的数据类
   1. feat_stack 解决稀疏矩阵切片访问问题，给定具体特征域和下标index，返回在0轴堆叠好的稀疏矩阵。
   2. feats_stack 返回所有特征域的特定切片。
   3. i 简化feats_stack方法名，提供一个友善的语法糖。
   4. sparse_feat_mm (query,key)-collide。某一特征query向量到window向量碰撞的次数。
   5. sparse_feats_mm (query,key)-collide。所有特征query向量到window向量碰撞的次数。
   6. query_collide (query,key)-collide。简化sparse_feats_mm方法名，提供一个友善的语法糖。
   7. window_collide self-collide。计算窗口内的collide数量。
   8. build_kernel_matrix 创建DPP核矩阵。
3. RecallDataDense 基于稠密矩阵存储的数据类。
   1. window_collide self-collide。计算窗口内的collide数量。
   2. build_kernel_matrix 创建DPP核矩阵。
4. Recall 该类模拟召回行为。每调用一次recall方法，生成一次recall视频数据。
   1. \_\_init__ 初始化
   2. recall 调用recall方法，返回一个RecallData类。

### evaluator.py
1. Evaluator 这个类负责生产batch数据、评估batch数据（打散算法的成功率）。
   1. \_\_init__ 初始化
   2. gen_batch_data 生成一批数据。
   3. eval_batch_data 在一批数据上，计算碰撞失败率。
   4. eval_batch_data_mp 调用多进程，在一批数据上，计算碰撞失败率。

### algo.py
1. DeterminantalPointProcessSlideWindowGPU 算法类
   1. \_\_init__ 初始化
   2. \_\_call__ 调用算法

### timer.py
1. time_logger 含参的双层计时装饰器

## Experimental Parameter
```python
TEST_NUM = 10240
FEAT_NAMES = ['author', 'category', 'music']
FEAT_NUMS = [1000, 30, 100]
RECALL_NUM = 20
RULE = [2, 3, 1]
```

## Contribution
1. 实现了一个比较通用的数据类和召回类，存储方式可以选择稀疏矩阵或稠密矩阵，固定窗口碰撞率、滑动窗口碰撞率都可以直接调用类方法快捷计算。
2. 实现了DPP滑窗打散算法，对比猴子随机打散baseline，提高了打散成功率。

![Monkey vs. DPP Slide Window Algorithm](./pic/Monkey%20vs.%20DPP%20Slide%20Window%20Algorithm.png)

3. 实现了可在GPU上并行计算的DPP滑窗算法。CPU代际升级也很难提升1倍的速度，而GPU实现对比CPU实现可提速 58.8 ~ 112.0 倍。

![CPU vs. GPU](./pic/CPU%20vs.%20GPU.png)

## Others
1. 评估（计算规则碰撞率）比较耗时，实现了多进程评估，但目前multiprocessing和torch.multiprocessing开多进程还有一些问题，如，稀疏矩阵暂不支持被multiprocessing序列化，多进程cuda tensor共享问题等等，所以，虽然实现了多进程并发，但评估时并未使用。
2. pytorch目前对稀疏矩阵的支持较差，如果需要使用稀疏矩阵存储格式，pytorch低版本可能报错缺失算子。
