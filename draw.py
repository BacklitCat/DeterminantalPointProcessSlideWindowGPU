import matplotlib.pyplot as plt
import numpy as np
# %%
"""
TEST_NUM: 10240, BATCH_SIZE: 1024
res_monkey: tensor([0.0003, 0.0290, 0.2431]), res_dpp: tensor([0.0000e+00, 0.0000e+00, 2.2536e-05])
time_gen: [1.9000129699707031], time_eval: [132.89892864227295], time_dpp: [2.998013734817505]
"""
x_label = ['author', 'category', 'music']
x = np.arange(len(x_label))

monkey = 1 - np.array([0.0003, 0.0290, 0.2431])
dpp = 1- np.array([0.0000e+00, 0.0000e+00, 2.2536e-05])

fig, ax = plt.subplots()
width = 0.2
rects1 = ax.bar(x - width/2, monkey, width, label='monkey')
rects2 = ax.bar(x + width/2, dpp, width, label='dpp')

for i, rect in enumerate(rects1):
    height = rect.get_height()
    if i == 0:
        plt.text(x=rect.get_x() + rect.get_width() / 2, y=height + 0.01, s=f'{height:.4f}  ', ha='center')
    else:
        plt.text(x=rect.get_x() + rect.get_width() / 2, y=height + 0.01, s=f'{height:.2f}', ha='center')

for i, rect in enumerate(rects2):
    height = rect.get_height()
    if i == 2:
        plt.text(x=rect.get_x() + rect.get_width() / 2, y=height + 0.01, s=f'{height:.6f}', ha='center')
    else:
        plt.text(x=rect.get_x() + rect.get_width() / 2, y=height + 0.01, s=f'{height:.2f}', ha='center')

ax.set_ylabel('Success rate (%)')
ax.set_title('Monkey vs. DPP Slide Window Algorithm\n(Number of tests: 10240)')
ax.set_xticks(x)
ax.set_xticklabels(x_label)
ax.legend(loc='lower left')

plt.savefig('./pic/Monkey vs. DPP Slide Window Algorithm.png')
plt.show()

#%%
"""
[GPU] GP102 P40 = 1080Ti
TEST_NUM: 10240, BATCH_SIZE: 1024
res_monkey: tensor([0.0003, 0.0290, 0.2431]), res_dpp: tensor([0.0000e+00, 0.0000e+00, 2.2536e-05])
time_gen: [1.9000129699707031], time_eval: [132.89892864227295], time_dpp: [2.998013734817505]

[CPU] i7-11700 4.9Ghz 
TEST_NUM: 10240, BATCH_SIZE: 1
res_dpp: tensor([4.5072e-05, 4.6875e-03, 3.4901e-02])
time_gen: [1.8594985008239746], time_eval: [34.645976305007935], time_dpp: [179.23463892936707]
"""
x_label = ['i7-11700 4.9Ghz', 'GP102 TESLA P40']
x = np.arange(len(x_label))
y = np.array([179.23463892936707, 2.998013734817505])

plt.bar(x, y, width=0.5, color=['C0', 'C1'])
for i in range(len(x_label)):
    plt.text(x=x[i], y=y[i] + 1, s=f'{y[i]:.6f}', ha='center')
plt.text(x=x[i], y=100, s=f'{(y[i-1]-y[i])/y[i]:.2%} up↑', ha='center')


plt.xticks(x, x_label)
plt.ylabel('Time (s)')
plt.title('CPU vs. GPU\n(Number of tests: 10240)')
plt.xlim(-1, 2)

plt.savefig('./pic/CPU vs. GPU.png')
plt.show()

#%%
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
x_label = ['i5-8250U 3.4Ghz', 'i7-11700 4.9Ghz', 'GP102 TESLA P40']
x = np.arange(len(x_label))
y = np.array([338.82056999206543, 179.23463892936707, 2.998013734817505])

plt.bar(x, y, width=0.5, color=['C2', 'C0', 'C1'])
for i in range(len(x_label)):
    plt.text(x=x[i], y=y[i] + 1, s=f'{y[i]:.4f}', ha='center')
    if i > 0:
        plt.text(x=x[i], y=(y[i-1]+y[i])/2, s=f'{(y[i-1]-y[i])/y[i]:.2%} up↑', ha='center')
plt.text(x=x[i], y=258, s=f'{(y[i-2]-y[i])/y[i]:.2%} up↑', ha='center')


plt.xticks(x, x_label, rotation=-6)
plt.ylabel('Time (s)')
plt.title('CPU vs. GPU\n(Number of tests: 10240)')
plt.xlim(-1, 3)

plt.savefig('./pic/CPU vs. GPU_.png')
plt.show()
