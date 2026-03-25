## CR-1
### 背景和需求
当前本项目跑通的是three_fin_2d，输入的风速是固定的，当前的任务是将风速参数化，也就是可以作为神经网络的输入参数，获得特定输入风速下，模型输出场内不同的点的风速。
基本的思路可以参考docs/PINN_Study_Notes.pdf的章节Chapter 8 Parameterisation (inlet_vel Example)，但这个是基于我之前的学习和理解，请把它当作输入，但是不要全信。
其实有一个现成的example做参数化：https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/advanced/parametrized_simulations.html ， 这是一个3D的heat sink，你可以参考这个页面的解释，以及下载实际的代码进行参考。

### 交付物
- 基于three_fin_2d的例子进行修改，形成新的代码软件
- 测试脚本，用来测试新训练出来的模型效果是否OK

### 完成标准
- run_case里可以加入参数执行不同的软件，带参数的和不带参数的。output也要做对应的调整。
- 先保证执行2轮成功，后续我自己来，但告诉我新模型大概是要训练几轮比较合适。
- 测试脚本需要能够一步执行，获得比较结果，方式是和原有的模型输出做比较，如果你有更好的办法也可以提出。