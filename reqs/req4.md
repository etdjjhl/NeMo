当前基于three_fin_2d的例子，有两套代码，一个是原始的heat_sink.py，还有一个是把入口风速参数化的heat_sink_param.py。现在想要加一个新的，基于原始的heat_sink.py将时间参数化，也就是说原本模型推理的是稳态，现在想要推理瞬态（Transient）。

你可以通过下面两种方式获得参考：
- https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html， 这个可以帮助你理解如何讲时间参数化，但不是完全照抄，因为这不是NS方程
- 自行搜索nvidia physics nemo相关的说明文档，来思考如何将时间参数化

你还需要思考如何验证这个瞬时模型的结果，比如：
- 将最终达到稳态的时间输入模型，将输出的结果和官方的openfoam结果做对比
- 多采集几个瞬时的点，对于趋势变化做判断
- 你更聪明，如果能想到更精确又省力的方法的话，那就最好了

交付物：
- 基于three_fin_2d的例子进行修改，形成新的代码软件
- 测试脚本，用来测试新训练出来的模型效果是否OK：有图片和有报告

其他要求：
- 符合当前的整体软件架构，比如入口是run_case.sh，outputs下等分文件夹，config.yaml文件得分开
- 在config.yaml里面，设计合理的batch值
- 监控的moitor看情况是否需要调整

请记住，你是专业的神经网络工程师，同时也是专业的软件架构师，请充分发挥你的聪明才智。