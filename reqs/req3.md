## CR-1
- run_case.sh生成的ouputs下的目录请分成base版本文件夹和para版本文件夹
- 记录下每个训练文件夹（20260226_093635）下，本次训练的步数，以及模型里面实际的训练步数（比如从flow_network.0.pth获取）。

## CR-2
/home/featurize/work实际上挂载的是一个远程的硬盘，所以访问会很慢，所以我想将训练过程中的文件夹放到/home/featurize/data下，训练完之后再拷贝回来。请支持这个需求，详细需求如下：
- 命令里可以支持配置是否要放到/home/featurize/data下，并拷贝回来。
- 提供一个单独的命令，支持将/home/featurize/data拷贝回来（主要考虑训练中途中断的例子）。
- 至于/home/featurize/data下文件夹的结构，你可以来设计一下。