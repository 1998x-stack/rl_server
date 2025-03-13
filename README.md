# rl_server
1 lib 通用功能文件夹。net,redis,config,utils
2 check_main 检测训练模型的指标
2 sample_main 采样用。对应于AI server
3 train_main 本地自带sample 训练主入口。用于启动 trainer。train_main_local 本地服务器，train_main_redis redis 带 sample 服务器,train_main_grad redis 带 grad 服务器
4 model 保存训练模型的参数
5 proto 协议的proto定义
    1) state: bytes
    2) action: bytes
    3) reward: bytes
    4) mask: bytes
    5) done: int32
    6) prob: bytes
    7) version: int32
6 logs 运行时 日志目录
7 grads_main grads 整合服务器。 train_main 上传 梯度