import torch

class Config:
    def __init__(self):
        # 数据路径
        self.data_path = "./data/20190115_13h17/"


        # 数据增强
        self.data_augmentation = True

        # 模型参数
        self.window_length = 1000
        self.input_size = self.window_length * 2  # 输入样本长度
        self.hidden_size = 128  # 隐藏层维度
        self.proj_size = 64  # 投影头输出维度
        self.num_classes = 21  # 类别数

        # 训练参数
        self.seed = 42  # 随机种子
        self.batch_size = 1024  # 批次大小
        self.num_epochs_unsupervised = 20  # 自监督训练epochs数
        self.num_epochs_supervised = 20  # 监督训练epochs数
        self.learning_rate = 0.01  # 初始学习率
        self.min_learning_rate = 0.0001  # 最小学习率
        self.temperature = 0.2 # 温度参数
        self.alpha = 0.9  # 联合损失函数中的权重系数s
        self.eval_interval = 10  # 评估间隔(epochs数)

        # 模型保存路径
        self.model_path = "./save/model.pth"
        
        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"