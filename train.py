import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import CLModel
from loss import SelfSupervisedLoss, SupervisedLoss
from utils import set_seed, evaluate
from data import IQDataset
from config import Config
from tqdm import tqdm

class IQAugmentation:
    def __init__(self, noise_std=0.01, time_shift_max=10, scale_range=(0.8, 1.2)):
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.scale_range = scale_range

    def __call__(self, iq_data):
        # 噪声注入
        noise = torch.randn_like(iq_data) * self.noise_std
        iq_data = iq_data + noise

        # 时移
        time_shift = torch.randint(-self.time_shift_max, self.time_shift_max, (1,)).item()
        iq_data = torch.roll(iq_data, shifts=(time_shift,), dims=1)

        # 缩放
        scale = torch.rand(1,device=config.device) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        iq_data = iq_data * scale

        return iq_data

def train(config):
    # 设置随机种子
    set_seed(config.seed)

    # 定义数据增强操作
    data_augmentation = IQAugmentation()

    # 加载数据集
    dataset = IQDataset(config.data_path)
        # 划分训练集、测试集和验证集

    train_dataset, test_dataset, val_dataset = random_split(
        dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False)


    # 初始化模型
    model = CLModel(config.input_size, config.hidden_size, config.proj_size, config.num_classes)
    model = model.to(config.device)

    # 定义损失函数和优化器
    criterion_ssl = SelfSupervisedLoss(config.temperature)
    criterion_sl = SupervisedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 自监督训练阶段
    for epoch in range(config.num_epochs_unsupervised):
        model.train()
        total_loss = 0.0
        for iq_data, _ in tqdm(train_loader):
            iq_data = iq_data.to(torch.float32).to(config.device)
            batch_size = iq_data.size(0)

            # 生成两个增强视图
            iq_data1 = data_augmentation(iq_data)
            iq_data2 = data_augmentation(iq_data)

            optimizer.zero_grad()
            proj_1 = model(iq_data1, mode='self_supervised')
            proj_2 = model(iq_data2, mode='self_supervised')
            loss = criterion_ssl(proj_1, proj_2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Unsupervised Epoch [{epoch+1}/{config.num_epochs_unsupervised}], Loss: {total_loss / len(train_loader):.4f}")

    # 监督训练阶段
    for epoch in range(config.num_epochs_supervised):
        model.train()
        total_loss = 0.0

        for iq_data, labels in tqdm(train_loader):
            iq_data = iq_data.to(torch.float32).to(config.device)
            labels = labels.to(torch.int64).to(config.device)
            optimizer.zero_grad()
            logits = model(iq_data, mode='supervised')
            loss = criterion_sl(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Supervised Epoch [{epoch+1}/{config.num_epochs_supervised}], Loss: {total_loss / len(train_loader):.4f}")

        # 在训练集上评估模型
        if (epoch + 1) % config.eval_interval == 0:
            accuracy = evaluate(model, train_loader, config.device, mode='supervised')
            print(f"Train Accuracy: {accuracy:.4f}")

                    # 在验证集上评估模型
        if (epoch + 1) % config.eval_interval == 0:
            accuracy = evaluate(model, val_loader, config.device)
            print(f"Validation Accuracy: {accuracy:.4f}")

    # 在测试集上评估模型
    accuracy = evaluate(model, test_loader, config.device)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), config.model_path)

if __name__ == "__main__":
    config = Config()
    print(config.device)
    train(config)