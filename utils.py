import torch
import numpy as np
from sklearn.metrics import accuracy_score

def set_seed(seed):
    """
    设置随机种子,保证实验可复现性
    :param seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, data_loader, device, mode='supervised'):
    """
    在测试集上评估模型性能
    :param model: 模型
    :param data_loader: 测试集数据加载器
    :param device: 设备(CPU或GPU)
    :param mode: 评估模式 ('supervised' 或 'self_supervised')
    :return: 准确率
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for iq_data, labels in data_loader:
            iq_data = iq_data.to(torch.float32).to(device)
            labels = labels.to(torch.int64).to(device)

            if mode == 'supervised':
                logits = model(iq_data, mode='supervised')
                _, predicted = torch.max(logits.data, 1)
            else:
                raise ValueError("Invalid mode: {}".format(mode))

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy