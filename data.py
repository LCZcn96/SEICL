import numpy as np
from torch.utils.data import Dataset
from config import Config


data_path = "./data/20190115_13h17/"
config = Config()

def read_data(data_path):
    nb_tx = 21
    nb_rx = 1
    tx_offset = 0
    offset = 0
    file_re = [[]]
    file_im = [[]]
    for j in range(tx_offset,nb_tx+tx_offset):
        file_re.append([])
        file_im.append([])
        for i in range(offset, nb_rx+offset):
            file_re[j-tx_offset].append('{0}re_{1:02d}_{2}.bin'.format(data_path, i,j))
            file_im[j-tx_offset].append('{0}im_{1:02d}_{2}.bin'.format(data_path, i,j))
            
    packet_size = 600

    cut_proportion = 1 #If we don't want to use the whole dataset
    print ("Nb de Tx : ", nb_tx)

    for rx in range(nb_rx):
        valeur_re = np.array([])
        valeur_im = np.array([])
        label = None 
        for tx in range(nb_tx):

            tmp_re = np.fromfile(file_re[tx][rx], dtype=np.float32, count=-1)
            tmp_im = np.fromfile(file_im[tx][rx], dtype=np.float32, count=-1)
            nb_p = (tmp_im.shape[0]//packet_size)
            cut_pos = int(nb_p * cut_proportion)
            
            valeur_re = np.append(valeur_re, tmp_re[:packet_size*(cut_pos)]) #Ensure case where files have partially written packets at the end      
            valeur_im = np.append(valeur_im, tmp_im[:packet_size*(cut_pos)])
            
            if label is None:
                label = np.stack(((tx)*np.ones(cut_pos), (rx)*np.ones(cut_pos)), axis = 1)
            else:
                label = np.append(label, np.stack(((tx)*np.ones(cut_pos), (rx)*np.ones(cut_pos)), axis = 1) , axis=0)
                
        nb_packets = valeur_re.shape[0] //packet_size

        if (rx == 0):
            vecteur_re = valeur_re[:packet_size*nb_packets].reshape(nb_packets, packet_size)
            vecteur_im = valeur_im[:packet_size*nb_packets].reshape(nb_packets, packet_size)
            labels = label

        else:
            vecteur_re = np.concatenate((vecteur_re, valeur_re[:packet_size*nb_packets].reshape(nb_packets, packet_size)), axis=0)
            vecteur_im = np.concatenate((vecteur_im, valeur_im[:packet_size*nb_packets].reshape(nb_packets, packet_size)), axis=0)
            labels = np.concatenate((labels, label), axis=0)
        
        valeur_re = valeur_re[:packet_size*nb_packets].reshape(nb_packets, packet_size)
        valeur_im = valeur_im[:packet_size*nb_packets].reshape(nb_packets, packet_size)
        labels_1d = labels[:, 0]

        return valeur_re,valeur_im,labels_1d

# 定义数据集类
class IQDataset(Dataset):
    def __init__(self, data_path, fixed_length=config.window_length, transform=None):
        self.data_re, self.data_im, self.labels = read_data(data_path)
        self.fixed_length = fixed_length
        self.transform = transform

    def __len__(self):
        return len(self.data_re)

    def __getitem__(self, idx):
        iq_data_re = self.data_re[idx]
        iq_data_im = self.data_im[idx]
        label = self.labels[idx]

        iq_data_re = self.pad_or_truncate(iq_data_re, self.fixed_length)
        iq_data_im = self.pad_or_truncate(iq_data_im, self.fixed_length)

        iq_data = np.hstack((iq_data_re, iq_data_im))

        if self.transform:
            iq_data = self.transform(iq_data)

        return iq_data, label

    def pad_or_truncate(self, data, length):
        if len(data) < length:
            pad_len = length - len(data)
            data = np.pad(data, (0, pad_len), mode='constant')
        else:
            data = data[:length]
        return data