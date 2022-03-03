import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from graph import get_adjacent_matrix
from torch.utils.data import DataLoader

def get_flow_data(flow_file: str) -> np.array:   # 这个是载入流量数据,返回numpy的多维数组
    """
    :param flow_file: str, path of .npz file to save the traffic flow data
    :return:
        np.array(N, T, D)
    """
    data = np.load(flow_file)

    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  # [N, T, D],transpose就是转置，让节点纬度在第0位，N为节点数，T为时间，D为节点特征
    # [:, :, 0]就是只取第一个特征，[:, :, np.newaxis]就是增加一个维度，因为：一般特征比一个多，即使是一个，保持这样的习惯，便于通用的处理问题

    return flow_data  # [N, T, D]

class LoadData(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
        """
        :param data_path: list, ["graph file name", "flow data file name"], path to save the data file names.
        :param num_nodes: int, number of nodes
        :param divide_days: list, [days of train data, days of test data], list to divide to original data
        :param time_interval: int, time interval between two traffic data records(mins)
        :param history_length: int , length of history data to be used
        :param train_mode: list, ["train", "test"]
        """
        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]  # 59-14 = 45 train_data
        self.test_days = divide_days[1]   # 7*2 = 14 test_data
        self.history_length = history_length  # 30/5 = 6
        self.time_interval = time_interval  # 5 min

        self.one_day_length = int(24*60/self.time_interval)

        self.graph = get_adjacent_matrix(distance_file = data_path[0], num_nodes = num_nodes)
        self.flow_norm, self.flow_data = self.pre_process_data(data = get_flow_data(data_path[1]), norm_dim =1)

    def __len__(self):  # size of dataset
        """
        :return: length of dataset (number of samples)
        """
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length # size of train dataset = train - history
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length  # test = test
        else:
            raise ValueError("Train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # get sample (x, y), index = [0, L1-1]
        """
        :param item: int, range between [ 0, len-1]
        :return:
            graph: torch.tensor,[N,N]
            data_x: torch.tensor, [N, H, D]
            data_y: torch.tensor, [N, 1, D]
        """
        if self.train_mode =="train":
            index = index
        elif self.train_mode == "test":
            index += self.train_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))
        data_x, data_y = LoadData.slice_Data(self.flow_data, self.history_length, index, self.train_mode)
        data_x = LoadData.to_tensor(data_x)  #(N, H, D)
        data_y = LoadData.to_tensor(data_y).unsqueeze(1) #(N, 1, D)
        return {"graph": LoadData.to_tensor(self.graph), "flow_x":data_x, "flow_y":data_y}
    @staticmethod
    def slice_Data(data, history_length, index, train_mode): #devide the size of dataset based on the history
        """
        :param data: np.array, normalized traffic data
        :param history_length: int, length of history dat tobe used
        :param index: int, index on temporal axis
        :param train_mode: str, ["train", "test"]
        :return:
            data_X: np.array, [N, H, D]
            data_y: np.array, [N, D]
        """
        if train_mode =="train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))
        data_x = data[:, start_index: end_index]
        data_y = data[:, end_index]

        return data_x, data_y
    @staticmethod
    def pre_process_data(data, norm_dim):   # normanized data
        """
        :param data: np.array
        :param norm_dim: int, normalized, dim = 1
        :return:
            norm_base:  list, [max_Data, min_data]
            norm_data: np.array, normalized data
        """
        norm_base = LoadData.normalize_base(data, norm_dim)
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):  #normlized base
        """
        :param data: np.array
        :param norm_dim: int, normalization dimension
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims = True)   #[N, T, D], norm_dim = 1, [N, 1, D], keepdims = True
        min_data = np.min(data, norm_dim, keepdims = True)

        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):  #max-min data
        """
        :param max_data: np.array, max data
        :param min_data: np.array, min data
        :param data: np.array, original traffic data without normalization
        :return:
            np.array, normalized
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data-mid) / base

        return normalized_data

    @staticmethod
    def recover_Data(max_data, min_data, data):  #visualization
        """
        :param max_data:  np.array, max data
        :param min_data: np.array, min data
        :param data: np.array, normalized data
        :return:
            recovered_Data: np.array, recovered data
        """
        mid = min_data
        base = max_data - min_data

        recovered_Data = data+base+mid
        return recovered_Data
    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype = torch.float)

from torch.autograd import Variable

if __name__ == '__main__':
    train_data = LoadData(data_path = ["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes= 307, divide_days= [45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=8)
    for _, data in enumerate(train_loader):
        input = data
        input = Variable(input)

    print(len(train_data))

    print(train_data[0]["flow_x"].size())
    print(train_data[0]["flow_y"].size())



