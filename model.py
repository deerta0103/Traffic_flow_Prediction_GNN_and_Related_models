import os
import time
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


from processing import LoadData
#fromutils import Evaluation
from utils import visualize_Result
from utils import compute_performance
from baseline import GATNet, GCN, ChebNet
from GATandVariant import GATNet2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_Data = LoadData(data_path = ["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes = 307, divide_days =[45, 14],
                      time_interval = 5, history_length=6, train_mode = "train")
train_loader = DataLoader(train_Data, batch_size = 64, shuffle = False, num_workers = 8)
test_data = LoadData(data_path = ["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes = 307, divide_days =[45, 14],
                      time_interval = 5, history_length=6, train_mode = "test")
test_loader = DataLoader(test_data, batch_size = 64, shuffle = False, num_workers = 8)

# model design
#net = GATNet(input_dim=6 * 1, hidden_dim=6, output_dim=1, n_heads=2)
#net = GCN(input_dim=6 * 1, hidden_dim=6, output_dim=1)
net = ChebNet(input_dim=6 * 1,  hidden_dim=6, output_dim=1, K=2)
#net = GATNet2(input_dim= 1,  hidden_dim=6, output_dim=1, n_heads = 2,T = 6)
#device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
net = net.to(device)

# loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(params = net.parameters())

# train
Epoch = 10
loss_train_plt = []

net.train()
for epoch in range(Epoch):
    epoch_loss = 0.0
    start_time = time.time()
    for data in train_loader:
        net.zero_grad()
        predict = net(data, device).to(torch.device("cpu"))
        data_y = data["flow_y"]
        loss = criterion(predict,data_y)
        
        epoch_loss = loss.item()
        
        loss.backward()

        optimizer.step()  # 更新参数
    end_time = time.time()
    loss_train_plt.append(10*epoch_loss / len(train_Data) / 64)
    
    print("Epoch: {:04d}, Loss: {:02.4f}, TIme: {:02.2f} mins".format(epoch, 1000 * epoch_loss/len(train_Data), (end_time-start_time)/60))

# Test
net.eval()
with torch.no_grad():
    MAE, MAPE, RMSE = [], [], []
    Target = np.zeros([307, 1, 1])  # [N, T, D]
    Predict = np.zeros_like(Target)
    
    total_loss = 0.0
    for data in test_loader: # for each batch
        predict_value = net(data, device).to(torch.device("cpu"))
        loss = criterion(predict_value, data["flow_y"])
        total_loss +=loss.item()

        predict_value = predict_value.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
        target_value = data["flow_y"].transpose(0, 2).squeeze(0)
        
        performance, data_save = compute_performance(predict_value, target_value, test_loader)
        
        # concatenate each time slot
        Predict = np.concatenate([Predict, data_save[0]], axis =1)
        Target = np.concatenate([Target, data_save[1]], axis = 1)
        
        MAE.append(performance[0])
        MAPE.append(performance[1])
        RMSE.append(performance[2])
        
    print("Test loss: {:02.4f}".format(1000 * total_loss / len(test_data)))
    
print("Performance: MAE {:2.2f}, MAPE {:2.2f}, RMSE {:2.2f}".format(np.mean(MAE), np.mean(MAPE), np.mean(RMSE)))

Predict = np.delete(Predict, 0, axis =1)
Target = np.delete(Target, 0, axis =1)

result_file = "GAT_result.h5"
file_obj = h5py.File(result_file, "w")

file_obj["predict"] = Predict
file_obj["target"] = Target



visualize_Result(h5_file = "GAT_result.h5",
                 nodes_id = 120,
                 time_se = [0, 24*12*2],
                 visualize_file = "gat_node_120",
                 loss = loss_train_plt)


    
    
    




