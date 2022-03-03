import matplotlib.pyplot as plt
from processing import LoadData
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import h5py

class Evaluation(object):
    """
    Evaluation metrics
    """
    def __init__(self):
        pass
    @staticmethod
    def mae_(target, output):
        return mean_absolute_error(target, output)

    @staticmethod
    def mape_(target, output):
        return np.mean(np.abs(target-output)/ (target+5))

    @staticmethod
    def rmse_(target, output):
        return np.sqrt(mean_squared_error(target, output))

    @staticmethod
    def total(target, output):
        mae = Evaluation.mae_(target,output)
        mape = Evaluation.mape_(target,output)
        rmse = Evaluation.rmse_(target,output)
        return mae, mape, rmse

def compute_performance(prediction, target, data):
    """
    :param prediction: np.array, the predicted results
    :param target: np.array, the ground truth
    :param data: the test dataset
    :return: 
        performance: np.array, Evaluation metrics(MAE, MAPE, RMSE), 
        recovered_data: np.array, Recovered results
    """
    try:
        dataset = data.dataset
    except:
        dataset = data

    prediction = LoadData.recover_Data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_Data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data


def visualize_Result(h5_file, nodes_id, time_se, visualize_file, loss):
    file_obj = h5py.File(h5_file, "r")
    prediction = file_obj["predict"][:][:, :, 0] #[N, T]
    target = file_obj["target"][:][:, :, 0]
    file_obj.close()
    
    plot_prediction = prediction[nodes_id][time_se[0]:time_se[1]]
    plot_target = target[nodes_id][time_se[0]:time_se[1]]
    # visilization for a day
    plt.figure()
    plt.grid(True, linestyle="-.", linewidth = 0.5)
    plt.plot(np.array([t for t in range(time_se[1]-time_se[0])]), plot_prediction, ls="-", marker = " ", color = "r")
    plt.plot(np.array([t for t in range(time_se[1]-time_se[0])]), plot_target, ls="-", marker = " ", color = "b")
    plt.legend(["prediction", "target"], loc = "upper right")
    plt.axis([0, time_se[1]- time_se[0],
              np.min(np.array([np.min(plot_prediction), np.min(plot_target)])),
              np.max(np.array([np.max(plot_prediction), np.max(plot_target)]))])
    
    plt.savefig(visualize_file + ".png")
    plt.show()

    plt.title("Training Loss")
    plt.xlabel("time/5mins")
    plt.ylabel("Traffic flow")
    plt.plot(loss, label = 'Training_loss')
    plt.legend()
    plt.savefig(visualize_file + "training loss.png" )
    plt.show()
    
    