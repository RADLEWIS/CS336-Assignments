import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# N_opt = k * C^alpha
# log (N_opt) = log(k) + alpha * log(C)
def func_log(log_C, alpha, log_k):
    return alpha * log_C + log_k

if __name__ == "__main__":
    path = "data/isoflops_curves.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    array = np.array(data)

    flops_param = {}
    flops_log_data = []
    param_log_data = []
    loss_data = []
    # D = C / 6N
    # log(D) = log(C) - log(6N)
    data_log_data = []
    for data_point in array:
        param = data_point["parameters"]
        flops = data_point["compute_budget"]
        final_loss = data_point["final_loss"]
        if not (flops in flops_param.keys() and flops_param[flops][1] < final_loss):
            flops_param[flops] = (param, final_loss)

    for flops, (param, loss) in flops_param.items():
        flops_log_data.append(np.log10(flops))
        param_log_data.append(np.log10(param))
        data_log_data.append(np.log10(flops) - np.log10(6 * param))
        loss_data.append(loss)

    flops_log_data = np.array(flops_log_data, dtype=np.float64)
    param_log_data = np.array(param_log_data, dtype=np.float64)
    data_log_data = np.array(data_log_data, dtype=np.float64)
    loss_data = np.array(loss_data, dtype=np.float64)

    # popt_l, pcov_l = curve_fit(func_log, flops_log_data, loss_data)
    popt_p, pcov_p = curve_fit(func_log, flops_log_data, param_log_data)
    popt_d, pcov_p = curve_fit(func_log, flops_log_data, data_log_data)

    predicted_flops = np.array([np.log10(1e23), np.log10(1e24)], dtype=np.float64)
    # predicted_loss = func_log(predicted_flops, *popt_l)
    predicted_params = func_log(predicted_flops, *popt_p)
    predicted_data = func_log(predicted_flops, *popt_d)
    flops_log_data_predict = np.append(flops_log_data, predicted_flops)

    plt.figure(0)
    plt.plot(flops_log_data, loss_data, "r-", label="loss")
    plt.scatter(flops_log_data, loss_data, color="red", marker="o")
    # plt.plot(flops_log_data_predict, func_log(flops_log_data_predict, *popt_l), "g--", label="loss fit: alpha=%5.3f, log_k=%5.3f" % tuple(popt_l))
    # plt.scatter(flops_log_data_predict, func_log(flops_log_data_predict, *popt_l), color="green", marker="o")
    '''
    for i in range(len(predicted_flops)):
        plt.annotate(
            f"({10 ** predicted_flops[i]:.2}, {10 ** predicted_loss[i]:.2})",
            (predicted_flops[i], predicted_loss[i]),
            textcoords="offset points",
            xytext=(-9, -18),
            fontsize=10, color="green"
        )
    '''
    plt.xlabel('flops log')
    plt.ylabel('loss')
    plt.legend()
    plt.title("optimal loss with different FLOPs")

    plt.figure(1)
    plt.plot(flops_log_data, param_log_data, "r-", label="data")
    plt.scatter(flops_log_data, param_log_data, color="red", marker="o")
    
    plt.plot(flops_log_data_predict, func_log(flops_log_data_predict, *popt_p), "g--", label="params fit: alpha=%5.3f, log_k=%5.3f" % tuple(popt_p))
    plt.scatter(flops_log_data_predict, func_log(flops_log_data_predict, *popt_p), color="green", marker="o")
    for i in range(len(predicted_flops)):
        plt.annotate(
            f"{10 ** predicted_params[i]:.2}",
            (predicted_flops[i], predicted_params[i]),
            textcoords="offset points",
            xytext=(-9, -18),
            fontsize=10, color="green"
        )
    plt.xlabel('flops log')
    plt.ylabel('params log')
    plt.legend()
    plt.title("optimal params with different FLOPs")

    plt.figure(2)
    plt.plot(flops_log_data, data_log_data, "r-", label="data")
    plt.scatter(flops_log_data, data_log_data, color="red", marker="o")

    plt.plot(flops_log_data_predict, func_log(flops_log_data_predict, *popt_d), "g--", label="tokens fit: alpha=%5.3f, log_k=%5.3f" % tuple(popt_d))
    plt.scatter(flops_log_data_predict, func_log(flops_log_data_predict, *popt_d), color="green", marker="o")
    for i in range(len(predicted_flops)):
        plt.annotate(
            f"{10 ** predicted_data[i]:.2}",
            (predicted_flops[i], predicted_data[i]),
            textcoords="offset points",
            xytext=(-9, -18),
            fontsize=10, color="green"
        )
    plt.xlabel('flops log')
    plt.ylabel('tokens log')
    plt.legend()
    plt.title("optimal tokens with different FLOPs")

    plt.show()

    print("flops_log_data: \n", flops_log_data)
    print("param_log_data:\n", param_log_data)
    print("data_log_data:\n", data_log_data)