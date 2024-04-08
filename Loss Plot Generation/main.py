import json
import Plotting

if __name__ == '__main__':
    losses = json.load(open('Loss Histories/loss_history_16.json'))
    Plotting.PlotLosses(losses, 50, 16)

    losses2 = json.load(open('Loss Histories/loss_history_32.json'))
    Plotting.PlotLosses(losses2, 50, 32)

    losses3 = json.load(open('Loss Histories/loss_history_64.json'))
    Plotting.PlotLosses(losses3, 50, 64)

    losses3 = json.load(open('Loss Histories/loss_history_128.json'))
    Plotting.PlotLosses(losses3, 50, 128)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
