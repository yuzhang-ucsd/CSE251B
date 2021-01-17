# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
from dataloader import load_data
from utils import *
from TrainProcedure import *

def main():
    do_show_fig = True
    do_save_fig = True
    CrossValid = True

    lr = 5 # learning rate
    M = 600 # Total Epoch
    Interval = 50 # Interval used for drawing the errorbars
    if CrossValid:
        # k-fold dataset
        k = 10
        # Load data from ./resized/ folder
        images, cnt = load_data(data_dir="./aligned/")
        Minivan = images.get('Minivan')
        Convertible = images.get('Convertible')
        minivan = flatten_img(Minivan)
        convertible = flatten_img(Convertible)
        cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, final_acc = CrossRun(k, minivan, convertible, lr, M)
    else:
        # Load data from ./resized/ folder
        images, cnt = load_data(data_dir="./resized/")
        Minivan = images.get('Minivan')
        Convertible = images.get('Convertible')
        minivan = flatten_img(Minivan)
        convertible = flatten_img(Convertible)
        cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, final_acc = OneRun(minivan, convertible, lr, M)

    # Plot: Cost & Accuracy against Epochs
    plotFunc(cost_train, acc_train, SetName = 'TrainSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Epoch = M, Interval = Interval)
    plotFunc(cost_val, acc_val, SetName = 'ValidSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Epoch = M, Interval = Interval)
    plotFunc(cost_test, acc_test, SetName = 'TestSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Epoch = M, Interval = Interval)
    if do_show_fig:
        plt.show()



if __name__ == '__main__':
    main()