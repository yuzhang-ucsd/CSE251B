# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
from dataloader import load_data
from utils import *
from TrainProcedure import *

def main():
    do_show_fig = True
    do_save_fig = False
    CrossValid = True
    SoftmaxQ6 = True

    lr = 3 # learning rate
    M = 300 # Total Epoch
    Interval = 50 # Interval used for drawing the errorbars
    Num_PC = 70

    if not SoftmaxQ6: #Q5
        if CrossValid:
            # k-fold dataset
            k = 10
            # Load data from ./resized/ folder
            images, cnt = load_data(data_dir="./aligned/")
            Minivan = images.get('Minivan')
            Convertible = images.get('Convertible')
            minivan = flatten_img(Minivan)
            convertible = flatten_img(Convertible)
            cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, final_acc = CrossRun(k, minivan, convertible, lr, M, Num_PC)
            print('The final test accuracy is %f' %(final_acc))
        else: #Q6
            # one Run
            # Load data from ./resized/ folder
            images, cnt = load_data(data_dir="./resized/")
            Minivan = images.get('Minivan')
            Convertible = images.get('Convertible')
            minivan = flatten_img(Minivan)
            convertible = flatten_img(Convertible)
            cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, final_acc = OneRun(minivan, convertible, lr, M, Num_PC)
            print('The final test accuracy is %f' %(final_acc))
    else:  #Q6
        # k-fold dataset
        k = 10
        # Load data from ./resized/ folder
        images, cnt = load_data(data_dir="./aligned/")
        Minivan = images.get('Minivan')
        Convertible = images.get('Convertible')
        Pickup = images.get('Pickup')
        Sedan = images.get('Sedan')
        minivan = flatten_img(Minivan)
        convertible = flatten_img(Convertible)
        pickup = flatten_img(Pickup)
        sedan = flatten_img(Sedan)
        cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, final_acc, Confusion_Matrix = Softmax(k, minivan, convertible, pickup, sedan, lr, M, Num_PC)
        print('The Confusion Matrix is\n')
        print(Confusion_Matrix)
        print('The final test accuracy is %f' %(final_acc))


    # Plot: Cost & Accuracy against Epochs
    plotFunc(cost_train, acc_train, SetName = 'TrainSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Softmax = SoftmaxQ6, Epoch = M, Interval = Interval)
    plotFunc(cost_val, acc_val, SetName = 'ValidSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Softmax = SoftmaxQ6, Epoch = M, Interval = Interval)
    plotFunc(cost_test, acc_test, SetName = 'TestSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Softmax = SoftmaxQ6, Epoch = M, Interval = Interval)
    if do_show_fig:
        plt.show()



if __name__ == '__main__':
    main()