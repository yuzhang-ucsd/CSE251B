import matplotlib.pyplot as plt
from dataloader import load_data
from utils import *
from TrainProcedure import *

def main():
    do_show_fig = True
    do_save_fig = True
    CrossValid = True
    SoftmaxQ6 = True
    StochasticDescent = True #useful if SoftmaxQ6 == True & CompareLrQ5c == False
    CompareLrQ5c = False # Ture if for Q5ciii (select different learning rates)
    if CompareLrQ5c:
        SoftmaxQ6 = False  #Disable this flag variable if CompareLrQ5c == True
        lr1 = 0.1
        lr2 = 3
        lr3 = 10
    lr = 3 # learning rate  lr = 0.02 when using SGD
    M = 300 # Total Epoch
    Interval = 50 # Interval used for drawing the errorbars
    Num_PC = 50 #Number of PCs

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
            if not CompareLrQ5c:
                cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, final_acc, std_final_acc = CrossRun(k, minivan, convertible, lr, M, Num_PC)
                print('The final test accuracy is %f (std = %f)' %(final_acc, std_final_acc))
            else:
                print('First Learning Rate')
                cost_train1, acc_train1, cost_val1, acc_val1, cost_test1, acc_test1, final_acc1, std_final_acc1 = CrossRun(k, minivan, convertible, lr1, M, Num_PC)
                print('Second Learning Rate')
                cost_train2, acc_train2, cost_val2, acc_val2, cost_test2, acc_test2, final_acc2, std_final_acc2 = CrossRun(k, minivan, convertible, lr2, M, Num_PC)
                print('Third Learning Rate')
                cost_train3, acc_train3, cost_val3, acc_val3, cost_test3, acc_test3, final_acc3, std_final_acc3 = CrossRun(k, minivan, convertible, lr3, M, Num_PC)
                print('The final test accuracy is %f (std = %f) for lr=%f' % (final_acc1, std_final_acc1, lr1))
                print('The final test accuracy is %f (std = %f) for lr=%f' % (final_acc2, std_final_acc2, lr2))
                print('The final test accuracy is %f (std = %f) for lr=%f' % (final_acc3, std_final_acc3, lr3))
        else:
            # one run
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
        cost_train, acc_train, cost_val, acc_val, cost_test, acc_test, final_acc, std_final_acc, Confusion_Matrix = Softmax(k, minivan, convertible, pickup, sedan, lr, M, Num_PC, StochasticDescent = StochasticDescent)
        print('The Confusion Matrix is\n')
        print(Confusion_Matrix)
        print('The final test accuracy is %f (std = %f)' %(final_acc, std_final_acc))

    # Plot: Cost & Accuracy against Epochs
    if not CompareLrQ5c:
        plotFunc(cost_test, acc_test, SetName = 'TestingSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Softmax = SoftmaxQ6, Epoch = M, Interval = Interval)
        plotFunc2(cost_train, cost_val, param='Error', do_save_fig=do_save_fig, CrossValid=CrossValid, Epoch = M, Softmax = SoftmaxQ6, Interval = Interval)
        plotFunc2(acc_train, acc_val, param='Accuracy', do_save_fig=do_save_fig, CrossValid=CrossValid, Epoch = M, Softmax = SoftmaxQ6, Interval = Interval)
    else:
        plotFunc3(cost_train1, cost_train2, cost_train3, lr1, lr2, lr3, SetName = 'TrainSet', do_save_fig = do_save_fig, CrossValid = CrossValid, Softmax = SoftmaxQ6, Epoch = M, Interval = Interval)
    if do_show_fig:
        plt.show()


if __name__ == '__main__':
    main()