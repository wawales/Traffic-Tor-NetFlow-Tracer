# -*- coding: utf-8 -*-
import torch
# import scipy.io as io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm
import random

dataset = []
dataset_test = []
model_name = "cnn_sigmoid2"

################################################################################
#6-11
# dataset_test.append(pickle.load(open('./test_data/London_Mumbai_4122.pkl', 'rb')))
# dataset_test.append(pickle.load(open('./test_data/Singapore_Virginia_3056.pkl', 'rb')))
################################################################################
#6-12
# dataset[0] += pickle.load(open( './../torData/6-12/0612_london_mumbai_2179.pkl', 'rb'))
# dataset[1] += pickle.load(open( './../torData/6-12/0612_singapore_virgina_1700.pkl', 'rb'))
dataset.append(pickle.load(open( './../torData/6-12/0612_london_mumbai_2179.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-12/0612_singapore_virgina_1700.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-12/0612_Siliconvalley_HongKong_1314.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-12/0612_Tokyo_Frankfurt_1322.pkl', 'rb')))
################################################################################
#6-13
dataset[0] += pickle.load(open( './../torData/6-13/0612_london_mumbai_2907.pkl', 'rb'))
dataset[1] += pickle.load(open( './../torData/6-13/0612_singapore_virginia_2909.pkl', 'rb'))
dataset[2] += pickle.load(open( './../torData/6-13/190612_Siliconvalley_HongKong_2528.pkl', 'rb'))
dataset[3] += pickle.load(open( './../torData/6-13/190612_Tokyo_Frankfurt_2014.pkl', 'rb'))
# dataset_test.append(pickle.load(open( './../torData/6-13/0612_london_mumbai_2907.pkl', 'rb')))
# dataset_test.append(pickle.load(open( './../torData/6-13/0612_singapore_virginia_2909.pkl', 'rb')))
# dataset_test.append(pickle.load(open( './../torData/6-13/190612_Siliconvalley_HongKong_2528.pkl', 'rb')))
# dataset_test.append(pickle.load(open( './../torData/6-13/190612_Tokyo_Frankfurt_2014.pkl', 'rb')))
################################################################################
#6-14!!!!!!!!
# dataset_test.append(pickle.load(open( './../torData/6-14/190614_London_Virgina_2500.pkl', 'rb')))!!!!!!!!!
dataset.append(pickle.load(open( './../torData/6-14/190614_Singapore_HongKong_3003.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-14/190614_Tokyo_Mubai_290.pkl', 'rb')))
################################################################################
#6-17
dataset.append(pickle.load(open( './../torData/6-17/190617_London_HongKong_3357.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-17/190617_Siliconvalley_Mumbai_2943.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-17/190617_Singapore_Frankfurt_3019.pkl', 'rb')))
################################################################################
#6-19!!!!!!!!!!!
# dataset_test.append(pickle.load(open( './../torData/6-19/190619_London_Frankfurt_3725.pkl', 'rb')))!!!!!!!!!!
dataset.append(pickle.load(open( './../torData/6-19/190619_Siliconvalley_Virginia_3302.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-19/190619_Singapore_Mumbai_2275.pkl', 'rb')))
# dataset_test.append(pickle.load(open( './../torData/6-19/190619_Tokyo_HongKong_1234.pkl', 'rb')))!!!!!!!!!!!
################################################################################
#6-21
dataset.append(pickle.load(open( './../torData/6-21/190621_London_Virginia_4615.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-21/190621_Singapore_HongKong_2389.pkl', 'rb')))
# dataset.append(pickle.load(open( './../torData/6-21/190621_Tokyo_Mumbai_3067.pkl', 'rb')))
dataset[4] += pickle.load(open( './../torData/6-21/190621_Singapore_HongKong_2389.pkl', 'rb'))
dataset[5] += pickle.load(open( './../torData/6-21/190621_Tokyo_Mumbai_3067.pkl', 'rb'))
################################################################################
#6-24
dataset.append(pickle.load(open( './../torData/6-24/190624_London_Frankfurt_1241.pkl', 'rb')))
# dataset.append(pickle.load(open( './../torData/6-24/190624_Silconvalley_Virginia_5221.pkl', 'rb')))
# dataset.append(pickle.load(open( './../torData/6-24/190624_Singapore_Mumbai_3706.pkl', 'rb')))
dataset.append(pickle.load(open( './../torData/6-24/190624_Tokyo_HongKong_2473.pkl', 'rb')))
dataset[9] += pickle.load(open( './../torData/6-24/190624_Silconvalley_Virginia_5221.pkl', 'rb'))
dataset[10] += pickle.load(open( './../torData/6-24/190624_Singapore_Mumbai_3706.pkl', 'rb'))
################################################################################
#7-10
dataset_test.append(pickle.load(open( './../torData/7-10/190710_London_Frankfurt_8592.pkl', 'rb')))
dataset_test.append(pickle.load(open( './../torData/7-10/190710_Silconvalley_Virginia_4665.pkl', 'rb')))
dataset_test.append(pickle.load(open( './../torData/7-10/190710_Singapore_Mumbai_3046.pkl', 'rb')))
dataset_test.append(pickle.load(open( './../torData/7-10/190710_Tokyo_HongKong_2049.pkl', 'rb')))
################################################################################

#7-12
# dataset_test.append(pickle.load(open( './../torData/7-12/190712_London_HongKong_5352.pkl', 'rb')))
# dataset_test.append(pickle.load(open( './../torData/7-12/190712_Silconvalley_Mumbai_4024.pkl', 'rb')))
# dataset_test.append(pickle.load(open( './../torData/7-12/190712_Singapore_Frankfurt_4745.pkl', 'rb')))
# dataset_test.append(pickle.load(open( './../torData/7-12/190712_Tokyo_Virginia_3832.pkl', 'rb')))


train_index = []
test_index = []
for i in range(len(dataset)):
    rr = list(range(len(dataset[i])))
    np.random.shuffle(rr)
    train_index.append(rr)
for i in range(len(dataset_test)):
    # rr = list(range(len(dataset_test[i])))
    rr = list(range(2000))
    np.random.shuffle(rr)
    test_index.append(rr)

def generate_data(dataset, dataset_test, train_index, test_index):
    flow_size = 200
    negetive_samples = 1
    negetive_samples_test = 1
    all_samples = 0
    all_samples_test = 0
    for num in range(len(dataset)):
        all_samples += len(train_index[num])
    for num2 in range(len(dataset_test)):
        all_samples_test += len(test_index[num2])
    train = np.zeros((all_samples * (negetive_samples + 1), 8, flow_size, 1))
    labels_train = np.zeros(all_samples * (negetive_samples + 1))

    index = 0
    for num in range(len(dataset)):
        for i in tqdm.tqdm(train_index[num]):
            train[index, 0, :, 0] = np.array(dataset[num][i]['client'][0]['<-'][:flow_size])
            train[index, 1, :, 0] = np.array(dataset[num][i]['server'][0]['->'][:flow_size])
            train[index, 2, :, 0] = np.array(dataset[num][i]['server'][0]['<-'][:flow_size])
            train[index, 3, :, 0] = np.array(dataset[num][i]['client'][0]['->'][:flow_size])
            train[index, 4, :, 0] = np.array(dataset[num][i]['client'][1]['<-'][:flow_size]) / 1000.0
            train[index, 5, :, 0] = np.array(dataset[num][i]['server'][1]['->'][:flow_size]) / 1000.0
            train[index, 6, :, 0] = np.array(dataset[num][i]['server'][1]['<-'][:flow_size]) / 1000.0
            train[index, 7, :, 0] = np.array(dataset[num][i]['client'][1]['->'][:flow_size]) / 1000.0

            labels_train[index] = 1
            index += 1
            neg = 0
            while neg < negetive_samples:
                while True:
                    if len(dataset) == 1:
                        num_false = num
                        break
                    num_false = random.randint(0,len(dataset)-1)
                    if num_false != num:
                        break
                while True:
                    idx = random.randint(0, len(train_index[num_false])-1)
                    if idx != i:
                        break

                train[index, 0, :, 0] = np.array(dataset[num_false][idx]['client'][0]['<-'][:flow_size])
                train[index, 1, :, 0] = np.array(dataset[num][i]['server'][0]['->'][:flow_size])
                train[index, 2, :, 0] = np.array(dataset[num][i]['server'][0]['<-'][:flow_size])
                train[index, 3, :, 0] = np.array(dataset[num_false][idx]['client'][0]['->'][:flow_size])
                train[index, 4, :, 0] = np.array(dataset[num_false][idx]['client'][1]['<-'][:flow_size]) / 1000.0
                train[index, 5, :, 0] = np.array(dataset[num][i]['server'][1]['->'][:flow_size]) / 1000.0
                train[index, 6, :, 0] = np.array(dataset[num][i]['server'][1]['<-'][:flow_size]) / 1000.0
                train[index, 7, :, 0] = np.array(dataset[num_false][idx]['client'][1]['->'][:flow_size]) / 1000.0

                labels_train[index] = 0
                index += 1
                neg += 1

    test = np.zeros((all_samples_test * (negetive_samples_test + 1), 8, flow_size, 1))
    labels_test = np.zeros(all_samples_test * (negetive_samples_test + 1))

    index = 0

    for num in range(len(dataset_test)):
        for i in tqdm.tqdm(test_index[num]):
            test[index, 0, :, 0] = np.array(dataset_test[num][i]['client'][0]['<-'][:flow_size])
            test[index, 1, :, 0] = np.array(dataset_test[num][i]['server'][0]['->'][:flow_size])
            test[index, 2, :, 0] = np.array(dataset_test[num][i]['server'][0]['<-'][:flow_size])
            test[index, 3, :, 0] = np.array(dataset_test[num][i]['client'][0]['->'][:flow_size])
            test[index, 4, :, 0] = np.array(dataset_test[num][i]['client'][1]['<-'][:flow_size]) / 1000.0
            test[index, 5, :, 0] = np.array(dataset_test[num][i]['server'][1]['->'][:flow_size]) / 1000.0
            test[index, 6, :, 0] = np.array(dataset_test[num][i]['server'][1]['<-'][:flow_size]) / 1000.0
            test[index, 7, :, 0] = np.array(dataset_test[num][i]['client'][1]['->'][:flow_size]) / 1000.0

            labels_test[index] = 1
            index += 1
            neg = 0
            while neg < negetive_samples_test:
                while True:
                    if len(dataset_test) == 1:
                        num_false = num
                        break
                    num_false = random.randint(0,len(dataset_test)-1)
                    if num_false != num:
                        break
                while True:
                    idx = random.randint(0, len(test_index[num_false])-1)
                    if idx != i:
                        break

                test[index, 0, :, 0] = np.array(dataset_test[num_false][idx]['client'][0]['<-'][:flow_size])
                test[index, 1, :, 0] = np.array(dataset_test[num][i]['server'][0]['->'][:flow_size])
                test[index, 2, :, 0] = np.array(dataset_test[num][i]['server'][0]['<-'][:flow_size])
                test[index, 3, :, 0] = np.array(dataset_test[num_false][idx]['client'][0]['->'][:flow_size])
                test[index, 4, :, 0] = np.array(dataset_test[num_false][idx]['client'][1]['<-'][:flow_size]) / 1000.0
                test[index, 5, :, 0] = np.array(dataset_test[num][i]['server'][1]['->'][:flow_size]) / 1000.0
                test[index, 6, :, 0] = np.array(dataset_test[num][i]['server'][1]['<-'][:flow_size]) / 1000.0
                test[index, 7, :, 0] = np.array(dataset_test[num_false][idx]['client'][1]['->'][:flow_size]) / 1000.0
                labels_test[index] = 0
                index += 1
                neg += 1

    return train, labels_train, test, labels_test
X_train, y_train, X_test, y_test = generate_data(dataset, dataset_test,train_index, test_index)
########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, (2, 20), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.RReLU(),
            nn.MaxPool2d((1, 5), stride=(1, 1)),
            # nn.MaxPool2d((1, 5), stride=(1, 3)),
            nn.Conv2d(32, 64, (4, 10), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.RReLU(),
            nn.MaxPool2d((1, 3), stride=(1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(9472, 256),
            nn.BatchNorm1d(256),
            # nn.RReLU(),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(128),,
            nn.RReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.RReLU(),
            nn.Linear(32, 2),
            nn.BatchNorm1d(2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(self.conv(x).shape)
        x = self.conv(x).view(-1, 9472)
        x2 = self.fc(x)
        return x2
net = Net()
########################################################################
#add model to CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
net = nn.DataParallel(net)
net.to(device)
########################################################################
# 3. Define a Loss function and optimizer

import torch.optim as optim

b = 0.5
# loss_weight = [1, 1]
# loss_weight = torch.FloatTensor(loss_weight).to(device)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer = optim.SGD(net.parameters(), lr=0.01)
def f(x, b):
    y = x[:, 1].div(x[:, 0].add(x[:, 1]))
    y[y > b] = 1
    y[y <= b] = 0
    y = y.clone().detach()
    return y
def g(x):
    # print(x.shape)
    y = x[:, 1].div(x[:, 0].add(x[:, 1]))
    y = y.clone().detach()
    return y
########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
# net.load_state_dict(torch.load('params_cnn_final.pkl'))
if len(X_train) > 0:
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    test_loss_min = 100
    batch_size = 256
    running_loss = 0.0
    train_loss_figure = []
    train_acc_figure = []
    test_loss_figure = []
    test_acc_figure = []
    # range_size = (len(X_train) + len(X_test)) / batch_size * 200
    # range_size = 6000
    range_size = 100
    rr = list(range(len(X_train)))
    batch_num = len(rr) // batch_size
    for i in np.arange(range_size):
        np.random.shuffle(rr)
        for j in range(batch_num):
            net.train()
            index = rr[j * batch_size: j * batch_size + batch_size]
            index = np.random.randint(0, len(X_train), batch_size)
            inputs = X_train[index]
            inputs = torch.from_numpy(inputs).view(-1,1,8,200).float()
            labels = y_train[index]
            labels = torch.from_numpy(labels).long()
            #copy tensor to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # predicted  = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



            # print statistics
            running_loss += loss.item()
            if (j + i * batch_num) % 20 == 19:
                # predicted = f(outputs, b)
                predicted = torch.max(outputs, 1)[1]
                # convert tensor to cpu
                predicted_cpu = predicted.cpu()
                labels_cpu = labels.cpu()
                train_accuracy = accuracy_score(labels_cpu, predicted_cpu)
                print('loss: %.6f  accuracy: %.6f  i: %d' % (running_loss / 20, train_accuracy, j + i * batch_num))
                train_loss_figure.append(running_loss / 20)
                train_acc_figure.append(train_accuracy)

                running_loss = 0.0
            if (j + i * batch_num) % 20 == 19 and len(dataset_test) != 0:
            # if len(dataset_test) != 0:
                net.eval()
                test_size = len(y_test)
                test_times = 1
                for j in np.arange(test_times):
                    index2 = list(range(test_size))
                    np.random.shuffle(index2)
                    test_input = X_test[index2]
                    test_input = torch.from_numpy(test_input).view(-1, 1, 8, 200).float()
                    test_labels = y_test[index2]
                    test_labels = torch.from_numpy(test_labels).long()
                    # copy tensor to GPU
                    test_input = test_input.to(device)
                    test_labels_GPU = test_labels.to(device)
                    test_output = net(test_input)
                    # test_predicted = f(outputs, b)
                    test_predicted = torch.max(test_output, 1)[1]
                    # convert tensor to cpu
                    test_predicted = test_predicted.cpu()
                    test_loss = criterion(test_output, test_labels_GPU)
                    test_loss1 = float(test_loss)
                    test_accuracy = accuracy_score(test_labels, test_predicted)
                    print('dev_loss: %.6f  accuracy: %.6f i: %d' % (test_loss, test_accuracy, j + i * batch_num))
                    del test_loss
                    # print(confusion_matrix(test_labels, test_predicted))
                    test_loss_figure.append(test_loss1)
                    test_acc_figure.append(test_accuracy)
                    # if test_loss1 < 0.35 and test_loss1 < test_loss_min:
                    #     torch.save(net.state_dict(), 'params_cnn.pkl')
                    #     print('Save Model epoch:%d' % (j + i * batch_num))
                    #     test_loss_min = test_loss1

    # print('Finished Training save batch: %d' % (save_batch))
    # torch.save(net.state_dict(), 'params_cnn_final.pkl')
    figure = []
    figure.append(train_loss_figure)
    figure.append(train_acc_figure)
    figure.append(test_loss_figure)
    figure.append(test_acc_figure)
    pickle.dump(figure, open('figure.pkl', 'wb'))
    print('dump figure.pkl')
########################################################################
net.load_state_dict(torch.load('params_cnn.pkl'))

# from collections import OrderedDict
# state_dict = torch.load('params_cnn.pkl')
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove module.
#     new_state_dict[name] = v
# net.load_state_dict(new_state_dict)
# net.cpu()
# print('Model loaded')
########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
if len(y_test) > 0:
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot as plt
    import seaborn as sns
    import pandas as pd
    net.eval()
    test_size = len(y_test)
    test_times = 1
    bb = 0.6
    metrics_figure = []
    index2 = list(range(test_size))
    np.random.shuffle(index2)
    test_input = X_test[index2]
    test_input = torch.from_numpy(test_input).view(-1, 1, 8, 200).float()
    test_labels = y_test[index2]
    test_labels = torch.from_numpy(test_labels).long()
    # copy tensor to GPU
    test_input = test_input.to(device)
    test_labels_GPU = test_labels.to(device)
    test_output = net(test_input)
    test_loss = criterion(test_output, test_labels_GPU)
    test_predict_prob = g(test_output)
    # pr-curve
    precision, recall, thresholds = precision_recall_curve(test_labels, test_predict_prob.cpu())
    pr_auc = auc(recall, precision)
    ap = average_precision_score(test_labels, test_predict_prob.cpu())
    # roc-curve
    fpr, tpr, thresholds = roc_curve(test_labels, test_predict_prob.cpu())
    roc_auc = roc_auc_score(test_labels, test_predict_prob.cpu())
    print("test_loss:%.3f  ap:%.3f  pr_auc:%.3f  roc_auc:%.3f   " % (test_loss, ap, pr_auc, roc_auc))
    # plot no skill
    plt.figure(1)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.xlabel('recall')
    plt.ylabel('precision')
    # show the plot
    plt.savefig('pr_curve')
    plt.figure(2)
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.savefig('roc_curve')
    for i in range(test_times):
        test_predicted = f(test_output, bb)
        # test_predicted = torch.max(test_output, 1)[1]
        # convert tensor to cpu
        test_predicted = test_predicted.cpu()
        test_accuracy = accuracy_score(test_labels, test_predicted)
        test_f1 = f1_score(test_labels, test_predicted)
        print('accuracy: %.3f  f1_score: %.3f   bb: %f' % (test_accuracy, test_f1, bb))
        # Creates a confusion matrix
        cm = confusion_matrix(test_labels, test_predicted)
        print(cm)
        # Transform to df for easier plotting
        cm_df = pd.DataFrame(cm, index=['false', 'true'], columns=['false', 'true'])
        plt.figure(3)
        sns.heatmap(cm_df, annot=True, fmt="d")
        plt.title('Accuracy:{0:.3f}'.format(test_accuracy))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("confusion matrix")

    metrics_figure = [recall, precision, fpr, tpr]
    pickle.dump(metrics_figure, open('metrics_figure.pkl', 'wb'))
    print('dump metrics_figure.pkl')

