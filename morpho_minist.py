import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy.special import expit


# 读取训练集
train_X = np.load('./train/train_minist.npy')  # 数字矩阵
train_label = pd.read_csv('./train/train_label.csv')
train_number = train_label['number']  # 数字标签
train_size = train_label['size']  # 粗细标签
# 读取测试集
test_X = np.load('./test/test_minist.npy')
test_label = pd.read_csv('./test/test_label.csv')
test_number = test_label['number']
test_size = test_label['size']
# 查看数据集规模
#print(f"训练集的尺度是：{train_X.shape}, 测试集的尺度是：{test_X.shape}")
# ----------------------------->第一题（必做）

def sigmoid(x):
    return expit(x)

def forward(X, W, b):#前向传播
    return sigmoid(np.dot(X, W) + b)

def loss(y, y_hat):#损失函数
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def backward(X, y, y_hat):#反向传播
    n = X.shape[0]
    dW = np.dot(X.T, y_hat - y) / n
    db = np.mean(y_hat - y)
    return dW, db

def update(W, b, dW, db, lr):#更新参数
    W -= lr * dW
    b -= lr * db
    return W, b

def train(X, y, lr, epochs):#训练
    W = np.zeros((X.shape[1],1))
    b = 0
    #print('initial')
    for i in range(epochs):
        idx = np.random.randint(X.shape[0])#随机梯度下降
        X_i = X[idx].reshape(1, -1)
        y_i = y[idx].reshape(-1, 1)
        y_hat = forward(X_i, W, b)
        #print(y_hat)
        loss_v = loss(y_i, y_hat)
        dW, db = backward(X_i, y_i, y_hat)
        W, b = update(W, b, dW, db, lr)
        #print('caculate')
        if i % 100 == 0:
            print(f'epoch {i}, loss {loss_v}')
    return W, b

def predict(X, W, b):#预测
    y_hat = forward(X, W, b)
    prediction = np.where(y_hat>0.5, 1, 0)
    return prediction

def draw_roc(y, y_hat):#画roc曲线
    fpr, tpr, thresholds = roc_curve(y, y_hat)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    


# train_X = train_X.reshape(train_X.shape[0], -1)
# test_X = test_X.reshape(test_X.shape[0], -1)
# train_size = train_size.values.reshape(-1, 1)
# test_size = test_size.values.reshape(-1, 1)

# W, b = train(train_X, train_size, 0.01, 1000)

# prediction = predict(test_X, W, b)

# print(f'accuracy {accuracy_score(test_size, prediction)}')
# print(f'precision {precision_score(test_size, prediction)}')
# print(f'recall {recall_score(test_size, prediction)}')
# print(f'f1 score {f1_score(test_size, prediction)}')
# print(f'roc_auc {roc_auc_score(test_size, prediction)}')
# draw_roc(test_size, prediction)


# ---------------------------->第二题（必做）
# TODO 2:使用Softmax回归拟合训练集的X数据和number标签,并对测试集进行预测
#

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def forward_softmax(X, W, b):
    return softmax(np.dot(X, W) + b)

def backward_softmax(X, y, y_hat):
    n = X.shape[0]
    dW = np.dot(X.T, y_hat - y) / n
    db = np.mean(y_hat - y, axis=0)
    return dW, db

def train_softmax(X, y, lr, epochs):
    num_classes = 10
    num_features = 784
    W = np.random.randn(num_features, num_classes) * 0.01
    b = np.zeros((1, num_classes))
    for i in range(epochs):
        idx = np.random.randint(X.shape[0])
        X_i = X[idx].reshape(1, -1)
        y_i = y[idx].reshape(1, -1)
        y_hat = forward_softmax(X_i, W, b)
        loss_v = cross_entropy_loss(y_i, y_hat)
        dW, db = backward_softmax(X_i, y_i, y_hat)
        W, b = update(W, b, dW, db, lr)
        if i % 100 == 0:
            print(f'epoch {i}, loss {loss_v}')
    return W, b

def predict_softmax(X, W, b):
    y_hat = forward_softmax(X, W, b)
    prediction = np.argmax(y_hat, axis=1)
    return prediction
def predict_proba(X, W, b):
    y_hat = forward_softmax(X, W, b)
    return y_hat

num_classes = 10
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)
y_train_one_hot = one_hot_encode(train_number.to_numpy(), num_classes)
y_test_one_hot = one_hot_encode(test_number.to_numpy(), num_classes)

W, b = train_softmax(train_X, y_train_one_hot, 0.01, 1000)
# print('w', W)   
# print('b', b)
prediction = predict_softmax(test_X, W, b)
auRoc_predict= predict_proba(test_X, W, b)
# print('predict', prediction)
print(f'accuracy {accuracy_score(test_number, prediction)}')
print(f'macro precision {precision_score(test_number, prediction, average="macro")}')
print(f'macro recall {recall_score(test_number, prediction, average="macro")}')
print(f'macro f1 score {f1_score(test_number, prediction, average="macro")}')
print(f'roc_auc {roc_auc_score(y_test_one_hot, auRoc_predict,multi_class="ovo")}')
cm = confusion_matrix(test_number, prediction)
print(cm)
