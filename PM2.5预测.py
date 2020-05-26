# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:50:30 2020

@author: lx
"""

########################## 基于 DNN\RNN 的 PM2.5预测  ##########################

import os                                        #用于查看与修改当前数据读取路径
import tensorflow as tf
from tensorflow import keras                          
import pandas as pd                              
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler #用于数据归一化

##更改路径至桌面
os.getcwd()
os.chdir('C:\\Users\\lx\\Desktop')       

##初步读取数据
data_ori=pd.read_csv("train.csv")
data_ori.head(5)
data_PM = data_ori[data_ori["Gas"]=="PM2.5"]      #提取PM2.5数据
data=data_PM.iloc[:,2:]
data_np=np.array(data).reshape(240*24,).astype(np.float32)
plt.plot(data)

##分为训练集和测试集
train=data[:int(240*24*0.7)]
plt.plot(train)
test=data[int(240*24*0.7):]
plt.plot(test)


##构建合适的数据结构
window_size=20
batch_size=32
shuffle_buffer_size=1000

def windowed_dataset(series,window_size,batch_size,shuffle_buffer):
    dataset=tf.data.Dataset.from_tensor_slices(series)
    dataset=dataset.window(window_size+1,shift=1,drop_remainder=True)
    dataset=dataset.flat_map(lambda window:window.batch(window_size+1))
    dataset=dataset.shuffle(shuffle_buffer).map(lambda window:(window[:-1],window[-1]))
    dataset=dataset.batch(batch_size).prefetch(1)
    return dataset

dataset = windowed_dataset(train,window_size,batch_size,shuffle_buffer_size)

##构建DNN模型框架（此例为两隐层的DNN）
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10,input_shape=[window_size],activation="relu"),
    tf.keras.layers.Dense(10,activation="relu"),
    tf.keras.layers.Dense(1)
])

##或构建RNN模型框架
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[window_size]),
    tf.keras.layers.SimpleRNN(40,return_sequences=True),
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*100.0)
])

##或构建LSTM模型框架
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=-1),input_shape=[window_size]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*100.0)
])

##查看模型框架
model.summary()

##选择loss与优化算法
model.compile(loss="mse",optimizer=tf.keras.optimizers.SGD(lr=8e-6,momentum=0.9))

##每epoch自动调整步长
Ir_schedule=tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-8*10**(epoch/20)) 

##开始初步训练！
history = model.fit(dataset,epochs=100,callbacks=[Ir_schedule])

##查看学习速率的变化对loss的影响，选择最佳步长
plt.semilogx(history.history["lr"],history.history["loss"])
plt.axis([0,1,0,100])

##选择合适步长后重新训练！！！
model.compile(loss="mse",optimizer=tf.keras.optimizers.SGD(lr=1e-6,momentum=0.9))
history = model.fit(dataset,epochs=100)

##可视化学习曲线
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(30,150)
    plt.show()
plot_learning_curves(history)


##预测(模型在测试集上的表现)
forecast = []
for time in range(len(data_np)-window_size):
    forecast.append(model.predict(data_np[time:time + window_size][np.newaxis]))
forecast = forecast[int(240*24*0.7)-window_size:]
results = np.array(forecast)[:,0,0]

plt.figure(figsize=(10,6))

plt.plot(test)  #在测试集上可视化模型预测结果
plt.plot(results)

tf.keras.metrics.mean_absolute_error(test,results).numpy() #查看在测试集上的MAE误差