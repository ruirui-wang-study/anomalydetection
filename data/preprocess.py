# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:31:20 2019

@author: ELİF NUR
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


def loadData(fromPath, LabelColumnName, labelCount
             ):  #This method to read the csv file and change the label feature

    data_ = pd.read_csv(fromPath,sep=',',skipinitialspace=True)

    if labelCount == 2:
        dataset = data_
        dataset[LabelColumnName] = dataset[LabelColumnName].apply({
            'DoS':
            'Anormal',
            'BENIGN':
            'Normal',
            'DDoS':
            'Anormal',
            'PortScan':
            'Anormal'
        }.get)
    else:
        dataset = data_
    # data_.set_index(' Flow Duration').plot()
    # data = dataset[LabelColumnName].value_counts()
    # data.plot(kind='pie')
    # duration = dataset[' Flow Duration']
    # duration.plot()

    # featureList = dataset.loc[:,[' Flow Duration','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Fwd IAT Mean','Fwd Packets/s',' Bwd Packets/s',' Packet Length Mean','Init_Win_bytes_forward',' Init_Win_bytes_backward']]
    # 2. 提取指定列作为特征
    selected_columns = ['EtherType', 'Protocol', 'CumIPv4Flag_X', 'TcpDstPort', 'UdpDstPort',
                        'CumPacketSize', 'FlowDuration', 'MaxPacketSize', 'MinPacketSize',
                        'NumPackets', 'CumTcpFlag_X']
    featureList = dataset[selected_columns]
    # featureList=dataset.drop([LabelColumnName], axis=1)
    return dataset, featureList


def datasetSplit(
        df,
        LabelColumnName):  #This method is to separate the dataset as X and y.
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    # X = df.loc[:,[' Flow Duration','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Fwd IAT Mean','Fwd Packets/s',' Bwd Packets/s',' Packet Length Mean','Init_Win_bytes_forward',' Init_Win_bytes_backward']]

    X = df.drop([LabelColumnName], axis=1)
    X = np.array(X)
    X = X.T
    for column in X:  #Control of values in X
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    X = X.T
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    y = df[[LabelColumnName]]
    return X, y


def train_test_dataset(
    df
):  #This method is to separate the dataset as X_train,X_test,y_train and y_test.
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    # X = df.loc[:,[' Flow Duration','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Fwd IAT Mean','Fwd Packets/s',' Bwd Packets/s',' Packet Length Mean','Init_Win_bytes_forward',' Init_Win_bytes_backward']]

    X = df.drop([' Label'], axis=1)
    y = df[[' Label']]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)
    X_train = np.array(X_train)
    X_train = X_train.T
    for column in X_train:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    X_train = X_train.T
    y_train = np.array(y_train)
    y_train = y_train.T
    for column in y_train:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    y_train = y_train.T
    X_test = np.array(X_test)
    X_test = X_test.T
    for column in X_test:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    X_test = X_test.T
    y_test = np.array(y_test)
    y_test = y_test.T
    for column in y_test:
        median = np.nanmedian(column)
        column[np.isnan(column)] = median
        column[column == np.inf] = 0
        column[column == -np.inf] = 0
    y_test = y_test.T

    return X_train, X_test, y_train, y_test
