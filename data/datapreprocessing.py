# 对CICIDS2017数据集的预处理，删除了含有空值和无穷值的行，并提取了指定的特征列。对于非数值的列，使用One-Hot Encoding进行标签编码，然后对数值型特征进行了标准化处理。
# 'Destination Port', 'Flow Duration', 'Total Fwd Packets',
#        'Total Backward Packets', 'Total Length of Fwd Packets',
#        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
#        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
#        'Fwd Packet Length Std', 'Bwd Packet Length Max',
#        'Bwd Packet Length Min', 'Bwd Packet Length Mean',
#        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
#        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
#        'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
#        'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
#        'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
#        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
#        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
#        'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
#        'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
#        'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
#        'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
#        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
#        'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
#        'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
#        'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
#        'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
#        'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
#        'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
#        'Idle Std', 'Idle Max', 'Idle Min', 'Label'
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import numpy as np



def dataprocess():
    # Load dataset
    data = pd.read_csv("dataset.csv",sep=',',skipinitialspace=True)
    # print(data.columns)   
    # Drop rows with NaN or infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Standardize numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Encode non-numerical columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])

    # Apply PCA for feature extraction
    X = data.drop(columns=['Label'])
    y = data['Label']
    # Apply PCA for feature extraction with 10 components
    n_components = 10
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca,y

# Convert PCA-transformed data back to DataFrame with column names
# columns_pca = [f"PC{i}" for i in range(1, n_components + 1)]
# X_pca_df = pd.DataFrame(data=X_pca, columns=columns_pca)
# print(X_pca_df.head())



# 'Flow Duration',

# # Step 1: Remove rows with NaN and infinity values
# data = data.dropna()  # Remove rows with NaN values
# data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]  # Remove rows with infinity values

# # Step 2: Extract specified columns as features
# selected_features = ['EtherType', 'Protocol', 'CumIPv4Flag_X', 'TcpDstPort', 'UdpDstPort','CumPacketSize', 'FlowDuration', 'MaxPacketSize', 'MinPacketSize', 'NumPackets', 'CumTcpFlag_X']
# X = data[selected_features]

# # Step 3: One-Hot Encoding for non-numeric columns (EtherType and Protocol)
# X = pd.get_dummies(X, columns=['EtherType', 'Protocol'])

# # Step 4: Standardize the numeric features
# scaler = StandardScaler()
# X[['CumIPv4Flag_X', 'TcpDstPort', 'UdpDstPort', 'CumPacketSize', 'FlowDuration',
#    'MaxPacketSize', 'MinPacketSize', 'NumPackets', 'CumTcpFlag_X']] = scaler.fit_transform(X[['CumIPv4Flag_X', 'TcpDstPort', 'UdpDstPort',
#                                                                                                    'CumPacketSize', 'FlowDuration', 'MaxPacketSize', 'MinPacketSize', 'NumPackets', 'CumTcpFlag_X']])

# # Extract label column
# y = data[' Label']

# # Now X contains the preprocessed features and y contains the labels
# print(X.head())
