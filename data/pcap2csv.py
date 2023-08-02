from scapy.all import *
import pandas as pd

# Function to extract features from a packet
def extract_features(packet):
    ether_type = packet[Ether].type
    proto = packet[IP].proto if IP in packet else None
    cum_ipv4_flag = len(packet[IP].flags) if IP in packet else 0
    tcp_dst_port = packet[TCP].dport if TCP in packet else None
    udp_dst_port = packet[UDP].dport if UDP in packet else None
    cum_packet_size = len(packet)
    flow_duration = packet.time - flow_start_time
    max_packet_size = packet_len if packet_len > max_packet_size else max_packet_size
    min_packet_size = packet_len if packet_len < min_packet_size else min_packet_size
    num_packets = 1
    cum_tcp_flag = len(packet[TCP].flags) if TCP in packet else 0

    return [ether_type, proto, cum_ipv4_flag, tcp_dst_port, udp_dst_port, cum_packet_size, flow_duration, max_packet_size, min_packet_size, num_packets, cum_tcp_flag]

# Load pcap file
pcap_file = "your_pcap_file.pcap"
packets = rdpcap(pcap_file)

# Initialize variables for feature extraction
flow_start_time = packets[0].time
max_packet_size = len(packets[0])
min_packet_size = len(packets[0])
features_list = []

# Extract features from each packet
for packet in packets:
    features = extract_features(packet)
    features_list.append(features)

# Create DataFrame with extracted features
column_names = ['EtherType', 'Protocol', 'CumIPv4Flag_X', 'TcpDstPort', 'UdpDstPort', 'CumPacketSize', 'FlowDuration', 'MaxPacketSize', 'MinPacketSize', 'NumPackets', 'CumTcpFlag_X']
df = pd.DataFrame(features_list, columns=column_names)

# Print the DataFrame
print(df)















from scapy.all import *

# 从pcap文件中读取数据包
def read_pcap_file(file_path):
    packets = rdpcap(file_path)
    return packets

# 提取指定特征
def extract_features(packets):
    features_list = []
    for packet in packets:
        if IP in packet:
            # 提取IP层信息
            ip_layer = packet[IP]

            # 提取传输层协议
            protocol = ip_layer.fields['proto']

            # 提取传输层目的端口号
            if protocol in [6, 17]:  # TCP or UDP
                dst_port = packet[protocol].dport
            else:
                dst_port = None

            # 提取IP标志字段
            cum_ipv4_flag_x = ip_layer.fields['flags']

            # 提取数据包大小
            packet_size = len(packet)

            # 提取时间戳信息（如果有）
            timestamp = packet.time

            # 将特征保存为字典
            features = {
                'EtherType': packet[Ether].fields['type'],
                'Protocol': protocol,
                'CumIPv4Flag_X': cum_ipv4_flag_x,
                'TcpDstPort': dst_port,
                'UdpDstPort': dst_port,
                'CumPacketSize': packet_size,
                'FlowDuration': timestamp,
                'MaxPacketSize': packet_size,
                'MinPacketSize': packet_size,
                'NumPackets': 1,
                'CumTcpFlag_X': None,
            }

            features_list.append(features)

    return features_list

# 读取pcap文件并提取特征
pcap_file = "your_pcap_file.pcap"
packets = read_pcap_file(pcap_file)
features_list = extract_features(packets)

# 将特征保存为DataFrame（可选）
df = pd.DataFrame(features_list)
print(df)


# 这个看起来靠谱一点
import pyshark

def extract_features_from_pcap(pcap_file):
    # List of features to extract
    features = ['eth.type', 'ip.proto', 'ip.flags', 'tcp.dstport', 'udp.dstport',
                'frame.len', 'frame.time_delta', 'tcp.len', 'tcp.len', 'ip.len', 'tcp.flags']

    # Create a DataFrame to store the extracted features
    data = pd.DataFrame(columns=features)

    # Read pcap file and extract features
    capture = pyshark.FileCapture(pcap_file, keep_packets=False)
    for pkt in capture:
        # Extract feature values from each packet
        row = {
            'eth.type': pkt.eth.type,
            'ip.proto': pkt.ip.proto,
            'ip.flags': pkt.ip.flags,
            'tcp.dstport': pkt.tcp.dstport,
            'udp.dstport': pkt.udp.dstport,
            'frame.len': pkt.length,
            'frame.time_delta': pkt.time_delta,
            'tcp.len': pkt.tcp.len,
            'tcp.len': pkt.tcp.len,
            'ip.len': pkt.ip.len,
            'tcp.flags': pkt.tcp.flags
        }
        # Append the extracted feature values to the DataFrame
        data = data.append(row, ignore_index=True)

    return data

# Replace 'your_pcap_file.pcap' with the path to your pcap file
pcap_file = 'your_pcap_file.pcap'
extracted_features = extract_features_from_pcap(pcap_file)

# Now 'extracted_features' contains the specified features extracted from the pcap file
