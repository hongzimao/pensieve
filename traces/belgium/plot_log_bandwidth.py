import numpy as np
import matplotlib.pyplot as plt


PACKET_SIZE = 1500.0  # bytes
TIME_INTERVAL = 5.0
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
N = 100
LINK_FILE = './logs/report_bus_0010.log'


time_ms = []
bytes_recv = []
recv_time = []
with open(LINK_FILE, 'rb') as f:
	for line in f:
		parse = line.split()
		time_ms.append(int(parse[1]))
		bytes_recv.append(float(parse[4]))
		recv_time.append(float(parse[5]))
time_ms = np.array(time_ms)
bytes_recv = np.array(bytes_recv)
recv_time = np.array(recv_time)
throughput_all = bytes_recv / recv_time

time_ms = time_ms - time_ms[0]
time_ms = time_ms / MILLISECONDS_IN_SECONDS
throughput_all = throughput_all * BITS_IN_BYTE / MBITS_IN_BITS * MILLISECONDS_IN_SECONDS

plt.plot(time_ms, throughput_all)
plt.xlabel('Time (second)')
plt.ylabel('Throughput (Mbit/sec)')
plt.show()
