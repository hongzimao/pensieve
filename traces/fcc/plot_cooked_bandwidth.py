import numpy as np
import matplotlib.pyplot as plt


PACKET_SIZE = 1500.0  # bytes
TIME_INTERVAL = 5.0
BITS_IN_BYTE = 8.0
MBITS_IN_BITS = 1000000.0
MILLISECONDS_IN_SECONDS = 1000.0
N = 100
LINK_FILE = './201606/cooked/trace_9996_http---www.youtube.com'


bandwidth_all = []
with open(LINK_FILE, 'rb') as f:
	for line in f:
		throughput = int(line.split()[0])
		bandwidth_all.append(throughput)

bandwidth_all = np.array(bandwidth_all)
bandwidth_all = bandwidth_all * BITS_IN_BYTE / MBITS_IN_BITS

time_all = np.array(range(len(bandwidth_all))) * TIME_INTERVAL
plt.plot(time_all, bandwidth_all)
plt.xlabel('Time (second)')
plt.ylabel('Throughput (Mbit/sec)')
plt.show()
