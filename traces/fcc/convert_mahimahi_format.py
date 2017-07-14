import os
import numpy as np


IN_FILE = './cooked/'
OUT_FILE = './mahimahi/'
FILE_SIZE = 2000
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
EXP_LEN = 5000.0  # millisecond


def main():
	files = os.listdir(IN_FILE)
	for trace_file in files:
		if os.stat(IN_FILE + trace_file).st_size >= FILE_SIZE:
			with open(IN_FILE + trace_file, 'rb') as f, open(OUT_FILE + trace_file, 'wb') as mf:
				millisec_time = 0
				mf.write(str(millisec_time) + '\n')
				for line in f:
					throughput = float(line.split()[0])
					pkt_per_millisec = throughput / BYTES_PER_PKT / MILLISEC_IN_SEC

					millisec_count = 0
					pkt_count = 0
					while True:
						millisec_count += 1
						millisec_time += 1
						to_send = (millisec_count * pkt_per_millisec) - pkt_count
						to_send = np.floor(to_send)

						for i in xrange(int(to_send)):
							mf.write(str(millisec_time) + '\n')

						pkt_count += to_send

						if millisec_count >= EXP_LEN:
							break


if __name__ == '__main__':
	main()
