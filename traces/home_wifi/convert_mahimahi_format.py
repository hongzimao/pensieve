import numpy as np


TRACE_FILE = './home_wifi_log'
MAHIMAHI_FILE = './home_wifi_mahimahi'
PACKET_SIZE = 1500  # bytes


with open(TRACE_FILE, 'rb') as f, open(MAHIMAHI_FILE, 'wb') as wf:

	pkt_to_send = 0

	for line in f:
		try:
			parse = line.split()
			pkt = float(parse[0])
			time_ms = float(parse[1])
			
			pkt_to_send += pkt
			while pkt_to_send >= PACKET_SIZE:
				wf.write(str(int(time_ms)) + '\n')
				pkt_to_send -= PACKET_SIZE

		except ValueError:
			pass 
