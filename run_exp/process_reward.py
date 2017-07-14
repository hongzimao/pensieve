import os
import numpy as np
import matplotlib.pyplot as plt


INPUT_FOLDER = './results/'
SCHEMES = ['BB', 'RB', 'BOLA', 'FESTIVE', 'RL']
BITRATE_MAX = 3.0  # Mbit/sec
K_IN_M = 1000.0
NUM_BINS = 200
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 


def main():

	all_total_reward = {}
	for scheme in SCHEMES:
		all_total_reward[scheme] = {}

	log_files = os.listdir(INPUT_FOLDER)
	for log_file in log_files:

		bit_rate = []
		rebuf_time = []

		with open(INPUT_FOLDER + log_file, 'rb') as f: 
			for line in f:
				parse = line.split()

				if len(parse) < 4:
					break

				bit_rate.append(float(parse[1]))
				rebuf_time.append(float(parse[3]))

		bit_rate = np.array(bit_rate) / K_IN_M
		rebuf_time = np.array(rebuf_time)

		smooth = np.abs(bit_rate[1:] - bit_rate[:-1]) 

		# --------------
		# linear scale
		# --------------

		# bit rate, rebuffer, smoothness
		# reward = bit_rate[1:] - 3 * rebuf_time[1:]  - smooth
		
		# bit rate, rebuffer
		# reward = bit_rate[1:] - 3 * rebuf_time[1:]
		
		# --------------
		# log scale
		# --------------

		log_bit_rate = np.log(bit_rate / BITRATE_MAX)
		log_smooth = np.abs(log_bit_rate[1:] - log_bit_rate[:-1])

		# log(bit_rate), rebuffer, smoothness
		reward = log_bit_rate[1:] - 2.5 * rebuf_time[1:] - log_smooth

		# log(bit_rate), rebuffer
		# reward = log_bit_rate[1:] - 2.5 * rebuf_time[1:]

		total_reward = np.sum(reward)

		for scheme in SCHEMES:
			if scheme in log_file:
				start_pt = len('log_') + len(scheme) + 1
				all_total_reward[scheme][log_file[start_pt:]] = total_reward
				break

	# align all records
	all_common_total_rewards = {}
	for scheme in SCHEMES:
		all_common_total_rewards[scheme] = []

	for log in all_total_reward[SCHEMES[0]]:

		log_all_existed = True
		for scheme in SCHEMES:
			if log not in all_total_reward[scheme]:
				log_all_existed = False
				break

		if log_all_existed:
			for scheme in SCHEMES:
				all_common_total_rewards[scheme].append(all_total_reward[scheme][log])

	mean_rewards = {}
	for scheme in SCHEMES:
		mean_rewards[scheme] = np.mean(all_common_total_rewards[scheme])

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in SCHEMES:
		ax.plot(all_common_total_rewards[scheme])
	
	SCHEMES_REW = []
	for scheme in SCHEMES:
		SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme]))

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])

	ax.legend(SCHEMES_REW, loc=4)
	
	plt.ylabel('total reward')
	plt.xlabel('trace index')
	plt.show()

	# ---- ---- ---- ----
	# CDF 
	# ---- ---- ---- ----

	fig = plt.figure()
	ax = fig.add_subplot(111)

	for scheme in SCHEMES:
		values, base = np.histogram(all_common_total_rewards[scheme], bins=NUM_BINS)
		cumulative = np.cumsum(values)
		ax.plot(base[:-1], cumulative)	

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])	

	ax.legend(SCHEMES_REW, loc=4)
	
	plt.ylabel('CDF')
	plt.xlabel('total reward')
	plt.show()

if __name__ == '__main__':
	main()