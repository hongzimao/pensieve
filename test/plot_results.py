import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FOLDER = './results/'
NUM_BINS = 10
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired
SIM_DP = 'sim_dp'
#SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]
# SCHEMES = ['sim_rl', SIM_DP]
SCHEMES = ['sim_rl', 'sim_bb', 'sim_mpc']

# bad_files = ["trace_5642_http---www.youtube.com",
# "ferry.nesoddtangen-oslo-report.2011-02-01_1000CET.log",
# "trace_32551_http---www.amazon.com",
# "trace_942598_http---www.youtube.com",
# "trace_5294_http---www.youtube.com"]
bad_files = ["bus.ljansbakken-oslo-report.2010-09-29_1827CEST"]

def main():
	time_all = {}
	bit_rate_all = {}
	buff_all = {}
	bw_all = {}
	raw_reward_all = {}

	for scheme in SCHEMES:
		time_all[scheme] = {}
		raw_reward_all[scheme] = {}
		bit_rate_all[scheme] = {}
		buff_all[scheme] = {}
		bw_all[scheme] = {}

	log_files = os.listdir(RESULTS_FOLDER)
	# import pdb; pdb.set_trace()
	# log_files1 = []
	# for log_file in log_files:
	# 	if SCHEMES[0] in log_file:
	# 		log_files1 = log_files1 + [log_file]
	# log_files = log_files1
	for log_file in log_files:

		time_ms = []
		bit_rate = []
		buff = []
		bw = []
		reward = []

		# print log_file

		with open(RESULTS_FOLDER + log_file, 'rb') as f:
			if SIM_DP in log_file:
				last_t = 0
				last_b = 0
				last_q = 1
				lines = []
				for line in f:
					lines.append(line)
					parse = line.split()
					if len(parse) >= 6:
						time_ms.append(float(parse[3]))
						bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
						buff.append(float(parse[4]))
						bw.append(float(parse[5]))

				for line in reversed(lines):
					parse = line.split()
					r = 0
					if len(parse) > 1:
						t = float(parse[3])
						b = float(parse[4])
						q = int(parse[6])
						if b == 4:
							rebuff = (t - last_t) - last_b
							assert rebuff >= -1e-4
							r -= REBUF_P * rebuff

						r += VIDEO_BIT_RATE[q] / K_IN_M
						r -= SMOOTH_P * np.abs(VIDEO_BIT_RATE[q] - VIDEO_BIT_RATE[last_q]) / K_IN_M
						# if r<-200:
						# 	import pdb; pdb.set_trace()
						reward.append(r)

						last_t = t
						last_b = b
						last_q = q

			else:
				for line in f:
					parse = line.split()
					if len(parse) <= 1:
						break
					time_ms.append(float(parse[0]))
					bit_rate.append(int(parse[1]))
					buff.append(float(parse[2]))
					bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
					reward.append(float(parse[6]))
					if float(parse[6])<-200:
						import pdb; pdb.set_trace()

		if SIM_DP in log_file:
			time_ms = time_ms[::-1]
			bit_rate = bit_rate[::-1]
			buff = buff[::-1]
			bw = bw[::-1]

		# import pdb; pdb.set_trace()
		time_ms = np.array(time_ms)
		# if time_ms.length >0
		try:
			time_ms -= time_ms[0]
		except Exception as e:
			print e

		# print log_file
		# import pdb; pdb.set_trace()
		for scheme in SCHEMES:
			if scheme in log_file:
				time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
				bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
				buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
				bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
				raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
				break

	# ---- ---- ---- ----
	# Reward records
	# ---- ---- ---- ----

	log_file_all = []
	reward_all = {}
	for scheme in SCHEMES:
		reward_all[scheme] = []

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		for scheme in SCHEMES:
			if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
				schemes_check = False
				break
		if schemes_check:
			log_file_all.append(l)
			for scheme in SCHEMES:
				reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]))
				if np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN]) < -200:
					print l
	# import pdb; pdb.set_trace()
	mean_rewards = {}
	for scheme in SCHEMES:
		mean_rewards[scheme] = np.mean(reward_all[scheme])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	# import pdb; pdb.set_trace()
	for scheme in SCHEMES:
		ax.plot(reward_all[scheme])
		# ax.plot([x/10 for x in reward_all[scheme]])

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
		# values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
		values, base = np.histogram([x for x in reward_all[scheme] if x>-100], bins=NUM_BINS)
		cumulative = np.cumsum(values)
		# import pdb; pdb.set_trace()

		# ax.plot(base[:-1], cumulative/40.0)
		ax.plot(base[:-1], cumulative)
		# ax.plot(reward_all[scheme])

	colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
	for i,j in enumerate(ax.lines):
		j.set_color(colors[i])

	ax.legend(SCHEMES_REW, loc=4)

	plt.ylabel('CDF')
	plt.xlabel('total reward')
	plt.show()


	# ---- ---- ---- ----
	# check each trace
	# ---- ---- ---- ----

	for l in time_all[SCHEMES[0]]:
		schemes_check = True
		# import pdb; pdb.set_trace()
		if any(i in l for i in bad_files):
		# if 1:
		            # # import pdb; pdb.set_trace()
		            # # if reward<-100:
		            # #     import pdb; pdb.set_trace()
		            # print "reward is " + str(reward)
			for scheme in SCHEMES:
				if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
					schemes_check = False
					break
			if schemes_check:
				fig = plt.figure()

				ax = fig.add_subplot(311)
				for scheme in SCHEMES:
					ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
				colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
				for i,j in enumerate(ax.lines):
					j.set_color(colors[i])
				plt.title(l)
				plt.ylabel('bit rate selection (kbps)')

				ax = fig.add_subplot(312)
				for scheme in SCHEMES:
					ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
				colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
				for i,j in enumerate(ax.lines):
					j.set_color(colors[i])
				plt.ylabel('buffer size (sec)')

				ax = fig.add_subplot(313)
				for scheme in SCHEMES:
					ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
				colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
				for i,j in enumerate(ax.lines):
					j.set_color(colors[i])
				plt.ylabel('bandwidth (mbps)')
				plt.xlabel('time (sec)')

				SCHEMES_REW = []
				for scheme in SCHEMES:
					SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))

				ax.legend(SCHEMES_REW, loc=9, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 2.0)))
				plt.show()


if __name__ == '__main__':
	main()
