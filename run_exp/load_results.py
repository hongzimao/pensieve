import os
import numpy as np
import matplotlib.pyplot as plt


RESULTS_FOLDER = './results/'


def main():
	total_reward_all = {}
	total_reward_all['BB'] = {}
	total_reward_all['RB'] = {}
	total_reward_all['FIXED'] = {}
	total_reward_all['FESTIVE'] = {}
	total_reward_all['BOLA'] = {}

	log_files = os.listdir(RESULTS_FOLDER)
	for log_file in log_files:

		R = 0
		with open(RESULTS_FOLDER + log_file, 'rb') as f:
			for line in f:
				parse = line.split()
				R += float(parse[-1])
		
		print log_file

		if 'BB' in log_file:
			total_reward_all['BB'][log_file[7:]] = R
		elif 'RB' in log_file:
			total_reward_all['RB'][log_file[7:]] = R
		elif 'FIXED' in log_file:
			total_reward_all['FIXED'][log_file[10:]] = R
		elif 'FESTIVE' in log_file:
			total_reward_all['FESTIVE'][log_file[12:]] = R
		elif 'BOLA' in log_file:
			total_reward_all['BOLA'][log_file[9:]] = R

		else:
			print "Error: log name doesn't contain proper abr schemes."
		
		
	log_file_all = []
	BB_reward_all = []
	RB_reward_all = []
	FIXED_reward_all = []
	FESTIVE_reward_all = []
	BOLA_reward_all = []

	for l in total_reward_all['BB']:
		if l in total_reward_all['RB'] and \
		   l in total_reward_all['FIXED'] and \
		   l in total_reward_all['FESTIVE'] and \
		   l in total_reward_all['BOLA']:
				log_file_all.append(l)
				BB_reward_all.append(total_reward_all['BB'][l])
				RB_reward_all.append(total_reward_all['RB'][l])
				FIXED_reward_all.append(total_reward_all['FIXED'][l])
				FESTIVE_reward_all.append(total_reward_all['FESTIVE'][l])
				BOLA_reward_all.append(total_reward_all['BOLA'][l])

	BB_total_reward = np.mean(BB_reward_all)
	RB_total_reward = np.mean(RB_reward_all)
	FIXED_total_reward = np.mean(FIXED_reward_all)
	FESTIVE_total_reward = np.mean(FESTIVE_reward_all)
	BOLA_total_reward = np.mean(BOLA_reward_all)

	plt.plot(BB_reward_all)
	plt.plot(RB_reward_all)
	plt.plot(FIXED_reward_all)
	plt.plot(FESTIVE_reward_all)
	plt.plot(BOLA_reward_all)

	plt.legend(['BB ' + str(BB_total_reward),
	            'RB ' + str(RB_total_reward),
	            'FIXED ' + str(FIXED_total_reward),
	            'FESTIVE ' + str(FESTIVE_total_reward), 
	            'BOLA ' + str(BOLA_total_reward)])
	
	plt.ylabel('total reward')
	plt.xlabel('trace index')
	plt.show()


if __name__ == '__main__':
	main()