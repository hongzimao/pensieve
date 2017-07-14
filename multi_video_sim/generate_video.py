import numpy as np
import matplotlib.pyplot as plt


RANDOM_SEED = 42
NUM_VIDEOS = 1000
MAX_NUM_BITRATES = 10
MIN_NUM_BITRATES = 3
MAX_NUM_CHUNKS = 100
MIN_NUM_CHUNKS = 20
# bit rate candidates
# [200, 300, 450, 750, 1200, 1850, 2850, 4300, 6000, 8000]  # Kbps
MEAN_VIDEO_SIZE = [0.1, 0.15, 0.38, 0.6, 0.93, 1.43, 2.15, 3.25, 4.5, 6]  # MB
STD_VIDEO_SIZE_NOISE = 0.1
VIDEO_FOLDER = './videos/'


np.random.seed(RANDOM_SEED)
all_bitrate_idx = np.array(range(MAX_NUM_BITRATES))
mask_bitrate_idx_to_shuffle = np.array(range(MAX_NUM_BITRATES))

for video_idx in xrange(NUM_VIDEOS):
	num_bitrates = np.random.randint(MIN_NUM_BITRATES, MAX_NUM_BITRATES + 1)
	num_chunks = np.random.randint(MIN_NUM_CHUNKS, MAX_NUM_CHUNKS + 1)

	np.random.shuffle(mask_bitrate_idx_to_shuffle)

	mask_bitrate_idx = mask_bitrate_idx_to_shuffle[:num_bitrates]
	mask_bitrate_idx.sort()

	if np.all(mask_bitrate_idx == [1, 3, 4, 5, 6, 7]):
		# avoid using the same bitrates as the ones we do testing
		np.random.shuffle(mask_bitrate_idx_to_shuffle)
		mask_bitrate_idx = mask_bitrate_idx_to_shuffle[:num_bitrates]
		mask_bitrate_idx.sort()

	with open(VIDEO_FOLDER + str(video_idx), 'wb') as f:
		f.write(str(num_bitrates) + '\t' + str(num_chunks) + '\n')
		for i in xrange(MAX_NUM_BITRATES):
			if i in mask_bitrate_idx:
				f.write('1' + '\t')
			else:
				f.write('0' + '\t')
		f.write('\n')

		for _ in xrange(num_chunks):
			for i in xrange(num_bitrates):
				mean = MEAN_VIDEO_SIZE[mask_bitrate_idx[i]]
				noise = np.random.normal(1, STD_VIDEO_SIZE_NOISE)
				f.write(str(mean * noise) + '\t')
			f.write('\n')	
