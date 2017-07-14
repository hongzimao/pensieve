import os
import numpy as np


VIDEO_SOURCE_FOLDER = '../video_server/'
VIDEO_FOLDER = 'video'
VIDEO_OUTPUT_FOLDER = './test_video/'
TOTAL_VIDEO_CHUNCK = 49
BITRATE_LEVELS = 6
MASK = [0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
M_IN_B = 1000000.0



video_chunk_sizes = []

for bitrate in xrange(BITRATE_LEVELS):

	video_chunk_sizes.append([])

	for chunk_idx in xrange(1, TOTAL_VIDEO_CHUNCK + 1):

		video_chunk_path = VIDEO_SOURCE_FOLDER + \
						   VIDEO_FOLDER + \
						   str(BITRATE_LEVELS - bitrate) + \
						   '/' + \
						   str(chunk_idx) + \
						   '.m4s'

		chunk_size = os.path.getsize(video_chunk_path) / M_IN_B
		video_chunk_sizes[bitrate].append(chunk_size)

	assert len(video_chunk_sizes[-1]) == TOTAL_VIDEO_CHUNCK

assert len(video_chunk_sizes) == BITRATE_LEVELS

with open(VIDEO_OUTPUT_FOLDER + '0', 'wb') as f:
	f.write(str(BITRATE_LEVELS) + '\t' + str(TOTAL_VIDEO_CHUNCK) + '\n')
	for m in MASK:
		f.write(str(m) + '\t')
	f.write('\n')
	for chunk_idx in xrange(TOTAL_VIDEO_CHUNCK):
		for i in xrange(BITRATE_LEVELS):
			f.write(str(video_chunk_sizes[i][chunk_idx]) + '\t')
		f.write('\n')
