import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './video_size_'
BUFFER_SIZE = 10
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.buffer = [0]*BUFFER_SIZE

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in xrange(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    # def get_video_chunk(self, quality):
    #     #self.mahimahi_ptr will denote the start of the buffer, once it is played it will be moved 1, with the array looping back?
    #     assert quality >= 0
    #     assert quality < BITRATE_LEVELS
    #     if quality < buffer_size:
    #         #download EL
    #         # self.video_size[quality]
    #         if self.buffer[quality] < len(VIDEO_BIT_RATE):
    #             video_chunk_size = self.video_size[self.buffer[quality]+1][self.video_chunk_counter + quality]
    #             self.buffer[quality] = self.buffer[quality] + 1
    #             self.z_t = self.z_t + self.buffer[quality]
    #
    #     # else:
    #     #     #we dont do anything idiot, it has to be in the buffer ffs
    #     #     #send BL
    #     #     throughput = self.cooked_bw[self.mahimahi_ptr] \
    #     #                  * B_IN_MB / BITS_IN_BYTE
    #     #     duration = self.cooked_time[self.mahimahi_ptr] \
    #     #                - self.last_mahimahi_time
    #     #
    #     #     packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
    #     #     #you have to update buffer along with time? how to do that?
    #     #     if self.mahimahi_ptr+1 >= len(self.cooked_bw):#condition because the video is ended??
    #     #         # loop back in the beginning
    #     #         # note: trace file starts with time 0
    #     #
    #     #     if video_chunk_counter_sent + packet_payload > video_chunk_size:
    #     #         #this is the final BL, even more than that maybe, so we have to end the loop until or should we
    #     #         #check for the time of chunk being run in the player
    #     #             # pass:
    #     #             # pass
    #     #     # self.buffer = self.buffer[1:]
    #
    #     video_chunk_size = self.video_size[quality][self.video_chunk_counter]
    #
    #     # use the delivery opportunity in mahimahi
    #     delay = 0.0  # in ms
    #     video_chunk_counter_sent = 0  # in bytes
    #
    #     while True:  # download video chunk over mahimahi
    #         #defining throughput, how much data is being sent
    #         throughput = self.cooked_bw[self.mahimahi_ptr+quality] \
    #                      * B_IN_MB / BITS_IN_BYTE
    #         #defining duration #how much time is the data representing
    #         duration = self.cooked_time[self.mahimahi_ptr+quality] \
    #                    - self.last_mahimahi_time
    #
    #         #total packet payload
    #         packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
    #
    #         #if condition for checking if the shit is ending
    #         if video_chunk_counter_sent + packet_payload > video_chunk_size:
    #
    #             fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
    #                               throughput / PACKET_PAYLOAD_PORTION
    #             delay += fractional_time
    #             self.last_mahimahi_time += fractional_time
    #             # assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
    #             break
    #
    #         #adding packet payload so that we know how much data is still there to be precessed
    #         video_chunk_counter_sent += packet_payload
    #         #why delay?
    #         delay += duration
    #         #last data bit time
    #         self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
    #         self.mahimahi_ptr += 1
    #
    #         if self.mahimahi_ptr >= len(self.cooked_bw):#condition because the video is ended??
    #             # loop back in the beginning
    #             # note: trace file starts with time 0
    #             self.mahimahi_ptr = 1
    #             self.last_mahimahi_time = 0
    #
    #     delay *= MILLISECONDS_IN_SECOND
    #     delay += LINK_RTT
    #
	# # add a multiplicative noise to the delay
	# # delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH) #check if this is right
    #     delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
    #     # rebuffer time
    #     rebuf = np.maximum(delay - self.buffer_size, 0.0)
    #
    #     # update the buffer
    #     self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)
    #
    #     # add in the new chunk
    #     self.buffer_size += VIDEO_CHUNCK_LEN
    #
    #     # sleep if buffer gets too large
    #     sleep_time = 0
    #     if self.buffer_size > BUFFER_THRESH:
    #         # exceed the buffer limit
    #         # we need to skip some network bandwidth here
    #         # but do not add up the delay
    #         drain_buffer_time = self.buffer_size - BUFFER_THRESH
    #         sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
    #                      DRAIN_BUFFER_SLEEP_TIME
    #         self.buffer_size -= sleep_time
    #
    #         while True:
    #             duration = self.cooked_time[self.mahimahi_ptr] \
    #                        - self.last_mahimahi_time
    #             if duration > sleep_time / MILLISECONDS_IN_SECOND:
    #                 self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
    #                 break
    #             sleep_time -= duration * MILLISECONDS_IN_SECOND
    #             self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
    #             self.mahimahi_ptr += 1
    #
    #             if self.mahimahi_ptr >= len(self.cooked_bw):
    #                 # loop back in the beginning
    #                 # note: trace file starts with time 0
    #                 self.mahimahi_ptr = 1
    #                 self.last_mahimahi_time = 0
    #
    #     # the "last buffer size" return to the controller
    #     # Note: in old version of dash the lowest buffer is 0.
    #     # In the new version the buffer always have at least
    #     # one chunk of video
    #     return_buffer_size = self.buffer_size
    #
    #     self.video_chunk_counter += 1
    #     video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter
    #
    #     end_of_video = False
    #     if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
    #         end_of_video = True
    #         self.buffer_size = 0
    #         self.video_chunk_counter = 0
    #
    #         # pick a random trace file
    #         self.trace_idx = np.random.randint(len(self.all_cooked_time))
    #         self.cooked_time = self.all_cooked_time[self.trace_idx]
    #         self.cooked_bw = self.all_cooked_bw[self.trace_idx]
    #
    #         # randomize the start point of the video
    #         # note: trace file starts with time 0
    #         self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
    #         self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
    #
    #     next_video_chunk_sizes = []
    #     for i in xrange(BITRATE_LEVELS):
    #         next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
    #
    #     return delay, \
    #         sleep_time, \
    #         return_buffer_size / MILLISECONDS_IN_SECOND, \
    #         rebuf / MILLISECONDS_IN_SECOND, \
    #         video_chunk_size, \
    #         next_video_chunk_sizes, \
    #         end_of_video, \
    #         video_chunk_remain

    def get_video_chunk(self, bit_rate_index):
        #self.mahimahi_ptr will denote the start of the buffer, once it is played it will be moved 1, with the array looping back?
        assert bit_rate_index >= 0
        # assert bit_rate_index < BITRATE_LEVELS
        if bit_rate_index < buffer_size:
            #download EL
            # self.video_size[bit_rate_index]
            if self.buffer[bit_rate_index] < len(VIDEO_BIT_RATE):
                video_chunk_size = self.video_size[self.buffer[bit_rate_index]+1][self.video_chunk_counter + bit_rate_index]
                self.buffer[bit_rate_index] = self.buffer[bit_rate_index] + 1
                self.z_t = self.z_t + self.buffer[bit_rate_index]

        video_chunk_size = self.video_size[self.buffer[bit_rate_index]][self.video_chunk_counter] #change this
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            #defining throughput, how much data is being sent
            throughput = self.cooked_bw[self.mahimahi_ptr + bit_rate_index] \
                         * B_IN_MB / BITS_IN_BYTE
            #defining duration #how much time is the data representing
            duration = self.cooked_time[self.mahimahi_ptr + bit_rate_index] \
                       - self.last_mahimahi_time

            #total packet payload
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            #if condition for checking if the bandwidth is finished, but in grad you need no bandwidth check, you just need EL check?
            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                # assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            #adding packet payload so that we know how much data is still there to be precessed
            video_chunk_counter_sent += packet_payload
            #why delay?
            delay += duration
            #last data bit time
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):#condition because the video is ended??
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

	# add a multiplicative noise to the delay
	# delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH) #check if this is right
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        # for i in xrange(BITRATE_LEVELS):
        #     next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain
