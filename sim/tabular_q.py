#
# Note: this code is not part of Pensieve.
# 
# It serves as a comparison for another line of RL approaches, 
# which utilize Q learning in the basic tabular form.
#
# More details can be found in the following papers. 
# - Design of a Q-Learning based client quality selection algorithm for HTTP Adaptive Video Streaming - https://www.researchgate.net/publication/258374689_Design_of_a_Q-Learning_based_client_quality_selection_algorithm_for_HTTP_Adaptive_Video_Streaming
# - Design and optimisation of a (FA)Q-learning-based HTTP adaptive streaming client - http://www.tandfonline.com/doi/pdf/10.1080/09540091.2014.885273?needAccess=true
# - A Learning-Based Algorithm for Improved Bandwidth-Awareness of Adaptive Streaming Clients - http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7140285
# - Online learning adaptation strategy for DASH clients - http://dl.acm.org/citation.cfm?id=2910603
#

import os
import sys
import numpy as np

import env
import fixed_env
import load_trace


BW_MIN = 0  # minimum bandwidth in Mbit/sec
BW_MAX = 10  # maximum bandwidth in Mbit/sec
D_BW = 1  # bandwidth granularity
BF_MIN = 0  # minimum buffer in sec
BF_MAX = 60  # maxiimum buffer in sec
D_BF = 1  # buffer granularity
BR_LV = 6  # number of bitrate levels
N_CHUNK = 50  # number of chunk until the end
LR_RATE = 1e-3  # learning rate
GAMMA = 0.99  # discount factor
DEFAULT_QUALITY = 1  # default video quality without agent
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
M_IN_K = 1000.0
BITS_IN_BYTE = 8.0
TEST_INTERVAL = 1000
EXP_RATE_MIN = 0
EXP_RATE_DECAY = 10000.0
TEST_LOG_PATH = './results/log_test'
TEST_LOG_FOLDER = './test_results/'
Q_TABLE_PATH = None


class Tabular_Q(object):
    def __init__(self):
        self.q_table = {}

        if Q_TABLE_PATH is not None:
            self.q_table = np.load(Q_TABLE_PATH)
        else:
            # initialize the q table
            for bw in np.linspace(BW_MIN, BW_MAX, (BW_MAX - BW_MIN) / D_BW + 1):
                for bf in np.linspace(BF_MIN, BF_MAX, (BF_MAX - BF_MIN) / D_BF + 1):
                    for br in xrange(BR_LV):
                        for c in xrange(N_CHUNK):
                            for a in xrange(BR_LV):
                                self.q_table[(bw, bf, br, c, a)] = 0.0

        self.exp_rate = 1.0

    def get_q_action(self, state, deterministic=False):
        bw = state[0]
        bf = state[1]
        br = state[2]
        c = state[3]

        if np.random.rand() < self.exp_rate and not deterministic:
            act = np.random.randint(BR_LV)
        else:
            max_q = - np.inf
            act = -1
            for a in xrange(BR_LV):
                q = self.q_table[(bw, bf, br, c, a)]
                if q > max_q:
                    act = a
                    max_q = q
            assert act != -1

        if self.exp_rate > EXP_RATE_MIN and not deterministic:
            self.exp_rate -= 1.0 / EXP_RATE_DECAY

        return act

    def train_q(self, state, act, reward, next_state, terminal):
        bw = state[0]
        bf = state[1]
        br = state[2]
        c = state[3]

        n_bw = next_state[0]
        n_bf = next_state[1]
        n_br = next_state[2]
        n_c = next_state[3]

        if terminal:
            max_next_q = 0
        else:
            max_next_q = - np.inf        
            for a in xrange(BR_LV):
                q = self.q_table[(n_bw, n_bf, n_br, n_c, a)]
                if q > max_next_q:
                    max_next_q = q

        curr_q = self.q_table[(bw, bf, br, c, act)]

        q_diff = GAMMA * max_next_q + reward - curr_q

        self.q_table[(bw, bf, br, c, act)] += LR_RATE * q_diff


def testing(tabular_q, epoch):

    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)

    all_cooked_time, all_cooked_bw, all_file_names = \
        load_trace.load_trace('./cooked_test_traces/')
    test_net_env = fixed_env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw)

    log_path = TEST_LOG_FOLDER + 'log_' + all_file_names[test_net_env.trace_idx]
    log_file = open(log_path, 'wb')

    time_stamp = 0
    video_count = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    state = [0, 0, 0, 0]

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            test_net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        last_bit_rate = bit_rate

        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
        log_file.flush()

        bw = float(video_chunk_size) / float(delay) / M_IN_K * BITS_IN_BYTE # Mbit/sec
        bw = min(int(bw / D_BW) * D_BW, BW_MAX)
        bf = min(int(buffer_size / D_BF) * D_BF, BF_MAX)
        br = bit_rate
        c = min(video_chunk_remain, N_CHUNK - 1)
        state = [bw, bf, br, c]

        bit_rate = tabular_q.get_q_action(state, deterministic=True)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            state = [0, 0, 0, 0]

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = TEST_LOG_FOLDER + 'log_' + all_file_names[test_net_env.trace_idx]
            log_file = open(log_path, 'wb')

    with open(TEST_LOG_PATH, 'ab') as log_file:
         # append test performance to the log
        rewards = []
        test_log_files = os.listdir(TEST_LOG_FOLDER)
        for test_log_file in test_log_files:
            reward = []
            with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
                for line in f:
                    parse = line.split()
                    try:
                        reward.append(float(parse[-1]))
                    except IndexError:
                        break
            rewards.append(np.sum(reward[1:]))

        rewards = np.array(rewards)

        rewards_min = np.min(rewards)
        rewards_5per = np.percentile(rewards, 5)
        rewards_mean = np.mean(rewards)
        rewards_median = np.percentile(rewards, 50)
        rewards_95per = np.percentile(rewards, 95)
        rewards_max = np.max(rewards)

        log_file.write(str(epoch) + '\t' +
                       str(rewards_min) + '\t' +
                       str(rewards_5per) + '\t' +
                       str(rewards_mean) + '\t' +
                       str(rewards_median) + '\t' +
                       str(rewards_95per) + '\t' +
                       str(rewards_max) + '\n')
        log_file.flush()



def main():
    np.random.seed(42)

    os.system('rm ' + TEST_LOG_PATH)

    ta_q = Tabular_Q()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()

    epoch = 0
    time_stamp = 0

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    state = [0, 0, 0, 0]

    while True:

        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        epoch += 1

        bw = float(video_chunk_size) / float(delay) / M_IN_K * BITS_IN_BYTE # Mbit/sec
        bw = min(int(bw / D_BW) * D_BW, BW_MAX)
        bf = min(int(buffer_size / D_BF) * D_BF, BF_MAX)
        br = bit_rate
        c = min(video_chunk_remain, N_CHUNK - 1)
        next_state = [bw, bf, br, c]

        ta_q.train_q(state, bit_rate, reward, next_state, end_of_video)

        state = next_state
        last_bit_rate = bit_rate

        bit_rate = ta_q.get_q_action(state)

        if end_of_video:
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            state = [0, 0, 0, 0]

        if epoch % TEST_INTERVAL == 0:
            testing(ta_q, epoch)
            np.save(TEST_LOG_PATH + '_q_table.npy', ta_q.q_table)


if __name__ == '__main__':
    main()