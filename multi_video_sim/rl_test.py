import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import tensorflow as tf
import a3c
import env


S_INFO = 7  # bit_rate, buffer_size, bandwidth_measurement, measurement_time, chunk_til_video_end
S_LEN = 10  # take how many frames in the past
A_DIM = 10
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 200  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [200,300,450,750,1200,1850,2850,4300,6000,8000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
M_IN_B = 1000000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './cooked_test_traces/'
TEST_VIDEO_FOLDER = './test_video/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]


def action_to_bitrate(action, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert action >= 0
    assert action < a_dim
    assert mask[action] == 1
    # index starts at 0, ':' is non-inclusive
    return np.sum(mask[:action])  

def bitrate_to_action(bitrate, mask, a_dim=A_DIM):
    assert len(mask) == a_dim
    assert bitrate >= 0 
    assert bitrate < np.sum(mask)
    cumsum_mask = np.cumsum(mask) - 1
    action = np.where(cumsum_mask == bitrate)[0][0]
    return action


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    net_env = env.Environment(fixed_env=True, 
                              trace_folder=TEST_TRACES, 
                              video_folder=TEST_VIDEO_FOLDER)

    log_path = LOG_FILE + '_' + net_env.all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action = bitrate_to_action(bit_rate, net_env.video_masks[net_env.video_idx])
        last_action = action

        s_batch = [np.zeros((S_INFO, S_LEN))]

        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, \
                rebuf, video_chunk_size, end_of_video, \
                video_chunk_remain, video_num_chunks, \
                next_video_chunk_size, mask = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            reward = VIDEO_BIT_RATE[action] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[action] -
                                               VIDEO_BIT_RATE[last_action]) / M_IN_K

            last_bit_rate = bit_rate
            last_action = action

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[action]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[action] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K
            state[4, -1] = video_chunk_remain / float(video_num_chunks)
            state[5, :] = -1
            nxt_chnk_cnt = 0
            for i in xrange(A_DIM):
                if mask[i] == 1:
                    state[5, i] = next_video_chunk_size[nxt_chnk_cnt] / M_IN_B
                    nxt_chnk_cnt += 1
            assert(nxt_chnk_cnt) == np.sum(mask)
            state[6, -A_DIM:] = mask

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))

            # the action probability should correspond to number of bit rates
            assert len(action_prob[0]) == np.sum(mask)

            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states
            action = bitrate_to_action(bit_rate, mask)

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                del s_batch[:]

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                action = bitrate_to_action(bit_rate, mask)
                last_action = action

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                entropy_record = []

                video_count += 1

                if video_count >= len(net_env.all_cooked_bw):
                    break

                log_path = LOG_FILE + '_' + net_env.all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()
