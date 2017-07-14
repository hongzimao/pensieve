# Too slow! Use dp.cc instead.

import numpy as np
import load_trace


MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
DELAY_FACTOR = 3
VIDEO_CHUNCK_LEN = 4000.0  # (ms), every time add this amount to buffer
BITRATE_LEVELS = 5
TOTAL_VIDEO_CHUNCK = 65
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # (sec)
PACKET_SIZE = 1500  # bytes
DT = 1  # time granularity
VIDEO_BIT_RATE = [350, 600, 1000, 2000, 3000]  # Kbps
DEFAULT_QUALITY = 0
M_IN_K = 1000.0
REBUF_PENALTY = 3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
VIDEO_SIZE_FILE = './video_size_'


# pre-compute all possible download time
def get_download_time(total_video_chunks, quan_time, quan_bw, dt, video_size, bitrate_levels):
    download_time = np.zeros([total_video_chunks, len(quan_time), bitrate_levels])
    for t in xrange(len(quan_time)):
        print t, len(quan_time)
        for n in xrange(total_video_chunks):
            for b in xrange(bitrate_levels):
                chunk_size = video_size[b][n]  # in bytes
                downloaded = 0.0
                time_spent = 0.0
                quan_idx = t
                while True:
                    curr_bw = quan_bw[quan_idx]  # in Mbit/sec
                    downloaded += curr_bw * dt / BITS_IN_BYTE * B_IN_MB * PACKET_PAYLOAD_PORTION
                    quan_idx += 1
                    if downloaded >= chunk_size or quan_idx >= len(quan_time):
                        break
                    time_spent += dt  # lower bound the time spent
                download_time[n, t, b] = time_spent

    return download_time


def restore_or_compute_download_time(download_time, video_chunk, quan_t_idx, bit_rate,
                                     quan_time, quan_bw, dt, video_size):
    if (video_chunk, quan_t_idx, bit_rate) in download_time:
        return download_time[video_chunk, quan_t_idx, bit_rate]
    else:
        chunk_size = video_size[bit_rate][video_chunk]  # in bytes
        downloaded = 0.0
        time_spent = 0.0
        quan_idx = quan_t_idx
        while True:
            curr_bw = quan_bw[quan_idx]  # in Mbit/sec
            downloaded += curr_bw * dt / BITS_IN_BYTE * B_IN_MB * PACKET_PAYLOAD_PORTION
            quan_idx += 1
            if downloaded >= chunk_size or quan_idx >= len(quan_time):
                break
            time_spent += dt  # lower bound the time spent
        download_time[video_chunk, quan_t_idx, bit_rate] = time_spent
        return time_spent


def main():
    all_cooked_time, all_cooked_bw = load_trace.load_trace()

    video_size = {}  # in bytes
    for bitrate in xrange(BITRATE_LEVELS):
        video_size[bitrate] = []
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))

    # assert len(all_cooked_time) == len(all_cooked_bw)
    # for cooked_data_idx in xrange(len(all_cooked_time))

    cooked_time = all_cooked_time[0]
    cooked_bw = all_cooked_bw[0]

    # -----------------------------------------
    # step 1: quantize the time and bandwidth
    # -----------------------------------------
    total_time_pt = int(np.ceil(cooked_time[-1] / DT))

    quan_time = np.linspace(np.floor(cooked_time[0]),
                            np.ceil(cooked_time[-1]),
                            total_time_pt+1)
    quan_bw = np.zeros(len(quan_time))

    curr_time_idx = 0
    for i in xrange(len(quan_bw)):
        while curr_time_idx < len(cooked_time) - 1 and \
              cooked_time[curr_time_idx] < quan_time[i]:
            curr_time_idx += 1
        quan_bw[i] = cooked_bw[curr_time_idx]

    # ----------------------------------------
    # step 2: cap the max time and max buffer
    # ----------------------------------------
    max_video_contents = np.sum(video_size[BITRATE_LEVELS - 1])  # in bytes
    total_bw = np.sum(quan_bw) * DT  # in MBit

    t_portion = max_video_contents / (total_bw * B_IN_MB * PACKET_PAYLOAD_PORTION / BITS_IN_BYTE)

    t_max = int(np.ceil(np.ceil(cooked_time[-1]) * t_portion))

    t_max_idx = int(np.ceil(t_max / DT))
    b_max_idx = t_max_idx

    full_quan_time = quan_time
    full_quan_bw = quan_bw

    for i in xrange(int(np.ceil(t_portion))):
        full_quan_time = np.append(full_quan_time,
                                   (quan_time[1:] + full_quan_time[-1]))
        full_quan_bw = np.append(full_quan_bw, quan_bw[1:])

    quan_time = full_quan_time
    quan_bw = full_quan_bw

    assert quan_time[-1] >= t_max

    # -----------------------------------------------------------
    # (optional) step 3: pre=compute the download time of chunks
    # download_time(chunk_idx, quan_time, bit_rate)
    # -----------------------------------------------------------
    all_download_time = {}

    # print "Pre-compute the download time table"
    # all_download_time = get_download_time(total_video_chunks=TOTAL_VIDEO_CHUNCK,
    #                                   quan_time=quan_time,
    #                                   quan_bw=quan_bw,
    #                                   dt=DT,
    #                                   video_size=video_size,
    #                                   bitrate_levels=BITRATE_LEVELS)

    # -----------------------------
    # step 4: dynamic programming
    # -----------------------------
    total_reward = {}
    last_dp_pt = {}

    # initialization, take default quality at start off
    download_time = \
        restore_or_compute_download_time(
            all_download_time, 0, 0, DEFAULT_QUALITY,
            quan_time, quan_bw, DT, video_size)
    first_chunk_finish_time = download_time + LINK_RTT / M_IN_K
    first_chunk_finish_idx = int(np.floor(first_chunk_finish_time / DT))
    buffer_size = int(VIDEO_CHUNCK_LEN / M_IN_K / DT)

    total_reward[(0, first_chunk_finish_idx, buffer_size, DEFAULT_QUALITY)] = \
        VIDEO_BIT_RATE[DEFAULT_QUALITY] / M_IN_K \
        - REBUF_PENALTY * first_chunk_finish_time
    last_dp_pt[(0, first_chunk_finish_idx, buffer_size, DEFAULT_QUALITY)] = (0, 0, 0, 0)

    for n in xrange(1, TOTAL_VIDEO_CHUNCK):
        print n, TOTAL_VIDEO_CHUNCK
        for t in xrange(t_max_idx):
            for b in xrange(b_max_idx):
                for m in xrange(BITRATE_LEVELS):
                    if (n - 1, t, b, m) in total_reward:
                        for new_bit_rate in xrange(BITRATE_LEVELS):
                            download_time = \
                                restore_or_compute_download_time(
                                    all_download_time, n, t, new_bit_rate,
                                    quan_time, quan_bw, DT, video_size)

                            buffer_size = quan_time[b]
                            rebuf = np.maximum(download_time - buffer_size, 0.0)

                            r = VIDEO_BIT_RATE[new_bit_rate] / M_IN_K \
                                - REBUF_PENALTY * rebuf \
                                - SMOOTH_PENALTY * np.abs(
                                    VIDEO_BIT_RATE[new_bit_rate] -
                                    VIDEO_BIT_RATE[m]) / M_IN_K

                            buffer_size = np.maximum(buffer_size - download_time, 0.0)
                            buffer_size += VIDEO_CHUNCK_LEN / M_IN_K

                            buffer_idx = int(buffer_size / DT)

                            new_time_idx = int(np.floor((quan_time[t] +
                                                         download_time +
                                                         LINK_RTT / M_IN_K) / DT))

                            new_total_reward = total_reward[(n - 1, t, b, m)] + r
                            if (n, new_time_idx, buffer_idx, new_bit_rate) not in total_reward:
                                total_reward[(n, new_time_idx, buffer_idx, new_bit_rate)] = \
                                        new_total_reward
                                last_dp_pt[(n, new_time_idx, buffer_idx, new_bit_rate)] = \
                                    (n - 1, t, b, m)
                            else:
                                if new_total_reward > total_reward[
                                   (n, new_time_idx, buffer_idx, new_bit_rate)]:
                                    total_reward[(n, new_time_idx, buffer_idx, new_bit_rate)] = \
                                        new_total_reward
                                last_dp_pt[(n, new_time_idx, buffer_idx, new_bit_rate)] = \
                                    (n - 1, t, b, m)

    # ---------------------------------
    # step 5: get the max total reward
    # ---------------------------------
    optimal_total_reward = - np.inf
    end_dp_pt = None
    for k in total_reward:
        if k[0] == TOTAL_VIDEO_CHUNCK - 1:
            if total_reward[k] > optimal_total_reward:
                optimal_total_reward = total_reward[k]
                end_dp_pt = last_dp_pt[k]

    print optimal_total_reward
    if end_dp_pt is not None:
        while end_dp_pt != (0, 0, 0, 0):
            print end_dp_pt
            end_dp_pt = last_dp_pt[end_dp_pt]


if __name__ == '__main__':
    main()