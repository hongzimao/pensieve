Make sure actual video files are stored in `video_server/video[1-6]`, then run
```
python get_video_sizes
```

Put training data in `sim/cooked_traces` and testing data in `sim/cooked_test_traces` (need to create folders). The trace format for simulation is `[time_stamp (sec), throughput (Mbit/sec)]`. Sample training/testing data we used can be downloaded separately from `train_sim_traces` and `test_sim_traces` in https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja?dl=0. More details of data preparation can be found in `traces/`.

To train a model, run 
```
python multi_agent.py
```

As reported by the A3C paper (http://proceedings.mlr.press/v48/mniha16.pdf) and a faithful implementation (https://openreview.net/pdf?id=Hk3mPK5gg), we also found the exploration factor in the actor network quite crucial for achieving good performance. A general strategy to train our system is to first set `ENTROPY_WEIGHT` in `a3c.py` to be a large value (in the scale of 1 to 5) in the beginning, then gradually reduce the value to `0.1` (after at least 100,000 iterations). 


The training process can be monitored in `sim/results/log_test` (validation) and `sim/results/log_central` (training). Tensorboard (https://www.tensorflow.org/get_started/summaries_and_tensorboard) is also used to visualize the training process, which can be invoked by running
```
python -m tensorflow.tensorboard --logdir=./results/
```
where the plot can be viewed at `localhost:6006` from a browser. 

Trained model will be saved in `sim/results/`. We provided a sample pretrained model with linear QoE as the reward signal. It can be loaded by setting `NN_MODEL = './results/pretrain_linear_reward.ckpt'` in `multi_agent.py`.