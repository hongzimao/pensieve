This set of code runs experiments over Mahimahi. It loads the traces from `cooked_traces/` and invokes ABR servers from `rl_servers/`. Selenium on top of a PyVirtualDisplay is used and the actual graphics is disabled.

Trained RL model needs to be saved in `rl_server/results/`. We provided a sample pretrained model with linear QoE as the reward signal. It can be loaded by setting `NN_MODEL = '../rl_server/results/pretrain_linear_reward.ckpt'` in `rl_server/rl_server_no_training.py`.

Traces need to be put in `cooked_traces/` (in Pensieve home directory), in Mahimahi format. The format details can be found if `traces/` and we provide some preprocessed data, which can be downloaded from `cooked_traces` at https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja?dl=0. 

RL, robustMPC, MPC are implemented in `rl_servers/`. Other ABR schemes, namely BB, RB, Festival, BOLA and DASH original, are natively supported in `dash.js/`, where the switch `abrAlgo` can be found in `dash.js/src/streaming/controllers/AbrController.js`. These algorithms are called from specific HTML files in `video_server/`.

To conduct the experiment, run
```
python run_all_traces.py
```

To view the results, modify `SCHEMES` in `plot_results.py` (it checks the file name of the log and matches to the corresponding ABR algorithm), then run 
```
python plot_results.py
```