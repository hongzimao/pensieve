This set of code runs experiments over real world networks. Selenium on top of a PyVirtualDisplay is used and the actual graphics is disabled.

Trained RL model needs to be saved in `rl_server/results/`. We provided a sample pretrained model with linear QoE as the reward signal. It can be loaded by setting `NN_MODEL = '../rl_server/results/pretrain_linear_reward.ckpt'` in `rl_server/rl_server_no_training.py`.

RL, robustMPC, MPC are implemented in `rl_servers/`. Other ABR schemes, namely BB, RB, Festival, BOLA and DASH original, are natively supported in `dash.js/`, where the switch `abrAlgo` can be found in `dash.js/src/streaming/controllers/AbrController.js`. These algorithms are called from specific HTML files in `video_server/`. Experiments run over RL, robustMPC, MPC and BOLA in random shuffles.

To conduct the experiment, modify `url` in `run_video.py` to the server address, and then run
```
python run_exp.py
```

To view the results, modify `SCHEMES` in `plot_results.py` (it checks the file name of the log and matches to the corresponding ABR algorithm), then run 
```
python plot_results.py
```