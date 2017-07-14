- To download the raw data, follow

```
# FCC broadband dataset
$ wget http://data.fcc.gov/download/measuring-broadband-america/2016/data-raw-2016-jun.tar.gz
$ tar -xvzf data-raw-2016-jun.tar.gz -C fcc

# Norway HSDPA bandwidth logs
$ wget -r --no-parent --reject "index.html*" http://home.ifi.uio.no/paalh/dataset/hsdpa-tcp-logs/

# Belgium 4G/LTE bandwidth logs (bonus)
$ wget http://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip
$ unzip logs_all.zip -d belgium
```

- For actual video streaming over Mahimahi (http://mahimahi.mit.edu), the format of the traces should be converted to the following
```
Each line gives a timestamp in milliseconds (from the beginning of the
trace) and represents an opportunity for one 1500-byte packet to be
drained from the bottleneck queue and cross the link. If more than one
MTU-sized packet can be transmitted in a particular millisecond, the
same timestamp is repeated on multiple lines.
```
An example mahimahi trace `run_exp/12mbps` depicts 12Mbit/sec bandwidth. More details of mahimahi trace file requirements can be found in https://github.com/ravinet/mahimahi/tree/master/traces. In our case, the script `convert_mahimahi_format.py` in `traces/norway/` and `traces/fcc/` can be used. Some samples of preprocessed data can be downloaded in `cooked_traces` from https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja?dl=0.

- For simulations, the format should be `[time_stamp (sec), throughput (Mbit/sec)]`. Some sample can be downloaded in `train_sim_traces` and `test_sim_traces` from https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja?dl=0.