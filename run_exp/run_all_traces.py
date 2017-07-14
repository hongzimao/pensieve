import os
import time
import json
import urllib
import subprocess


TRACE_PATH = '../cooked_traces/' 

with open('./chrome_retry_log', 'wb') as f:
	f.write('chrome retry log\n')

os.system('sudo sysctl -w net.ipv4.ip_forward=1')

ip_data = json.loads(urllib.urlopen("http://ip.jsontest.com/").read())
ip = str(ip_data['ip'])

ABR_ALGO = 'BB'
PROCESS_ID = 0
command_BB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'RB'
PROCESS_ID = 1
command_RB = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'FIXED'
PROCESS_ID = 2
command_FIXED = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'FESTIVE'
PROCESS_ID = 3
command_FESTIVE = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'BOLA'
PROCESS_ID = 4
command_BOLA = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'fastMPC'
PROCESS_ID = 5
command_fastMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'robustMPC'
PROCESS_ID = 6
command_robustMPC = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

ABR_ALGO = 'RL'
PROCESS_ID = 7
command_RL = 'python run_traces.py ' + TRACE_PATH + ' ' + ABR_ALGO + ' ' + str(PROCESS_ID) + ' ' + ip

proc_BB = subprocess.Popen(command_BB, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
proc_RB = subprocess.Popen(command_RB, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
proc_FIXED = subprocess.Popen(command_FIXED, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
proc_FESTIVE = subprocess.Popen(command_FESTIVE, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
proc_BOLA = subprocess.Popen(command_BOLA, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
proc_fastMPC = subprocess.Popen(command_fastMPC, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
proc_robustMPC = subprocess.Popen(command_robustMPC, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)
proc_RL = subprocess.Popen(command_RL, stdout=subprocess.PIPE, shell=True)
time.sleep(0.1)

proc_BB.wait()
proc_RB.wait()
proc_FIXED.wait()
proc_FESTIVE.wait()
proc_BOLA.wait()
proc_fastMPC.wait()
proc_robustMPC.wait()
proc_RL.wait()
