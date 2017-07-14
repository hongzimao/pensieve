import os
import sys
import signal
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from pyvirtualdisplay import Display
from time import sleep

# TO RUN: download https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz
# run sudo apt-get install python-setuptools
# run sudo apt-get install xvfb
# after untar, run sudo python setup.py install
# follow directions here: https://pypi.python.org/pypi/PyVirtualDisplay to install pyvirtualdisplay

# For chrome, need chrome driver: https://code.google.com/p/selenium/wiki/ChromeDriver
# chromedriver variable should be path to the chromedriver
# the default location for firefox is /usr/bin/firefox and chrome binary is /usr/bin/google-chrome
# if they are at those locations, don't need to specify


def timeout_handler(signum, frame):
	raise Exception("Timeout")

abr_algo = sys.argv[1]
run_time = int(sys.argv[2])
exp_id = sys.argv[3]

# ---------------------------------------------------
# ---- change localhost in url to server address ----
# ---------------------------------------------------
#          |
#          v
url = 'localhost/' + 'myindex_' + abr_algo + '.html'

# timeout signal
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(run_time + 30)
	
try:
	# copy over the chrome user dir
	default_chrome_user_dir = '../abr_browser_dir/chrome_data_dir'
	chrome_user_dir = '/tmp/chrome_user_dir_real_exp_' + abr_algo
	os.system('rm -r ' + chrome_user_dir)
	os.system('cp -r ' + default_chrome_user_dir + ' ' + chrome_user_dir)
	
	# start abr algorithm server
	if abr_algo == 'RL':
		command = 'exec /usr/bin/python ../rl_server/rl_server_no_training.py ' + exp_id
	elif abr_algo == 'fastMPC':
		command = 'exec /usr/bin/python ../rl_server/mpc_server.py ' + exp_id
	elif abr_algo == 'robustMPC':
		command = 'exec /usr/bin/python ../rl_server/robust_mpc_server.py ' + exp_id
	else:
		command = 'exec /usr/bin/python ../rl_server/simple_server.py ' + abr_algo + ' ' + exp_id
	
	proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	sleep(2)
	
	# to not display the page in browser
	display = Display(visible=0, size=(800,600))
	display.start()
	
	# initialize chrome driver
	options=Options()
	chrome_driver = '../abr_browser_dir/chromedriver'
	options.add_argument('--user-data-dir=' + chrome_user_dir)
	options.add_argument('--ignore-certificate-errors')
	driver=webdriver.Chrome(chrome_driver, chrome_options=options)
	
	# run chrome
	driver.set_page_load_timeout(10)
	driver.get(url)
	
	sleep(run_time)
	
	driver.quit()
	display.stop()
	
	# kill abr algorithm server
	proc.send_signal(signal.SIGINT)
	# proc.kill()
	
	print 'done'
	
except Exception as e:
	try: 
		display.stop()
	except:
		pass
	try:
		driver.quit()
	except:
		pass
	try:
		proc.send_signal(signal.SIGINT)
	except:
		pass
	
	print e	

