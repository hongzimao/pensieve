import os

start_dir = os.getcwd()

# mahimahi
os.system("sudo sysctl -w net.ipv4.ip_forward=1")
os.system("sudo add-apt-repository -y ppa:keithw/mahimahi")
os.system("sudo apt-get -y update")
os.system("sudo apt-get -y install mahimahi")

# apache server
os.system("sudo apt-get -y install apache2")

# selenium
os.system("wget 'https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz'")
os.system("sudo apt-get -y install python-setuptools python-pip xvfb xserver-xephyr tightvncserver unzip")
os.system("tar xvzf selenium-2.39.0.tar.gz")
selenium_dir = start_dir + "/selenium-2.39.0"
os.chdir( selenium_dir )
os.system("sudo python setup.py install" )
os.system("sudo sh -c \"echo 'DBUS_SESSION_BUS_ADDRESS=/dev/null' > /etc/init.d/selenium\"")

# py virtual display
os.chdir( start_dir )
os.system("sudo pip install pyvirtualdisplay")
os.system("wget 'https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb' ")
os.system("sudo dpkg -i google-chrome-stable_current_amd64.deb")
os.system("sudo apt-get -f -y install")

# tensorflow
os.system("sudo apt-get -y install python-pip python-dev")
os.system("sudo pip install tensorflow")

# tflearn
os.system("sudo pip install tflearn")
os.system("sudo apt-get -y install python-h5py")
os.system("sudo apt-get -y install python-scipy")

# matplotlib
os.system("sudo apt-get -y install python-matplotlib")

# copy the webpage files to /var/www/html
os.chdir( start_dir )
os.system("sudo cp video_server/myindex_*.html /var/www/html")
os.system("sudo cp video_server/dash.all.min.js /var/www/html")
os.system("sudo cp -r video_server/video* /var/www/html")
os.system("sudo cp video_server/Manifest.mpd /var/www/html")

# make results directory
os.system("mkdir cooked_traces")
os.system("mkdir rl_server/results")
os.system("mkdir run_exp/results")
os.system("mkdir real_exp/results")

# need to copy the trace and pre-trained NN model
print "Need to put trace files in 'pensieve/cooked_traces'."
print "Need to put pre-trained NN model in 'pensieve/rl_server/results'."
