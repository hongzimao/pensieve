**************
OVERVIEW
**************



This folder contains 1 subfolders:



1. player_code: contains the source code for dash.js player (version master-v1.2.0)







***************
INITIAL SETUP
***************



To test a single player with a single server, we could do the following steps:



Step 1: Get video server working



1. Install nodejs (http://nodejs.org/)



2. setup a local http server using the video "Envivio" from http://dashif.org/reference/players/javascript/1.3.0/samples/dash-if-reference-player/index.html

The player works with other video as well, however, the current version only works with this video (The MPC table is hard-coded).



Step 2: Install the dash video player:



1. Go to player_code folder, install Dependencies
	
2.1. [install nodejs](http://nodejs.org/)
	
2.2. [install grunt](http://gruntjs.com/getting-started)
    		
* npm install -g grunt-cli
	
2.3. [install grunt-template-jasmine-istanbul](https://github.com/maenu/grunt-template-jasmine-istanbul)
    		
* npm install grunt-template-jasmine-istanbul --save-dev
	
2.4. install some other dependencies:
    		
* npm install grunt-contrib-connect grunt-contrib-watch grunt-contrib-jshint grunt-contrib-uglify grunt-jsdoc-plugin
		
* npm install grunt-jsdoc



2. Build / Run tests ("compile"):
	
grunt --config Gruntfile.js --force

	
or just use ./compile.sh



3. Copy the "compiled" file dash.all.js to the server_code folder



4. Open chrome browser, and enter http://localhost:8080/myindex.html to see the video!



**********************
PLAYER CODE STRUCTURE
**********************



The key pieces of the player code is in the following:



1. player_code/app/js/streaming/BufferController.js
This is in charge of requesting new chunks, record bitrate, send log to server, etc. E.g. line 447-482 sends log of QoE to server once the video playback is finished.



2. player_code/app/js/streaming/AbrController.js

This is in charge of selecting bitrate adaptation algorithms and choose bitrates. It calls FastMPC or RB or BB algorithm to compute the optimal bitrate. The key logic is in "getPlaybackQuality" function, especially line 312-384.



3. player_code/app/js/streaming/algo/
This folder contains the key bitrate adaptation logic. In particular, FastMPC.js contains the fast mpc table, BandwidthPredictor.js contains the implementation of a simple harmonic mean predictor. Note that fast MPC table is pre-computed and hardcoded for the example video in server_code folder. If we would like to change the video, we need to recompute the fast MPC table and hardcode the new table into FastMPC.js. 


