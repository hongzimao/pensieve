# filename=$1
grunt --config Gruntfile.js --force
cp ./dash.all.js ../video_server/
# cp ./dash.all.js ./compiled_code/$1_dash.all.js
