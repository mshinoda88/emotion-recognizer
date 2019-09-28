#!/bin/bash

fileid="1ZQxP_wKK36b13i3Z_ZZ377HfTCFPGMxf"
filename="yolov3-wider_16000.weights"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${fileid}" -o ${filename}

fileid="1cErV8C7Jj7SGco12IzJIrXUP_u-KHq05"
filename="yolov3-face.cfg"
wget "https://drive.google.com/uc?export=download&id=${fileid}" -O $filename

