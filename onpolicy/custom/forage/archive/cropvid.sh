#!/bin/bash

# ./cropvid.sh MAForage_20240723_68472617.mp4 135 55 517 435 MAForage_20240723_68472617_cropped.mp4

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <input_file> <x1> <y1> <x2> <y2> <output_file>"
    echo "Example: $0 input.mp4 100 50 500 300 output.mp4"
    echo "Coordinates (x1, y1) represent the top left and (x2, y2) represent the bottom right of the cropping rectangle."
    exit 1
fi

input_file=$1
x1=$2
y1=$3
x2=$4
y2=$5
output_file=$6

width=$((x2 - x1))
height=$((y2 - y1))

ffmpeg -y -loglevel error -i "$input_file" -vf "crop=${width}:${height}:${x1}:${y1}" -c:a copy "$output_file"
echo "Written" $output_file