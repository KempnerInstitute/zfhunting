# Train with parallel jobs + sleep 5 seconds
MAX_PARALLEL_JOBS=3
CURRENT_PARALLEL_JOBS=0
for NUM_AGENTS in 4; do 
  python train_foraging.py --num_agents $NUM_AGENTS \
    --experiment_name ${NUM_AGENTS}_agents \
    --num_patches $(($NUM_AGENTS * 2)) \
    --max_food $(($NUM_AGENTS * 10)) \
    > train_${NUM_AGENTS}.log 2>&1 &
  sleep 5
  CURRENT_PARALLEL_JOBS=$((CURRENT_PARALLEL_JOBS + 1))
  if [ "$CURRENT_PARALLEL_JOBS" -ge "$MAX_PARALLEL_JOBS" ]; then
    wait -n
    CURRENT_PARALLEL_JOBS=$((CURRENT_PARALLEL_JOBS - 1))
  fi
done
wait




# Crop all videos
for INPUT_VID in *.mp4; do
  OUTPUT_VID="${INPUT_VID%.mp4}_cropped.mp4"
  # OUTPUT_GIF="${INPUT_VID%.mp4}_cropped.gif"
	if [[ "$INPUT_VID" != *_cropped.mp4 ]]; then        
	    ./cropvid.sh $INPUT_VID 135 55 517 435 $OUTPUT_VID
      # ffmpeg -i $OUTPUT_VID $OUTPUT_GIF
	fi
done