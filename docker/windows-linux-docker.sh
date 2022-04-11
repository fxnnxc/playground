docker run \
    -it \
    --net=host \
    --privileged \
    -v="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --env DISPLAY \
    --gpus 3 \
    --memory 60g \
    --cpus 12 \
    --env="QT_X11_NO_MITSHM=1" \
    --device /dev/dri \
    deeping:ros_noetic