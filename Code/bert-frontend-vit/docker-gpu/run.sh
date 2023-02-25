sudo docker run --rm --gpus all\
                     --pid=host\
                     --mount src=$1,target=/workspace,type=bind\
                     -w /workspace\
                     -e "CI_BUILD_HOME=/workspace"\
                     -e "CI_BUILD_USER=$(id -u -n)"\
                     -e "CI_BUILD_UID=$(id -u)"\
                     -e "CI_BUILD_GROUP=$(id -g -n)"\
                     -e "CI_BUILD_GID=$(id -g)"\
                     -h $2\
                     --name $2\
                     -it --net=host\
                     $2\
                     /bin/bash
