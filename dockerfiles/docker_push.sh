#export DOCKER_ID_USER="mauvilsa";
#docker login;

CUDA="8.0";
OS="ubuntu16.04";
REV=$(git log --date=iso ../laia/Version.lua Dockerfile laia-docker | sed -n '/^Date:/{s|^Date: *||;s| .*||;s|-|.|g;p;}' | sort -r | head -n 1);
docker push mauvilsa/laia:${REV}-cuda$CUDA-$OS;
