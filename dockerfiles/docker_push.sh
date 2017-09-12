#export DOCKER_ID_USER="mauvilsa";
#docker login;

CUDA="8.0";
OS="ubuntu16.04";
REV=$(sed -rn '/^Version.DATE/{ s|.*Date: *([0-9]+)-([0-9]+)-([0-9]+).*|\1.\2.\3|; p; }' ../laia/Version.lua);
docker push mauvilsa/laia:${REV}-cuda$CUDA-$OS;
