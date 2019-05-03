#!/bin/bash
filename="app/models/frcnn_detector/weights/model_frcnn_vgg.hdf5"
file_id="1KDS9XREi7-cM-ibGML0GLoekCzCkZR1z"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
curl -b ./cookie.txt -L -o ${filename} $url
