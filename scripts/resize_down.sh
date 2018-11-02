#!/bin/bash
#To see what these filters do click here
#http://www.imagemagick.org/Usage/filter/nicolas/

image=$1
filter=$2

#This is the no filter case
#Filter defaults to Lanczos/Mitchell, so will need to artifically add the "None" filter
if [ -z "$filter" || "$filter" == "none" ]
then
    convert -resize 500X333 -filter point $1 out_$1
fi

#Filter types given here:https://imagemagick.org/script/command-line-options.php#filter
declare -A array1=( 
 [catrom]=1  [triangle]=1 [lanczos]=1
)
 
#If there is a filter check that it is valid
if [[ -n "${array1[$filter]}" ]]
then
    convert -resize 500X333 -filter $filter $image out_$image
else
    echo "[ERROR!] Make sure filter is one of the following\n [catrom, triangle, lanczos, or none]"
fi
