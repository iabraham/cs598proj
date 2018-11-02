#!/bin/bash
image=$1
filter=$2

#This is the no filter case
if [ -z "$filter" ]
then
    convert -resize 500X333 $1 out_$1
fi

#Linear and Bilinear did not work
declare -A array1=( 
 [catrom]=1  [triangle]=1 [hermite]=1 [lanczos]=1
)
 
#If there is a filter check that it is valid
if [[ -n "${array1[$filter]}" ]]
then
    convert -resize 500X333 -filter $filter $image out_$image
else
    echo "[ERROR!] Make sure filter is one of the following [catrom, linear, triangle, bilinear, or lanczos]"
fi
