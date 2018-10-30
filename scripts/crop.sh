#!/bin/bash
image=$1

convert $1 -crop 500x333+0+0 out_$1
