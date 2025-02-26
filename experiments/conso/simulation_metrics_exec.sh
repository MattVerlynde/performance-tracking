#!/bin/bash

command=${@:4}
touch $1
touch $2

t0=$(date -u +%Y-%m-%dT%T.%9NZ)
echo ${t0} >> $2

sleep 180

echo $(date -u +%Y-%m-%dT%T.%9NZ) >> $2


# for i in $(seq 1 10);
# do
#     $command -n $i
# done

$command -n 1

echo $(date -u +%Y-%m-%dT%T.%9NZ) >> $2

for i in $(seq 1 $3);
do
    $command -n $i
    t1=$(date -u +%Y-%m-%dT%T.%9NZ)
    echo ${t1} >> $2
done
