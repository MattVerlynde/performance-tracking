#!/bin/bash

command=${@:3}
touch $1
touch $2

t0=$(date -u +%Y-%m-%dT%T.%9NZ)
echo ${t0} >> $2

sleep 180

echo $(date -u +%Y-%m-%dT%T.%9NZ) >> $2

sleep 1

echo $(date -u +%Y-%m-%dT%T.%9NZ) >> $2

$command
t1=$(date -u +%Y-%m-%dT%T.%9NZ)
echo ${t1} >> $2
