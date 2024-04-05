#!/bin/bash

command=${@:3}
output=$1
password=$2

t1=$(date -u +%Y-%m-%dT%T.%9NZ)

$command

t2=$(date -u +%Y-%m-%dT%T.%9NZ)

query="data=from(bucket: \"telegraf_bucket\")
    |> range(start: ${t1}, stop: ${t2})
    |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
    |> yield(name: \"mean\")"

echo $query > query

export HISTIGNORE='*sudo -S*'
echo "$password" | sudo -S -k docker cp query influxdb:/query
rm query

echo "$password" | sudo -S -k docker exec -it influxdb sh -c 'influx query -f query -r' > $output