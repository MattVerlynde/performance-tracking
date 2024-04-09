#!/bin/bash

command=${@:2}
output=$1

t1=$(date -u +%Y-%m-%dT%T.%9NZ)

$command

t2=$(date -u +%Y-%m-%dT%T.%9NZ)

query="data=from(bucket: \"telegraf_bucket\")
    |> range(start: ${t1}, stop: ${t2})
    |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
    |> yield(name: \"mean\")"

echo $query > query

sudo docker cp query influxdb:/query
rm query

sudo docker exec -it influxdb sh -c 'influx query -f query -r' > $output