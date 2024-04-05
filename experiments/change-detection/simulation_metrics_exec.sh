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

docker cp query influxdb:/query -u 0
rm query
echo Query copied to container

docker exec -u 0 -i influxdb sh -c 'influx query -f query -r' > $output