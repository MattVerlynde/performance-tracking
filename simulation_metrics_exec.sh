#!/bin/bash

command=$@

t1=$(date -u +%Y-%m-%dT%T.%9NZ)
echo "*************************************************"
echo "Time start: $t1"
echo "*************************************************"
echo "Running python script: $command"
echo "*************************************************"

$command

t2=$(date -u +%Y-%m-%dT%T.%9NZ)

echo "*************************************************"
echo "Time stop: $t2"
echo "*************************************************"

echo "$(($(date -d "$t2" +%s%9N) - $(date -d "$t1" +%s%9N)))" >> execution_time

query="data=from(bucket: \"telegraf_bucket\")
    |> range(start: ${t1}, stop: ${t2})
    |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
    |> yield(name: \"mean\")"

echo $query > query

sudo docker cp query influxdb:/query

sudo docker exec -it influxdb sh -c 'influx query -f query -r' > metrics_output

echo "*************************************************"
echo "File metrics_output created"
echo "*************************************************"
head metrics_output
echo "*************************************************"

firefox -url "http://localhost:9090/d/edh1jtjp0b4lcb/38860f63-1c6f-5143-a2a7-cfc431003966?orgId=1&from=$(($(date -d "$t1" +%s%3N) - 1))&to=$(($(date -d "$t2" +%s%3N) + 1))"