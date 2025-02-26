#!/bin/bash

t0=$(head -n1 $1)
t1=$(tail -n1 $1)

qcpu="data=from(bucket: \"telegraf_bucket\")
  |> range(start: ${t0}, stop: ${t1})
  |> filter(fn: (r) => r[\"_measurement\"] == \"cpu\")
  |> filter(fn: (r) => r[\"_field\"] == \"usage_user\")
  |> filter(fn: (r) => r[\"cpu\"] == \"cpu-total\")
  |> yield()"

qmem="data=from(bucket: \"telegraf_bucket\")
  |> range(start: ${t0}, stop: ${t1})
  |> filter(fn: (r) => r[\"_measurement\"] == \"mem\")
  |> filter(fn: (r) => r[\"_field\"] == \"used_percent\")
  |> yield()"

qenergy="data=from(bucket: \"telegraf_bucket\")
  |> range(start: ${t0}, stop: ${t1})
  |> filter(fn: (r) => r[\"_measurement\"] == \"mqtt_consumer\")
  |> filter(fn: (r) => r[\"_field\"] == \"value\")
  |> filter(fn: (r) => r[\"topic\"] == \"zwave/Smart_switch_PC/50/0/value/66049\")
  |> yield()"

qtemp="data=from(bucket: \"telegraf_bucket\")
  |> range(start: ${t0}, stop: ${t1})
  |> filter(fn: (r) => r[\"_measurement\"] == \"temp\")
  |> filter(fn: (r) => r[\"sensor\"] == \"coretemp_package_id_0\")
  |> yield()"

qreads="data=from(bucket: \"telegraf_bucket\")
  |> range(start: ${t0}, stop: ${t1})
  |> filter(fn: (r) => r[\"_measurement\"] == \"diskio\")
  |> filter(fn: (r) => r[\"_field\"] == \"reads\")
  |> filter(fn: (r) => r[\"name\"] == \"nvme0n1p3\")
  |> yield()"


for query in "$qcpu" "$qmem" "$qenergy" "$qtemp" "$qreads";
do
    echo $query > query
    sudo docker cp query influxdb:/query
    sudo docker exec -it influxdb sh -c 'influx query -f query -r' | tail -n+5 >> $2
    echo "#####" >> $2
done
rm query
