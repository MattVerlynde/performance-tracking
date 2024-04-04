#!/bin/bash

while getopts 'f:p:Pi:w:c:' OPTION; do
  case "$OPTION" in
    f)
      name_file="$OPTARG"
      ;;
    P) 
      by_pid=true
      get_pid=true
      ;;
    p) 
      by_pid=true
      get_pid=false
      npid="$OPTARG"
      ;;
    i) 
      name_image="$OPTARG"
      ;;
    w) 
      window_size="$OPTARG"
      ;;
    c) 
      n_jobs_cov="$OPTARG"
      ;;
    \?)
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
  esac
done

: ${name_file:?Missing -f}

for i in {1..10}
do
  t1=$(date -u +%Y-%m-%dT%T.%9NZ)
  echo "*************************************************"
  echo "Time start: $t1"
  echo "*************************************************"
  echo "Running python script: $name_file"
  echo "*************************************************"

  python3 $name_file $name_image $window_size $n_jobs_cov
  t2=$(date -u +%Y-%m-%dT%T.%9NZ)

  echo "*************************************************"
  echo "Time stop: $t2"
  echo "*************************************************"

  echo "$name_image,$window_size,$n_jobs_cov,$(($(date -d "$t2" +%s%9N) - $(date -d "$t1" +%s%9N)))" >> execution_time

  if [ "$by_pid" = true ]; then
    if [ "$get_pid" = true ]; then
      npid=$(cat python_process.pid)
    fi
    echo "Process ID: ${npid}"
    query="data=from(bucket: \"telegraf_bucket\")
      |> range(start: ${t1}, stop: ${t2})
      |> filter(fn: (r) => r[\"_measurement\"] == \"procstat\")
      |> filter(fn: (r) => r[\"pid\"] == \"${npid}\")
      |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
      |> yield(name: \"mean\")"
  else
    query="data=from(bucket: \"telegraf_bucket\")
      |> range(start: ${t1}, stop: ${t2})
      |> aggregateWindow(every: 1s, fn: mean, createEmpty: false)
      |> yield(name: \"mean\")"
  fi

  echo $query > query

  sudo docker cp query influxdb:/query

  sudo docker exec -it influxdb sh -c 'influx query -f query -r' > metrics_output
  mv metrics_output $name_image.$window_size.$n_jobs_cov.$i.metrics_output
  echo "*************************************************"
  echo "File $name_image.$window_size.$n_jobs_cov.$i.metrics_output created"
  echo "*************************************************"
  # head $name_image.$window_size.$n_jobs_cov.$i.metrics_output
  # echo "*************************************************"

done