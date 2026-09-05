#!/bin/bash
for i in Flex*
do
    echo "{\"model\": \""$i"\""$(for j in $i/task-*.json; do k=${j/*task-[0-9][0-9][0-9]-/}; echo -n ", \""${k/-metrics.json/}"\": "$(jq '.metrics.acc_raw // .metrics.exact_match' "$j")""; done; echo -n "}")
done
