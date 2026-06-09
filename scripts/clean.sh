#!/bin/bash

NODES_LIST="clsgpu01,clsgpu02,clsgpu03,clsgpu04,clsgpu05,clsgpu06,clsgpu07,clsgpu08,clsgpu09,clsgpu10,clsgpu11,clsgpu12,clsgpu13,clsgpu14,clsgpu15,clsgpu16,clsgpu17,clsgpu18,clsgpu19"

for node in $(echo $NODES_LIST | tr ',' ' '); do
	echo "Cleaning jobs on $node..."
	ssh $node "ps -ef | grep python3.13 | grep -v grep | cut -c 9-15 | xargs kill -9" &
	sleep 1s
done
