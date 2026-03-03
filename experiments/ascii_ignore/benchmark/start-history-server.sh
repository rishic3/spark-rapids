#!/bin/bash

LOG_DIR=$(pwd)/event_logs
PORT=18080

while getopts "l:p:h" opt; do
    case $opt in
        l)
            LOG_DIR=$OPTARG
            ;;
        p)
            PORT=$OPTARG
            ;;
        h)
            echo "Usage: $0 [-l log_directory] [-p port]"
            echo "  -l: Log directory (default: $(pwd)/event_logs)"
            echo "  -p: Port number (default: 18080)"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Usage: $0 [-l log_directory] [-p port]"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

if [ -z "$SPARK_HOME" ]; then
    export SPARK_HOME=/opt/spark-3.5.5
fi

export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$LOG_DIR -Dspark.history.ui.port=$PORT"
$SPARK_HOME/sbin/start-history-server.sh

sleep 1
PID=$(pgrep -f "org.apache.spark.deploy.history.HistoryServer")

if [ -n "$PID" ]; then
    echo "Spark History Server running with PID: $PID"
else
    echo "Warning: Could not find Spark History Server PID"
fi
