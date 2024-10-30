#!/bin/sh

echo Before killing servers:
ps -ax | grep -i bok

# Kill existing bokeh server(s)
kill -9 `ps -waxf | grep bokeh | grep -v grep | awk '{print $2}'`

echo After killing servers:
ps -ax | grep -i bok

echo Starting server:
bokeh serve visualise.py &

sleep 2
open http://localhost:5006/visualise

