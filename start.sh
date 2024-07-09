#!/usr/bin/env bash
  2 
  3 Python=/root/miniconda3/
  4 export PATH=$Python/bin:$PATH
  5 export LC_ALL=C
  6 export PYTHONDONTWRITEBYTECODE=1
  7 export PYTHONIOENCODING=UTF-8
  8 
  9 LocalDir=$(cd "$(dirname $0)";pwd)
 10 cd  $LocalDir
 11 
 12 worker=$1
 13 port=$2
 14 gunicorn server:app -w $worker -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$port --keep-alive 300
