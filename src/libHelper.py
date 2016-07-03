#!/usr/bin/env python
#encoding: utf8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from datetime import datetime
import time


def timestamp2str(timestamp):
    dt=datetime.fromtimestamp(float(timestamp))
    return datetime.strftime(dt,'%Y-%m-%d-%H')


def str2timestamp(dateString):
    dt=time.strptime(dateString, '%Y-%m-%d-%H')
    return int(time.mktime(dt))
    