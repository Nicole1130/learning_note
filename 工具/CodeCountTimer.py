# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:40:44 2020

@author: zhuwenyan
"""

import requests
from WorkWeixinRobot.work_weixin_robot import WWXRobot

from datetime import datetime
import os
from apscheduler.schedulers.blocking import BlockingScheduler


url = 'http://localhost:8087/count/getThisWeekCodeCount' # 接口地址
ROBOT_KEY = '89a76049-a623-415c-a2ac-6e6e3c41efd6' # 群机器人的key
# id：显示名称
nameDic = {
    "111": "名称1",
    "222": "名称2",
  }


def getCodeCount():
    try:
        r = requests.get(url).json() 
        data =  r['data']
        text = "本周代码提交情况："
        for name in nameDic:
            text += '\n'
            if name in data:
                text += nameDic[name]
                text += ' : '
                text += str(data[name])
            else:
                text += nameDic[name]
                text += ' : '
                text += str(0) 
        return text
    except Exception as e:
        print("网络访问受限")


def sendMSG():
    rbt = WWXRobot(key=ROBOT_KEY) # 群机器人
    text = getCodeCount()
    rbt.send_text(content=text) # 发送消息


def tick():
    print('Tick! The time is: %s' % datetime.now())

if __name__ == '__main__':
    scheduler = BlockingScheduler()
    scheduler.add_job(tick, 'interval',seconds='05') # 隔5s调用一次
#    scheduler.add_job(tick, 'cron', second='05') # 每当秒为05时调用
#    scheduler.add_job(job_func, 'cron', day_of_week='fri', hour=14, minute=22) # 每周五14点22分调用

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass
