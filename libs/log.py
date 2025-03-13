# -*- coding: utf-8 -*-
"""
:Author: XM
:Coding: UTF-8
:Version: 1.0
"""
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

"""
log system 内容存放路径为 ../logs
"""
import os
import time
import traceback, sys

class Log:
    def __init__(self,dir_name):
        """
        检测 ../logs下是否存在对应的文件目录
        没有则创建
        """
        self.dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)),'../logs/',dir_name)#"../logs/" + dir_name
        try:
            if not os.path.exists(self.dir_name):
                os.makedirs(self.dir_name)
        except:
            print("create log dir: "+ self.dir_name + " error")
    
    def log_info(self,message:str,print_screen:bool=False):
        #检测文件是否存在
        file_name = self.dir_name + "/" + time.strftime("%Y-%m-%d", time.localtime()) + ".log"
        message = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + message
        #进行打屏
        if print_screen:
            print(message)
            
        message = message + "\n"
        try:
            fa = open(file_name,"a")
            fa.write(message)
        except:
            print("write log message: "+ message + " error")
        finally:
            fa.close()

    #异常日志
    def log_exception(self,print_screen:bool=False):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error = "Exception: " + repr(traceback.format_exception(exc_type, exc_value, exc_traceback))  # 将异常信息转为字符串
        self.log_info(error,print_screen)
    