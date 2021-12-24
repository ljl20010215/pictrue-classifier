# 2021/12/9 15:12

class MoreParameter(Exception):
    def __init__(self,err='运行脚本需要更多的参数'):
        Exception.__init__(self,err)

class LessParameter(Exception):
    def __init__(self,err='运行脚本需要更少的参数'):
        Exception.__init__(self,err)
