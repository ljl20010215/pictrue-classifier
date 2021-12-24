# 2021/12/12 21:56

class Globals:
    def __init__(self):
        self.progress_rate1 = (-1, 1, '')           # 存储任务执行进度
        self.progress_rate2 = (-1, 1, '')
        self.flag=-1                                # 标志任务进度是否complete
        self.acc=0.0                                # 任务结束后，保存正确率显示在UI界面中
        self.pkl_path=''                            # 保存pkl文件路径
        self.pkl_name=''
    def reset(self):
        self.progress_rate1 = (-1, 1, '')
        self.progress_rate2 = (-1, 1, '')
        self.flag=-1
        self.acc = 0.0

g_values=Globals()