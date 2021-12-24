# 2021/12/11 8:09
import tkinter
from Global import *
import threading
import time
import os
from DivideData import GetTrainAndTest
from classify import classifier
from Train import Trainer
import windnd
from tkinter import ttk,StringVar,Entry,Button,Label,Scrollbar,messagebox
from tkinter.filedialog import askopenfilename, askopenfilenames, askdirectory, asksaveasfilename

class Gui:
    def __init__(self):
        self.cls_name=[]        # 存放分类时的数目
        self.flag=-1            # 标志训练过程 还是 划分数据过程，从而产生不同的UI
        self.times_num=1            # 训练多少轮
        self.selected_path=''       # 存储在treeview中被选中的文件路径
        self.tree_item=[]               # 存放训练数据
        self.tk = tkinter.Tk()
        self.class_num=2                # 进行分类时选择分类数

        self.label6=ttk.Label(self.tk,text='分类数')
        self.class_box=ttk.Combobox(self.tk)

        # self.progress = Label(self.tk,text='进度：xxxx')  # 计录任务执行的进度
        self.label1 = ttk.Label(self.tk, text='训练数据源', borderwidth=1)
        self.treeview = ttk.Treeview(self.tk)
        self.goal_path = StringVar()                   # 存放图片集路径
        self.divide_rate = StringVar()
        self.divide_rate.set(0.8)
        self.entry = Entry(self.tk, textvariable=self.goal_path)           # 输入路径
        self.label4=Label(self.tk,text='划分比率')
        self.rate = Entry(self.tk,textvariable=self.divide_rate)       # 输入的划分比

        self.label5=Label(self.tk,text='训练次数')
        self.times=ttk.Combobox(self.tk)

        self.label2=Label(self.tk,text='选择图片集路径 OR 划分训练保存路径')
        self.fileChoose = Button(self.tk, text='...', font=('黑体', 15), justify='center', command=self.choose_file)  # 选择文件

        self.start_classify_button = Button(self.tk, text='Start To Classify',command=self.thread_classify)  # 开始分类的按钮
        self.VScroll1 = Scrollbar(self.treeview, orient='vertical', command=self.treeview.yview)        # 为TreeView添加滚动条
        self.train_button=Button(self.tk,text='Start To Train',command=self.thread_train)         # 开始训练的按钮
        self.divide_Data_button=Button(self.tk,text='Divide Data',command=self.thread_divide)      # 划分数据按钮

        self.place()
        self.tk.mainloop()

    # 对所以组件进行布局
    def place(self):
        self.tk.geometry("600x300+200+200")
        self.tk.title('图片分类程序')
        self.tk.resizable(False, False)  # 禁止用户改变窗口大小

        self.label1.place(relx=0.02, rely=0.01, relwidth=0.2, relheight=0.08)
        self.treeview.place(relx=0.02, rely=0.1, relwidth=0.35, relheight=0.8)

        self.goal_path.set('D:\\')
        self.entry.place(relx=0.40, rely=0.1, relwidth=0.4, relheight=0.07)
        # self.rate.place(relx=0.3, rely=0.1, relwidth=0.4, relheight=0.07)

        self.label2.place(relx=0.40, rely=0.05, relwidth=0.42, relheight=0.05)

        self.fileChoose.place(relx=0.81, rely=0.1, relwidth=0.1, relheight=0.07)

        self.times['value']=['1','10','20','30','40','50','60']
        self.times.current(0)
        self.times['state']='readonly'

        self.times.bind('<<ComboboxSelected>>',self.get_time)

        self.label5.place(relx=0.6, rely=0.2, relwidth=0.2, relheight=0.06)
        self.times.place(relx=0.6,rely=0.30,relwidth=0.2,relheight=0.08)

        self.label6.place(relx=0.42, rely=0.2, relwidth=0.2, relheight=0.06)
        self.class_box.place(relx=0.4,rely=0.30,relwidth=0.2,relheight=0.08)
        self.class_box['value']=['2','3','4','5','6','7','8','9','10']
        self.class_box.current(0)
        self.class_box['state']='readonly'
        self.class_box.bind('<<ComboboxSelected>>', self.get_cls)

        self.label4.place(relx=0.4,rely=0.4,relwidth=0.4,relheight=0.05)
        self.rate.place(relx=0.4,rely=0.48,relwidth=0.4,relheight=0.08)

        self.start_classify_button.place(relx=0.38, rely=0.63, relwidth=0.2, relheight=0.1)
        self.train_button.place(relx=0.6,rely=0.63,relwidth=0.2,relheight=0.1)
        self.divide_Data_button.place(relx=0.48,rely=0.8,relwidth=0.2,relheight=0.1)

        # 给treeview添加滚动条
        self.VScroll1.place(relx=0.971, rely=0.028, relwidth=0.024, relheight=0.958)
        self.treeview.configure(yscrollcommand=self.VScroll1.set)

        self.treeview.bind("<<TreeviewSelect>>", self.trefun)  # 当图片被选中时，执行trefun

        # self.progress.place(relx=0.8,rely=0.9,relwidth=0.2,relheight=0.1)

        windnd.hook_dropfiles(self.treeview, func=self.dragged_files)  # 给treeview添加拖拽

    # 选中下拉框中的某个数时调用
    def get_time(self,*args):
        num=self.times.get()
        self.times_num=int(num)
        print(self.times_num)
    # 选中分类数时调用
    def get_cls(self,*args):
        num=self.class_box.get()
        self.class_num=int(num)
        print(self.class_num)

    # 选择图片集路径
    def choose_file(self):
        file_path = askdirectory(title='请选择文件', initialdir=r'D:\\')
        self.goal_path.set(file_path)

    # 拖拽文件到treeView时将文件名罗列到里面去
    def dragged_files(self,file_path,key=0):
        # if os.path.isdir(file_path):
        if key==0:
            file_path = str(file_path[0], 'utf-8')
        if os.path.isdir(file_path):
            path_items = [os.path.join(file_path,name) for name in os.listdir(file_path) if not name.endswith('.tar')]
            head = file_path.split('\\')[-1]
            root = self.treeview.insert('', len(self.tree_item), file_path, text=head, values=('1'))
            for index, item in enumerate(path_items):
                head=item.split('\\')[-1]
                second=self.treeview.insert(root, index, item, text=head, value=(str(index)))
                for x,pic_name in enumerate(os.listdir(item)):
                    self.treeview.insert(second,x,os.path.join(item,pic_name),text=pic_name,value=str(x))

        else:
            head = file_path.split('\\')[-1]
            self.treeview.insert('', len(self.tree_item), file_path, text=head, value=len(self.tree_item))
        self.tree_item.append(file_path)

    # 选择训练数据
    def trefun(self,event):
        sels = event.widget.selection()
        file_path=(str(sels[0]))
        print(file_path)
        self.selected_path=file_path

    # 检测输入的合法性
    def check(self,key):
        if key==1:              # 点击分类按钮时
            path_check=os.path.exists(self.goal_path.get()) and os.path.isdir(self.goal_path.get())
            is_empty=len(os.listdir(self.goal_path.get()))==0
            if len(self.selected_path)==0:
                return False
            num=self.selected_path.split('_')[-1].split('.')[-2]
            if num=='':
                return False
            num_check=int(num) == self.class_num          # pkl文件的分类数是否和选择的分类数匹配
            print(int(self.selected_path[-5]))
            return path_check and not is_empty and self.selected_path.endswith('.pkl') and num_check
        if key==2:              # 点击训练按钮时
            # goal_check=os.path.exists(self.goal_path.get()) and os.path.isdir(self.goal_path.get())     # 将训练得到的结果文件保存到哪
            path_check = os.path.exists(self.selected_path) and os.path.isdir(self.selected_path)  # 检测选择的训练集是否合法
            number_check = int(self.times.get())  # 检测训练次数
            s_ptrain=os.path.join(self.selected_path, 'Train')
            s_ptest=os.path.join(self.selected_path, 'Test')
            s_plib=os.path.join(self.selected_path, 'Lib')
            return path_check and number_check and os.path.exists(s_ptrain) and os.path.exists(s_ptest) and os.path.exists(s_plib)

        if key==3:              # 点击划分数据按钮时
            try:
                rate_check=float(self.divide_rate.get()) < 1 and float(self.divide_rate.get()) > 0
            except Exception :
                return False
            goal_check=os.path.exists(self.goal_path.get()) and os.path.isdir(self.goal_path.get())             # 检测训练集保存路径
            path_check=os.path.exists(self.selected_path) and os.path.isdir(self.selected_path)     # 检测总的数据集
            return path_check and goal_check and rate_check

    # 禁止使用按钮
    def ban_button(self):
        self.train_button['state']='disable'
        self.divide_Data_button['state']='disable'
        self.start_classify_button['state']='disable'

    # 开启按钮
    def normal_button(self):
        self.train_button['state'] = 'normal'
        self.divide_Data_button['state'] = 'normal'
        self.start_classify_button['state'] = 'normal'

    # 更新UI界面
    def update_progress(self):
        label = Label(self.tk)
        label.place(relx=0.68, rely=0.9, relwidth=0.32, relheight=0.1)
        str=''
        if self.flag==0:                # 进行分类时检测
            self.ban_button()
            while g_values.progress_rate1[0]!=g_values.progress_rate1[1]-1 or g_values.progress_rate2[0]!=g_values.progress_rate2[1]-1 or g_values.flag != 1:
                label.configure(text=f'进度：{g_values.progress_rate1[0]}/{g_values.progress_rate1[1]-1},{g_values.progress_rate2[0]}/{g_values.progress_rate2[1]-1}')
            str='completed!'
            g_values.reset()
        if self.flag==1:                # 进行训练时检测
            self.ban_button()
            while(g_values.progress_rate1[0]!=g_values.progress_rate1[1]-1 or g_values.progress_rate2[0]!=g_values.progress_rate2[1]-1) or g_values.flag != 1:
                label.configure(text=f'进度：{g_values.progress_rate1[0]}/{g_values.progress_rate1[1]-1} '
                                     f'{g_values.progress_rate1[2]} {g_values.progress_rate2[0]}/{g_values.progress_rate2[1]-1} {format(g_values.acc,"0.3f")}')
            str=f'completed! best acc_rate={format(g_values.acc,"0.3f")}'

        if self.flag==2:                # 划分数据时检测
            self.ban_button()
            while(g_values.progress_rate1[0]!=g_values.progress_rate1[1]-1) or g_values.flag != 1:
                label.configure(text=f'进度：{g_values.progress_rate1[0]}/{g_values.progress_rate1[1]}')
            str=f'completed!'
            g_values.reset()

        label.configure(text=str)
        self.flag=-1
        self.normal_button()


    # 多线程，一个线程用来更新UI界面，一个用来执行后端逻辑
    def thread_divide(self):
        if self.check(3):
            self.flag=0
            UI = threading.Thread(target=self.update_progress)           # 更新UI
            DT= threading.Thread(target=self.divide_data)
            UI.setDaemon(True)  # 设置为守护线程，在主线程结束后该线程也将结束
            DT.setDaemon(True)
            UI.start()
            DT.start()
        else:
            messagebox.showerror(title="format error", message="请确认您输入的内容格式正确以及路径合法且存在")

    def divide_data(self):
        GetTrainAndTest(self.selected_path, float(self.divide_rate.get()), self.goal_path.get())
        self.dragged_files(os.path.join(self.goal_path.get(),'Data'),key=1)

    def thread_train(self):
        if self.check(2):
            self.flag=1
            UI = threading.Thread(target=self.update_progress)
            T = threading.Thread(target=self.train_data)  # 执行底层逻辑
            T.setDaemon(True)  # 设置为守护线程，在主线程结束后该线程也将结束
            UI.setDaemon(True)
            UI.start()
            T.start()

        else:
            messagebox.showerror(title="format error", message="请确认您输入的内容格式正确以及路径合法且存在")

    def train_data(self):
        self.cls_name=Trainer(ROOT_TRAIN=os.path.join(self.selected_path,'Train'), ROOT_TEST=os.path.join(self.selected_path, 'Test'),
                GOAL_PATH=os.path.join(self.selected_path, 'Lib')).main_train(self.times_num).reset()
        self.treeview.insert('', len(self.tree_item), g_values.pkl_path, text=g_values.pkl_name, value=len(self.tree_item))
        self.tree_item.append(os.path.join(self.selected_path, 'Lib','best_model.pkl'))
        g_values.reset()

    def thread_classify(self):
        if self.check(1):
            self.flag=2
            T = threading.Thread(target=self.classify)
            T.setDaemon(True)
            T.start()
            UI = threading.Thread(target=self.update_progress)
            UI.setDaemon(True)
            UI.start()
        else:
            messagebox.showerror(title="format error", message="请确认您输入的内容格式正确以及路径合法且存在")

    def classify(self):
        for x in range(self.class_num):                 # 分类时，创建的分类目录由分类数决定
            self.cls_name.append(f'cls_{x}')
        classifier(GOAL_PATH=self.goal_path.get(),MODEL_PATH=self.selected_path,cls_name=self.cls_name,class_num=self.class_num).classify()


if __name__=='__main__':
    g=Gui()
