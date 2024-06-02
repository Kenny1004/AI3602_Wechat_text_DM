import pandas as pd
from datetime import datetime


def read_file(data_path):##读取csv文件
    data=pd.read_csv(data_path)
    return data

def filter_sender_receiver(data,Issend): #筛选信息对象#Issend 0:reciever 1:sender
    data=data[data['IsSender']==Issend]
    return data

def get_time(time):
    return datetime.fromtimestamp(time)

data_path='data.csv' ##a CSV file
df=read_file(data_path)
df=filter_sender_receiver(df,0)

print(df.columns) #输出第一行的关键字类型


print(df.shape[0])##消息有多少条

for time in df['CreateTime']: #直接用df['CreateTime][x]来提取元素可能会出错，第x条消息可能被过滤掉了
    #######如果要用索引的话可以把这里的东西都加进一个List里，就可以从0,1,2,3一直用索引来拿元素了。
    print(get_time(time))#把那个长的要死的事件转化为看得懂的时间
    print(get_time(time).month)
    break

#for x in df['StrContent']:
    #print(x)
#