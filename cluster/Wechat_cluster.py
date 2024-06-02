import pandas as pd
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random
from openai import OpenAI
client = OpenAI(
    api_key="##",
    base_url="https://lonlie.plus7.plus/v1"
)



def filter_sender_receiver(data,Issend): #筛选信息对象#Issend 0:reciever 1:sender
    data=data[data['IsSender']==Issend]
    return data


def embedding_the_data(data):##transform string to numpy vector
    device='cuda'
    model = SentenceTransformer("/home/zlhu/data2/zlhu/bge-large-zh-v1.5",device=device)
    embeddings=[]
    for message in tqdm(data['StrContent']):
        embedding = model.encode(message, normalize_embeddings=True,batch_size=512)
        embeddings.append(embedding)
        
    embeddings=np.array(embeddings)
    output_embedding_path="data/embedding_data.npy"
    np.save(output_embedding_path,embeddings)

def cluster_the_data(string_data,num_clusters):
    embeddings=np.load("data/embedding_data.npy") #PCA from 1024 dimensions to lower
    pca=PCA(n_components=0.8)
    low_dimension_data=pca.fit_transform(embeddings) #about 300 dimensions

    
    kmeans=KMeans(n_clusters=num_clusters,n_init=30,max_iter=600) #Kmeans
    kmeans.fit(low_dimension_data)
    labels=kmeans.labels_

    Result=[[] for _ in range(num_clusters)]
   
    for i in range(len(labels)):
        label=labels[i]
        Result[label].append(string_data.iloc[i]['StrContent'])

    sum=0
    print("Finish cluster")
    for i in range(num_clusters):
        print("Cluster {} num: {}".format(i,len(Result[i])))
        sum+=len(Result[i])
    
    assert sum==string_data.shape[0]

    return Result

def generate_summary(prompt):
    rule="""你是一个聊天文本数据分析师。
    我搜集了两个人的微信聊天记录，并把这些消息聚类聚成了30类，并且在其中每一个聚类采样了十条消息。发给了你。
    请你根据采样的消息，概括每一类聊天的主题且总结该类别的关键词。在概括完后，你需要分析这两个人的聊天风格、两个人的关系。如果你在聊天中有什么额外发现，也可以提供出来。请全部使用中文回答。
    """
    completion = client.chat.completions.create(
                messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": rule+prompt
                }
                ],
                model="gpt-4o"
                )
    message = completion.choices[0].message
    print(message.content)
    return message.content

if __name__=='__main__':
    data_path='data/wechat_data.csv'
    data=pd.read_csv(data_path)
    data =data.dropna(subset=['StrContent'])
    #data=filter_sender_receiver(data,1) ##Get sender's message

    #embedding_the_data(data) ##embedding
    num_clusters=30

    print("##开始聚类")
    result=cluster_the_data(data,num_clusters)
    num_samples=10 ##sample # message from each label

    prompt=""""""
    for i in range(num_clusters):
        prompt+="############\nCluster  {}:  Num: {} \n".format(i,len(result[i]))
        sample_list=random.sample(result[i],num_samples)
        for x in sample_list:
            prompt+="""{}\n""".format(x)
    print("#########聚类结果如下######")
    print(prompt)
    
    res=generate_summary(prompt)
    
    output="""##聚类结果：\n{}\n##聊天分析：\n{}""".format(prompt,res)

    output_path="result.json"
    with open(output_path, "w", encoding="utf-8") as fin:
        json.dump(output, fin, ensure_ascii=False, indent=4)
        print("Data is saved in {}".format(output_path))