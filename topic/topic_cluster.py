import os
import re
import sys
sys.path.append(sys.path[0]+'/..')
import json
from bertopic import BERTopic
from config.config import Config
from sklearn.datasets import fetch_20newsgroups
# text = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

def top_50(event_id, post_text, topic_post):
    return None

def Post_Level_TianBian(dataset_dir):
    # need original-text
    # https://maartengr.github.io/BERTopic/index.html#common
    return None

def Topic_Level_BEARD(dataset_dir):
    topic_post = {}
    post_text = {}
    event_list = os.listdir(dataset_dir)
    for event_id in event_list:
        event_path = os.path.join(dataset_dir,event_id)
        for _,dirs,_ in os.walk(event_path):
            post_list = dirs
            break
        temp = []
        for post_id in post_list:
            temp.append(post_id)
            post_path = os.path.join(event_path,post_id)
            with open(post_path+'/original.jsonl') as f:
                data = json.load(f)
                text = data['full_text'].replace("\n","")
                press = r'https?:\/\/[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+\.?(\/[-a-zA-Z0-9]*)*'
                text = re.sub(press, '', text)
                post_text[post_id] = text
        topic_post[event_id] = temp
    return post_text,topic_post

def topic_cluster_Eva(path):
    ids, text = [],[]
    post_text,topic_post = Topic_Level_BEARD(config.dataset_dir)
    for post_id in post_text:
        ids.append(post_id)
        text.append(post_text[post_id])    
    # Model
    topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
    topics, _ = topic_model.fit_transform(text)
    print(topic_model.get_topic_info())
    result = {}
    for i in range(0,len(ids)):
        result[ids[i]] = topics[i]
    print(result)
    # Compare result with topic_post(Evaluate)
    Acc = 0
    return Acc

if __name__ == '__main__':
    config = Config()
    accuracy = topic_cluster_Eva(config.dataset_dir)
    