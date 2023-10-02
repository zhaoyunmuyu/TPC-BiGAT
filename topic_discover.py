'''
 Filename:  topic_discover.py
 Description:  热点话题发现(公式待修改)
 Created:  2022年11月7日 20时27分
 Author:  Li Shao
'''

import os
import sys
import json
import time
from datetime import datetime
sys.path.append(sys.path[0]+'/..')
from config.config import Config

# 时间戳转换
def timestamp(timestr):
    dateFormatter = "%a %b %d %H:%M:%S %z %Y"
    temp = datetime.strptime(timestr, dateFormatter)
    struct_time = time.strptime(str(temp),'%Y-%m-%d %H:%M:%S%z')
    return time.mktime(struct_time)

# 每个topic与热度相关的特征json
def hot_feature(topic_path):
    topic_dict = {}
    topic_dict['id'] = topic_path.split('/')[-1]
    # 获取最早时间
    ori_times = []
    for _, dirs, _ in os.walk(topic_path):
        for post_id in dirs:
            post_path = os.path.join(topic_path, post_id)
            with open(post_path+'/original.jsonl') as f:
                data = json.load(f)
                ori_times.append(timestamp(data['created_at']))
        break
    ori_times.sort()
    topic_dict['ori_time'] = ori_times[0]
    topic_dict['data'] = []
    for _,dirs,_ in os.walk(topic_path):
        for post_id in dirs:
            post_dict = {}
            post_path = os.path.join(topic_path, post_id)
            with open(post_path+'/original.jsonl') as f:
                data = json.load(f)
                post_dict['id'] = post_id
                post_dict['time'] = timestamp(data['created_at'])
                post_dict['ret_num'] = data['retweet_count']
                post_dict['fav_num'] = data['favorite_count']
                post_dict['url_num'] = len(data['entities']['urls'])
                try:
                    post_dict['pic_num'] = len(data['entities']['media'])
                except:
                    post_dict['pic_num'] = 0
                post_dict['is_verified'] = data['user']['verified']
                post_dict['follower_num'] = data['user']['followers_count']
                post_dict['quote'] = []
            for _,dirs,_ in os.walk(post_path):
                for layer in dirs:
                    layer_path = os.path.join(post_path, layer)
                    layer_list = os.listdir(layer_path)
                    for quote_id in layer_list:
                        quote_dict = {}
                        with open(os.path.join(layer_path, quote_id)) as f:
                            data = json.load(f)
                            quote_dict['id'] = data['id_str']
                            quote_dict['time'] = timestamp(data['created_at'])
                            quote_dict['ret_num'] = data['retweet_count']
                            quote_dict['fav_num'] = data['favorite_count']
                            quote_dict['url_num'] = len(data['entities']['urls'])
                            try:
                                quote_dict['pic_num'] = len(data['entities']['media'])
                            except:
                                quote_dict['pic_num'] = 0
                            quote_dict['is_verified'] = data['user']['verified']
                            quote_dict['follower_num'] = data['user']['followers_count']
                        post_dict['quote'].append(quote_dict)
                break
            topic_dict['data'].append(post_dict)
        break
    return topic_dict

# 计算hot-point
def calculate_hotpoint(topic_path, time_delay):
    posts_hot_point = []
    with open(topic_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        try:
            ori_time = data['ori_time']
        except:
            print(topic_path)
        # print(topic_path)
        for post_dict in data['data']:
            time = post_dict['time']
            if(time - ori_time) < time_delay:
                ret_num = post_dict['ret_num']
                fav_num = post_dict['fav_num']
                url_num = post_dict['url_num']
                pic_num = post_dict['pic_num']
                is_verified = post_dict['is_verified']
                follower_num = post_dict['follower_num']
                # 公式待修改
                post_hot_point = ret_num + fav_num + url_num + pic_num + is_verified + follower_num
                for quote_dict in post_dict['quote']:
                    time = quote_dict['time']
                    if(time - ori_time) < time_delay:
                        ret_num = quote_dict['ret_num']
                        fav_num = quote_dict['fav_num']
                        url_num = quote_dict['url_num']
                        pic_num = quote_dict['pic_num']
                        is_verified = quote_dict['is_verified']
                        follower_num = quote_dict['follower_num']
                        # 公式待修改
                        temp = ret_num + fav_num + url_num + pic_num + is_verified + follower_num
                        post_hot_point += temp
                posts_hot_point.append(post_hot_point)
    # 简单求和
    topic_hot_point = sum(posts_hot_point)
    return topic_hot_point

def get_hot_topics(config):
    print('start topic discover')
    discover_rate = config.discover_rate
    # 保存hotpoint_feature
    data_path = config.dataset_dir
    topic_dir = config.topic_dir
    if not os.path.exists(topic_dir):  
        os.makedirs(topic_dir)
    topic_list = os.listdir(data_path)
    num = 0 
    for topic_id in topic_list:
        num += 1
        # print(num,'/',len(topic_list),topic_id)
        if os.path.exists(os.path.join(topic_dir, topic_id+'.json')):
            continue
        topic_path = os.path.join(data_path,topic_id)
        topic_dict = hot_feature(topic_path)
        with open(os.path.join(topic_dir, topic_id+'.json'), 'w', encoding='utf-8') as file:
            file.write(json.dumps(topic_dict, indent=2))
    # 计算hotpoint&save
    topic_hotpoint = {}
    time_delay = config.time_delay
    hot_point_path = os.path.join(topic_dir+'/..', config.dataset+'_'+str(time_delay)+'.json')
    for topic_id in os.listdir(topic_dir):
        topic_path = os.path.join(topic_dir, topic_id)
        topic_hotpoint[topic_id.split('.')[0]] = calculate_hotpoint(topic_path, time_delay)
    with open(hot_point_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(topic_hotpoint, indent=2))
    # 选择hot-topic
    hot_topics = []
    with open(hot_point_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        temp = sorted(data.items(),key=lambda item:item[1], reverse=1)
        for i in range(0, int(len(temp)*discover_rate)):
            hot_topics.append(temp[i][0])
    return hot_topics

if __name__ == '__main__':
    config = Config()
    hot_topics = get_hot_topics(config)
    print(hot_topics)