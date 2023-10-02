import json
import shutil
import jsonlines
from utils.text_filter import text_filter
import time
import fitlog
import csv
from config.config import Config
from process.make_graph import make_graph
from process.dataset import *
from model.BiGCN import BiGCN
from model.GAT import BiGAT
from model.one_GAT import one_GAT
from model.chahi_gat import chahi_GAT
# from model.experiment.BIGAT import BiGAT
from model.train import train_model_4class, train_model_2class
from utils.dataset_split import dataset_split, LoadKFoldData
from topic.topic_discover import get_hot_topics
from utils.make_pic import model_loss_acc

config = Config()
now_time = str(time.ctime())
fitlog.set_log_dir(config.project_dir+"/logs/")
# fitlog.commit(__file__)
fitlog.add_hyper(now_time, name="time")
fitlog.add_hyper(config.dataset, name="dataset")
fitlog.add_hyper(config.time_delay, name="time_delay")
fitlog.add_hyper(config.level, name="level")
fitlog.add_hyper(config.bert_embedding_dim, name="bert_embedding_dim")
fitlog.add_hyper(config.hidden_size, name="hidden_size")
fitlog.add_hyper(config.out_size, name="out_size")
fitlog.add_hyper(config.dropout, name="dropout")
fitlog.add_hyper(config.droprate, name="droprate")
fitlog.add_hyper(config.TDdroprate, name="TDdroprate")
fitlog.add_hyper(config.BUdroprate, name="BUdroprate")
fitlog.add_hyper(config.validation_split, name="validation_split")
fitlog.add_hyper(config.learning_rate, name="learning_rate")
fitlog.add_hyper(config.weight_decay, name="weight_decay")
fitlog.add_hyper(config.epochs, name="epochs")
fitlog.add_hyper(config.discover_rate, name="discover_rate")

# Role0: Make graph Structure
# make_graph(config)

# # Role1: Topic cluster
# topic已存在

# # Role2: Choose Potential Hot Topic(Topic_level & Have_time)
hot_topics = get_hot_topics(config)

# # Role3: Split dataset
# BEARD
train_set, test_set = dataset_split(config, hot_topics, config.cache_save_dir, config.validation_split)
# # TianBian
# topics = list(map(lambda id:id.split('.')[0], os.listdir(config.result_graph_dir)))
# train_set, test_set = dataset_split(config, topics, config.cache_save_dir, config.validation_split)
# 测试
# with open(config.cache_save_dir+'/train_valid_du.txt', encoding="utf-8") as p:
#     t_set = p.readlines()
# train_set = [line.strip("\n") for line in t_set]
# with open(config.cache_save_dir+'/test_du.txt', encoding="utf-8") as f:
#     t_set = f.readlines()
# test_set = [line.strip("\n") for line in t_set]
# def gettopic(post):
#     with open(config.cache_save_dir+'/train.txt', encoding="utf-8") as f:
#         text = f.readlines()
#         for item in text:
#             if post in item:
#                 return item
#     with open(config.cache_save_dir+'/test.txt', encoding="utf-8") as p:
#         text = p.readlines()
#         for item in text:
#             if post in item:
#                 return item
#     return None
# data1 = csv.reader(open(config.cache_save_dir+'/train_valid.csv'))
# file1 = open(config.cache_save_dir+'/train_valid_du.txt',mode='w+',encoding='utf-8')
# for line in data1:
#     item = gettopic(line[0])
#     if item:
#         file1.write(item)
# data2 = csv.reader(open(config.cache_save_dir+'/test.csv'))
# file2 = open(config.cache_save_dir+'/test_du.txt',mode='w+',encoding='utf-8')
# for line in data2:
#     item = gettopic(line[0])
#     if item:
#         file2.write(item)

# # # Role4: Train model
# model = BiGAT_door(in_feats = config.bert_embedding_dim,hid_feats = config.hidden_size,out_feats = config.out_size, num_class = config.num_class, tddroprate=config.TDdroprate,budroprate=config.BUdroprate)
model = BiGAT(in_feats = config.bert_embedding_dim,hid_feats = config.hidden_size,heads = 1, out_feats = config.out_size,num_class = config.num_class,tddroprate = config.TDdroprate, budroprate = config.BUdroprate)
# model = BiGCN(in_feats = config.bert_embedding_dim,hid_feats = config.hidden_size,out_feats = config.out_size,num_class = config.num_class,dropout = config.dropout)# set_name = 'BU'
fitlog.add_hyper(model.get_name(), name="model")
# fitlog.add_hyper(model.get_name()+'|'+set_name, name="model")
if config.num_class == 4:
    for iter in range(config.iterations):
        train_losses,val_losses,train_accses,val_accs,accs,F1_1,F1_2,F1_3,F1_4 = train_model_4class(config, model, train_set, test_set, iter)
    fitlog.add_best_metric(accs, name="accs")
    fitlog.add_best_metric(F1_1, name="F1_1")
    fitlog.add_best_metric(F1_2, name="F1_2")
    fitlog.add_best_metric(F1_3, name="F1_3")
    fitlog.add_best_metric(F1_4, name="F1_4")
elif config.num_class == 2:
    for iter in range(config.iterations):
        train_losses,val_losses,train_accses,val_accs,accs,F1_1,F1_2 = train_model_2class(config, model, train_set, test_set, iter)
    fitlog.add_best_metric(accs, name="accs")
    fitlog.add_best_metric(F1_1, name="F1_1")
    fitlog.add_best_metric(F1_2, name="F1_2")
else:
    print('unknown num_class')

# # Role5: Show result
for step in range(1,config.epochs+1):
    fitlog.add_loss(train_losses[step-1],name="train-loss",step=step)
    fitlog.add_loss(val_losses[step-1],name="val-loss",step=step)
    fitlog.add_metric({"train":{"acc":train_accses[step-1]}}, step=step)
    fitlog.add_metric({"val":{"acc":val_accs[step-1]}}, step=step)

experiment_id = model.get_name()+'_'+config.dataset+'_'+now_time
model_loss_acc(train_losses,train_accses,val_losses,val_accs,config.imgs_dir,experiment_id)
filelist = os.listdir(config.cache_save_dir)
for item in filelist:
    if item[0:3] == 'log':
        src = os.path.join(config.cache_save_dir, item)
        dst = os.path.join(config.cache_save_dir, experiment_id)
        os.rename(src,dst)
        break
fitlog.finish()