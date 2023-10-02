import os
import torch
import logging
from datetime import datetime
import sys

class Config():
    def __init__(self):    
        self.dataset = 'BEARD' # BEARD, TianBian-Twitter15, TianBian-Twitter16, TianBian-Weibo
        self.num_class = 4 if self.dataset == 'TianBian-Twitter16' or self.dataset == 'TianBian-Twitter15' else 2
        
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data', self.dataset, 'structure')
        self.graph_dir = os.path.join(self.project_dir, 'graph', self.dataset)
        self.topic_dir = os.path.join(self.project_dir, 'topic', self.dataset)
        self.result_dir = os.path.join(self.project_dir, 'result')
        self.imgs_dir = os.path.join(self.project_dir, 'pic')
        self.cache_save_dir = os.path.join(self.project_dir, 'cache')
        if not os.path.exists(self.cache_save_dir):
            os.makedirs(self.cache_save_dir)
        
        self.post_delay = 0
        self.time_delay = 0    # 2*60*60
        # Post, Topic-allP, Topic-eachP, other
        self.level = 'Post' if self.dataset != 'TianBian-Twitter16' and self.dataset != 'TianBian-Twitter15' else 'other'
        self.result_graph_dir = os.path.join(self.graph_dir, str(self.post_delay)) if self.level == 'other' else os.path.join(self.graph_dir, self.level, str(self.time_delay))
       
        self.bert_embedding_dim = 5000 if self.dataset == 'TianBian-Twitter16' or self.dataset == 'TianBian-Twitter15' else 768
        self.hidden_size = 0 # 300
        self.out_size = 0 # 128
        self.dropout = 0
        self.droprate = 0
        self.TDdroprate = 0
        self.BUdroprate = 0

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.validation_split = 0
        self.learning_rate = 0
        self.weight_decay = 0
        self.batch_size = 64
        self.number_workers = 0
        self.iterations = 0
        self.epochs = 0
        self.patience = 0
        self.discover_rate = 0

        # 日志初始化
        logger_init(log_file_name='log', log_level=logging.WARNING, log_dir=self.cache_save_dir)
        self.logger = logging.getLogger('RunningLogger')
        self.logger.setLevel(logging.INFO)

    def show_config(self):
        for name,value in vars(self).items():
            print(name+":",value)

    def log_config(self,model_name):
        return f'Parameters:\n\
                   dataset:{self.dataset};\n\
                   level:{self.level};\n\
                   time_delay:{self.time_delay};\n\
                   validation_split:{self.validation_split};\n\
                   learning_rate:{self.learning_rate};\n\
                   weight_decay:{self.weight_decay};\n\
                   batch_size:{self.batch_size};\n\
                   model:{model_name}'

# 日志初始化函数
def logger_init(log_file_name,log_level,log_dir,only_file=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    # 只打印到日志文件/控制台与日志同时打印
    if only_file:
        logging.basicConfig(filename=log_path,level=log_level,format=formatter,datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,format=formatter,datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),      # 输入到日志文件中
                                      logging.StreamHandler(sys.stdout)]) # 输出到控制台

if __name__ == '__main__':
    config = Config()
    config.show_config()
