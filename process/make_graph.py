import sys
sys.path.append(sys.path[0]+'/..')
from config.config import Config
from process.BEARD_graph import extract_BEARD_dataset
from process.TianBian_graph import extract_TianBian_dataset

def make_graph(config):
    if config.dataset == 'BEARD':
        extract_BEARD_dataset(config)
    elif config.dataset[0:8] == 'TianBian':
        obj = config.dataset.split('-')[1]
        print(obj)
        extract_TianBian_dataset(config, obj)

if __name__ == '__main__':
    config = Config()
    make_graph(config)
