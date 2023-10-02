# TPC-BiGAT

This repository contains code related to my paper *Rumor Detection on Potential Hot Topics with Bi-Directional Graph Attention Network*.

## Outline

In this paper:

- We cluster single tweet into topics for topic-level rumor detection. 
- We introduce the neural topic model into rumor detection task for the first time to automatically cluster textual information. 
- We design a topic heat model to filter irrelevant information and reduce the impact of them. 
- We model the rumor propagation process as a graph structure and use a bi-directional graph attention network to learn it. 
- Extensive experiments illustrate that the proposed method achieves a better detection performance compared to existing baselines.

## How to use

### Environment Requirements

Before running this project, make sure you have the following environment installed:

- Python 3.7 and above
- Required dependency libraries (can be found in `requirements.txt`)

### Installation of dependencies

```
pip install -r requirements.txt
```

### Running code

Depending on the model, dataset, and training parameters used, relevant configurations are made in the `config.py` file, and then the main program entry file is executed to start it.

```
python main.py
```

## Dataset

Three datasets are used in this paper.

[^Twitter15 and Twitter16]: Jing Ma, Wei Gao, and Kam-Fai Wong. Detect rumors in microblog posts using propagation structure via kernel learning. Association for Computational Linguistics, 2017.
[^BEARD]: Fengzhu Zeng and Wei Gao. Early Rumor Detection Using Neural Hawkes Process with a New Benchmark Dataset. In Proceedings of the 2022 Conference of the North American  Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4105–4117, Seattle, United States, July 2022. Association for Computational Linguistics.

## Contribute

If you have any suggestions or have found a bug, feel free to submit an issue or just fork and make a Pull Request.

## Contact details

If you have any questions, you can contact me at [lishao@stu.scu.edu.com](lishao@stu.scu.edu.com).
