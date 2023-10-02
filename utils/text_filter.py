'''
 Filename:  evaluate.py
 Description:  复杂指标衡量
 Created:  2022年11月5日 13时05分
 Author:  Li Shao
 PS: 还存在一些过滤问题(url)、存在重复文本
'''
import re

def text_filter(desstr, restr=''):
    try:
        emoji = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        emoji = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    other_lg = re.compile(u'[^\u0000-\u05C0\u2100-\u214F]+')
    url = re.compile(u'https?:\/\/[a-zA-Z0-9][-a-zA-Z0-9]{0,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{0,62})+\.?(\/[-a-zA-Z0-9]*)*')
    text = url.sub(restr, desstr)
    text = emoji.sub(restr, text)
    text = other_lg.sub(restr, text)
    return text
