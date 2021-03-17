import sys
import json
import re
import numpy as np

pc_table = dict()
cp_table = dict()
pi_table = dict()
ci_table = dict()
all_c_table = dict()
all_cc_table = dict()
path_map = 'train/map.txt'
path_stats = 'train/stats.txt'


def load_characters():
    """
    加载汉字、拼音，并建立汉字、拼音、序号的互相关联
    :return:
    """
    with open('拼音汉字表_12710172/拼音汉字表.txt', 'r', encoding='gbk') as f:
        lines = f.readlines()
        pinyin_num = lines.__len__() + 1
        character_num = 0
        pi_table['$'] = 0
        all_c_table['$'] = 0
        ci_table['$'] = 0
        for i, line in enumerate(lines):
            line = line[:-1]
            pinyin = line.split(' ')[0]
            characters = line.split(' ')[1:]
            pc_table[pinyin] = [character for character in characters]
            pi_table[pinyin] = i + 1
            for c in characters:
                all_c_table[c] = 0
                character_num += 1
                cp_table.setdefault(c, []).append(pi_table[pinyin])
                if not ci_table.__contains__(c):
                    ci_table[c] = character_num
    return pinyin_num, character_num  # 计算上开始符号$


def load_news(index):
    """
    加载新闻
    :param index: 新闻文件的序号
    :return: 返回一个新闻内容的列表
    """
    re_exp = '[^\u4e00-\u9fff]+'
    contents = []
    with open('sina_news_gbk/2016-%d.txt' % index, 'r', encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            contents.append(re.sub(re_exp, '$', json.loads(line)['html']))
    return contents


def load_results():
    with open(path_map, 'r', encoding='utf-8') as f:
        pc_table = json.loads(f.readline())
        cp_table = json.loads(f.readline())
        pi_table = json.loads(f.readline())
        ci_table = json.loads(f.readline())
    with open(path_stats, 'r', encoding='utf-8') as f:
        all_c_table = json.loads(f.readline())
        all_cc_table = json.loads(f.readline())

    return pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table


def collect(p_freq=None, pp_freq=None, c_freq=None, cc_freq=None):
    strings = []
    news_ids = [2, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in news_ids:
        strings += load_news(i)
    for string in strings:
        string = ''.join(['$', string])
        for i, char in enumerate(string[1:]):
            try:
                all_c_table[char] += 1
                chars = string[i-1] + char
                if all_cc_table.__contains__(chars):
                    all_cc_table[chars] += 1
                else:
                    all_cc_table[chars] = 1
            except KeyError as e:
                pass
    save()


def save(p_freq=None, c_freq=None, pp_freq=None, cc_freq=None):
    with open(path_map, 'w', encoding='utf-8') as f:
        f.write(json.dumps(pc_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(cp_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(pi_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(ci_table, ensure_ascii=False) + '\n')
    with open(path_stats, 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_c_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(all_cc_table, ensure_ascii=False) + '\n')
    if pp_freq:
        np.savetxt('train/p_freq.txt', p_freq)
        np.savetxt('train/c_freq.txt', c_freq)
        np.savetxt('train/pp_freq.txt', pp_freq)
        np.savetxt('train/cc_freq.txt', cc_freq)


def train():
    load_characters()
    # p_freq = np.zeros(pinyin_num, dtype=np.int)
    # pp_freq = np.zeros((pinyin_num, pinyin_num), dtype=np.int)
    # c_freq = np.zeros(character_num, dtype=np.int)
    # cc_freq = np.zeros((character_num, character_num), dtype=np.int)
    collect()


if __name__ == '__main__':
    train()