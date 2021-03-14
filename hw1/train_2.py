import sys
import json, re

pc_table = dict()
all_c_table = dict()
path = 'train/pinyin_table.txt'


def load_characters():
    with open('拼音汉字表_12710172/拼音汉字表.txt', 'r', encoding='gbk') as f:
        lines = f.readlines()
        # table_file = open(path, 'w', encoding='utf-8')
        for line in lines:
            line = line[:-1]
            pinyin = line.split(' ')[0]
            characters = line.split(' ')[1:]
            pc_table[pinyin] = [{character: 0} for character in characters]
        # table_file.close()
        # print(table)
    with open('拼音汉字表_12710172/一二级汉字表.txt', 'r', encoding='gbk') as f:
        characters = f.readline()
        for c in characters:
            all_c_table[c] = 0
    all_c_table['$'] = 0


def load_news(index):
    print('2016-0%d' % index)
    re_exp = '[^\u4e00-\u9fff]+'
    contents = []
    with open('sina_news_gbk/2016-02.txt', 'r', encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            contents.append(re.sub(re_exp, '$', json.loads(line)['html']))
    return contents


def load_results():
    pass


def collect():
    strings = load_news(2)
    total = 0
    for str in strings:
        for char in str:
            try:
                all_c_table[char] += 1
                total += 1
            except KeyError as e:
                pass
                # print('not found %s' % char)
    print(all_c_table)
    print(total)
    save()


def save():
    with open(path, 'w', encoding='gbk') as f:
        f.write(json.dumps(pc_table, ensure_ascii=False))
        f.write('\n')
        f.write(json.dumps(all_c_table, ensure_ascii=False))


def train():
    load_characters()
    collect()
