import os
import json
import re
import math
import pickle

path_map = 'train/map.txt'
path_stats = 'train/stats.txt'


def load_characters():
    """
    加载汉字、拼音，并建立汉字、拼音、序号的互相关联
    :return:
    """
    pc_table = dict()
    cp_table = dict()
    pi_table = dict()
    ci_table = dict()
    all_c_table = dict()
    all_cc_table = dict()
    with open('拼音汉字表_12710172/拼音汉字表.txt', 'r', encoding='gbk') as f:
        lines = f.readlines()
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
    return pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table


def load_news(index, ci_table):
    """
    加载新闻
    :param index: 新闻文件的序号
    :return: 返回一个新闻内容的列表
    """
    contents = []
    try:
        with open('sina_news_gbk/2016-%d.txt' % index, 'r', encoding='gbk') as f:
            lines = f.readlines()
            for line in lines:
                src = json.loads(line)['html']
                src = ''.join(
                    word if word in ci_table else '$'
                    for word in src
                )
                src = f'${src}$'
                src = re.sub(r'(\$)\1+', '$', src)
                contents.append(src)
    except SystemError as e:
        pass
    return contents


def load_results():
    """
    加载已训练的结果；如果没有结果（with open失败），则调用load_characters()来加载拼音、汉字
    :return:
    """
    try:
        with open(path_map, 'r', encoding='utf-8') as f:
            pc_table = json.loads(f.readline())
            cp_table = json.loads(f.readline())
            pi_table = json.loads(f.readline())
            ci_table = json.loads(f.readline())
        with open(path_stats, 'r', encoding='utf-8') as f:
            all_c_table = json.loads(f.readline())
            all_cc_table = json.loads(f.readline())
        return pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table
    except FileNotFoundError as e:
        return load_characters()


def collect(pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table):
    """
    在已有的结果上继续收集数据进行训练
    :param ci_table:
    :param pc_table:
    :param cp_table:
    :param pi_table:
    :param all_c_table: 每个汉字出现的次数
    :param all_cc_table: 每两个汉字组合出现的次数（如果未出现则不出现）
    :return: nothing
    """
    strings = []
    news_ids = [2, 4, 5, 6, 7, 8, 9, 10, 11]
    for i in news_ids:
        strings += load_news(i, ci_table)
    for string in strings:
        for i in range(len(string[1:])):
            try:
                char = string[i]
                all_c_table[char] += 1
                chars = string[i - 1] + char
                if chars in all_cc_table:
                    all_cc_table[chars] += 1
                else:
                    all_cc_table[chars] = 1
            except KeyError as e:
                pass
    save(pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table)


def save(pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table):
    with open(path_map, 'w', encoding='utf-8') as f:
        f.write(json.dumps(pc_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(cp_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(pi_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(ci_table, ensure_ascii=False) + '\n')
    with open(path_stats, 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_c_table, ensure_ascii=False) + '\n')
        f.write(json.dumps(all_cc_table, ensure_ascii=False) + '\n')


def train():
    pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table = load_results()
    collect(pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table)
    return pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table


class Predictor:
    def __init__(self):
        self.params = dict()
        self.record = list()
        pc_table, cp_table, pi_table, ci_table, all_c_table, all_cc_table = load_results()
        self.params['all_cc'] = all_cc_table
        self.params['all_c'] = all_c_table
        self.params['pc'] = pc_table
        self.params['pc']['$'] = '$'
        self.lam = 0.9
        self.all_c_num = sum(self.params['all_c'].values())
        print(self.all_c_num)
        pass

    def predict(self, line):
        line = f'$ {line}'
        pinyins = line.split(' ')
        result = ''
        all_cc = self.params['all_cc']
        all_c = self.params['all_c']
        self.record = []
        print(pinyins)
        for i in range(len(pinyins[:-1])):
            pinyin = pinyins[i]
            pinyin = pinyin.lower()
            next_py = pinyins[i + 1]
            self.record.append({})
            if i:
                for nxt_char in self.params['pc'][next_py]:
                    self.record[i][nxt_char] = {'prev': '', 'p': math.inf}
                    for now_char in self.params['pc'][pinyin]:
                        word = now_char + nxt_char
                        if word not in all_cc or now_char not in all_c:
                            continue
                        log_prob = \
                            -math.log(all_cc[word] / all_c[now_char]) + self.record[i - 1][now_char]['p']
                        if log_prob < self.record[i][nxt_char]['p']:
                            self.record[i][nxt_char]['p'] = log_prob
                            self.record[i][nxt_char]['prev'] = now_char
            else:
                for character in self.params['pc'][next_py]:
                    log_prob = -math.log((all_c[character] + 1) / self.all_c_num)
                    self.record[i][character] = {'prev': '$', 'p': log_prob}
        char = sorted(self.record[-1].items(), key=lambda item: item[1]['p'])
        char = char[0][0]
        i = self.record.__len__() - 1
        while char != '$':
            print(sorted(self.record[i].items(), key=lambda item: item[1]['p']))
            result = char + result
            char = self.record[i][char]['prev']
            i -= 1
        print(result)
        return result


if __name__ == '__main__':
    train()
