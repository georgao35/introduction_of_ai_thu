import pickle
import json
import re
import math
import time


class Pinyin:
    def __init__(self):
        self.pc_table = dict()
        self.cp_table = dict()
        self.pi_table = dict()
        self.ci_table = dict()
        self.all_c_table = dict()
        self.all_cc_table = dict()
        self.all_ccc_table = dict()
        self.stat_path = '../train/stats.txt'
        self.stat3path = '../train/stats3.txt'
        self.map_path = '../train/map.txt'
        self.load_pc()
        self.load_stats()
        self.reg = 0.9999
        self.alpha = 0.3
        self.epsilon = 1e-8
        self.num_choose = 6
        self.num_select = 2 * self.num_choose  # 选取进一步扩展的字数

    def collect(self):
        news_ids = [2, 4, 5, 6, 7, 8, 9, 10, 11]
        strings = []
        for index in news_ids:
            with open('sina_news_gbk/2016-%d.txt' % index, 'r', encoding='gbk') as f:
                lines = f.readlines()
                for line in lines:
                    src = json.loads(line)['html']
                    src = ''.join(
                        word if word in self.ci_table else '$'
                        for word in src
                    )
                    src = f'${src}$'
                    src = re.sub(r'(\$)\1+', '$', src)
                    strings.append(src)
        for string in strings:
            for i in range(len(string[1:])):
                try:
                    char = string[i]
                    self.all_c_table[char] += 1
                    chars = string[i - 1] + char
                    if chars in self.all_cc_table:
                        self.all_cc_table[chars] += 1
                    else:
                        self.all_cc_table[chars] = 1
                except KeyError as e:
                    pass
                try:
                    chars = string[i - 1] + string[i] + string[i + 1]
                    if chars in self.all_ccc_table:
                        self.all_ccc_table[chars] += 1
                    else:
                        self.all_ccc_table[chars] = 1
                except KeyError:
                    pass
        self.save()

    def load_pc(self):
        try:
            with open(self.map_path, 'r', encoding='utf-8') as f:
                self.pc_table = json.loads(f.readline())
                self.cp_table = json.loads(f.readline())
                self.pi_table = json.loads(f.readline())
                self.ci_table = json.loads(f.readline())
            self.pc_table['$'] = '$'
        except FileNotFoundError:
            pass

    def load_stats(self):
        with open(self.stat_path, 'r', encoding='utf-8') as f:
            self.all_c_table = json.loads(f.readline())
            self.all_cc_table = json.loads(f.readline())
        with open(self.stat3path, 'rb') as f:
            self.all_ccc_table = pickle.load(f)

    def save(self):
        with open(self.stat3path, 'wb') as f:
            pickle.dump(self.all_ccc_table, f)

    def train(self):
        self.collect()

    def predict(self, line):
        all_ccc = self.all_ccc_table
        all_cc = self.all_cc_table
        all_c = self.all_c_table
        all_c_num = sum(self.all_c_table.values())

        line = f'$ {line} $'
        pinyins = line.split(' ')
        result = ''
        record = []
        record2 = []
        for i in range(len(pinyins[:-1])):
            pinyin = pinyins[i]
            pinyin = pinyin.lower()
            next_py = pinyins[i + 1]
            record.append({})
            record2.append({})
            if i:
                for nxt_char in self.pc_table[next_py]:
                    record[i][nxt_char] = []
                    record2[i][nxt_char] = {'prev': '', 'p': math.inf}
                    tmp = dict()
                    for now_char in self.pc_table[pinyin]:
                        word = now_char + nxt_char
                        prob = self.epsilon
                        if word in all_cc:
                            prob += all_cc[word] / all_c[now_char] * self.reg
                        prob += all_c[nxt_char] / all_c_num * (1-self.reg)
                        # print(prob)
                        log_prob = -math.log(prob) + record2[i-1][now_char]['p']
                        tmp[now_char] = log_prob
                    further_calculate = sorted(tmp.items(), key=lambda item: item[1])[:self.num_select]
                    for now_char in further_calculate:
                        now_char = now_char[0]
                        record[i][nxt_char].append({'prev': now_char, 'p': math.inf})
                        for prev in record[i-1][now_char]:
                            # print(prev['prev'], type(now_char), type(nxt_char))
                            word3 = prev['prev'] + now_char + nxt_char
                            word = prev['prev'] + now_char
                            prob = self.epsilon
                            if word3 in all_ccc and word in all_cc:
                                prob += all_ccc[word3] / all_cc[word] * self.reg
                            prob += all_c[nxt_char] / all_c_num * (1-self.reg)
                            log_prob = -math.log(prob) + prev['p']
                            if log_prob < record[i][nxt_char][-1]['p']:
                                record[i][nxt_char][-1]['p'] = log_prob
                            if log_prob < record2[i][nxt_char]['p']:
                                record2[i][nxt_char]['p'] = log_prob
                                record2[i][nxt_char]['prev'] = now_char
                    record[i][nxt_char]=sorted(record[i][nxt_char], key=lambda item: item['p'])[:self.num_choose]
            else:
                for character in self.pc_table[next_py]:
                    prob = (all_c[character] + 1) / all_c_num
                    if '$' + character in all_cc:
                        prob = prob * self.alpha + (1 - self.alpha) * all_cc['$' + character] / all_c['$']
                    log_prob = - math.log(prob)
                    record2[i][character] = {'prev': '$', 'p': log_prob}
                    record[i][character] = [{'prev': '$', 'p': log_prob}]
        i = len(record2)-1
        char = record2[i]['$']['prev']
        while char != '$':
            result = char + result
            i -= 1
            char = record2[i][char]['prev']
        return result


if __name__ == '__main__':
    pyboard = Pinyin()
    print(pyboard.predict("ji qi xue xi ji qi ying yong"))
