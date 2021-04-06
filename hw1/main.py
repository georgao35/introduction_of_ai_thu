import argparse
import os
from train_2 import train, Predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--train")
    input_path = parser.parse_args().input_file
    output_path = parser.parse_args().output_file
    try:
        with open(input_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
            predictor = Predictor()
            with open(output_path, 'w', encoding='gbk') as out:
                for line in lines:
                    out.write(predictor.predict(line) + '\n')
    except SystemError as e:
        pass


def test():
    total = 0
    total_correct = 0
    total_sen = 0
    total_correct_sen = 0
    with open('输入法测试集.txt', 'r', encoding='utf-8') as f:
        while True:
            try:
                input_line = f.readline()
                if not input_line:
                    break
                while not input_line[-1].isprintable():
                    input_line = input_line[:-1]
                length = input_line.split(' ').__len__()
                result = f.readline()
                guess = predictor.predict(input_line)
                correct = True
                for i in range(length):
                    total += 1
                    if guess[i] == result[i]:
                        total_correct += 1
                    else:
                        correct = False
                if correct:
                    total_correct_sen += 1
                total_sen += 1
            except Exception as e:
                pass
    print(total, total_correct, total_correct/total)
    print(total_sen, total_correct_sen, total_correct_sen/total_sen)


if __name__ == '__main__':
    # main()
    predictor = Predictor()
    # print(predictor.predict('ji qi xue xi ji qi ying yong'))
    test()
