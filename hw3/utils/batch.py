import random


class Batch:
    def __init__(self, data: list, batch_num: int):
        self.data = data
        self.data_size = len(data)
        self.batch_size = batch_num
        self.len = (self.data_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batch_size = self.batch_size
        random.shuffle(self.data)

        for i in range(self.len):
            data_selected = self.data[batch_size * i: batch_size * (i+1)]
            yield data_selected
