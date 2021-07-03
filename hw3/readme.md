- 使用`pip install -r requirements.txt`安装所需依赖库
- 使用命令`python train.py`可以训练网络，可以在文件中调整参数和选项
- 如果有训练好的模型的话，使用命令`python main.py`可以只进行测试，可以在文件中调整预训练文件路径。预训练文件储存的是pytorch网络的`state_dict()`
- 采用了V2数据库，数据已经放在data文件夹下，需要在data文件夹下有数据才能训练

### 模型参数

- mlp、cnn、rnn三种网络分别在baseline.py, cnn.py, rnn.py文件中，每个参数的意义可参见docstring
- 神经网络的初始化在`init_network()`函数中，可以在其中调整初始化时候的参数
- 学习时的参数如learning_rate、weight_decay 可以在`train()`函数中的optimizer初始化时进行调整
- 如果要用gensim训练词向量则可以在train.py中调用`train_w2vec()`；否则可以用`load_w2vec(filepath)`并传递模型位置
- 在训练结束后会给出此次训练的结果，并且会保存在models中此次训练的网络参数。

### 测试训练模型

- 在models文件夹中有几个预训练好的模型，可以进行测试，输出测试指标，运行方法为`python main.py`，需要到文件中更改测试模型的类型，可以为MLP, CNN, RNN;
- 在测试前要将main中生成网络的参数调为相同，否则不能读入参数；
- 若要测试已经训练好的模型，则需要保存模型的state_dict，并用DataLoader处理输入数据
