# Transformer
<br><br>
![model](image/model.png)
<br><br>

'''
Epoch: 1, Train loss: 5.342, Val loss: 4.104, Epoch time = 51.305s
Epoch: 2, Train loss: 3.759, Val loss: 3.306, Epoch time = 55.960s
Epoch: 3, Train loss: 3.156, Val loss: 2.888, Epoch time = 65.804s
Epoch: 4, Train loss: 2.767, Val loss: 2.629, Epoch time = 71.922s
Epoch: 5, Train loss: 2.478, Val loss: 2.442, Epoch time = 74.421s
Epoch: 6, Train loss: 2.249, Val loss: 2.307, Epoch time = 72.754s
Epoch: 7, Train loss: 2.056, Val loss: 2.217, Epoch time = 78.290s
Epoch: 8, Train loss: 1.895, Val loss: 2.108, Epoch time = 76.270s
Epoch: 9, Train loss: 1.754, Val loss: 2.053, Epoch time = 79.638s
Epoch: 10, Train loss: 1.632, Val loss: 1.996, Epoch time = 83.422s
Epoch: 11, Train loss: 1.523, Val loss: 1.965, Epoch time = 87.027s
Epoch: 12, Train loss: 1.418, Val loss: 1.939, Epoch time = 86.907s
Epoch: 13, Train loss: 1.328, Val loss: 1.928, Epoch time = 90.075s
Epoch: 14, Train loss: 1.250, Val loss: 1.940, Epoch time = 96.738s
Epoch: 15, Train loss: 1.172, Val loss: 1.936, Epoch time = 96.887s
Epoch: 16, Train loss: 1.101, Val loss: 1.915, Epoch time = 97.977s
Epoch: 17, Train loss: 1.035, Val loss: 1.895, Epoch time = 97.573s
Epoch: 18, Train loss: 0.976, Val loss: 1.911, Epoch time = 97.933s
'''

### Sample Result of trained model
'''
Eine Gruppe von Menschen steht vor einem Iglu .
A group of people standing in front of an igloo . 
'''

### 2.1 Model Specification

* total parameters = 55,207,087
* model size       = 215.7MB
* lr scheduling    = ReduceLROnPlateau

#### 2.1.1 configuration

* batch_size = 128
* max_len = 256
* d_model = 512
* n_layers = 6
* n_heads = 8
* ffn_hidden = 2048
* drop_prob = 0.1
* init_lr = 0.1
* factor = 0.9
* patience = 10
* warmup = 100
* adam_eps = 5e-9
* epoch = 1000
* clip = 1
* weight_decay = 5e-4

## 3. Reference
- [Attention is All You Need, 2017 - Google](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Data & Optimization Code Reference - Bentrevett](https://github.com/bentrevett/pytorch-seq2seq/)