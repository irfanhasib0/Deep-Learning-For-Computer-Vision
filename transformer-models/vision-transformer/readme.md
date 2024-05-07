# Vision Transformer 
This work is on progress ...

Paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). 

Referene :
 [ViT](https://medium.com/towards-data-science/implementing-vision-transformer-vit-from-scratch-3e192c6155f0) for step-by-step guide on implementing ViT in detail.


```python
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}
```
