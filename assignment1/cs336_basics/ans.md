Prob2:
(a). utf8能够一一对应; 网络上绝大部分信息采用utf8编码
(b). input: "你好" "你"字在utf8有3个字节, 110xxxxx / 1110xxxx / 11110xxx 这样的格式, 不能逐字节解码
(c). 110xxxxx 110xxxxx (不遵循多字节规则) (第一位是110xxxxx, 第二位应该是10xxxxxx)


Problem(train_bpe_tinystories):
(a). Longest token: b' accomplishment' (0min19sec)

Problem(train_bpe_expts_owt):
(a). Longest token: b'-------------------------' (61min53sec)

Problem(tokenizer_experiments):
(a): TinyStories: 约4.04299481097109 
    OpenWebText: 
(b): 3.314927768860353 (词汇出现的频率大致相同, 但是不同来源的文本风格不同, 词语偏好不同, 所有compression ratio略低)
    ___
(c): in tokens per sec: 17798
    ___

Problem(transformer_accounting):
(a). 
    vocab_size = 50257
    context_length = 1024
    num_layers = 48
    d_model = 1600
    num_heads = 25
    d_ff = 6400

    token embedding: vocab_size * d_model = 80,411,200
    within a single transformer_block:
        ln1: d_model
        attn: w_q, w_k, w_v, w_o: d_model * d_model * 4
        ln2: d_model
        ffn: w1, w2, w3: d_ff * d_model * 3
        Total: 1600 * 2 + 1600 * 1600 * 4 + 6400 * 1600 * 3 = 40,963,200
    norm: d_model = 1600
    output embedding: vocab_size * d_model = 80,411,200

    Total weight num: 2,127,057,600
    Memory: 8,508,230,400 B ≈ 7.92 GB

(b).

    token embedding: 2 * vocab_size * d_model * context_length = 164,682,137,600
    within a single transformer_block:
        ln1: 0
        attn: 2 * d_model * context_length * context_length + 2 * d_model * context_length * context_length + 2 * d_model * context_length * d_model
        ln2: 0
        ffn: (2 * d_model * d_ff * 2 + 2 * d_model * d_ff) * context_length
        Total: d * c(26d + 4c)
    output embedding: 2 * vocab_size * d_model * context_length = 164,682,137,600
    Total: 3,923,043,942,400 FLOPS
    Total: d * c(4v + n(26d + 4c))

(c). 
    FFN

(d) & (e).

| Model | Embedding | Attention | FFN |
| --- | --- | --- | --- |
| GPT-2 small | 158,094,852,096 | 53,760,491,520 | 173,946,175,488 |
| GPT-2 medium | 210,793,136,128 | 154,618,822,656 | 618,475,290,624 |
| GPT-2 large   | 263,491,420,160 | 279,172,874,240 | 1,288,490,188,800 |
| GPT-2 XL(16,384 seq_len) | 5,269,828,403,200 | 86,489,903,923,200 | 48,318,382,080,000 |

    as model size increases, FFN greatly increases
    as context length increases, Attn greatly increases

Problem (adamwAccounting):
(a).
    params:
        - Transformer block
        - RMSNorm     d_model * 2
        - MHA         4 * d_model * d_model
        - FFN         12 * d_model ** 2
        = d_model (16 * d_model + 2)
        - RMSNorm       d_model
        - Embedding     2 * d_model * vocab_size
        - Cross_entropy -
    = d_model(2 * vocab_size + 1 + num_layers(2 + 16 * d_model))

    activations:
        - Transformer block
        - RMSNorm     2 * batch_size * context_length * d_model
        - MHA         3 * batch_size * context_length * d_model + 2 * batch_size * context_length * context_length + 2 * batch_size * context_length * d_model 
        - FFN         batch_size * context_length * 4 * d_model + batch_size * context_length * 4 * d_model + batch_size * context_length * d_model
        = batch_size * context_length(16 * d_model + 2 * context_length) * num_layers
        - RMSNorm       batch_size * context_length * d_model
        - Embedding     batch_size * context_length * vocab_size
        - Cross_entropy batch_size * context_length * vocab_size
    = batch_size * context_length((16 * num_layers + 1) * d_model + 2 * vocab_size + 2 * num_layers * context_length)

    gradients
    = d_model(2 * vocab_size + 1 + num_layers(2 + 16 * d_model))

    optimizer state
    = 2 * d_model(2 * vocab_size + 1 + num_layers(2 + 16 * d_model))

    Total
    = 4 * d_model(2 * vocab_size + 1 + num_layers(2 + 16 * d_model)) + batch_size * context_length((16 * num_layers + 1) * d_model + 2 * vocab_size + 2 * num_layers * context_length)
    memory should times 4 

(b).
    1,463,519,232 * batch_size + 8,508,230,400
    around 52.8

(c).
    3 + 4 + 6 = 13
    13 * d_model(2 * vocab_size + 1 + num_layers(2 + 16 * d_model))
    先看element-wise, 每次循环做大约13次flops, 然后乘以总参数量

(d).
    forward: b * d * c(4v + n(26d + 4c)) ≈ 1.6e21
    backward: 2 * forward
    optimize: batch_size * 13 * d_model(2 * vocab_size + 1 + num_layers(2 + 16 * d_model))
    Total: 4.82e21
    around 5723 days
    
    quick check:
        # tokens: batch_size * context_length
        # params: d_model(2 * vocab_size + 16 * d_model + 3)
