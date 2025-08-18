# uv run --active python cs336_basics/train.py
import wandb, time
from transformer import *
from optimizer import *
from datetime import datetime

params = {
    "steps": 1000,
    "vocab_size": 10000,
    "d_model": 512, 
    "num_heads": 16, 
    "num_layers": 4,
    "d_ff": 1344,
    "theta": 10000,
    "lr": 1e-3,
    "lr_max": 1e-3,
    "lr_min": 1e-5,
    "T_w": 100,
    "T_c": 1000,
    "weight_decay": 2e-2,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "max_norm": 1.0,
    "batch_size": 32,
    "context_length": 256,
    "train_data_path": "data/TinyStoriesV2-GPT4-_token_ids.npy",
    "valid_data_path": "data/TinyStoriesV2-GPT4-_token_ids.npy" ,
    "checkpoints_path": "checkpoints/",
    "checkpoints_period": 100,
    "log_path": "logs/log",
    "log_period": 5,
    "device": 'cpu', 
    "dtype": torch.float32,
}

def train(
        steps: int,
        vocab_size: int,
        d_model: int, 
        num_heads: int, 
        num_layers: int,
        d_ff: int,
        theta: float,
        lr: float,
        lr_max: float,
        lr_min: float,
        T_w: float,
        T_c: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        max_norm: float,
        batch_size: int,
        context_length: int,
        train_data_path,
        valid_data_path,
        checkpoints_path,
        checkpoints_period,
        log_path,
        log_period,
        device: torch.device | None=None, 
        dtype: torch.dtype | None=None,
        ):
    
    transformer_lm = Transformer_LM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta, device, dtype)
    transformer_lm = torch.compile(transformer_lm) # torch.compile对einsum支持有限
    optimizer = AdamW(transformer_lm.parameters(), lr, weight_decay, betas, eps)
    wandb.init(project="my-experiment")
    wandb.config.update({"epochs": steps, "batch_size": batch_size})
    train_text = np.memmap(train_data_path, dtype=np.uint16, mode="r")
    valid_text = np.memmap(valid_data_path, dtype=np.uint16, mode="r")
    l_ = []

    # log_time
    start_time = time.time()
    now = datetime.now()
    date_str = now.strftime("%m-%d %H") 
    log_path = log_path + date_str + ".txt"
    # 预记录信息
    with open(log_path, "a", encoding="utf-8") as f:
        log_str = f"time: {now.strftime("%m-%d %H-%M-%S")}, step:{params['steps']}, lr_max: {params['lr']}\n"
        f.write(log_str)
    for t in range(steps):
        transformer_lm.train()
        train_features, train_labels = get_batch(train_text, batch_size, context_length, device)
        # 转换成torch.long类型才能用作索引
        train_features = train_features.to(dtype=torch.long, device=device)
        train_labels   = train_labels.to(dtype=torch.long, device=device)
        pred = transformer_lm(train_features)
        
        l = cross_entropy_loss(pred, train_labels)
        optimizer.zero_grad()
        l.backward()
        l_.append(l.item())

        gradient_clipping(transformer_lm.parameters(), max_norm)
        optimizer.step()
        new_lr = lr_cosine_schedule(t, lr_max, lr_min, T_w, T_c)

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # 记录当前时间, 学习率损失, 与参数&梯度范数
        ts = time.time()
        p_l2_norm, g_l2_norm = monitor_norm(transformer_lm.parameters())
        wandb.log({"step_time": ts - start_time, "train_loss": l, "lr": new_lr, "param_l2_norm": p_l2_norm, "grad_l2_norm": g_l2_norm}, step=t)

        if t % log_period == 0 or t == steps - 1:
            train_loss = sum(l_) / len(l_)
            l_ = []
            
            valid_features, valid_labels = get_batch(valid_text, batch_size, context_length, device)
            valid_features = valid_features.to(dtype=torch.long, device=device)
            valid_labels   = valid_labels.to(dtype=torch.long, device=device)

            transformer_lm.eval()
            with torch.no_grad():
                pred = transformer_lm(valid_features)
                valid_loss = cross_entropy_loss(pred, valid_labels)
            
            with open(log_path, "a", encoding="utf-8") as f:
                log_str = f"step: {t}, train_loss: {train_loss}, valid_loss: {valid_loss}.\n"
                f.write(log_str)
            wandb.log({"valid_loss": valid_loss}, step=t)
        if t % checkpoints_period == 0 or t == steps - 1:
            pth_name = f"checkpoint_{t}steps_{l}_{date_str}.pth"
            checkpoint_path = checkpoints_path + pth_name
            save_checkpoint(transformer_lm, optimizer, steps, checkpoint_path)

if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = True
    train(**params)