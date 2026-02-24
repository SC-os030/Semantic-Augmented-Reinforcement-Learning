import os
import sys
try:
    import google.colab

    IN_COLAB = True
    print(">>> 检测到 Colab 环境，正在安装依赖...")
    os.system('pip install -q transformers datasets accelerate matplotlib seaborn')
except ImportError:
    IN_COLAB = False
    print(">>> 检测到本地环境，跳过自动安装 (请确保已 pip install transformers datasets accelerate matplotlib)")

# 设置环境变量
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR = "experiment_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 数据集选择开关 ---
    # 可选值: 'gsm8k', 'svamp', 'multiarith'
    # 修改这里来切换实验对象！
    DATASET_NAME = 'gsm8k'

    # --- 模型配置 ---
    # Student: 负责生成答案 (隐式推理)
    STUDENT_MODEL_NAME = "google/flan-t5-base"
    # Teacher: 负责编码显式 CoT (语义对齐)
    TEACHER_MODEL_NAME = "prajjwal1/bert-tiny"

    # --- 核心超参数 ---
    LATENT_K = 10
    BATCH_SIZE = 8  #依据硬件情况进行调整

    # 隐式推理：只生成答案，长度很短
    MAX_SOURCE_LEN = 512
    MAX_TARGET_LEN = 64

    # --- 训练参数 (防崩坏优化版) ---
    LR_STAGE1 = 1e-4  # Stage 1: 语义注入 (正常学习率)
    LR_STAGE2 = 1e-6  # Stage 2: RL 微调 (极低学习率，防止崩坏)

    # 训练轮数
    EPOCHS_STAGE1 = 5  # 预热轮数
    EPOCHS_STAGE2 = 5  # RL 轮数

    LAMBDA_VAL = 0.2  # 奖励混合系数
    TEMP = 0.07  # InfoNCE 温度

    # --- 数据限制 ---
    # 设置为 -1 使用全量数据 (推荐)
    # 设置为 50 用于快速测试代码逻辑
    DATA_LIMIT = -1

    RUN_STAGE1 = True
    RUN_STAGE2 = True

    # 模型保存路径
    STAGE1_CHECKPOINT = os.path.join(RESULTS_DIR, f"stage1_{DATASET_NAME.lower()}.pth")
    BEST_MODEL_CHECKPOINT = os.path.join(RESULTS_DIR, f"best_model_{DATASET_NAME.lower()}.pth")


print(f"Running on: {Config.DEVICE}")
print(f"Dataset: {Config.DATASET_NAME}")
print(f"Model: {Config.STUDENT_MODEL_NAME}")


# ==========================================
# 2. Dataset Processing (数据处理核心)
# ==========================================

def process_data_item(question, cot, final_val, s_tokenizer, t_tokenizer):
    """
    构造隐式推理的输入输出对：
    - Input: Question
    - Teacher Input: Question + CoT (用于提取语义向量)
    - Target: ONLY Answer (强制模型隐式推理，不生成过程)
    """
    # Prompt: 引导隐式思考
    input_text = f"Question: {question}\nLet's think implicitly:"

    # Target: 仅包含答案！这是实现 8x 加速的关键
    target_text = f"The answer is {final_val}."

    # Teacher Input: 完整的 CoT (用于 Stage 1 InfoNCE 对齐)
    teacher_text = f"Question: {question} Let's think step by step: {cot} The answer is {final_val}."

    # Tokenization
    s_inputs = s_tokenizer(input_text, max_length=Config.MAX_SOURCE_LEN, truncation=True, padding="max_length",
                           return_tensors="pt")
    t_inputs = t_tokenizer(teacher_text, max_length=Config.MAX_SOURCE_LEN, truncation=True, padding="max_length",
                           return_tensors="pt")
    full_labels = s_tokenizer(target_text, max_length=Config.MAX_TARGET_LEN, truncation=True, padding="max_length",
                              return_tensors="pt")

    return {
        "input_ids": s_inputs["input_ids"].squeeze(0),
        "attention_mask": s_inputs["attention_mask"].squeeze(0),
        "cot_ids": t_inputs["input_ids"].squeeze(0),
        "cot_mask": t_inputs["attention_mask"].squeeze(0),
        "full_labels": full_labels["input_ids"].squeeze(0),
        "gt": final_val,
        "raw_q": question
    }


class GSM8KCustomDataset(Dataset):
    def __init__(self, data, s_tokenizer, t_tokenizer):
        self.data = data
        self.s_tokenizer = s_tokenizer
        self.t_tokenizer = t_tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # GSM8K 格式处理
        parts = item['answer'].split("####")
        cot = parts[0].strip()
        final_val = parts[1].strip()
        return process_data_item(item['question'], cot, final_val, self.s_tokenizer, self.t_tokenizer)


class SVAMPCustomDataset(Dataset):
    def __init__(self, data, s_tokenizer, t_tokenizer):
        self.data = data
        self.s_tokenizer = s_tokenizer
        self.t_tokenizer = t_tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # SVAMP 格式兼容
        body = item.get('Body') or item.get('body')
        q_text = item.get('Question') or item.get('question')
        eq = item.get('Equation') or item.get('equation')
        ans = item.get('Answer') or item.get('answer')

        question = f"{body} {q_text}"
        cot = f"The equation is {eq}."
        return process_data_item(question, cot, str(ans), self.s_tokenizer, self.t_tokenizer)


class MultiArithCustomDataset(Dataset):
    def __init__(self, data, s_tokenizer, t_tokenizer):
        self.data = data
        self.s_tokenizer = s_tokenizer
        self.t_tokenizer = t_tokenizer

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # MultiArith 格式处理
        final_val = str(item['final_answer'][0]) if isinstance(item['final_answer'], list) else str(
            item['final_answer'])
        cot = f"Calculation: {item['equation']}"
        return process_data_item(item['question'], cot, final_val, self.s_tokenizer, self.t_tokenizer)


# ==========================================
# 3. Data Loading Factory (工厂模式)
# ==========================================
class DataHandler:
    def __init__(self):
        dataset_name = Config.DATASET_NAME.lower()
        print(f">>> Initializing DataHandler for: {dataset_name} ...")

        self.s_tokenizer = AutoTokenizer.from_pretrained(Config.STUDENT_MODEL_NAME)
        self.t_tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_MODEL_NAME)

        # --- 1. GSM8K ---
        if dataset_name == 'gsm8k':
            try:
                self.dataset = load_dataset("gsm8k", "main")
            except Exception as e:
                print(f"Load failed, trying mirror... {e}")
                self.dataset = load_dataset("gsm8k", "main", trust_remote_code=True)
            self.DatasetClass = GSM8KCustomDataset

        # --- 2. SVAMP ---
        elif dataset_name == 'svamp':
            try:
                # 尝试加载本地
                self.dataset = load_dataset("csv", data_files="SVAMP.csv")
                print(">>> Loaded local SVAMP.csv")
            except:
                print(">>> Downloading SVAMP from Hugging Face Hub...")
                self.dataset = load_dataset("ChilleD/SVAMP")
            self.DatasetClass = SVAMPCustomDataset

        # --- 3. MultiArith ---
        elif dataset_name == 'multiarith':
            try:
                self.dataset = load_dataset("json", data_files="MultiArith.json")
                print(">>> Loaded local MultiArith.json")
            except:
                print(">>> Downloading MultiArith from Hugging Face Hub...")
                self.dataset = load_dataset("ChilleD/MultiArith")
            self.DatasetClass = MultiArithCustomDataset
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # 统一数据集分割 (如果只有 train)
        if 'test' not in self.dataset:
            print(">>> Splitting train set into train/test (90/10)...")
            self.dataset = self.dataset['train'].train_test_split(test_size=0.1, seed=42)

        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]

        # 调试模式截断
        if Config.DATA_LIMIT > 0:
            print(f"!!! DEBUG MODE: Using only {Config.DATA_LIMIT} samples !!!")
            self.train_data = self.train_data.select(range(min(Config.DATA_LIMIT, len(self.train_data))))
            self.test_data = self.test_data.select(range(min(20, len(self.test_data))))
        else:
            print(f">>> Full Scale Mode: Using {len(self.train_data)} training samples.")

    def get_dataloaders(self, split="train", shuffle=True):
        data = self.train_data if split == "train" else self.test_data
        dataset = self.DatasetClass(data, self.s_tokenizer, self.t_tokenizer)
        # Windows/CPU 兼容性设置
        num_workers = 0 if os.name == 'nt' else 2
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle, num_workers=num_workers)


# ==========================================
# 4. Model Framework (双塔架构)
# ==========================================
class SemanticReasoningFramework(nn.Module):
    def __init__(self):
        super().__init__()
        # Student: Flan-T5
        self.student = T5ForConditionalGeneration.from_pretrained(Config.STUDENT_MODEL_NAME)
        # Teacher: BERT (Frozen)
        self.teacher = AutoModel.from_pretrained(Config.TEACHER_MODEL_NAME)
        for p in self.teacher.parameters(): p.requires_grad = False

        # Projector: 维度对齐 (Student Dim -> Teacher Dim)
        self.proj = nn.Linear(self.student.config.d_model, self.teacher.config.hidden_size)

    def forward_stage1(self, input_ids, attention_mask, full_labels, cot_ids, cot_mask):
        """
        Stage 1 Forward: SFT Loss + InfoNCE Loss
        """
        bs = input_ids.size(0)

        # --- 1. InfoNCE: 语义对齐 ---
        # 构造前 K 个 Decoder Input IDs
        dec_ids = torch.full((bs, Config.LATENT_K), self.student.config.decoder_start_token_id).to(Config.DEVICE)

        # 获取 Student 隐状态 (Latent State)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_ids,
                               output_hidden_states=True)
        # 取最后一个隐层并平均，然后投影
        latent_vec = self.proj(torch.mean(outputs.decoder_hidden_states[-1], dim=1))

        # 获取 Teacher 显式向量 (Explicit CoT Embedding)
        with torch.no_grad():
            teacher_vec = self.teacher(input_ids=cot_ids, attention_mask=cot_mask).last_hidden_state[:, 0, :]

        # 计算 InfoNCE Loss
        feat = torch.nn.functional.normalize(latent_vec, dim=1)
        target = torch.nn.functional.normalize(teacher_vec, dim=1)
        logits = torch.matmul(feat, target.T) / Config.TEMP
        labels = torch.arange(bs).to(Config.DEVICE)
        loss_nce = nn.CrossEntropyLoss()(logits, labels)

        # --- 2. SFT: 答案生成监督 ---
        loss_sft = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=full_labels).loss

        # 总损失
        return loss_nce + loss_sft

    def generate_answer(self, input_ids, attention_mask):
        """
        隐式推理生成: 只生成答案，不生成 CoT
        """
        return self.student.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=32,  # 短生成，验证高效率
            return_dict_in_generate=True, output_scores=True,
            do_sample=False  # 贪婪解码，防止崩坏
        )


# ==========================================
# 5. Utils (Reward & Metrics)
# ==========================================
def compute_outcome_reward(pred_texts, gt_vals):
    """
    计算奖励:
    +1.0: 包含正确数字
    -0.1: 格式正确但数字错误 (Format Reward)
    -1.0: 格式错误/乱码
    """
    rewards = []
    for pred, gt in zip(pred_texts, gt_vals):
        gt_clean = str(gt).replace(',', '').strip()

        # 1. 严格匹配答案
        if re.search(r'\b' + re.escape(gt_clean) + r'\b', pred):
            rewards.append(1.0)
        # 2. 格式保底 (防止 RL 崩坏成乱码)
        # 只要包含了 "answer" 单词或者任何数字，就认为是尝试了，只给轻微惩罚
        elif "answer" in pred.lower() or re.search(r'\d', pred):
            rewards.append(-0.1)
        # 3. 乱码惩罚
        else:
            rewards.append(-1.0)
    return np.array(rewards)


# ==========================================
# 6. Main Execution Loop
# ==========================================
def run_experiment():
    print(f"\n{'=' * 40}\n Experiment Start: {Config.DATASET_NAME.upper()} \n{'=' * 40}")

    # 1. 加载数据
    dh = DataHandler()
    train_loader = dh.get_dataloaders("train")
    test_loader = dh.get_dataloaders("test", shuffle=False)

    # 2. 初始化模型
    model_wrapper = SemanticReasoningFramework().to(Config.DEVICE)

    # 记录日志
    history = {"loss": [], "test_acc": []}
    best_acc = 0.0

    # --- STAGE 1: Representation Reshaping (Semantic Augmentation) ---
    if Config.RUN_STAGE1:
        print("\n>>> [Stage 1] Semantic Augmentation (SFT + InfoNCE)")
        optimizer = AdamW(filter(lambda p: p.requires_grad, model_wrapper.parameters()), lr=Config.LR_STAGE1)

        # 学习率调度器
        num_training_steps = len(train_loader) * Config.EPOCHS_STAGE1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50,
                                                    num_training_steps=num_training_steps)

        model_wrapper.train()
        for epoch in range(Config.EPOCHS_STAGE1):
            loop = tqdm(train_loader, desc=f"S1 Epoch {epoch + 1}/{Config.EPOCHS_STAGE1}")
            avg_loss = 0
            for batch in loop:
                optimizer.zero_grad()
                # 移动数据到 GPU
                ids = batch['input_ids'].to(Config.DEVICE)
                mask = batch['attention_mask'].to(Config.DEVICE)
                cot_ids = batch['cot_ids'].to(Config.DEVICE)
                cot_mask = batch['cot_mask'].to(Config.DEVICE)
                full_labels = batch['full_labels'].to(Config.DEVICE)

                # 前向传播
                loss = model_wrapper.forward_stage1(ids, mask, full_labels, cot_ids, cot_mask)
                loss.backward()
                optimizer.step()
                scheduler.step()

                avg_loss += loss.item()
                loop.set_postfix({"Loss": f"{loss.item():.4f}"})

            history["loss"].append(avg_loss / len(train_loader))

        # 保存 Stage 1 权重
        print(f">>> Saving Stage 1 model to {Config.STAGE1_CHECKPOINT}")
        torch.save(model_wrapper.student.state_dict(), Config.STAGE1_CHECKPOINT)

    # --- STAGE 2: Policy Refinement (RL) ---
    if Config.RUN_STAGE2:
        print("\n>>> [Stage 2] Policy Refinement (RL with Format Reward)")
        # 重新初始化优化器，使用极低学习率
        optimizer = AdamW(model_wrapper.student.parameters(), lr=Config.LR_STAGE2)

        for epoch in range(Config.EPOCHS_STAGE2):
            model_wrapper.train()
            loop = tqdm(train_loader, desc=f"S2 Epoch {epoch + 1}/{Config.EPOCHS_STAGE2}")

            for i, batch in enumerate(loop):
                optimizer.zero_grad()
                ids = batch['input_ids'].to(Config.DEVICE)
                mask = batch['attention_mask'].to(Config.DEVICE)

                # 1. Rollout (生成)
                gen_out = model_wrapper.generate_answer(ids, mask)
                seqs = gen_out.sequences

                # 解码生成的文本
                txts = dh.s_tokenizer.batch_decode(seqs, skip_special_tokens=True)

                # 2. Compute Reward (计算奖励)
                rewards = compute_outcome_reward(txts, batch['gt'])
                r_tensor = torch.tensor(rewards).to(Config.DEVICE).float().unsqueeze(1)

                # 3. Update Policy (策略更新)
                # 重新前向传播计算 Log Prob (Loss)
                # 注意：这里 seqs[:, 1:] 是因为 T5 的 labels 需要移位，去掉 start token
                model_outputs = model_wrapper.student(input_ids=ids, attention_mask=mask, labels=seqs)

                # Policy Gradient Loss: -Reward * LogProb
                # model_outputs.loss 是 CrossEntropy (即 -LogProb)
                loss = (r_tensor * model_outputs.loss).mean()

                loss.backward()

                # 【关键】梯度裁剪防止崩坏
                torch.nn.utils.clip_grad_norm_(model_wrapper.student.parameters(), max_norm=1.0)

                optimizer.step()

                # 监控输出样例
                if i == 0:
                    print(f"\n[Monitor] Q: {batch['raw_q'][0][:40]}...")
                    print(f"  Pred: {txts[0]}")
                    print(f"  GT: {batch['gt'][0]} | Reward: {rewards[0]}")

            # --- Evaluation ---
            print(">>> Evaluating...")
            model_wrapper.student.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in test_loader:
                    ids = batch['input_ids'].to(Config.DEVICE)
                    mask = batch['attention_mask'].to(Config.DEVICE)

                    gen_out = model_wrapper.generate_answer(ids, mask)
                    txts = dh.s_tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)

                    # 只有 Reward=1.0 才算 Accuracy 正确
                    r = compute_outcome_reward(txts, batch['gt'])
                    correct += (r == 1.0).sum().item()
                    total += len(r)

            acc = correct / total
            history["test_acc"].append(acc)
            print(f">>> Test Accuracy: {acc * 100:.2f}%")

            # 保存最佳模型
            if acc >= best_acc:
                best_acc = acc
                print(f"!!! New Best Accuracy! Saving to {Config.BEST_MODEL_CHECKPOINT}")
                torch.save(model_wrapper.student.state_dict(), Config.BEST_MODEL_CHECKPOINT)

    # --- Plotting Results ---
    print("\n>>> Plotting Results...")
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    if history["loss"]:
        plt.plot(history["loss"], label="S1 Loss", color='blue')
        plt.title("Stage 1: Alignment Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    if history["test_acc"]:
        plt.plot(history["test_acc"], label="Test Acc", color='orange', marker='o')
        plt.title("Stage 2: RL Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        # 标注峰值
        max_acc = max(history["test_acc"])
        plt.text(len(history["test_acc"]) - 1, max_acc, f"Peak: {max_acc * 100:.1f}%")

    save_path = os.path.join(Config.RESULTS_DIR, f"final_results_{Config.DATASET_NAME}.png")
    plt.savefig(save_path)
    print(f">>> Experiment Finished! Best Acc: {best_acc * 100:.2f}%")
    print(f">>> Results saved to {save_path}")


if __name__ == "__main__":
    run_experiment()