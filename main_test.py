import os
# 1. 设置环境变量抑制干扰日志
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import numpy as np
import re
import matplotlib.pyplot as plt
from openai import OpenAI
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# ==========================================
# 1. Configuration (过拟合验证版)
# ==========================================
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR = "experiment_results"
    LOCAL_DATA_PATH = "dataset"

    OPENAI_API_KEY = ""
    OPENAI_MODEL = "gpt-4o-mini"

    STUDENT_MODEL_NAME = "google/flan-t5-base"
    TEACHER_MODEL_NAME = "prajjwal1/bert-tiny"

    LATENT_K = 3
    BATCH_SIZE = 1

    LR_STAGE1 = 1e-4
    LR_STAGE2 = 2e-5

    EPOCHS_STAGE1 = 5
    EPOCHS_STAGE2 = 20

    LAMBDA_VAL = 0.2
    TEMP = 0.07

    DATA_LIMIT = 1

    RUN_STAGE1 = True
    RUN_STAGE2 = True
    STAGE1_CHECKPOINT = "stage1_student.pth"


os.makedirs(Config.RESULTS_DIR, exist_ok=True)
print(f"Running on device: {Config.DEVICE}")


# ==========================================
# 2. Dataset Handler
# ==========================================
class GSM8KCustomDataset(Dataset):
    def __init__(self, data, student_tokenizer, teacher_tokenizer):
        self.data = data
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = "I have 1 apple. I buy 1 more apple. How many apples do I have?"
        final_val = "2"
        cot = "I start with 1 apple. I buy 1 more. 1 + 1 = 2."

        # Prompt
        input_text = f"Question: {question}\nLet's think step by step. The answer is:"

        student_inputs = self.student_tokenizer(
            input_text, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        )
        cot_inputs = self.teacher_tokenizer(
            cot, max_length=128, truncation=True, padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids": student_inputs["input_ids"].squeeze(0),
            "attention_mask": student_inputs["attention_mask"].squeeze(0),
            "cot_input_ids": cot_inputs["input_ids"].squeeze(0),
            "cot_attention_mask": cot_inputs["attention_mask"].squeeze(0),
            "ground_truth_answer": final_val,
            "raw_question": question
        }


class GSM8KHandler:
    def __init__(self):
        print(f">>> Loading GSM8K data...")
        try:
            try:
                self.dataset = load_from_disk(Config.LOCAL_DATA_PATH)
            except:
                self.dataset = load_dataset("gsm8k", "main", cache_dir=Config.LOCAL_DATA_PATH)
        except Exception as e:
            raise e

        self.student_tokenizer = AutoTokenizer.from_pretrained(Config.STUDENT_MODEL_NAME)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(Config.TEACHER_MODEL_NAME)

        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]

        if Config.DATA_LIMIT > 0:
            print(f"!!! DEBUG MODE: Using only {Config.DATA_LIMIT} samples !!!")
            self.train_data = self.train_data.select(range(min(Config.DATA_LIMIT, len(self.train_data))))
            self.test_data = self.test_data.select(range(min(Config.DATA_LIMIT, len(self.test_data))))

    def get_dataloaders(self, split="train", shuffle=True):
        data = self.train_data if split == "train" else self.test_data
        dataset = GSM8KCustomDataset(data, self.student_tokenizer, self.teacher_tokenizer)
        return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=shuffle)


# ==========================================
# 3. Model Framework
# ==========================================
class SemanticReasoningFramework(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = T5ForConditionalGeneration.from_pretrained(Config.STUDENT_MODEL_NAME)
        self.teacher = AutoModel.from_pretrained(Config.TEACHER_MODEL_NAME)
        for p in self.teacher.parameters(): p.requires_grad = False
        self.proj = nn.Linear(self.student.config.d_model, self.teacher.config.hidden_size)

    def get_latent_representation(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.full((batch_size, Config.LATENT_K), self.student.config.decoder_start_token_id).to(
            Config.DEVICE)
        outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                               output_hidden_states=True)
        last_hidden = outputs.decoder_hidden_states[-1]
        latent_vec = torch.mean(last_hidden, dim=1)
        return self.proj(latent_vec)

    def get_teacher_representation(self, cot_ids, cot_mask):
        with torch.no_grad():
            outputs = self.teacher(input_ids=cot_ids, attention_mask=cot_mask)
            return outputs.last_hidden_state[:, 0, :]

    def generate_with_latent(self, input_ids, attention_mask):
        return self.student.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=Config.LATENT_K + 64, return_dict_in_generate=True, output_scores=True,
            do_sample=False, num_beams=1
        )


# ==========================================
# 4. Reward System
# ==========================================
class GPT4ProcessEvaluator:
    def __init__(self):
        self.use_api = False
        if "sk-" in Config.OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.use_api = True
            except:
                pass

    def get_batch_scores(self, questions, latent_texts):
        if not self.use_api: return np.array([0.1] * len(questions))
        scores = []
        for q, lat in zip(questions, latent_texts):
            try:
                prompt = f"Question: {q}\nThought: '{lat}'\nIs this relevant? Return 0.0 to 1.0."
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=3
                )
                val = float(response.choices[0].message.content.strip())
                scores.append(val)
            except:
                scores.append(0.0)
        return np.array(scores)


def compute_outcome_reward(pred_texts, gt_texts):
    rewards = []
    for pred, gt in zip(pred_texts, gt_texts):
        # 宽容匹配：只要包含正确数字即可
        if gt in pred:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return np.array(rewards)


def info_nce_loss(feat, target, temp=Config.TEMP):
    feat = F.normalize(feat, dim=1)
    target = F.normalize(target, dim=1)
    logits = torch.matmul(feat, target.T) / temp
    labels = torch.arange(logits.size(0)).to(Config.DEVICE)
    return F.cross_entropy(logits, labels)


# ==========================================
# 5. Main Loop
# ==========================================
def run_experiment():
    print(f"\n{'=' * 40}\n Experiment Start (Overfit Mode) \n{'=' * 40}")

    data_handler = GSM8KHandler()
    train_loader = data_handler.get_dataloaders("train")
    model_wrapper = SemanticReasoningFramework().to(Config.DEVICE)
    history = {"s1_loss": [], "s2_reward": []}

    # --- STAGE 1 ---
    if Config.RUN_STAGE1:
        print("\n>>> [Stage 1] Semantic Augmentation")
        optimizer = AdamW(filter(lambda p: p.requires_grad, model_wrapper.parameters()), lr=Config.LR_STAGE1)
        for epoch in range(Config.EPOCHS_STAGE1):
            model_wrapper.train()
            loop = tqdm(train_loader, desc=f"S1 Epoch {epoch + 1}")
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attn_mask = batch['attention_mask'].to(Config.DEVICE)
                cot_ids = batch['cot_input_ids'].to(Config.DEVICE)
                cot_mask = batch['cot_attention_mask'].to(Config.DEVICE)
                loss = info_nce_loss(model_wrapper.get_latent_representation(input_ids, attn_mask),
                                     model_wrapper.get_teacher_representation(cot_ids, cot_mask))
                loss.backward()
                optimizer.step()
                loop.set_postfix({"Loss": f"{loss.item():.4f}"})
                history["s1_loss"].append(loss.item())
        torch.save(model_wrapper.student.state_dict(), os.path.join(Config.RESULTS_DIR, Config.STAGE1_CHECKPOINT))

    # --- STAGE 2 ---
    if Config.RUN_STAGE2:
        print("\n>>> [Stage 2] Policy Refinement")
        if not Config.RUN_STAGE1:
            try:
                model_wrapper.student.load_state_dict(
                    torch.load(os.path.join(Config.RESULTS_DIR, Config.STAGE1_CHECKPOINT)))
            except:
                pass

        optimizer = AdamW(model_wrapper.student.parameters(), lr=Config.LR_STAGE2)
        evaluator = GPT4ProcessEvaluator()

        for epoch in range(Config.EPOCHS_STAGE2):
            model_wrapper.train()
            loop = tqdm(train_loader, desc=f"S2 Epoch {epoch + 1}")

            for i, batch in enumerate(loop):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attn_mask = batch['attention_mask'].to(Config.DEVICE)
                gen_output = model_wrapper.generate_with_latent(input_ids, attn_mask)
                sequences = gen_output.sequences

                latent_texts = data_handler.student_tokenizer.batch_decode(sequences[:, 1:Config.LATENT_K + 1],
                                                                           skip_special_tokens=True)
                full_texts = data_handler.student_tokenizer.batch_decode(sequences[:, 1:], skip_special_tokens=True)

                r_proc = evaluator.get_batch_scores(batch['raw_question'], latent_texts)
                r_out = compute_outcome_reward(full_texts, batch['ground_truth_answer'])
                r_total = (1 - Config.LAMBDA_VAL) * r_out + Config.LAMBDA_VAL * r_proc
                r_tensor = torch.tensor(r_total).to(Config.DEVICE).float().unsqueeze(1)

                outputs = model_wrapper.student(input_ids=input_ids, attention_mask=attn_mask,
                                                labels=sequences[:, 1:].contiguous())
                loss = (r_tensor * outputs.loss).mean()
                loss.backward()
                optimizer.step()

                history["s2_reward"].append(np.mean(r_total))
                loop.set_postfix({"AvgR": f"{np.mean(r_total):.2f}"})

                print(f"\n[Monitor] Q: {batch['raw_question'][0][:30]}...")
                print(f"  Pred: {full_texts[0]}")
                print(f"  GT: {batch['ground_truth_answer'][0]} | R_Out: {r_out[0]}")
                if r_out[0] > 0: print("🎉🎉🎉 SUCCESS! The model solved it! 🎉🎉🎉")

    # --- Plotting ---
    if history["s1_loss"]:
        plt.figure()
        plt.plot(history["s1_loss"])
        plt.savefig(os.path.join(Config.RESULTS_DIR, "stage1_loss.png"))
    if history["s2_reward"]:
        plt.figure()
        plt.plot(history["s2_reward"])
        plt.savefig(os.path.join(Config.RESULTS_DIR, "stage2_reward.png"))

    print("\n>>> Done!")


if __name__ == "__main__":
    run_experiment()
