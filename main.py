import os
import re
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModel,
    AdamW
)

# ==========================================
# Config
# ==========================================

class Config:

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    DATASET = "gsm8k"

    STUDENT = "google/flan-t5-base"
    TEACHER = "prajjwal1/bert-tiny"

    MAX_SRC = 512
    MAX_TGT = 64

    BATCH = 8

    LR_STAGE1 = 1e-4
    LR_STAGE2 = 1e-6

    EPOCH_STAGE1 = 5
    EPOCH_STAGE2 = 5

    LATENT_K = 10

    TEMP = 0.07

    PPO_CLIP = 0.2
    KL_COEF = 0.02

    RESULTS = "results"

    os.makedirs(RESULTS, exist_ok=True)


# ==========================================
# Dataset
# ==========================================

def parse_answer(ans):

    if "####" in ans:
        cot, final = ans.split("####")
        return cot.strip(), final.strip()

    return "", ans.strip()


class MathDataset(Dataset):

    def __init__(self, tokenizer):

        dataset = load_dataset("gsm8k","main")

        self.data = dataset["train"]

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        cot, ans = parse_answer(item["answer"])

        prompt = f"Question: {item['question']}\nLet's think implicitly:"

        target = f"The answer is {ans}."

        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_SRC,
            return_tensors="pt"
        )

        lab = self.tokenizer(
            target,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_TGT,
            return_tensors="pt"
        )

        labels = lab["input_ids"].squeeze()

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {

            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels,
            "gt": ans
        }


# ==========================================
# Model
# ==========================================

class SemanticFramework(nn.Module):

    def __init__(self):

        super().__init__()

        self.student = T5ForConditionalGeneration.from_pretrained(Config.STUDENT)

        self.teacher = AutoModel.from_pretrained(Config.TEACHER)

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.projector = nn.Linear(
            self.student.config.d_model,
            self.teacher.config.hidden_size
        )

        self.value_head = nn.Linear(self.student.config.d_model,1)

    # -----------------------------------
    # Stage1
    # -----------------------------------

    def stage1(self,batch):

        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]

        out = self.student(
            input_ids=ids,
            attention_mask=mask,
            labels=labels
        )

        sft_loss = out.loss

        decoder_ids = torch.full(
            (ids.size(0),Config.LATENT_K),
            self.student.config.decoder_start_token_id
        ).to(ids.device)

        latent = self.student(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=decoder_ids,
            output_hidden_states=True
        )

        latent_vec = latent.decoder_hidden_states[-1].mean(dim=1)

        latent_vec = self.projector(latent_vec)

        with torch.no_grad():

            teacher_vec = self.teacher(ids).last_hidden_state[:,0]

        feat = F.normalize(latent_vec,dim=1)
        tgt = F.normalize(teacher_vec,dim=1)

        logits = torch.matmul(feat,tgt.T)/Config.TEMP

        labels_nce = torch.arange(len(feat)).to(ids.device)

        nce_loss = nn.CrossEntropyLoss()(logits,labels_nce)

        return sft_loss+nce_loss

    # -----------------------------------
    # Generate
    # -----------------------------------

    def generate(self,ids,mask):

        return self.student.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=32,
            do_sample=True,
            top_p=0.9
        )

    # -----------------------------------
    # logprob
    # -----------------------------------

    def logprob(self,ids,mask,seq):

        outputs = self.student(
            input_ids=ids,
            attention_mask=mask,
            labels=seq
        )

        logits = outputs.logits

        logp = F.log_softmax(logits,dim=-1)

        token_logp = logp.gather(
            -1,
            seq.unsqueeze(-1)
        ).squeeze(-1)

        return token_logp.mean(dim=1)

    # -----------------------------------
    # value
    # -----------------------------------

    def value(self,ids,mask):

        out = self.student.encoder(
            input_ids=ids,
            attention_mask=mask
        )

        hidden = out.last_hidden_state[:,0]

        return self.value_head(hidden).squeeze()


# ==========================================
# Reward
# ==========================================

def compute_reward(preds,gts):

    rewards=[]

    for p,g in zip(preds,gts):

        g=str(g).strip()

        if re.search(r'\b'+re.escape(g)+r'\b',p):

            rewards.append(1.0)

        elif re.search(r'\d',p):

            rewards.append(-0.1)

        else:

            rewards.append(-1.0)

    return np.array(rewards)


# ==========================================
# PPO loss
# ==========================================

def ppo_loss(old_logp,new_logp,adv):

    ratio=torch.exp(new_logp-old_logp)

    clip=torch.clamp(ratio,1-Config.PPO_CLIP,1+Config.PPO_CLIP)

    loss=-torch.min(ratio*adv,clip*adv)

    return loss.mean()


# ==========================================
# Stage1 training
# ==========================================

def train_stage1(model,loader):

    optim=AdamW(model.parameters(),lr=Config.LR_STAGE1)

    model.train()

    for epoch in range(Config.EPOCH_STAGE1):

        loop=tqdm(loader)

        for batch in loop:

            batch={k:v.to(Config.DEVICE) if torch.is_tensor(v) else v for k,v in batch.items()}

            loss=model.stage1(batch)

            optim.zero_grad()

            loss.backward()

            optim.step()

            loop.set_description(f"S1 loss {loss.item():.4f}")

    torch.save(model.student.state_dict(),"results/stage1.pt")


# ==========================================
# Stage2 PPO
# ==========================================

def train_stage2(model,loader,tokenizer):

    optim=AdamW(model.parameters(),lr=Config.LR_STAGE2)

    for epoch in range(Config.EPOCH_STAGE2):

        model.train()

        loop=tqdm(loader)

        for batch in loop:

            ids=batch["input_ids"].to(Config.DEVICE)
            mask=batch["attention_mask"].to(Config.DEVICE)

            seq=model.generate(ids,mask)

            text=tokenizer.batch_decode(seq,skip_special_tokens=True)

            rewards=compute_reward(text,batch["gt"])

            rewards=torch.tensor(rewards).float().to(Config.DEVICE)

            values=model.value(ids,mask)

            adv=rewards-values

            adv=(adv-adv.mean())/(adv.std()+1e-8)

            old_logp=model.logprob(ids,mask,seq).detach()

            new_logp=model.logprob(ids,mask,seq)

            policy_loss=ppo_loss(old_logp,new_logp,adv)

            value_loss=F.mse_loss(values,rewards)

            kl=(old_logp-new_logp).mean()

            loss=policy_loss+0.5*value_loss+Config.KL_COEF*kl

            optim.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optim.step()

            loop.set_description(f"PPO {loss.item():.4f}")

    torch.save(model.student.state_dict(),"results/final_model.pt")


# ==========================================
# Evaluate
# ==========================================

def evaluate(model,loader,tokenizer):

    model.eval()

    correct=0
    total=0

    with torch.no_grad():

        for batch in loader:

            ids=batch["input_ids"].to(Config.DEVICE)
            mask=batch["attention_mask"].to(Config.DEVICE)

            seq=model.generate(ids,mask)

            text=tokenizer.batch_decode(seq,skip_special_tokens=True)

            r=compute_reward(text,batch["gt"])

            correct+=(r==1).sum()
            total+=len(r)

    print("Accuracy:",correct/total)


# ==========================================
# Main
# ==========================================

def main():

    tokenizer=AutoTokenizer.from_pretrained(Config.STUDENT)

    dataset=MathDataset(tokenizer)

    loader=DataLoader(dataset,batch_size=Config.BATCH,shuffle=True)

    model=SemanticFramework().to(Config.DEVICE)

    print("\nStage1 training")

    train_stage1(model,loader)

    print("\nStage2 PPO")

    train_stage2(model,loader,tokenizer)

    print("\nEvaluation")

    evaluate(model,loader,tokenizer)


if __name__=="__main__":

    main()
