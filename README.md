# Semantic-Augmented Reinforcement Learning for Fixed-Length Latent Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

This repository contains the official implementation of the paper **"Semantic-Augmented Reinforcement Learning for Fixed-Length Latent Reasoning"**.

## 🚀 Abstract

Large Language Models (LLMs) excel at reasoning via Chain-of-Thought (CoT) but suffer from high computational costs and latency. This paper introduces a **Semantic-Augmented Implicit Reasoning Framework** that compresses explicit reasoning steps into fixed-length latent representations. 

By leveraging **Cross-Modal Contrastive Learning (Stage 1)** and **Reinforcement Learning with Hybrid Rewards (Stage 2)**, our method enables small-scale models (e.g., Flan-T5-Base) to solve complex arithmetic problems with **8.0x inference speedup** while maintaining high accuracy.

## 🛠️ Architecture

The framework consists of two training stages:
1.  **Representation Reshaping**: Aligning latent states with explicit CoT reasoning using InfoNCE loss.
2.  **Policy Optimization**: Refining the decision boundary using RL (PPO-style) with outcome and format-based rewards.
