ChaseFire is a proof-of-concept project that explores making transformer-based fake news detection accessible by strategically optimizing Microsoft's DeBERTa-v3-base model. It poses an important research question: Can we build a competitive fake news detection model with limited computational resources through architectural scaling and training innovations?

Model:
Aggressive Model Scaling for Accessibility: The original 12-layer, 12-attention-headed DeBERTa-v3-base is scaled down to 6 layers and 6 attention heads to reduce computational and memory requirements, making training feasible on consumer-grade GPUs without sacrificing core representational power.

Dual-Head Unified Architecture: The model combines masked language modeling (MLM) pretraining and downstream binary classification for fake news detection within a single framework, facilitating efficient model reuse and training.

Memory Optimization Techniques: Use of gradient checkpointing and mixed precision (bfloat16) training reduces training time and memory footprint, enabling practical experimentation in constrained environments.

Efficient Pretraining Strategy: Pretraining uses a focused 20% subset of WikiText-103, balancing domain exposure and computational feasibility to demonstrate foundational language understanding.

Integrated Training and Evaluation Pipeline: A streamlined codebase supports both MLM pretraining and fine-tuning steps with sophisticated scheduling, optimization, and automated logging via Weights & Biases.

Proof-of-Concept Significance:
While the current implementation represents an initial exploration with limited pretraining epochs and partial dataset use, this setup demonstrates the promise of this scaled architecture. The model establishes a potential pathway for building lightweight transformer models suited for fake news detection in resource-limited scenarios.

Architecture:
Lightweight DeBERTa Design

Training Pipeline
Stage 1: Masked Language Modeling (MLM) pretraining on WikiText-103 subset

Stage 2: Binary classification fine-tuning on WELFake dataset

Optimization: AdamW optimizer with linear learning rate scheduling

Monitoring: Experiment tracking and logging integrated via Weights & Biases

Performance Insights
Metric	Value
Accuracy	83.16%
F1-Score	82.27%
Precision	89.78%
Recall	75.91%
ROC-AUC	84.64%
MCC	0.673
Note: These metrics reflect the current training scope and serve as a baseline for future enhancements.

Implementation Features:
Efficient Memory Usage: Gradient checkpointing and mixed precision reduce GPU memory usage

Flexible, Modular Codebase: Easily extendable PyTorch implementation with clear separation of pretraining and fine-tuning modules

Dataset Handling: Supports both masked language modeling data and classification data seamlessly

Comprehensive Metrics: Includes precision, recall, F1, ROC-AUC, and MCC for robust performance evaluation

Experimental Setup
Pretraining Dataset: 20% subset of WikiText-103 (masked language modeling)

Fine-tuning Dataset: WELFake (fake news classification)

Batch Size: 32 for memory optimization

Learning Rate: 1e-5 with AdamW and linear scheduler

🎯 Future Directions
Complete Pretraining on full WikiText-103 with extended epochs

Fine-tuning Refinements including LoRA parameter-efficient tuning

Cross-validation and Benchmarking against state-of-the-art fake news classifiers

Model Scaling Experiments to find optimal depth/head configurations
