***This Git is forked from [Patch-Diffusion](https://github.com/Zhendong-Wang/Patch-Diffusion) with some changes to dockerfile to accomodate FlashAttention-2. So for docker image instruction, visit [their requirement tab](https://github.com/Zhendong-Wang/Patch-Diffusion/tree/main?tab=readme-ov-file#requirements)***

# Main ideas = Metaformer + CANVAS + DiffusionNAG
- [Metaformer-v1](https://arxiv.org/pdf/2111.11418.pdf) gave 1 important insight: If a network block is already decently arranged, any kernel we replace can provide good performance. ***As it is easier and often to be the right choice to iteratively improve performance than reinventing the wheel***, in this case that is swapping out existing kernel with better kernel, I turn toward to [CANVAS](https://github.com/tsinghua-ideal/Canvas) which is a NAS algorithm to find candidate kernel instead of candidate network block. CANVAS uses random sampling, so to make it better we will use [DiffusionNAG with a Neural Predictor](https://arxiv.org/abs/2305.16943) to provide guidance. After we've found the ideal kernel, we iteratively use that kernel to swap out existing Conv block.
- From [How Powerful are Performance Predictors in NAS](https://openreview.net/attachment?id=6RB77-6-_oI&name=supplementary_material), we know that SoTL-E + Jacov.Covariance is a pretty solid Neural Predictor that can be used on candidate network block that has unknown kernel. However, it takes 1000 random archs to train SoTL-E to get such good result in the paper so we need significant speedups in trianing. ***Note: We DO NOT freeze layer to avoid the same issue that plagued hypernet NAS.***

# Training speedups: 
- [Patch-Diffusion](https://github.com/Zhendong-Wang/Patch-Diffusion): ***This is where this github is forked from***. 2x faster training, 64x64 dataset has +0.1->0.5 FID compared to EDM. Pretrained ADM checkpoint on [Huggingface](https://huggingface.co/zhendongw/patch-diffusion/tree/main). [EDM](https://github.com/NVlabs/edm/tree/main) mostly improve FID by modding sampling algorithm and precondition U-Net (i.e. DDPM, NCSN, ADM) but not changing the architecture per se.
- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) and see [blog post for FlashAttention-2](https://crfm.stanford.edu/2023/07/17/flash2.html) and [animation for FlashAttention-1, FlashAttention-2, Self-Attention](https://www.youtube.com/playlist?list=PLBWdTDczuFNrxnetcH-5CVLRP1u4Ua87m)
- Since this is a retraining problem, we can use LoRA to only pick the most relevant entries in the weight matrix rather than all entries in the weight matrix to retrain.
- [Knowledge Distillation on U-Net](https://arxiv.org/pdf/2305.15798.pdf) for faster finetuning where equation (1) can be replaced with Patch-Diffusion loss or EDM loss: 

# Training times
All info about duration for training from scratch is in [Table 7 in Supplementary Materials of EDM](https://openreview.net/attachment?id=k7FuTOWMOc7&name=supplementary_material). A rule of thumb is 10000 training iteration per image for finetuning on A DIFFERENT DATASET as shown in [a Github PR](https://github.com/lucidrains/denoising-diffusion-pytorch/issues/121), Patch-Diffusion uses 20000 training iteration per image for finetuning - harder task than retraining on THE SAME DATASET. However, it really depends on how diverse is the dataset and resolution. Let's follow Patch-Diffusion for retraining FFHQ, Celeb-A but we will scale training iteration based on the ratio of training iteration to train from scratch between different datasets. From EDM code we know that Training Iteration = Duration (MImg) / Batch size per 1 iteration, thus: <br>
> Duration for FFHQ, CelebA-64x64 = (20000 x 1) x 256 / 10^6 = 5.12 MImg. <br>
> Duration for CIFAR10-32x32 = (20000x0.5) x 512 / 10^6 = 5.12 MImg. In table 7, training iteration per image for FFHQ and CelebA are also x2 training iteration per image for CIFAR10. <br>
> Duration for ImageNet-64x64 = (20000x0.78) x 4096 / 10^6 = 63.9 MImg <br>

Estimate training day = (100 x 1 x (16/8) x (5.12/200)) / (8 + 1.77) = 0.52 day for 8 A100. Alternatively, we can train NAS Performance Predictor like in [How Powerful are Performance Predictors in NAS](https://openreview.net/attachment?id=6RB77-6-_oI&name=supplementary_material) paper, meaning estimate training day = 5.2 days instead.
> 1 is the duration to train CelebA and FFHQ with Patch-Diffusion for 16 V100.
> 5.12/200 for the ratio of duration (MImg) between Retraining and Training from scratch. <br>
> (16/8) total days for training Patch-Diffusion for 8 V100 instead. This is so that we can estimate like-for-like between 8 V100 and 8 A100 <br>
> x8 for Mosaic Diffusion. <br>
> x1.77 for bandwidth-bound speedup from V100 to A100. <br>

For ImageNet-64x64: (100 x 13 x (32/8) x (63.9/2500)) / (8 + 1.77) = 13.6 days for 8 A100: If we train on 1000 random arch, it'd be 136 days so not feasible. <br>
For CIFAR10-32x32: (1000 x 2 x (8/8) x 1/2 x (5.12/200)) / (8 + 1.77) = 2.62 days: Almost halved of training 1000 random archs on CelebA and FFHQ. <br>


# Changelog
### 0.0: Add FlashAttention-2 
See [this](https://github.com/Dao-AILab/flash-attention/issues/108) and [this](https://github.com/HazyResearch/diffusers/commit/fd45ca2afb26d013e954ccbeba8b639c4783b270?diff=unified&w=0) Git PR.

### 1.0: Adding CANVAS
- Kernel within existing network architecture is instantiated as placeholder in line 235 in sample.py
> m = torchvision.models.resnet18() <br> kernels = placeholder.get_placeholders(m, example_input, check_shapes=True)
- Then sample() in sample.py to sample kernel from search space
> kernel = canvas.sample(m, torch.zeros((1, 3, 224, 224)).cuda(), 'cuda:0') <br>
> print(kernel.module)          # Show generated torch.nn.Module class. <br>
> print(kernel.graphviz)        # Show generated GraphViz code.
- Then replace() in sample.py will replace current placeholder with sampled kernels
> canvas.replace(m, kernel.module)   # Replace all kernels.

### 1.1: Adding DiffusionNAG to replace random sampling in CANVAS
