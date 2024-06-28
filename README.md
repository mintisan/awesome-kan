# Awesome KAN(Kolmogorov-Arnold Network)


![Awesome](https://awesome.re/badge.svg) [![Contributions](https://img.shields.io/github/issues-pr-closed-raw/mintisan/awesome-kan.svg?label=contributions)](https://github.com/mintisan/awesome-kan/pulls) [![Commits](https://img.shields.io/github/last-commit/mintisan/awesome-kan.svg?label=last%20contribution)](https://github.com/gigwegbe/tinyml-papers-and-projects/commits/main) ![GitHub stars](https://img.shields.io/github/stars/mintisan/awesome-kan.svg?style=social)



A curated list of awesome libraries, projects, tutorials, papers, and other resources related to Kolmogorov-Arnold Network (KAN). This repository aims to be a comprehensive and organized collection that will help researchers and developers in the world of KAN!

![image](https://github.com/mintisan/awesome-kan/assets/9136049/5ce213da-5fe5-49bf-a210-01144a70c14e)


## Table of Contents

- [Awesome KAN(Kolmogorov-Arnold Network)](#awesome-kankolmogorov-arnold-network)
  - [Table of Contents](#table-of-contents)
  - [Papers](#papers)
    - [Theorem](#theorem)
  - [Library](#library)
    - [ConvKANs](#convkans)
    - [Benchmark](#benchmark)
    - [Non-Python](#non-python)
    - [Alternative](#alternative)
  - [Project](#project)
  - [Discussion](#discussion)
  - [Tutorial](#tutorial)
    - [YouTube](#youtube)
  - [Contributing](#contributing)
  - [License](#license)
  - [Star History](#star-history)


## Papers

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) : Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users. Through two examples in mathematics and physics, KANs are shown to be useful collaborators helping scientists (re)discover mathematical and physical laws. In summary, KANs are promising alternatives for MLPs, opening opportunities for further improving today's deep learning models which rely heavily on MLPs.
- [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/html/2405.07200v1)
- [Kolmogorov Arnold Informed neural network: A physics-informed deep learning framework for solving PDEs based on Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.11045) | [code](https://github.com/yizheng-wang/research-on-solving-partial-differential-equations-of-solid-mechanics-based-on-pinn) ｜ ![Github stars](https://img.shields.io/github/stars/yizheng-wang/research-on-solving-partial-differential-equations-of-solid-mechanics-based-on-pinn.svg)
- [Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.13155) | [code](https://github.com/AntonioTepsich/Convolutional-KANs) ｜ ![Github stars](https://img.shields.io/github/stars/AntonioTepsich/Convolutional-KANs.svg)
- [Smooth Kolmogorov Arnold networks enabling structural knowledge representation](https://arxiv.org/abs/2405.11318)
- [TKAN: Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07344) ｜ [code](https://github.com/remigenet/tkan) ｜ ![Github stars](https://img.shields.io/github/stars/remigenet/tkan.svg)
- [ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU](https://arxiv.org/abs/2406.02075) ｜ [code](https://github.com/quiqi/relu_kan) ｜ ![Github stars](https://img.shields.io/github/stars/quiqi/relu_kan.svg)
- [U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation](https://arxiv.org/abs/2406.02918)｜ [code](https://github.com/CUHK-AIM-Group/U-KAN) ｜ ![Github stars](https://img.shields.io/github/stars/CUHK-AIM-Group/U-KAN.svg)
- [Kolmogorov-Arnold Networks (KANs) for Time Series Analysis](https://arxiv.org/pdf/2405.08790)
- [Wav-KAN: Wavelet Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.12832)
- [A First Look at Kolmogorov-Arnold Networks in Surrogate-assisted Evolutionary Algorithms](https://arxiv.org/abs/2405.16494) | [code](https://github.com/hhyqhh/KAN-EA)｜ ![Github stars](https://img.shields.io/github/stars/hhyqhh/KAN-EA.svg)
- [A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting](https://arxiv.org/abs/2406.02486) ｜ [code](https://github.com/remigenet/TKAT) ｜ ![Github stars](https://img.shields.io/github/stars/remigenet/tkat.svg)
- [fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions](https://arxiv.org/abs/2406.07456) | [code](https://github.com/alirezaafzalaghaei/fKAN) | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/fKAN.svg)
- [BSRBF-KAN: A combination of B-splines and Radial Basic Functions in Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.11173) | [code](https://github.com/hoangthangta/BSRBF_KAN) | ![Github stars](https://img.shields.io/github/stars/hoangthangta/BSRBF_KAN.svg)
- [GraphKAN: Enhancing Feature Extraction with Graph Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.13597) | [code](https://github.com/Ryanfzhang/GraphKan) | ![Github stars](https://img.shields.io/github/stars/Ryanfzhang/GraphKan.svg)
- [rKAN: Rational Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.14495) | [code](https://github.com/alirezaafzalaghaei/rKAN) | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/rKAN.svg)
- [SigKAN: Signature-Weighted Kolmogorov-Arnold Networks for Time Series](https://arxiv.org/abs/2406.17890) ｜ [code](https://github.com/remigenet/SigKAN) ｜ ![Github stars](https://img.shields.io/github/stars/remigenet/sigkan.svg)

### Theorem

- 1957-[The original Kolmogorov Arnold paper](https://cs.uwaterloo.ca/~y328yu/classics/Kolmogorov57.pdf)
- 2009-[On a constructive proof of Kolmogorov’s superposition theorem](https://ins.uni-bonn.de/media/public/publication-media/remonkoe.pdf?pk=82)
- 2021-[The Kolmogorov-Arnold representation theorem revisited](https://arxiv.org/abs/2007.15884)
- 2021-[The Kolmogorov Superposition Theorem can Break the Curse of Dimension When Approximating High Dimensional Functions](https://arxiv.org/pdf/2112.09963)

## Library

- [pykan](https://github.com/KindXiaoming/pykan) : Offical implementation for Kolmogorov Arnold Networks ｜ ![Github stars](https://img.shields.io/github/stars/KindXiaoming/pykan.svg)
- [efficient-kan](https://github.com/Blealtan/efficient-kan) : An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network (KAN). ｜ ![Github stars](https://img.shields.io/github/stars/Blealtan/efficient-kan.svg)
- [FastKAN](https://github.com/ZiyaoLi/fast-kan) : Very Fast Calculation of Kolmogorov-Arnold Networks (KAN)  ｜ ![Github stars](https://img.shields.io/github/stars/ZiyaoLi/fast-kan.svg)
- [FasterKAN](https://github.com/AthanasiosDelis/faster-kan) : FasterKAN = FastKAN + RSWAF bases functions and benchmarking with other KANs. Fastest KAN variation as of 5/13/2024, 2 times slower than MLP in backward speed.  ｜ ![Github stars](https://img.shields.io/github/stars/AthanasiosDelis/faster-kan.svg)
- [TorchKAN](https://github.com/1ssb/torchkan) : Simplified KAN Model Using Legendre approximations and Monomial basis functions for Image Classification for MNIST. Achieves 99.5% on MNIST using Conv+LegendreKAN.   ｜ ![Github stars](https://img.shields.io/github/stars/1ssb/torchkan.svg)
- [FourierKAN](https://github.com/GistNoesis/FourierKAN/) : Pytorch Layer for FourierKAN. It is a layer intended to be a substitution for Linear + non-linear activation |  ![Github stars](https://img.shields.io/github/stars/GistNoesis/FourierKAN.svg)
- [Vision-KAN](https://github.com/chenziwenhaoshuai/Vision-KAN) : PyTorch Implementation of Vision Transformers with KAN layers, built on top ViT. 95% accuracy on CIFAR100 (top-5), 80% on ImageNet1000 (training in progress) | ![Github stars](https://img.shields.io/github/stars/chenziwenhaoshuai/Vision-KAN.svg)
- [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) : Kolmogorov-Arnold Networks (KAN) using Chebyshev polynomials instead of B-splines. ｜ ![Github stars](https://img.shields.io/github/stars/SynodicMonth/ChebyKAN.svg)
- [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks) : Implementation of Graph Neural Network version of Kolmogorov Arnold Networks (GraphKAN) ｜ ![Github stars](https://img.shields.io/github/stars/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks.svg)
- [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) : Kolmogorov–Arnold Networks with modified activation (using fully connected network to represent the activation) ｜ ![Github stars](https://img.shields.io/github/stars/Zhangyanbo/FCN-KAN.svg)
- [X-KANeRF](https://github.com/lif314/X-KANeRF) : KAN based NeRF with various basis functions like B-Splines, Fourier, Radial Basis Functions, Polynomials, etc ｜ ![Github stars](https://img.shields.io/github/stars/lif314/X-KANeRF.svg)
- [Large Kolmogorov-Arnold Networks](https://github.com/Indoxer/LKAN) : Variations of Kolmogorov-Arnold Networks (including CUDA-supported KAN convolutions) ｜ ![Github stars](https://img.shields.io/github/stars/Indoxer/LKAN.svg)
- [xKAN](https://github.com/mlsquare/xKAN) : Kolmogorov-Arnold Networks with various basis functions like B-Splines, Fourier, Chebyshev, Wavelets etc ｜ ![Github stars](https://img.shields.io/github/stars/mlsquare/xKAN.svg)
- [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN) : Kolmogorov-Arnold Networks (KAN) using Jacobi polynomials instead of B-splines. ｜ ![Github stars](https://img.shields.io/github/stars/SpaceLearner/JacobiKAN.svg)
- [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks) : Implementation of Graph Neural Network version of Kolmogorov Arnold Networks (GraphKAN) ｜ ![Github stars](https://img.shields.io/github/stars/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks.svg)
- [OrthogPolyKAN](https://github.com/Boris-73-TA/OrthogPolyKANs) : Kolmogorov-Arnold Networks (KAN) using orthogonal polynomials instead of B-splines. ｜ ![Github stars](https://img.shields.io/github/stars/Boris-73-TA/OrthogPolyKANs.svg)
- [kansformers](https://github.com/akaashdash/kansformers) : Kansformers: Transformers using KANs | ![Github stars](https://img.shields.io/github/stars/akaashdash/kansformers.svg)
- [Deep-KAN](https://github.com/Sid2690/Deep-KAN): Better implementation of Kolmogorov Arnold Network  | ![Github stars](https://img.shields.io/github/stars/Sid2690/Deep-KAN.svg)
- [RBF-KAN](https://github.com/Sid2690/RBF-KAN): RBF-KAN is a PyTorch module that implements a Radial Basis Function Kolmogorov-Arnold Network  | ![Github stars](https://img.shields.io/github/stars/Sid2690/RBF-KAN.svg)
- [KolmogorovArnold.jl](https://github.com/vpuri3/KolmogorovArnold.jl) : Very fast Julia implementation of KANs with RBF and RSWAF basis. Extra speedup is gained by writing custom gradients to share work between forward and backward pass. ｜ ![Github stars](https://img.shields.io/github/stars/vpuri3/KolmogorovArnold.jl)
- [Wav-KAN](https://github.com/zavareh1/Wav-KAN): Wav-KAN: Wavelet Kolmogorov-Arnold Networks  | ![Github stars](https://img.shields.io/github/stars/zavareh1/Wav-KAN)
- [KANX](https://github.com/stergiosba/kanx) : Fast Implementation (Approximation) of Kolmogorov-Arnold Network in JAX  | ![Github stars](https://img.shields.io/github/stars/stergiosba/kanx.svg)
- [jaxKAN](https://github.com/srigas/jaxKAN) : Adaptation of the original KAN (with full regularization) in JAX + Flax | ![Github stars](https://img.shields.io/github/stars/srigas/jaxKAN.svg)
- [efficient-kan-jax](https://github.com/dorjeduck/efficient-kan-jax) : JAX port of efficient-kan | ![Github stars](https://img.shields.io/github/stars/dorjeduck/efficient-kan-jax.svg)
- [cuda-Wavelet-KAN](https://github.com/Da1sypetals/cuda-Wavelet-KAN) : CUDA implementation of Wavelet KAN.  | ![Github stars](https://img.shields.io/github/stars/Da1sypetals/cuda-Wavelet-KAN.svg)
- [FlashKAN](https://github.com/dinesh110598/FlashKAN/): Grid size-independent computation of Kolmogorov Arnold networks | ![Github stars](https://img.shields.io/github/stars/dinesh110598/FlashKAN.svg)
- [BSRBF_KAN](https://github.com/hoangthangta/BSRBF_KAN/): Combine B-Spline (BS) and Radial Basic Function (RBF) in Kolmogorov-Arnold Networks (KANs) | ![Github stars](https://img.shields.io/github/stars/hoangthangta/BSRBF_KAN.svg)
- [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN/): Kolmogorov-Arnold Networks (KAN) using Taylor series instead of Fourier | ![Github stars](https://img.shields.io/github/stars/Muyuzhierchengse/TaylorKAN.svg)
- [fKAN](https://github.com/alirezaafzalaghaei/fKAN/): fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/fKAN.svg)
- [Initial Investigation of Kolmogorov-Arnold Networks (KANs) as Feature Extractors for IMU Based Human Activity Recognition](https://arxiv.org/abs/2406.11914)
- [rKAN](https://github.com/alirezaafzalaghaei/rKAN/): rKAN: Rational Kolmogorov-Arnold Networks | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/rKAN.svg)
- [TKAN](https://github.com/remigenet/TKAN): Temporal Kolmogorov-Arnold Networks Tensorflow 2.x layer implementations | ![Github stars](https://img.shields.io/github/stars/remigenet/tkan.svg)
- [TKAT](https://github.com/remigenet/TKAT): Temporal Kolmogorov-Arnold Transformer Tensorflow 2.x model implementation | ![Github stars](https://img.shields.io/github/stars/remigenet/tkat.svg)
- [SigKAN](https://github.com/remigenet/SigKAN): Path Signature-Weighted Kolmogorov-Arnold Networks tensorflow 2.x layer implementations, based on iisignature | ![Github stars](https://img.shields.io/github/stars/remigenet/sigkan.svg)
- [xmm-prototype](https://github.com/Da1sypetals/XMM-prototype): (_💕 Needing contribution if you master optimizing CUDA code!❤️❤️❤️_) A tool for customizing your own KAN that runs on CUDA with minimal lines of code.
 
### ConvKANs

- [Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs) : This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to the Convolutional Layers, changing the classic linear transformation of the convolution to non linear activations in each pixel. ｜ ![Github stars](https://img.shields.io/github/stars/AntonioTepsich/Convolutional-KANs.svg)
- [TorchConv KAN](https://github.com/IvanDrokin/torch-conv-kan) : A Convolutional Kolmogorov-Arnold Networks Collection ｜ ![Github stars](https://img.shields.io/github/stars/IvanDrokin/torch-conv-kan.svg)
- [Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan) : This repository implements Convolutional Kolmogorov-Arnold Layers with various basis functions. The repository includes implementations of 1D, 2D, and 3D convolutions with different kernels, ResNet-like, Unet-like, and DenseNet-like models, training code based on accelerate/PyTorch, and scripts for experiments with CIFAR-10/100, Tiny ImageNet and ImageNet1k. Pretrained weights on ImageNet1k are also available ｜ ![Github stars](https://img.shields.io/github/stars/IvanDrokin/torch-conv-kan.svg)
- [convkan](https://github.com/StarostinV/convkan) : Implementation of convolutional layer version of KAN (drop-in replacement of Conv2d) ｜ ![Github stars](https://img.shields.io/github/stars/StarostinV/convkan.svg)
- [KA-Conv](https://github.com/XiangboGaoBarry/KA-Conv) : Kolmogorov-Arnold Convolutional Networks with Various Basis Functions (Optimization for Efficiency and GPU memory usage) | ![Github stars](https://img.shields.io/github/stars/XiangboGaoBarry/KA-Conv.svg)
 - [KAN-Conv2D](https://github.com/omarrayyann/KAN-Conv2D) : Drop-in Convolutional KAN built on multiple implementations ([Original pykan](https://github.com/KindXiaoming/pykan) / [efficient-kan](https://github.com/Blealtan/efficient-kan) / [FastKAN](https://github.com/ZiyaoLi/fast-kan)) to support the original paper hyperparameters. | ![Github stars](https://img.shields.io/github/stars/omarrayyann/KAN-Conv2D.svg)
 - [CNN-KAN](https://github.com/jakariaemon/CNN-KAN) : A modified CNN architecture using Kolmogorov-Arnold Networks | ![Github stars](https://img.shields.io/github/stars/jakariaemon/CNN-KAN.svg)
 - [ConvKAN3D](https://github.com/FirasBDarwish/ConvKAN3D) : 3D Convolutional Layer built on top of the efficient-kan implementation (importable Python package from PyPi), drop-in replacement of Conv3d.

### Benchmark

- [KAN-benchmarking](https://github.com/Jerry-Master/KAN-benchmarking) : Benchmark for efficiency in memory and time of different KAN implementations. | ![Github stars](https://img.shields.io/github/stars/Jerry-Master/KAN-benchmarking.svg)
- [seydi1370/Basis_Functions](https://github.com/seydi1370/Basis_Functions) : This packaege investigates the performance of 18 different polynomial basis functions, grouped into several categories based on their mathematical properties and areas of application. The study evaluates the effectiveness of these polynomial-based KANs on the MNIST dataset for handwritten digit classification. | ![Github stars](https://img.shields.io/github/stars/seydi1370/Basis_Functions.svg)


### Non-Python

- [KolmogorovArnold.jl](https://github.com/vpuri3/KolmogorovArnold.jl) : Very fast Julia implementation of KANs with RBF and RSWAF basis. Extra speedup is gained by writing custom gradients to share work between forward and backward pass. ｜ ![Github stars](https://img.shields.io/github/stars/vpuri3/KolmogorovArnold.jl)
- [kan-polar](https://github.com/mpoluektov/kan-polar) : Kolmogorov-Arnold Networks in MATLAB ｜ ![Github stars](https://img.shields.io/github/stars/mpoluektov/kan-polar.svg)
- [kamo](https://github.com/dorjeduck/kamo) : Kolmogorov-Arnold Networks in Mojo ｜ ![Github stars](https://img.shields.io/github/stars/dorjeduck/kamo.svg)
- [Building a Kolmogorov-Arnold Neural Network in C](https://rabmcmenemy.medium.com/building-a-kolmogorov-arnold-neural-network-in-c-fac89f2b2330)




### Alternative

- [high-order-layers-torch](https://github.com/jloveric/high-order-layers-torch) : High order piecewise polynomial neural networks using Chebyshev polynomials at Gauss Lobatto nodes (lagrange polynomials). Includes convolutional layers as well HP refinement for non convolutional layers, linear initialization and various applications in the linked repos with varrying levels of success. Euler equations of fluid dynamics, nlp, implicit representation and more | ![Github stars](https://img.shields.io/github/stars/jloveric/high-order-layers-torch.svg)

## Project

- [KAN-GPT](https://github.com/AdityaNG/kan-gpt) : The PyTorch implementation of Generative Pre-trained Transformers (GPTs) using Kolmogorov-Arnold Networks (KANs) for language modeling ｜ ![Github stars](https://img.shields.io/github/stars/AdityaNG/kan-gpt.svg)
- [KAN-GPT-2](https://github.com/CG80499/KAN-GPT-2) : Training small GPT-2 style models using Kolmogorov-Arnold networks.(despite the KAN model having 25% fewer parameters!). ｜ ![Github stars](https://img.shields.io/github/stars/CG80499/KAN-GPT-2.svg)
- [KANeRF](https://github.com/Tavish9/KANeRF) : Kolmogorov-Arnold Network (KAN) based NeRF ｜ ![Github stars](https://img.shields.io/github/stars/Tavish9/KANeRF.svg)
- [Vision-KAN](https://github.com/chenziwenhaoshuai/Vision-KAN) : KAN for Vision Transformer ｜ ![Github stars](https://img.shields.io/github/stars/chenziwenhaoshuai/Vision-KAN.svg)
- [Simple-KAN-4-Time-Series](https://github.com/MSD-IRIMAS/Simple-KAN-4-Time-Series) : A simple feature-based time series classifier using Kolmogorov–Arnold Networks ｜ ![Github stars](https://img.shields.io/github/stars/MSD-IRIMAS/Simple-KAN-4-Time-Series.svg)
- [KANU_Net](https://github.com/JaouadT/KANU_Net) : U-Net architecture with Kolmogorov-Arnold Convolutions (KA convolutions)  ｜ ![Github stars](https://img.shields.io/github/stars/JaouadT/KANU_Net.svg)
- [kanrl](https://github.com/riiswa/kanrl) : Kolmogorov-Arnold Network for Reinforcement Leaning, initial experiments ｜ ![Github stars](https://img.shields.io/github/stars/riiswa/kanrl.svg)
- [kan-diffusion](https://github.com/kabachuha/kan-diffusion) : Applying KANs to Denoising Diffusion Models with two-layer KAN able to restore images almost as good as 4-layer MLP (and 30% less parameters). ｜ ![Github stars](https://img.shields.io/github/stars/kabachuha/kan-diffusion.svg)
- [KAN4Rec](https://github.com/TianyuanYang/KAN4Rec) : Implementation of Kolmogorov-Arnold Network (KAN) for Recommendations ｜ ![Github stars](https://img.shields.io/github/stars/TianyuanYang/KAN4Rec.svg)
- [CF-KAN](https://github.com/jindeok/CF-KAN) : Kolmogorov-Arnold Network (KAN) implementation for collaborative filtering (CF) | ![Github stars](https://img.shields.io/github/stars/jindeok/CF-KAN.svg)
- [X-KANeRF](https://github.com/lif314/X-KANeRF) : X-KANeRF: KAN-based NeRF with Various Basis Functions to explain the the NeRF formula ｜ ![Github stars](https://img.shields.io/github/stars/lif314/X-KANeRF.svg)
- [KAN4Graph](https://github.com/yueliu1999/KAN4Graph) : Implementation of Kolmogorov-Arnold Network (KAN) for Graph Neural Networks (GNNs) and Tasks on Graphs ｜ ![Github stars](https://img.shields.io/github/stars/yueliu1999/KAN4Graph.svg)
- [ImplicitKAN](https://github.com/belkakari/implicit-kan) : Kolmogorov-Arnold Network (KAN) as an implicit function for images and other modalities ｜ ![Github stars](https://img.shields.io/github/stars/belkakari/implicit-kan.svg)
- [ThangKAN](https://github.com/hoangthangta/ThangKAN) : Kolmogorov-Arnold Network (KAN) for text classification over GLUE tasks ｜ ![Github stars](https://img.shields.io/github/stars/hoangthangta/ThangKAN.svg)
- [JianpanHuang/KAN](https://github.com/JianpanHuang/KAN) : This repository contains a demo of regression task (curve fitting) using an efficient Kolmogorov-Arnold Network. ｜ ![Github stars](https://img.shields.io/github/stars/JianpanHuang/KAN.svg)
- [Fraud Detection in Supply Chains Using Kolmogorov Arnold Networks](https://github.com/ChrisD-7/Fraud-Detection-in-Supply-Chains-with-Kolmogorov-Arnold-Networks/) ｜ ![Github stars](https://img.shields.io/github/stars/ChrisD-7/Fraud-Detection-in-Supply-Chains-with-Kolmogorov-Arnold-Networks.svg)
- [CL-KAN-ViT](https://github.com/saeedahmadicp/KAN-CL-ViT) :  Kolmogorov-Arnold Network (KAN) based vision transformer for class-based continual learning to mitigate catastrophic forgetting | ![Github stars](https://img.shields.io/github/stars/saeedahmadicp/KAN-CL-ViT.svg)
- [KAN-Autoencoder](https://github.com/SekiroRong/KAN-AutoEncoder) : KAE KAN-based AutoEncoder (AE, VAE, VQ-VAE, RVQ, etc.) | ![Github stars](https://img.shields.io/github/stars/SekiroRong/KAN-AutoEncoder.svg)



## Discussion

- [KAN Hacker news discussion](https://news.ycombinator.com/item?id=40219205)
- [**Can Kolmogorov–Arnold Networks (KAN) beat MLPs?**](https://pub.towardsai.net/can-kolmogorov-arnold-networks-kan-beat-mlps-060fc34da9ce)
- [Twitter thinks they killed MLPs. But what are Kolmogorov-Arnold Networks?](https://medium.com/@mikeyoung_97230/twitter-thinks-they-killed-mlps-but-what-are-kolmogorov-arnold-networks-b1ec6131891e)
- [[D] Kolmogorov-Arnold Network is just an MLP](https://www.reddit.com/r/MachineLearning/comments/1clcu5i/d_kolmogorovarnold_network_is_just_an_mlp/)
- [KAN: Kolmogorov–Arnold Networks: A review](https://vikasdhiman.info/reviews/KAN_a_review.pdf) : This review raises 4 major criticisms of the paper [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756). "MLPs have learnable activation functions as well", "The content of the paper does not justify the name, Kolmogorov-Arnold networks (KANs)", "KANs are MLPs with spline-basis as the activation function" and "KANs do not beat the curse of dimensionality" unlike claimed.

## Tutorial

- [**KAN Author's twitter introduction**](https://twitter.com/ZimingLiu11/status/1785483967719981538)
- [pg2455/KAN-Tutorial](https://github.com/pg2455/KAN-Tutorial)  ｜ ![Github stars](https://img.shields.io/github/stars/pg2455/KAN-Tutorial.svg)
- [A Simplified Explanation Of The New Kolmogorov-Arnold Network (KAN) from MIT](https://medium.com/@isaakmwangi2018/a-simplified-explanation-of-the-new-kolmogorov-arnold-network-kan-from-mit-cbb59793a040)
- [The Math Behind KAN — Kolmogorov-Arnold Networks](https://towardsdatascience.com/the-math-behind-kan-kolmogorov-arnold-networks-7c12a164ba95)
- [A from-scratch implementation of Kolmogorov-Arnold Networks (KAN)…and MLP](https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/) | [GitHub Code](https://github.com/lollodealma/ml_without_tears/tree/master/kan_mlp_from_scratch)
- [team-daniel/KAN](https://github.com/team-daniel/KAN) : Implementation on how to use Kolmogorov-Arnold Networks (KANs) for classification and regression tasks.｜ ![Github stars](https://img.shields.io/github/stars/team-daniel/KAN.svg)
- [vincenzodentamaro/keras-FastKAN](https://github.com/vincenzodentamaro/keras-FastKAN) : Tensorflow Keras implementation of FastKAN Kolmogorov Arnold Network｜ ![Github stars](https://img.shields.io/github/stars/vincenzodentamaro/keras-FastKAN.svg)
- [Official Tutorial Notebooks](https://github.com/KindXiaoming/pykan/tree/master/tutorials)
- [imodelsX examples with KAN](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/kan.ipynb) : Scikit-learn wrapper for tabular data for KAN (Kolmogorov Arnold Network)
- [What is the new Neural Network Architecture?(KAN) Kolmogorov-Arnold Networks Explained](https://medium.com/@zahmed333/what-is-the-new-neural-network-architecture-kan-kolmogorov-arnold-networks-explained-d2787b013ade)
- [KAN: Kolmogorov–Arnold Networks — A Short Summary](https://kargarisaac.medium.com/kan-kolmogorov-arnold-networks-a-short-summary-a1aef1336990)
- [What is the significance of the Kolmogorov axioms for Mathematical Probability?](https://www.cantorsparadise.com/what-is-the-significance-of-the-kolmogorov-axioms-for-mathematical-probability-ba4eb5551e7e)
- [Andrey Kolmogorov — one of the greatest mathematicians of the XXth century](https://valeman.medium.com/andrey-kolmogorov-one-of-the-greatest-mathematicians-of-the-xxst-century-4167ad02d10)
- [Unpacking Kolmogorov-Arnold Networks](https://pub.towardsai.net/unpacking-kolmogorov-arnold-networks-84ff98463370) : Edge-Based Activation: Exploring the Mathematical Foundations and Practical Implications of KANs
- [Why is the (KAN) Kolmogorov-Arnold Networks so promising](https://engyasin.github.io/posts/why-the-new-kolmogorov-arnold-networks-so-promising/)
- [Demystifying Kolmogorov-Arnold Networks: A Beginner-Friendly Guide with Code](https://daniel-bethell.co.uk/posts/kan/)
- [KANvas](https://kanvas.deepverse.tech/#/kan) : Provide quick & intuitive interaction for people to try KAN
- [KAN-Tutorial](https://github.com/pg2455/KAN-Tutorial/): Understanding Kolmogorov-Arnold Networks: A Tutorial Series on KAN using Toy Examples

### YouTube


- [**KAN: Kolmogorov-Arnold Networks | Ziming Liu(KAN Author)**](https://www.youtube.com/watch?v=AUDHb-tnlB0&ab_channel=ValenceLabs)
- [**Deep Dive on Kolmogorov–Arnold Neural Networks | Ziming Liu(KAN Author)**](https://youtu.be/95pVYknTAv0?si=BRc5P5FRCw1Y4Q1i&t=1)
- [Why the world NEEDS Kolmogorov Arnold Networks](https://www.youtube.com/watch?v=vzUkThsQa9E&ab_channel=ThatMathThing)
- [Kolmogorov-Arnold Networks: MLP vs KAN, Math, B-Splines, Universal Approximation Theorem](https://www.youtube.com/watch?v=-PFIkkwWdnM&t=1515s&ab_channel=UmarJamil)
- [Didn't Graduate Guide to: Kolmogorov-Arnold Networks](https://www.youtube.com/watch?app=desktop&v=3XAW0kqbH2Q&feature=youtu.be&ab_channel=DeepFriedPancake)
- [超越谷歌DeepMind的最新大作：KAN全网最详细解读！](https://www.youtube.com/watch?v=OEvJE-O1R2k)
- [Kolmogorov Arnold Networks (KAN) Paper Explained - An exciting new paradigm for Deep Learning?](https://www.youtube.com/watch?v=7zpz_AlFW2w&ab_channel=NeuralBreakdownwithAVB)
- [KAN: Kolmogorov-Arnold Networks Explained](https://www.youtube.com/watch?v=CkCijaXqAOM)
- [Kolmogorov-Arnold Networks (KANs) and Lennard Jones](https://www.youtube.com/watch?v=_0q7scVScBI&ab_channel=JohnKitchin)
- [Simply explained! KAN: Kolmogorov–Arnold Networks is interpretable! Mathematics and Physics](https://www.youtube.com/watch?v=q8qFYMycNKE)
- [用KAN拟合环境光渲染的查找表](https://www.youtube.com/watch?v=xZ2TyGAYefQ&ab_channel=MinminGong) | [code](https://github.com/gongminmin/KlayGE/tree/develop/KlayGE/Samples/src/EnvLighting)

## Contributing

We welcome your contributions! Please follow these steps to contribute:

1. Fork the repo.
2. Create a new branch (e.g., `feature/new-kan-resource`).
3. Commit your changes to the new branch.
4. Create a Pull Request, and provide a brief description of the changes/additions.

Please make sure that the resources you add are relevant to the field of Kolmogorov-Arnold Network. Before contributing, take a look at the existing resources to avoid duplicates.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mintisan/awesome-kan)](https://star-history.com/#mintisan/awesome-kan&Date)
