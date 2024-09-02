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
    - [Library-based](#library-based)
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
- [KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205)
- [KAN or MLP: A Fairer Comparison](https://arxiv.org/abs/2407.16674) : Under the same number of parameters or FLOPs, we find KAN  outperforms MLP only in symbolic formula representing, but remains inferior to MLP on other tasks of machine learning, computer vision, NLP, and audio processing. We also conduct ablation studies on KAN and find that its advantage in symbolic formula representation mainly stems from its B-spline activation function. | [code](https://github.com/yu-rp/KANbeFair) ｜ ![Github stars](https://img.shields.io/github/stars/yu-rp/kanbefair.svg)
- [DropKAN: Regularizing KANs by masking post-activations](https://arxiv.org/abs/2407.13044) : DropKAN (Dropout Kolmogorov-Arnold Networks) is a regularization method that prevents co-adaptation of activation function weights in Kolmogorov-Arnold Networks (KANs). DropKAN operates by randomly masking some of the post-activations within the KANs computation graph, while scaling-up the retained post-activations. We show that this simple procedure that require minimal coding effort has a regularizing effect and consistently lead to better generalization of KANs. | [code](https://github.com/Ghaith81/dropkan) ｜ ![Github stars](https://img.shields.io/github/stars/Ghaith81/dropkan.svg)
- [Rethinking the Function of Neurons in KANs](https://arxiv.org/abs/2407.20667) : The neurons of Kolmogorov-Arnold Networks (KANs) perform a simple summation motivated by the Kolmogorov-Arnold representation theorem, Our findings indicate that substituting the sum with the average function in KAN neurons results in significant performance enhancements compared to traditional KANs. Our study demonstrates that this minor modification contributes to the stability of training by confining the input to the spline within the effective range of the activation function. | [code](https://github.com/Ghaith81/dropkan) ｜ ![Github stars](https://img.shields.io/github/stars/Ghaith81/dropkan.svg)
- [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/html/2405.07200v1)
- [Kolmogorov Arnold Informed neural network: A physics-informed deep learning framework for solving PDEs based on Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.11045) | [code](https://github.com/yizheng-wang/research-on-solving-partial-differential-equations-of-solid-mechanics-based-on-pinn) ｜ ![Github stars](https://img.shields.io/github/stars/yizheng-wang/research-on-solving-partial-differential-equations-of-solid-mechanics-based-on-pinn.svg)
- [Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.13155) | [code](https://github.com/AntonioTepsich/Convolutional-KANs) ｜ ![Github stars](https://img.shields.io/github/stars/AntonioTepsich/Convolutional-KANs.svg)
- [Kolmogorov-Arnold Convolutions: Design Principles and Empirical Studies](https://arxiv.org/abs/2407.01092) | [code](https://github.com/IvanDrokin/torch-conv-kan) ｜ ![Github stars](https://img.shields.io/github/stars/IvanDrokin/torch-conv-kan.svg)
- [Smooth Kolmogorov Arnold networks enabling structural knowledge representation](https://arxiv.org/abs/2405.11318)
- [TKAN: Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07344) ｜ [code](https://github.com/remigenet/tkan) ｜ ![Github stars](https://img.shields.io/github/stars/remigenet/tkan.svg)
- [ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU](https://arxiv.org/abs/2406.02075) ｜ [code](https://github.com/quiqi/relu_kan) ｜ ![Github stars](https://img.shields.io/github/stars/quiqi/relu_kan.svg)
- [U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation](https://arxiv.org/abs/2406.02918)｜ [code](https://github.com/CUHK-AIM-Group/U-KAN) ｜ ![Github stars](https://img.shields.io/github/stars/CUHK-AIM-Group/U-KAN.svg)
- [Kolmogorov-Arnold Networks (KANs) for Time Series Analysis](https://arxiv.org/pdf/2405.08790)
- [Wav-KAN: Wavelet Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.12832)
- [A First Look at Kolmogorov-Arnold Networks in Surrogate-assisted Evolutionary Algorithms](https://arxiv.org/abs/2405.16494) | [code](https://github.com/hhyqhh/KAN-EA)｜ ![Github stars](https://img.shields.io/github/stars/Jinfeng-Xu/FKAN-GCF.svg)
- [FourierKAN-GCF: Fourier Kolmogorov-Arnold Network--An Effective and Efficient Feature Transformation for Graph Collaborative Filtering](https://arxiv.org/abs/2406.01034) ｜ [code](https://github.com/Jinfeng-Xu/FKAN-GCF) ｜ ![Github stars](https://img.shields.io/github/stars/Jinfeng-Xu/FKAN-GCF.svg)
- [A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting](https://arxiv.org/abs/2406.02486) ｜ [code](https://github.com/remigenet/TKAT) ｜ ![Github stars](https://img.shields.io/github/stars/remigenet/tkat.svg)
- [fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions](https://arxiv.org/abs/2406.07456) | [code](https://github.com/alirezaafzalaghaei/fKAN) | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/fKAN.svg)
- [BSRBF-KAN: A combination of B-splines and Radial Basic Functions in Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.11173) | [code](https://github.com/hoangthangta/BSRBF_KAN) | ![Github stars](https://img.shields.io/github/stars/hoangthangta/BSRBF_KAN.svg)
- [GraphKAN: Enhancing Feature Extraction with Graph Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.13597) | [code](https://github.com/Ryanfzhang/GraphKan) | ![Github stars](https://img.shields.io/github/stars/Ryanfzhang/GraphKan.svg)
- [rKAN: Rational Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.14495) | [code](https://github.com/alirezaafzalaghaei/rKAN) | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/rKAN.svg)
- [SigKAN: Signature-Weighted Kolmogorov-Arnold Networks for Time Series](https://arxiv.org/abs/2406.17890) ｜ [code](https://github.com/remigenet/SigKAN) ｜ ![Github stars](https://img.shields.io/github/stars/remigenet/sigkan.svg)
- [Demonstrating the Efficacy of Kolmogorov-Arnold Networks in Vision Tasks](https://arxiv.org/abs/2406.14916) | [code](https://github.com/jmj2316/KAN-in-VIsion) ｜ ![Github stars](https://img.shields.io/github/stars/jmj2316/KAN-in-VIsion.svg)
- [KANQAS: Kolmogorov-Arnold Network for Quantum Architecture Search](https://arxiv.org/abs/2406.17630) | [code](https://github.com/Aqasch/KANQAS_code)
- [DeepOKAN: Deep Operator Network Based on Kolmogorov Arnold Networks for Mechanics Problems](https://arxiv.org/abs/2405.19143)
- [A deep machine learning algorithm for construction of the Kolmogorov–Arnold representation](https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742))
- [Inferring turbulent velocity and temperature fields and their statistics from Lagrangian velocity measurements using physics-informed Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.15727)
- [A Comprehensive Survey on Kolmogorov Arnold Networks (KAN)](https://arxiv.org/abs/2407.11075)
- [Sparks of Quantum Advantage and Rapid Retraining in Machine Learning](https://doi.org/10.48550/arXiv.2407.16020) | [code](https://github.com/wtroy2/Quantum-KAN) | ![Github stars](https://img.shields.io/github/stars/wtroy2/Quantum-KAN.svg)
- [Adaptive Training of Grid-Dependent Physics-Informed Kolmogorov-Arnold Networks](https://doi.org/10.48550/arXiv.2407.17611) | [code](https://github.com/srigas/jaxKAN) | ![Github stars](https://img.shields.io/github/stars/srigas/jaxkan.svg)
- [Gaussian Process Kolmogorov-Arnold Networks](https://arxiv.org/abs/2407.18397)
- [Kolmogorov--Arnold networks in molecular dynamics](https://arxiv.org/abs/2407.17774)
- [Kolmogorov-Arnold Network for Online Reinforcement Learning](https://arxiv.org/abs/2408.04841v1)  [code](https://github.com/victorkich/Kolmogorov-PPO) | ![Github stars](https://img.shields.io/github/stars/victorkich/Kolmogorov-PPO.svg)
- [TC-KANRecon: High-Quality and Accelerated MRI Reconstruction via Adaptive KAN Mechanisms and Intelligent Feature Scaling](https://arxiv.org/abs/2408.05705) [code](https://github.com/lcbkmm/tc-kanrecon) | ![Github stars](https://img.shields.io/github/stars/lcbkmm/tc-kanrecon.svg)
- [Kolmogorov-Arnold Networks for Time Series: Bridging Predictive Power and Interpretability](https://arxiv.org/abs/2406.02496)
- [KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting?](https://arxiv.org/pdf/2408.11306) | [code](https://github.com/2448845600/KAN4TSF) | ![Github stars](https://img.shields.io/github/stars/2448845600/KAN4TSF.svg)
  
### Theorem

- 1957-[On the representation of continuous functions of several variables by superpositions of continuous functions of a smaller number of variables](https://cs.uwaterloo.ca/~y328yu/classics/Kolmogorov57.pdf) : The original Kolmogorov Arnold paper
- 1957-[On functions of three variables](https://cs.uwaterloo.ca/~y328yu/classics/Arnold57.pdf)
- 2009-[On a constructive proof of Kolmogorov’s superposition theorem](https://ins.uni-bonn.de/media/public/publication-media/remonkoe.pdf?pk=82)
- 2021-[The Kolmogorov-Arnold representation theorem revisited](https://arxiv.org/abs/2007.15884)
- 2021-[The Kolmogorov Superposition Theorem can Break the Curse of Dimension When Approximating High Dimensional Functions](https://arxiv.org/pdf/2112.09963)

## Library

- [pykan](https://github.com/KindXiaoming/pykan) : Offical implementation for Kolmogorov Arnold Networks ｜ ![Github stars](https://img.shields.io/github/stars/KindXiaoming/pykan.svg)
- [efficient-kan](https://github.com/Blealtan/efficient-kan) : An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network (KAN). ｜ ![Github stars](https://img.shields.io/github/stars/Blealtan/efficient-kan.svg)
- [FastKAN](https://github.com/ZiyaoLi/fast-kan) : Very Fast Calculation of Kolmogorov-Arnold Networks (KAN)  ｜ ![Github stars](https://img.shields.io/github/stars/ZiyaoLi/fast-kan.svg)
- [FasterKAN](https://github.com/AthanasiosDelis/faster-kan) : FasterKAN = FastKAN + RSWAF bases functions and benchmarking with other KANs. Fastest KAN variation as of 5/13/2024, 2 times slower than MLP in backward speed.  ｜ ![Github stars](https://img.shields.io/github/stars/AthanasiosDelis/faster-kan.svg)
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
- [FlashKAN](https://github.com/dinesh110598/FlashKAN/): Grid size-independent computation of Kolmogorov Arnold networks | ![Github stars](https://img.shields.io/github/stars/dinesh110598/FlashKAN.svg)
- [BSRBF_KAN](https://github.com/hoangthangta/BSRBF_KAN/): Combine B-Spline (BS) and Radial Basic Function (RBF) in Kolmogorov-Arnold Networks (KANs) | ![Github stars](https://img.shields.io/github/stars/hoangthangta/BSRBF_KAN.svg)
- [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN/): Kolmogorov-Arnold Networks (KAN) using Taylor series instead of Fourier | ![Github stars](https://img.shields.io/github/stars/Muyuzhierchengse/TaylorKAN.svg)
- [fKAN](https://github.com/alirezaafzalaghaei/fKAN/): fKAN: Fractional Kolmogorov-Arnold Networks with trainable Jacobi basis functions | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/fKAN.svg)
- [Initial Investigation of Kolmogorov-Arnold Networks (KANs) as Feature Extractors for IMU Based Human Activity Recognition](https://arxiv.org/abs/2406.11914)
- [rKAN](https://github.com/alirezaafzalaghaei/rKAN/): rKAN: Rational Kolmogorov-Arnold Networks | ![Github stars](https://img.shields.io/github/stars/alirezaafzalaghaei/rKAN.svg)
- [TKAN](https://github.com/remigenet/TKAN): Temporal Kolmogorov-Arnold Networks Keras3 layer implementations multibackend (Jax, Tensorflow, Torch) | ![Github stars](https://img.shields.io/github/stars/remigenet/tkan.svg)
- [TKAT](https://github.com/remigenet/TKAT): Temporal Kolmogorov-Arnold Transformer Tensorflow 2.x model implementation | ![Github stars](https://img.shields.io/github/stars/remigenet/tkat.svg)
- [SigKAN](https://github.com/remigenet/SigKAN): Path Signature-Weighted Kolmogorov-Arnold Networks tensorflow 2.x layer implementations, based on iisignature | ![Github stars](https://img.shields.io/github/stars/remigenet/sigkan.svg)
- [KAN-SGAN](https://github.com/hoangthangta/KAN-SGAN/): Semi-supervised learning with Generative Adversarial Networks (GANs) using Kolmogorov-Arnold Network Layers (KANLs) | ![Github stars](https://img.shields.io/github/stars/hoangthangta/KAN-SGAN.svg)
- [FC_KAN](https://github.com/hoangthangta/FC_KAN): Function Combinations in Kolmogorov-Arnold Networks | ![Github stars](https://img.shields.io/github/stars/hoangthangta/FC_KAN.svg)

### Library-based

- [TorchKAN](https://github.com/1ssb/torchkan) : Simplified KAN Model Using Legendre approximations and Monomial basis functions for Image Classification for MNIST. Achieves 99.5% on MNIST using Conv+LegendreKAN.   ｜ ![Github stars](https://img.shields.io/github/stars/1ssb/torchkan.svg)
- [efficient-kan-jax](https://github.com/dorjeduck/efficient-kan-jax) : JAX port of efficient-kan | ![Github stars](https://img.shields.io/github/stars/dorjeduck/efficient-kan-jax.svg)
- [jaxKAN](https://github.com/srigas/jaxKAN) : Adaptation of the original KAN (with full regularization) in JAX + Flax | ![Github stars](https://img.shields.io/github/stars/srigas/jaxKAN.svg)
- [cuda-Wavelet-KAN](https://github.com/Da1sypetals/cuda-Wavelet-KAN) : CUDA implementation of Wavelet KAN.  | ![Github stars](https://img.shields.io/github/stars/Da1sypetals/cuda-Wavelet-KAN.svg)
- [keras_efficient_kan](https://github.com/remigenet/keras_efficient_kan): A full keras implementation of efficient_kan tested with tensorflow, pytorch and jax backend | ![Github stars](https://img.shields.io/github/stars/remigenet/keras_efficient_kan.svg)
- [Quantum KAN](https://github.com/wtroy2/Quantum-KAN): KANs optimizable through quantum annealing | ![Github stars](https://img.shields.io/github/stars/wtroy2/Quantum-KAN.svg)
- [KAN: Kolmogorov–Arnold Networks in MLX for Apple silicon](https://github.com/Goekdeniz-Guelmez/mlx-kan) : KAN (Kolmogorov–Arnold Networks) in the MLX framework| ![Github stars](https://img.shields.io/github/stars/Goekdeniz-Guelmez/mlx-kan.svg)
 
### ConvKANs

- [Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs) : This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to the Convolutional Layers, changing the classic linear transformation of the convolution to non linear activations in each pixel. ｜ ![Github stars](https://img.shields.io/github/stars/AntonioTepsich/Convolutional-KANs.svg)
- [Torch Conv KAN](https://github.com/IvanDrokin/torch-conv-kan) : This repository implements Convolutional Kolmogorov-Arnold Layers with various basis functions. The repository includes implementations of 1D, 2D, and 3D convolutions with different kernels, ResNet-like, Unet-like, and DenseNet-like models, training code based on accelerate/PyTorch, and scripts for experiments with CIFAR-10/100, Tiny ImageNet and ImageNet1k. Pretrained weights on ImageNet1k are also available ｜ ![Github stars](https://img.shields.io/github/stars/IvanDrokin/torch-conv-kan.svg)
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
- [Julia-Wav-KAN](https://github.com/PritRaj1/Julia-Wav-KAN) : A Julia implementation of Wavelet Kolmogorov-Arnold Networks. ｜ ![Github stars](https://img.shields.io/github/stars/PritRaj1/Julia-Wav-KAN.svg)
- [Building a Kolmogorov-Arnold Neural Network in C](https://rabmcmenemy.medium.com/building-a-kolmogorov-arnold-neural-network-in-c-fac89f2b2330)
- [C# and C++ implementations, benchmarks, tutorials](http://openkan.org/)
- [FluxKAN.jl](https://github.com/cometscome/FluxKAN.jl) : An easy to use Flux implementation of the Kolmogorov Arnold Network. This is a Julia version of TorchKAN.



### Alternative

- [high-order-layers-torch](https://github.com/jloveric/high-order-layers-torch) : High order piecewise polynomial neural networks using Chebyshev polynomials at Gauss Lobatto nodes (lagrange polynomials). Includes convolutional layers as well HP refinement for non convolutional layers, linear initialization and various applications in the linked repos with varrying levels of success. Euler equations of fluid dynamics, nlp, implicit representation and more | ![Github stars](https://img.shields.io/github/stars/jloveric/high-order-layers-torch.svg)
- [Training based on Kaczmarz, not Broyden method](http://openkan.org/) : The training process is independent of the basis functions, the provided link shows alternative to Broyden method originally suggested in MIT paper. It outperforms MLP siginificantly, benchmarks provided. 

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
- [OpenKAN](http://openkan.org/)
- [KAN-DQN](https://github.com/andythetechnerd03/KAN-It-Play-Flappy-Bird) : An experiment where KAN replaces MLP in Deep Q-Network to play Flappy Bird as a Reinforcement Learning agent. | ![GitHub Repo stars](https://img.shields.io/github/stars/andythetechnerd03/KAN-It-Play-Flappy-Bird)



## Discussion

- HN-[KAN Hacker news discussion](https://news.ycombinator.com/item?id=40219205)
- HN-[	A new type of neural network is more interpretable](https://news.ycombinator.com/item?id=41162676)
- HN-[Trying Kolmogorov-Arnold Networks in Practice](https://news.ycombinator.com/item?id=40855028)
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
- [KAN-Continual_Learning_tests](https://github.com/MrPio/KAN-Continual_Learning_tests) : Collection of tests performed during the study of the new Kolmogorov-Arnold Neural Networks (KAN) ｜ ![Github stars](https://img.shields.io/github/stars/MrPio/KAN-Continual_Learning_tests.svg)
- [The Annotated Kolmogorov Network (KAN)](https://alexzhang13.github.io/blog/2024/annotated-kan/): An annotated code guide implementation of KAN, like the Annotated Transformer.

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
- [Video with alternative approach to training process](https://www.youtube.com/watch?v=eS_k6L638k0)

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
