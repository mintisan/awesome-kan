# Awesome KAN(Kolmogorov-Arnold Network)


![Awesome](https://awesome.re/badge.svg) [![Contributions](https://img.shields.io/github/issues-pr-closed-raw/mintisan/awesome-kan.svg?label=contributions)](https://github.com/mintisan/awesome-kan/pulls) [![Commits](https://img.shields.io/github/last-commit/mintisan/awesome-kan.svg?label=last%20contribution)](https://github.com/gigwegbe/tinyml-papers-and-projects/commits/main) ![GitHub stars](https://img.shields.io/github/stars/mintisan/awesome-kan.svg?style=social)


A curated list of awesome libraries, projects, tutorials, papers, and other resources related to Kolmogorov-Arnold Network (KAN). This repository aims to be a comprehensive and organized collection that will help researchers and developers in the world of KAN!

![image](https://github.com/mintisan/awesome-kan/assets/9136049/5ce213da-5fe5-49bf-a210-01144a70c14e)


## Table of Contents

- [Papers](#papers)
- [Library](#library)
  - [ConvKANs](#convkans)
- [Project](#project)
- [Discussion](#discussion)
- [Tutorial](#tutorial)
  - [YouTube](#youtube) 
- [Contributing](#contributing)


## Papers

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) : Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users. Through two examples in mathematics and physics, KANs are shown to be useful collaborators helping scientists (re)discover mathematical and physical laws. In summary, KANs are promising alternatives for MLPs, opening opportunities for further improving today's deep learning models which rely heavily on MLPs.
- [Chebyshev Polynomial-Based Kolmogorov-Arnold Networks](https://arxiv.org/html/2405.07200v1) : The document discusses a novel neural network architecture called the Chebyshev Kolmogorov-Arnold Network (Chebyshev KAN) for approximating complex nonlinear functions. It combines the theoretical foundations of the Kolmogorov-Arnold Theorem with the powerful approximation capabilities of Chebyshev polynomials. The Chebyshev KAN layer approximates a target multivariate function by representing it as a weighted sum of Chebyshev polynomials, leveraging the Kolmogorov-Arnold Theorem's guarantee of the existence of a superposition of univariate functions to represent any continuous multivariate function. The paper provides a detailed mathematical explanation of the implementation, training, and optimization of the Chebyshev KAN layer, as well as experimental results demonstrating its effectiveness in approximating complex fractal-like functions.
- [TKAN: Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07344) ｜ [code](https://github.com/remigenet/tkan) ｜ ![Github stars](https://img.shields.io/github/stars/remigenet/tkan.svg)

## Library

- [pykan](https://github.com/KindXiaoming/pykan) : Offical implementation for Kolmogorov Arnold Networks ｜ ![Github stars](https://img.shields.io/github/stars/KindXiaoming/pykan.svg)
- [efficient-kan](https://github.com/Blealtan/efficient-kan) : An efficient pure-PyTorch implementation of Kolmogorov-Arnold Network (KAN). ｜ ![Github stars](https://img.shields.io/github/stars/Blealtan/efficient-kan.svg)
- [FastKAN](https://github.com/ZiyaoLi/fast-kan) : Very Fast Calculation of Kolmogorov-Arnold Networks (KAN)  ｜ ![Github stars](https://img.shields.io/github/stars/ZiyaoLi/fast-kan.svg)
- [FasterKAN](https://github.com/AthanasiosDelis/faster-kan) : FasterKAN = FastKAN + RSWAF bases functions and benchmarking with other KANs. Fastest KAN variation as of 5/13/2024, 2 times slower than MLP in backward speed.  ｜ ![Github stars](https://img.shields.io/github/stars/AthanasiosDelis/faster-kan.svg)
- [TorchKAN](https://github.com/1ssb/torchkan) : Simplified KAN Model Using Legendre approximations and Monomial basis functions for Image Classification for MNIST. Achieves 99.5% on MNIST using Conv+LegendreKAN.   ｜ ![Github stars](https://img.shields.io/github/stars/1ssb/torchkan.svg)
- [FourierKAN](https://github.com/GistNoesis/FourierKAN/) : Pytorch Layer for FourierKAN. It is a layer intended to be a substitution for Linear + non-linear activation |  ![Github stars](https://img.shields.io/github/stars/GistNoesis/FourierKAN.svg)
- [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) : Kolmogorov-Arnold Networks (KAN) using Chebyshev polynomials instead of B-splines. ｜ ![Github stars](https://img.shields.io/github/stars/SynodicMonth/ChebyKAN.svg)
- [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks) : Implementation of Graph Neural Network version of Kolmogorov Arnold Networks (GraphKAN) ｜ ![Github stars](https://img.shields.io/github/stars/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks.svg)
- [FCN-KAN](https://github.com/Zhangyanbo/FCN-KAN) : Kolmogorov–Arnold Networks with modified activation (using fully connected network to represent the activation) ｜ ![Github stars](https://img.shields.io/github/stars/Zhangyanbo/FCN-KAN.svg)
- [X-KANeRF](https://github.com/lif314/X-KANeRF) : KAN based NeRF with various basis functions like B-Splines, Fourier, Radial Basis Functions, Polynomials, etc ｜ ![Github stars](https://img.shields.io/github/stars/lif314/X-KANeRF.svg)
- [Large Kolmogorov-Arnold Networks](https://github.com/Indoxer/LKAN) : Variations of Kolmogorov-Arnold Networks (including CUDA-supported KAN convolutions) ｜ ![Github stars](https://img.shields.io/github/stars/Indoxer/LKAN.svg)
- [FastKAN](https://github.com/ZiyaoLi/fast-kan) : Very Fast Calculation of Kolmogorov-Arnold Networks (KAN)  ｜ ![Github stars](https://img.shields.io/github/stars/ZiyaoLi/fast-kan.svg)
- [xKAN](https://github.com/mlsquare/xKAN) : Kolmogorov-Arnold Networks with various basis functions like B-Splines, Fourier, Chebyshev, Wavelets etc ｜ ![Github stars](https://img.shields.io/github/stars/mlsquare/xKAN.svg)
- [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN) : Kolmogorov-Arnold Networks (KAN) using Jacobi polynomials instead of B-splines. ｜ ![Github stars](https://img.shields.io/github/stars/SpaceLearner/JacobiKAN.svg)
- [kan-polar](https://github.com/mpoluektov/kan-polar) : Kolmogorov-Arnold Networks in MATLAB ｜ ![Github stars](https://img.shields.io/github/stars/mpoluektov/kan-polar.svg)
- [GraphKAN](https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks) : Implementation of Graph Neural Network version of Kolmogorov Arnold Networks (GraphKAN) ｜ ![Github stars](https://img.shields.io/github/stars/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks.svg)
- [OrthogPolyKAN](https://github.com/Boris-73-TA/OrthogPolyKANs) : Kolmogorov-Arnold Networks (KAN) using orthogonal polynomials instead of B-splines. ｜ ![Github stars](https://img.shields.io/github/stars/Boris-73-TA/OrthogPolyKANs.svg)
- [kansformers](https://github.com/akaashdash/kansformers) : Kansformers: Transformers using KANs | ![Github stars](https://img.shields.io/github/stars/akaashdash/kansformers.svg)
- [Deep-KAN](https://github.com/Sid2690/Deep-KAN): Better implementation of Kolmogorov Arnold Network 
- [RBF-KAN](https://github.com/Sid2690/RBF-KAN): RBF-KAN is a PyTorch module that implements a Radial Basis Function Kolmogorov-Arnold Network 
- [KolmogorovArnold.jl](https://github.com/vpuri3/KolmogorovArnold.jl) : Very fast Julia implementation of KANs with RBF and RSWAF basis. Extra speedup is gained by writing custom gradients to share work between forward and backward pass. ｜ ![Github stars](https://img.shields.io/github/stars/vpuri3/KolmogorovArnold.jl


### ConvKANs

- [Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs) : This project extends the idea of the innovative architecture of Kolmogorov-Arnold Networks (KAN) to the Convolutional Layers, changing the classic linear transformation of the convolution to non linear activations in each pixel. ｜ ![Github stars](https://img.shields.io/github/stars/AntonioTepsich/Convolutional-KANs.svg)
- [Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan) : This repository implements Convolutional Kolmogorov-Arnold Layers with various basis functions. ｜ ![Github stars](https://img.shields.io/github/stars/IvanDrokin/torch-conv-kan.svg)
- [convkan](https://github.com/StarostinV/convkan) : Implementation of convolutional layer version of KAN (drop-in replacement of Conv2d) ｜ ![Github stars](https://img.shields.io/github/stars/StarostinV/convkan.svg)
- [KA-Conv](https://github.com/XiangboGaoBarry/KA-Conv) : Kolmogorov-Arnold Convolutional Networks with Various Basis Functions (Optimization for Efficiency and GPU memory usage) | ![Github stars](https://img.shields.io/github/stars/XiangboGaoBarry/KA-Conv.svg)


### Alternative

- [high-order-layers-torch](https://github.com/jloveric/high-order-layers-torch) : High order piecewise polynomial neural networks using Chebyshev polynomials at Gauss Lobatto nodes (lagrange polynomials). Includes convolutional layers as well HP refinement for non convolutional layers, linear initialization and various applications in the linked repos with varrying levels of success. Euler equations of fluid dynamics, nlp, implicit representation and more | ![Github stars](https://img.shields.io/github/stars/jloveric/high-order-layers-torch.svg)

## Project

- [KAN-GPT](https://github.com/AdityaNG/kan-gpt) : The PyTorch implementation of Generative Pre-trained Transformers (GPTs) using Kolmogorov-Arnold Networks (KANs) for language modeling ｜ ![Github stars](https://img.shields.io/github/stars/AdityaNG/kan-gpt.svg)
- [KAN-GPT-2](https://github.com/CG80499/KAN-GPT-2) : Training small GPT-2 style models using Kolmogorov-Arnold networks.(despite the KAN model having 25% fewer parameters!). ｜ ![Github stars](https://img.shields.io/github/stars/CG80499/KAN-GPT-2.svg)
- [KANeRF](https://github.com/Tavish9/KANeRF) : Kolmogorov-Arnold Network (KAN) based NeRF ｜ ![Github stars](https://img.shields.io/github/stars/Tavish9/KANeRF.svg)
- [Simple-KAN-4-Time-Series](https://github.com/MSD-IRIMAS/Simple-KAN-4-Time-Series) : A simple feature-based time series classifier using Kolmogorov–Arnold Networks ｜ ![Github stars](https://img.shields.io/github/stars/MSD-IRIMAS/Simple-KAN-4-Time-Series.svg)
- [kanrl](https://github.com/riiswa/kanrl) : Kolmogorov-Arnold Network for Reinforcement Leaning, initial experiments ｜ ![Github stars](https://img.shields.io/github/stars/riiswa/kanrl.svg)
- [kan-diffusion](https://github.com/kabachuha/kan-diffusion) : Applying KANs to Denoising Diffusion Models with two-layer KAN able to restore images almost as good as 4-layer MLP (and 30% less parameters). ｜ ![Github stars](https://img.shields.io/github/stars/kabachuha/kan-diffusion.svg)
- [KAN4Rec](https://github.com/TianyuanYang/KAN4Rec) : Implementation of Kolmogorov-Arnold Network (KAN) for Recommendations ｜ ![Github stars](https://img.shields.io/github/stars/TianyuanYang/KAN4Rec.svg)
- [X-KANeRF](https://github.com/lif314/X-KANeRF) : X-KANeRF: KAN-based NeRF with Various Basis Functions to explain the the NeRF formula ｜ ![Github stars](https://img.shields.io/github/stars/lif314/X-KANeRF.svg)
- [KAN4Graph](https://github.com/yueliu1999/KAN4Graph) : Implementation of Kolmogorov-Arnold Network (KAN) for Graph Neural Networks (GNNs) and Tasks on Graphs ｜ ![Github stars](https://img.shields.io/github/stars/yueliu1999/KAN4Graph.svg)
- [ImplicitKAN](https://github.com/belkakari/implicit-kan) : Kolmogorov-Arnold Network (KAN) as an implicit function for images and other modalities ｜ ![Github stars](https://img.shields.io/github/stars/belkakari/implicit-kan.svg)


## Discussion

- [KAN Hacker news discussion](https://news.ycombinator.com/item?id=40219205)
- [**Can Kolmogorov–Arnold Networks (KAN) beat MLPs?**](https://pub.towardsai.net/can-kolmogorov-arnold-networks-kan-beat-mlps-060fc34da9ce)
- [Twitter thinks they killed MLPs. But what are Kolmogorov-Arnold Networks?](https://medium.com/@mikeyoung_97230/twitter-thinks-they-killed-mlps-but-what-are-kolmogorov-arnold-networks-b1ec6131891e)
- [[D] Kolmogorov-Arnold Network is just an MLP](https://www.reddit.com/r/MachineLearning/comments/1clcu5i/d_kolmogorovarnold_network_is_just_an_mlp/)

## Tutorial

- [**KAN Author's twitter introduction**](https://twitter.com/ZimingLiu11/status/1785483967719981538)
- [A Simplified Explanation Of The New Kolmogorov-Arnold Network (KAN) from MIT](https://medium.com/@isaakmwangi2018/a-simplified-explanation-of-the-new-kolmogorov-arnold-network-kan-from-mit-cbb59793a040)
- [team-daniel/KAN](https://github.com/team-daniel/KAN) : Implementation on how to use Kolmogorov-Arnold Networks (KANs) for classification and regression tasks.｜ ![Github stars](https://img.shields.io/github/stars/team-daniel/KAN.svg)
- [vincenzodentamaro/keras-FastKAN](https://github.com/vincenzodentamaro/keras-FastKAN) : Tensorflow Keras implementation of FastKAN Kolmogorov Arnold Network｜ ![Github stars](https://img.shields.io/github/stars/vincenzodentamaro/keras-FastKAN.svg)
- [Official Tutorial Notebooks](https://github.com/KindXiaoming/pykan/tree/master/tutorials)
- [imodelsX examples with KAN](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/kan.ipynb) : Scikit-learn wrapper for tabular data for KAN (Kolmogorov Arnold Network)
- [What is the new Neural Network Architecture?(KAN) Kolmogorov-Arnold Networks Explained](https://medium.com/@zahmed333/what-is-the-new-neural-network-architecture-kan-kolmogorov-arnold-networks-explained-d2787b013ade)
- [KAN: Kolmogorov–Arnold Networks — A Short Summary](https://kargarisaac.medium.com/kan-kolmogorov-arnold-networks-a-short-summary-a1aef1336990)
- [What is the significance of the Kolmogorov axioms for Mathematical Probability?](https://www.cantorsparadise.com/what-is-the-significance-of-the-kolmogorov-axioms-for-mathematical-probability-ba4eb5551e7e)
- [Andrey Kolmogorov — one of the greatest mathematicians of the XXth century](https://valeman.medium.com/andrey-kolmogorov-one-of-the-greatest-mathematicians-of-the-xxst-century-4167ad02d10)
- [Unpacking Kolmogorov-Arnold Networks](https://pub.towardsai.net/unpacking-kolmogorov-arnold-networks-84ff98463370) : Edge-Based Activation: Exploring the Mathematical Foundations and Practical Implications of KANs

### YouTube


- [**KAN: Kolmogorov-Arnold Networks | Ziming Liu(KAN Author)**](https://www.youtube.com/watch?v=AUDHb-tnlB0&ab_channel=ValenceLabs)
- [Kolmogorov-Arnold Networks: MLP vs KAN, Math, B-Splines, Universal Approximation Theorem](https://www.youtube.com/watch?v=-PFIkkwWdnM&t=1515s&ab_channel=UmarJamil)
- [Didn't Graduate Guide to: Kolmogorov-Arnold Networks](https://www.youtube.com/watch?app=desktop&v=3XAW0kqbH2Q&feature=youtu.be&ab_channel=DeepFriedPancake)
- [超越谷歌DeepMind的最新大作：KAN全网最详细解读！](https://www.youtube.com/watch?v=OEvJE-O1R2k)
- [Kolmogorov Arnold Networks (KAN) Paper Explained - An exciting new paradigm for Deep Learning?](https://www.youtube.com/watch?v=7zpz_AlFW2w&ab_channel=NeuralBreakdownwithAVB)
- [KAN: Kolmogorov-Arnold Networks Explained](https://www.youtube.com/watch?v=CkCijaXqAOM)
- [Kolmogorov-Arnold Networks (KANs) and Lennard Jones](https://www.youtube.com/watch?v=_0q7scVScBI&ab_channel=JohnKitchin)

## Contributing

We welcome your contributions! Please follow these steps to contribute:

1. Fork the repo.
2. Create a new branch (e.g., `feature/new-kan-resource`).
3. Commit your changes to the new branch.
4. Create a Pull Request, and provide a brief description of the changes/additions.

Please make sure that the resources you add are relevant to the field of Kolmogorov-Arnold Network. Before contributing, take a look at the existing resources to avoid duplicates.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
