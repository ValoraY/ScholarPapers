## 📑 Guoqi Li Papers

论文按年份分组（点击年份或空白区域可展开/折叠该年份的论文）


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2025</summary>

<div class="paper-card">

<h3 class="paper-title">MVA: Linear Attention with High-order Query-Keys Integration and Multi-level Vocabulary Decomposition</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://raw.githubusercontent.com/mlresearch/v267/main/assets/ning25b/ning25b.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Linear attention offers the advantages of linear inference time and fixed memory usage compared to Softmax attention. However, training large-scale language models with linear attention from scratch remains prohibitively expensive and exhibits significant performance gaps compared to Softmax-based models. To address these challenges, we focus on transforming pre-trained Softmax-based language models into linear attention models. We unify mainstream linear attention methods using a high-order QK integration theory and a multi-level vocabulary decomposition. Specifically, the QK integration theory explains the efficacy of combining linear and sparse attention from the perspective of information collection across different frequency bands. The multilevel vocabulary decomposition exponentially expands memory capacity by recursively exploiting compression loss from compressed states. To further improve performance and reduce training costs, we adopt a soft integration strategy with attention scores, effectively combining a sliding window mechanism. With less than 100M tokens, our method fine-tunes models to achieve linear complexity while retaining 99% of their original performance. Compared to state-of-the-art linear attention model and method, our approach improves MMLU scores by 1.2 percentage points with minimal fine-tuning. Furthermore, even without the sliding window mechanism, our method achieves state-of-the-art performance on all test sets with 10B tokens.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SpikingBrain: Spiking Brain-inspired Large Models</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2509.05276" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware. Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Training remains stable for weeks on hundreds of MetaX C550 GPUs, with the 7B model reaching a Model FLOPs Utilization of 23.4 percent. The proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spikingbrain technical report: Spiking brain-inspired large models</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://noticias.ai/wp-content/uploads/2025/09/2509.05276v1.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Mainstream Transformer-based large language models (LLMs) face significant efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly. These constraints limit their ability to process long sequences effectively. In addition, building large models on non-NVIDIA computing platforms poses major challenges in achieving stable and efficient training and deployment. To address these issues, we introduce SpikingBrain, a new family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX 1 GPU cluster and focuses on three core aspects: i) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; ii) Algorithmic Optimizations: an efficient, conversion-based training pipeline compatible with existing LLMs, along with a dedicated spike coding framework; iii) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to the MetaX hardware.Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using exceptionally low data resources (continual pre-training of∼ 150B tokens). Our models also significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B achieves …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Speed always wins: A survey on efficient architectures for large language models</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2508.09834" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Large Language Models (LLMs) have delivered impressive results in language understanding, generation, reasoning, and pushes the ability boundary of multimodal models. Transformer models, as the foundation of modern LLMs, offer a strong baseline with excellent scaling properties. However, the traditional transformer architecture requires substantial computations and poses significant obstacles for large-scale training and practical deployment. In this survey, we offer a systematic examination of innovative LLM architectures that address the inherent limitations of transformers and boost the efficiency. Starting from language modeling, this survey covers the background and technical details of linear and sparse sequence modeling methods, efficient full attention variants, sparse mixture-of-experts, hybrid model architectures incorporating the above techniques, and emerging diffusion LLMs. Additionally, we discuss applications of these techniques to other modalities and consider their wider implications for developing scalable, resource-aware foundation models. By grouping recent studies into the above category, this survey presents a blueprint of modern efficient LLM architectures, and we hope this could help motivate future research toward more efficient, versatile AI systems.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A multisynaptic spiking neuron for simultaneously encoding spatiotemporal dynamics</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.nature.com/articles/s41467-025-62251-6" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are biologically more plausible and computationally more powerful than artificial neural networks due to their intrinsic temporal dynamics. However, vanilla spiking neurons struggle to simultaneously encode spatiotemporal dynamics of inputs. Inspired by biological multisynaptic connections, we propose the Multi-Synaptic Firing (MSF) neuron, where an axon can establish multiple synapses with different thresholds on a postsynaptic neuron. MSF neurons jointly encode spatial intensity via firing rates and temporal dynamics via spike timing, and generalize Leaky Integrate-and-Fire (LIF) and ReLU neurons as special cases. We derive optimal threshold selection and parameter optimization criteria for surrogate gradients, enabling scalable deep MSF-based SNNs without performance degradation. Extensive experiments across various benchmarks show that MSF neurons significantly …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Efficient Identification of Cable Partial Discharge: A Spiking Neural Network Approach</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11149056/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Cable partial discharge (PD) identification is a critical technology for ensuring insulation reliability in power systems. To address the high computational resource demands of traditional deep learning methods in edge device deployment, this article proposes a low-power event-driven spiking neural network (SNN)-based approach for cable PD identification. By reconstructing the temporal feature extraction mechanism of AlexNet and leveraging the sparse computation and biologically inspired event-driven properties of SNNs, the model replaces traditional activation functions with integrate-and-fire (IF) neurons. A three-layer convolutional encoder is designed to extract spatiotemporal dynamic features from phase-resolved partial discharge (PRPD) patterns, combined with differentiable surrogate gradients for end-to-end training. Experiments on a real-world cable PD dataset demonstrate that, compared to traditional …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Scaling Linear Attention with Sparse State Expansion</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2507.16577" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The Transformer architecture, despite its widespread success, struggles with long-context scenarios due to quadratic computation and linear memory growth. While various linear attention variants mitigate these efficiency constraints by compressing context into fixed-size states, they often degrade performance in tasks such as in-context retrieval and reasoning. To address this limitation and achieve more effective context compression, we propose two key innovations. First, we introduce a row-sparse update formulation for linear attention by conceptualizing state updating as information classification. This enables sparse state updates via softmax-based top-$k$ hard classification, thereby extending receptive fields and reducing inter-class interference. Second, we present Sparse State Expansion (SSE) within the sparse framework, which expands the contextual state into multiple partitions, effectively decoupling parameter size from state capacity while maintaining the sparse classification paradigm. Supported by efficient parallelized implementations, our design achieves effective classification and highly discriminative state representations. We extensively validate SSE in both pure linear and hybrid (SSE-H) architectures across language modeling, in-context retrieval, and mathematical reasoning benchmarks. SSE demonstrates strong retrieval performance and scales favorably with state size. Moreover, after reinforcement learning (RL) training, our 2B SSE-H model achieves state-of-the-art mathematical reasoning performance among small reasoning models, scoring 64.5 on AIME24 and 50.2 on AIME25, significantly outperforming similarly sized open-source Transformers. These results highlight SSE as a promising and efficient architecture for long-context modeling.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Mmdend: Dendrite-inspired multi-branch multi-compartment parallel spiking neuron for sequence modeling</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://aclanthology.org/2025.acl-long.1332/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Vanilla spiking neurons are simplified from complex biological neurons with dendrites, soma, and synapses, into single somatic compartments. Due to limitations in performance and training efficiency, vanilla spiking neurons face significant challenges in modeling long sequences. In terms of performance, the oversimplified dynamics of spiking neurons omit long-term temporal dependencies. Additionally, the long-tail membrane potential distribution and binary activation discretization errors further limit their capacity to model long sequences. In terms of efficiency, the serial mechanism of spiking neurons leads to excessively long training times for long sequences. Though parallel spiking neurons are an efficient solution, their number of parameters is often tied to the hidden dimension or sequence length, which makes current parallel neurons unsuitable for large architectures. To address these issues, we propose** MMDEND**: a Multi-Branch Multi-Compartment Parallel Spiking Dendritic Neuron. Its proportion-adjustable multi-branch, multi-compartment structure enables long-term temporal dependencies. Additionally, we introduce a Scaling-Shifting Integer Firing (SSF) mechanism that fits the long-tail membrane potential distribution, retains efficiency, and mitigates discretization errors. Compared with parallel neurons, MMDEND achieves better long-sequence modeling capability with fewer parameters and lower energy consumption. Visualization also confirms that the SSF mechanism effectively fits long-tail distributions.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Enabling scale and rotation invariance in convolutional neural networks with retina like transformation</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608025002746" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Traditional convolutional neural networks (CNNs) struggle with scale and rotation transformations, resulting in reduced performance on transformed images. Previous research focused on designing specific CNN modules to extract transformation-invariant features. However, these methods lack versatility and are not adaptable to a wide range of scenarios. Drawing inspiration from human visual invariance, we propose a novel brain-inspired approach to tackle the invariance problem in CNNs. If we consider a CNN as the visual cortex, we have the potential to design an “eye” that exhibits transformation invariance, allowing CNNs to perceive the world consistently. Therefore, we propose a retina module and then integrate it into CNNs to create transformation-invariant CNNs (TICNN), achieving scale and rotation invariance. The retina module comprises a retina-like transformation and a transformation-aware neural …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Knowledge distillation for spiking neural networks: aligning features and saliency</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://iopscience.iop.org/article/10.1088/2634-4386/ade821/meta" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) are renowned for their energy efficiency and bio-fidelity, but their widespread adoption is hindered by challenges in training, primarily due to the non-differentiability of spiking activations and limited representational capacity. Existing approaches, such as ANN-to-SNN conversion and surrogate gradient learning, either suffer from prolonged simulation times or suboptimal performance. To address these challenges, we provide a novel perspective that frames knowledge distillation as a hybrid training strategy, effectively combining knowledge transfer from pretrained models with spike-based gradient learning. This approach leverages the complementary benefits of both paradigms, enabling the development of high-performance, low-latency SNNs. Our approach features a lightweight affine projector that facilitates flexible representation alignment across diverse network architectures …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SVL: Spike-based Vision-language Pretraining for Efficient 3D Open-world Understanding</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2505.17674" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) provide an energy-efficient way to extract 3D spatio-temporal features. However, existing SNNs still exhibit a significant performance gap compared to Artificial Neural Networks (ANNs) due to inadequate pre-training strategies. These limitations manifest as restricted generalization ability, task specificity, and a lack of multimodal understanding, particularly in challenging tasks such as multimodal question answering and zero-shot 3D classification. To overcome these challenges, we propose a Spike-based Vision-Language (SVL) pretraining framework that empowers SNNs with open-world 3D understanding while maintaining spike-driven efficiency. SVL introduces two key components: (i) Multi-scale Triple Alignment (MTA) for label-free triplet-based contrastive learning across 3D, image, and text modalities, and (ii) Re-parameterizable Vision-Language Integration (Rep-VLI) to enable lightweight inference without relying on large text encoders. Extensive experiments show that SVL achieves a top-1 accuracy of 85.4% in zero-shot 3D classification, surpassing advanced ANN models, and consistently outperforms prior SNNs on downstream tasks, including 3D classification (+6.1%), DVS action recognition (+2.1%), 3D detection (+1.1%), and 3D segmentation (+2.1%) with remarkable efficiency. Moreover, SVL enables SNNs to perform open-world 3D question answering, sometimes outperforming ANNs. To the best of our knowledge, SVL represents the first scalable, generalizable, and hardware-friendly paradigm for 3D open-world understanding, effectively bridging the gap between SNNs and ANNs in complex open-world understanding tasks. Code is available https://github.com/bollossom/SVL.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and  Complexity</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2505.10352" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) have shown competitive performance to Artificial Neural Networks (ANNs) in various vision tasks, while offering superior energy efficiency. However, existing SNN-based Transformers primarily focus on single-image tasks, emphasizing spatial features while not effectively leveraging SNNs&#x27; efficiency in video-based vision tasks. In this paper, we introduce SpikeVideoFormer, an efficient spike-driven video Transformer, featuring linear temporal complexity . Specifically, we design a spike-driven Hamming attention (SDHA) which provides a theoretically guided adaptation from traditional real-valued attention to spike-driven attention. Building on SDHA, we further analyze various spike-driven space-time attention designs and identify an optimal scheme that delivers appealing performance for video tasks, while maintaining only linear temporal complexity. The generalization ability and efficiency of our model are demonstrated across diverse downstream video tasks, including classification, human pose tracking, and semantic segmentation. Empirical results show our method achieves state-of-the-art (SOTA) performance compared to existing SNN approaches, with over 15\% improvement on the latter two tasks. Additionally, it matches the performance of recent ANN-based methods while offering significant efficiency gains, achieving , and improvements on the three tasks. https://github.com/JimmyZou/SpikeVideoFormer
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Topology optimization of random memristors for input-aware dynamic SNN</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.science.org/doi/abs/10.1126/sciadv.ads5340" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Machine learning has advanced unprecedentedly, exemplified by GPT-4 and SORA. However, they cannot parallel human brains in efficiency and adaptability due to differences in signal representation, optimization, runtime reconfigurability, and hardware architecture. To address these challenges, we introduce pruning optimization for input-aware dynamic memristive spiking neural network (PRIME). PRIME uses spiking neurons to emulate brain’s spiking mechanisms and optimizes the topology of random memristive SNNs inspired by structural plasticity, effectively mitigating memristor programming stochasticity. It also uses the input-aware early-stop policy to reduce latency and leverages memristive in-memory computing to mitigate von Neumann bottleneck. Validated on a 40-nm, 256-K memristor-based macro, PRIME achieves comparable classification accuracy and inception score to software baselines, with …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">BIG-FUSION: Brain-Inspired Global-Local Context Fusion Framework for Multimodal Emotion Recognition in Conversations</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/32149" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Considering the importance of capturing both global conversational topics and local speaker dependencies for multimodal emotion recognition in conversations, current approaches first utilize sequence models like Transformer to extract global context information, then apply Graph Neural Networks to model local speaker dependencies for local context information extraction, coupled with Graph Contrastive Learning (GCL) to enhance node representation learning. However, this sequential design introduces potential biases: the extracted global context information inevitably influences subsequent processing, compromising the independence and diversity of the original local features; current graph augmentation methods in GCL cannot consider both global and local context information in conversations to evaluate the node importance, hindering the learning of key information. Inspired by the human brain excels at handling complex tasks by efficiently integrating local and global information processing mechanisms, we propose an aligned global-local context fusion framework for sequence-based design to address these problems. This design includes a dual-attention Transformer and a dual-evaluation method for graph augmentation in GCL. The dual-attention Transformer combines global attention for overall context extraction with sliding-window attention for local context capture, both enhanced by spiking neuron dynamics. The dual-evaluation method in GCL comprises global importance evaluation to identify nodes crucial for overall conversation context, and local importance evaluation to detect nodes significant for local semantics …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spike2former: Efficient spiking transformer for high-performance image segmentation</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/32126" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) have a low-power advantage but perform poorly in image segmentation tasks. The reason is that directly converting neural networks with complex architectural designs for segmentation tasks into spiking versions leads to performance degradation and non-convergence. To address this challenge, we first identify the modules in the architecture design that lead to the severe reduction in spike firing, make targeted improvements, and propose Spike2Former architecture. Second, we propose normalized integer spiking neurons to solve the training stability problem of SNNs with complex architectures. We set a new state-of-the-art for SNNs in various semantic segmentation datasets, with a significant improvement of+ 12.7% mIoU and 5.0 x efficiency on ADE20K,+ 14.3% mIoU and 5.2 x efficiency on VOC2012, and+ 9.1% mIoU and 6.6 x efficiency on CityScapes.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Efficient 3d recognition with event-driven spike sparse convolution</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/34212" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) provide an energy-efficient way to extract 3D spatio-temporal features. Point clouds are sparse 3D spatial data, which suggests that SNNs should be well-suited for processing them. However, when applying SNNs to point clouds, they often exhibit limited performance and fewer application scenarios. We attribute this to inappropriate preprocessing and feature extraction methods. To address this issue, we first introduce the Spike Voxel Coding (SVC) scheme, which encodes the 3D point clouds into a sparse spike train space, reducing the storage requirements and saving time on point cloud preprocessing. Then, we propose a Spike Sparse Convolution (SSC) model for efficiently extracting 3D sparse point cloud features. Combining SVC and SSC, we design an efficient 3D SNN backbone (E-3DSNN), which is friendly with neuromorphic hardware. For instance, SSC can be implemented on neuromorphic chips with only minor modifications to the addressing function of vanilla spike convolution. Experiments on ModelNet40, KITTI, and Semantic KITTI datasets demonstrate that E-3DSNN achieves state-of-the-art (SOTA) results with remarkable efficiency. Notably, our E-3DSNN (1.87 M) obtained 91.7% top-1 accuracy on ModelNet40, surpassing the current best SNN baselines (14.3 M) by 3.0%. To our best knowledge, it is the first direct training 3D SNN backbone that can simultaneously handle various 3D computer vision tasks (eg, classification, detection, and segmentation) with an event-driven nature.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Human Activity Recognition using RGB-Event based Sensors: A Multi-modal Heat Conduction Model and A Benchmark Dataset</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2504.05830" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Human Activity Recognition (HAR) primarily relied on traditional RGB cameras to achieve high-performance activity recognition. However, the challenging factors in real-world scenarios, such as insufficient lighting and rapid movements, inevitably degrade the performance of RGB cameras. To address these challenges, biologically inspired event cameras offer a promising solution to overcome the limitations of traditional RGB cameras. In this work, we rethink human activity recognition by combining the RGB and event cameras. The first contribution is the proposed large-scale multi-modal RGB-Event human activity recognition benchmark dataset, termed HARDVS 2.0, which bridges the dataset gaps. It contains 300 categories of everyday real-world actions with a total of 107,646 paired videos covering various challenging scenarios. Inspired by the physics-informed heat conduction model, we propose a novel multi-modal heat conduction operation framework for effective activity recognition, termed MMHCO-HAR. More in detail, given the RGB frames and event streams, we first extract the feature embeddings using a stem network. Then, multi-modal Heat Conduction blocks are designed to fuse the dual features, the key module of which is the multi-modal Heat Conduction Operation layer. We integrate RGB and event embeddings through a multi-modal DCT-IDCT layer while adaptively incorporating the thermal conductivity coefficient via FVEs into this module. After that, we propose an adaptive fusion module based on a policy routing strategy for high-performance classification. Comprehensive experiments demonstrate that our method consistently performs well, validating its effectiveness and robustness. The source code and benchmark dataset will be released on https://github.com/Event-AHU/HARDVS/tree/HARDVSv2
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Autoregressive image generation with randomized parallel decoding</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2503.10568" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
We introduce ARPG, a novel visual autoregressive model that enables randomized parallel generation, addressing the inherent limitations of conventional raster-order approaches, which hinder inference efficiency and zero-shot generalization due to their sequential, predefined token generation order. Our key insight is that effective random-order modeling necessitates explicit guidance for determining the position of the next predicted token. To this end, we propose a novel decoupled decoding framework that decouples positional guidance from content representation, encoding them separately as queries and key-value pairs. By directly incorporating this guidance into the causal attention mechanism, our approach enables fully random-order training and generation, eliminating the need for bidirectional attention. Consequently, ARPG readily generalizes to zero-shot inference tasks such as image inpainting, outpainting, and resolution expansion. Furthermore, it supports parallel inference by concurrently processing multiple queries using a shared KV cache. On the ImageNet-1K 256 benchmark, our approach attains an FID of 1.83 with only 32 sampling steps, achieving over a 30 times speedup in inference and a 75 percent reduction in memory consumption compared to representative recent autoregressive models at a similar scale.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Event-based video reconstruction via spatial-temporal heterogeneous spiking neural network</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10925495/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras detect per-pixel brightness changes and output asynchronous event streams with high temporal resolution, high dynamic range, and low latency. However, the unstructured nature of event streams means that humans cannot analyze and interpret them in the same way as natural images. Event-based video reconstruction is a widely used method aimed at reconstructing intuitive videos from event streams. Most reconstruction methods based on traditional artificial neural networks (ANNs) have high energy consumption, which counteracts the low-power advantage of event cameras. Spiking neural networks (SNNs) are a new generation of event-driven neural networks that encode information via discrete spikes, which leads to greater computational efficiency. Previous methods based on SNNs overlooked the asynchronous nature of event streams, leading to reconstructions that suffer from artifacts …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Enhancing Robustness of Spiking Neural Networks Through Retina-Like Coding and Memory-Based Neurons</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5174012" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are emerging as a promising alternative to traditional artificial neural networks (ANNs), offering advantages such as lower power consumption and biological interpretability. Despite recent progress in training SNNs and their performance in computer vision tasks, there remains a question of SNN robustness to corrupted images in real-world scenarios. To address this problem, we introduce CIFAR10-C and IMAGENET-C datasets from the ANN field as benchmarks and further propose novel methods to improve SNN corruption robustness. Specifically, we propose a retina-like coding to simulate dynamic human visual perception, providing a foundation for extracting robust features through varied temporal input. Meanwhile, we introduce a memory-based spiking neuron (MSN) that integrates memory units to learn robust features, along with a parallel version (pMSN) to facilitate parallel computing and achieve superior performance. Experimental results demonstrate that our method improves SNN recognition accuracy and robustness, achieving average accuracies of 87.04\% on the CIFAR10-C dataset and 40.37\% on the IMAGENET-C dataset, surpassing the state-of-the-art SNN method’s 85.95\% and 39.11\%, respectively. These findings highlight the potential of our approach to enhance the robustness of SNNs in real-world scenarios. Our codes will be released.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">AuthSim: Towards Authentic and Effective Safety-critical Scenario Generation for Autonomous Driving Tests</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2502.21100" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Generating adversarial safety-critical scenarios is a pivotal method for testing autonomous driving systems, as it identifies potential weaknesses and enhances system robustness and reliability. However, existing approaches predominantly emphasize unrestricted collision scenarios, prompting non-player character (NPC) vehicles to attack the ego vehicle indiscriminately. These works overlook these scenarios&#x27; authenticity, rationality, and relevance, resulting in numerous extreme, contrived, and largely unrealistic collision events involving aggressive NPC vehicles. To rectify this issue, we propose a three-layer relative safety region model, which partitions the area based on danger levels and increases the likelihood of NPC vehicles entering relative boundary regions. This model directs NPC vehicles to engage in adversarial actions within relatively safe boundary regions, thereby augmenting the scenarios&#x27; authenticity. We introduce AuthSim, a comprehensive platform for generating authentic and effective safety-critical scenarios by integrating the three-layer relative safety region model with reinforcement learning. To our knowledge, this is the first attempt to address the authenticity and effectiveness of autonomous driving system test scenarios comprehensively. Extensive experiments demonstrate that AuthSim outperforms existing methods in generating effective safety-critical scenarios. Notably, AuthSim achieves a 5.25% improvement in average cut-in distance and a 27.12% enhancement in average collision interval time, while maintaining higher efficiency in generating effective safety-critical scenarios compared to existing methods. This underscores its significant advantage in producing authentic scenarios over current methodologies.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Without Paired Labeled Data: End-to-End Self-Supervised Learning for Drone-view Geo-Localization</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2502.11381" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Drone-view Geo-Localization (DVGL) aims to achieve accurate localization of drones by retrieving the most relevant GPS-tagged satellite images. However, most existing methods heavily rely on strictly pre-paired drone-satellite images for supervised learning. When the target region shifts, new paired samples are typically required to adapt to the distribution changes. The high cost of annotation and the limited transferability of these methods significantly hinder the practical deployment of DVGL in open-world scenarios. To address these limitations, we propose a novel end-to-end self-supervised learning method with a shallow backbone network, called the dynamic memory-driven and neighborhood information learning (DMNIL) method. It employs a clustering algorithm to generate pseudo-labels and adopts a dual-path contrastive learning framework to learn discriminative intra-view representations. Furthermore, DMNIL incorporates two core modules, including the dynamic hierarchical memory learning (DHML) module and the information consistency evolution learning (ICEL) module. The DHML module combines short-term and long-term memory to enhance intra-view feature consistency and discriminability. Meanwhile, the ICEL module utilizes a neighborhood-driven dynamic constraint mechanism to systematically capture implicit cross-view semantic correlations, consequently improving cross-view feature alignment. To further stabilize and strengthen the self-supervised training process, a pseudo-label enhancement strategy is introduced to enhance the quality of pseudo supervision. Extensive experiments on three public benchmark datasets demonstrate that the proposed method consistently outperforms existing self-supervised methods and even surpasses several state-of-the-art supervised methods. Our code is available at https://github.com/ISChenawei/DMNIL.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2502.09449" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Temporal processing is fundamental for both biological and artificial intelligence systems, as it enables the comprehension of dynamic environments and facilitates timely responses. Spiking Neural Networks (SNNs) excel in handling such data with high efficiency, owing to their rich neuronal dynamics and sparse activity patterns. Given the recent surge in the development of SNNs, there is an urgent need for a comprehensive evaluation of their temporal processing capabilities. In this paper, we first conduct an in-depth assessment of commonly used neuromorphic benchmarks, revealing critical limitations in their ability to evaluate the temporal processing capabilities of SNNs. To bridge this gap, we further introduce a benchmark suite consisting of three temporal processing tasks characterized by rich temporal dynamics across multiple timescales. Utilizing this benchmark suite, we perform a thorough evaluation of recently introduced SNN approaches to elucidate the current status of SNNs in temporal processing. Our findings indicate significant advancements in recently developed spiking neuron models and neural architectures regarding their temporal processing capabilities, while also highlighting a performance gap in handling long-range dependencies when compared to state-of-the-art non-spiking models. Finally, we discuss the key challenges and outline potential avenues for future research.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">EfficientLLM: Scalable Pruning-Aware Pretraining for Architecture-Agnostic Edge Language Models</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2502.06663" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Modern large language models (LLMs) driven by scaling laws, achieve intelligence emergency in large model sizes. Recently, the increasing concerns about cloud costs, latency, and privacy make it an urgent requirement to develop compact edge language models. Distinguished from direct pretraining that bounded by the scaling law, this work proposes the pruning-aware pretraining, focusing on retaining performance of much larger optimized models. It features following characteristics: 1) Data-scalable: we introduce minimal parameter groups in LLM and continuously optimize structural pruning, extending post-training pruning methods like LLM-Pruner and SparseGPT into the pretraining phase. 2) Architecture-agnostic: the LLM architecture is auto-designed using saliency-driven pruning, which is the first time to exceed SoTA human-designed LLMs in modern pretraining. We reveal that it achieves top-quality edge language models, termed EfficientLLM, by scaling up LLM compression and extending its boundary. EfficientLLM significantly outperforms SoTA baselines with parameters, such as MobileLLM, SmolLM, Qwen2.5-0.5B, OLMo-1B, Llama3.2-1B in common sense benchmarks. As the first attempt, EfficientLLM bridges the performance gap between traditional LLM compression and direct pretraining methods, and we will fully open source at https://github.com/Xingrun-Xing2/EfficientLLM.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Scaling spike-driven transformer with efficient spike firing approximation training</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10848017/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The ambition of brain-inspired Spiking Neural Networks (SNNs) is to become a low-power alternative to traditional Artificial Neural Networks (ANNs). This work addresses two major challenges in realizing this vision: the performance gap between SNNs and ANNs, and the high training costs of SNNs. We identify intrinsic flaws in spiking neurons caused by binary firing mechanisms and propose a Spike Firing Approximation (SFA) method using integer training and spike-driven inference. This optimizes the spike firing pattern of spiking neurons, enhancing efficient training, reducing power consumption, improving performance, enabling easier scaling, and better utilizing neuromorphic chips. We also develop an efficient spike-driven Transformer architecture and a spike-masked autoencoder to prevent performance degradation during SNN scaling. On ImageNet-1k, we achieve state-of-the-art top-1 accuracy of 78.5 …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Temporal-Aware Spiking Transformer Hashing Based on 3D-DWT</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2501.06786" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
With the rapid growth of dynamic vision sensor (DVS) data, constructing a low-energy, efficient data retrieval system has become an urgent task. Hash learning is one of the most important retrieval technologies which can keep the distance between hash codes consistent with the distance between DVS data. As spiking neural networks (SNNs) can encode information through spikes, they demonstrate great potential in promoting energy efficiency. Based on the binary characteristics of SNNs, we first propose a novel supervised hashing method named Spikinghash with a hierarchical lightweight structure. Spiking WaveMixer (SWM) is deployed in shallow layers, utilizing a multilevel 3D discrete wavelet transform (3D-DWT) to decouple spatiotemporal features into various low-frequency and high frequency components, and then employing efficient spectral feature fusion. SWM can effectively capture the temporal dependencies and local spatial features. Spiking Self-Attention (SSA) is deployed in deeper layers to further extract global spatiotemporal information. We also design a hash layer utilizing binary characteristic of SNNs, which integrates information over multiple time steps to generate final hash codes. Furthermore, we propose a new dynamic soft similarity loss for SNNs, which utilizes membrane potentials to construct a learnable similarity matrix as soft labels to fully capture the similarity differences between classes and compensate information loss in SNNs, thereby improving retrieval performance. Experiments on multiple datasets demonstrate that Spikinghash can achieve state-of-the-art results with low energy consumption and fewer parameters.
</p>

</div>
</details>


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2024</summary>

<div class="paper-card">

<h3 class="paper-title">Neuromorphic-enabled video-activated cell sorting</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.nature.com/articles/s41467-024-55094-0" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Imaging flow cytometry allows image-activated cell sorting (IACS) with enhanced feature dimensions in cellular morphology, structure, and composition. However, existing IACS frameworks suffer from the challenges of 3D information loss and processing latency dilemma in real-time sorting operation. Herein, we establish a neuromorphic-enabled video-activated cell sorter (NEVACS) framework, designed to achieve high-dimensional spatiotemporal characterization content alongside high-throughput sorting of particles in wide field of view. NEVACS adopts event camera, CPU, spiking neural networks deployed on a neuromorphic chip, and achieves sorting throughput of 1000 cells/s with relatively economic hybrid hardware solution (~$10 K for control) and simple-to-make-and-use microfluidic infrastructures. Particularly, the application of NEVACS in classifying regular red blood cells and blood-disease-relevant …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">DAFDet: A Unified Dynamic SAR Target Detection Architecture With Asymptotic Fusion Enhancement and Feature Encoding Decoupling</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10813601/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In many military and civilian applications, synthetic aperture radar (SAR) image target detection plays a vital role. However, current methods for SAR target detection generally fail to balance speed and accuracy, thus making it impossible to deploy them to real-world engineering applications. In addition, strong scattering, multiscales, high density, complex background interference, and speckle noise make it remarkably challenging to extract effective target information and disentangle background noise from the target information, ultimately resulting in high missing and false alarm rates. To address these issues, a unified dynamic SAR target detection architecture (DAFDet) with asymptotic fusion enhancement and feature encoding decoupling is proposed in this article. First, a dynamic architecture is constructed by cascading two identical detectors and integrating a designed decision maker. This decision maker can …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">An efficient sequential decentralized federated progressive channel pruning strategy for smart grid electricity theft detection</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10807710/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
This article aims to develop a lightweight, decentralized federated learning (FL)-based strategy for electricity theft detection (ETD). Different from most of the existing ETD solutions, which typically deploy centralized deep learning models, our proposed method utilizes well-pruned lightweight networks and operates in a completely decentralized manner while maintaining the performance of the ETD model. Specifically, to protect data privacy, a novel sequential decentralized FL (SDFL) framework was designed, eliminating the centralized parameter aggregation node in traditional FL. Each client communicates model parameters only with its neighbors and trains its model locally. In addition, to facilitate deployment on edge devices, model pruning techniques are integrated with the sequential transmission characteristics of the SDFL framework. A progressive channel pruning technique is proposed, gradually reducing …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking transformer with experts mixture</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/137101016144540ed3191dc2b02f09a5-Abstract-Conference.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) provide a sparse spike-driven mechanism which is believed to be critical for energy-efficient deep learning. Mixture-of-Experts (MoE), on the other side, aligns with the brain mechanism of distributed and sparse processing, resulting in an efficient way of enhancing model capacity and conditional computation. In this work, we consider how to incorporate SNNs’ spike-driven and MoE’s conditional computation into a unified framework. However, MoE uses softmax to get the dense conditional weights for each expert and TopK to hard-sparsify the network, which does not fit the properties of SNNs. To address this issue, we reformulate MoE in SNNs and introduce the Spiking Experts Mixture Mechanism (SEMM) from the perspective of sparse spiking activation. Both the experts and the router output spiking sequences, and their element-wise operation makes SEMM computation spike-driven and dynamic sparse-conditional. By developing SEMM into Spiking Transformer, the Experts Mixture Spiking Attention (EMSA) and the Experts Mixture Spiking Perceptron (EMSP) are proposed, which performs routing allocation for head-wise and channel-wise spiking experts, respectively. Experiments show that SEMM realizes sparse conditional computation and obtains a stable improvement on neuromorphic and static datasets with approximate computational overhead based on the Spiking Transformer baselines.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">MetaLA: Unified optimal linear approximation to softmax attention map</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/8329a45669017898bb0cc09d27f8d2bb-Abstract-Conference.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Various linear complexity models, such as Linear Transformer (LinFormer), State Space Model (SSM), and Linear RNN (LinRNN), have been proposed to replace the conventional softmax attention in Transformer structures. However, the optimal design of these linear models is still an open question. In this work, we attempt to answer this question by finding the best linear approximation to softmax attention from a theoretical perspective. We start by unifying existing linear complexity models as the linear attention form and then identify three conditions for the optimal linear attention design:(1) Dynamic memory ability;(2) Static approximation ability;(3) Least parameter approximation. We find that none of the current linear models meet all three conditions, resulting in suboptimal performance. Instead, we propose Meta Linear Attention (MetaLA) as a solution that satisfies these conditions. Our experiments on Multi-Query Associative Recall (MQAR) task, language modeling, image classification, and Long-Range Arena (LRA) benchmark demonstrate that MetaLA is more effective than the existing linear models.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Flexible and scalable deep dendritic spiking neural networks with multiple nonlinear branching</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2412.06355" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Recent advances in spiking neural networks (SNNs) have a predominant focus on network architectures, while relatively little attention has been paid to the underlying neuron model. The point neuron models, a cornerstone of deep SNNs, pose a bottleneck on the network-level expressivity since they depict somatic dynamics only. In contrast, the multi-compartment models in neuroscience offer remarkable expressivity by introducing dendritic morphology and dynamics, but remain underexplored in deep learning due to their unaffordable computational cost and inflexibility. To combine the advantages of both sides for a flexible, efficient yet more powerful model, we propose the dendritic spiking neuron (DendSN) incorporating multiple dendritic branches with nonlinear dynamics. Compared to the point spiking neurons, DendSN exhibits significantly higher expressivity. DendSN&#x27;s flexibility enables its seamless integration into diverse deep SNN architectures. To accelerate dendritic SNNs (DendSNNs), we parallelize dendritic state updates across time steps, and develop Triton kernels for GPU-level acceleration. As a result, we can construct large-scale DendSNNs with depth comparable to their point SNN counterparts. Next, we comprehensively evaluate DendSNNs&#x27; performance on various demanding tasks. By modulating dendritic branch strengths using a context signal, catastrophic forgetting of DendSNNs is substantially mitigated. Moreover, DendSNNs demonstrate enhanced robustness against noise and adversarial attacks compared to point SNNs, and excel in few-shot learning settings. Our work firstly demonstrates the possibility of training bio-plausible dendritic SNNs with depths and scales comparable to traditional point SNNs, and reveals superior expressivity and robustness of reduced dendritic neuron models in deep learning, thereby offering a fresh perspective on advancing neural network design.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Brain-Inspired Computing: A Systematic Survey and Future Trends (vol 112, pg 544, 2024)</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://scholar.google.com/scholar?cluster=9465807871306407395&hl=en&oi=scholarr" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Abstract unavailable. This publication does not provide a summary using scholarly.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SNN-BERT: Training-efficient Spiking Neural Networks for energy-efficient BERT</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608024005549" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) are naturally suited to process sequence tasks such as NLP with low power, due to its brain-inspired spatio-temporal dynamics and spike-driven nature. Current SNNs employ” repeat coding” that re-enter all input tokens at each timestep, which fails to fully exploit temporal relationships between the tokens and introduces memory overhead. In this work, we align the number of input tokens with the timestep and refer to this input coding as” individual coding”. To cope with the increase in training time for individual encoded SNNs due to the dramatic increase in timesteps, we design a Bidirectional Parallel Spiking Neuron (BPSN) with following features: First, BPSN supports spike parallel computing and effectively avoids the issue of uninterrupted firing; Second, BPSN excels in handling adaptive sequence length tasks, which is a capability that existing work does not have …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A High Energy-Efficiency Multi-core Neuromorphic Architecture for Deep SNN Training</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2412.05302" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
There is a growing necessity for edge training to adapt to dynamically changing environment. Neuromorphic computing represents a significant pathway for high-efficiency intelligent computation in energy-constrained edges, but existing neuromorphic architectures lack the ability of directly training spiking neural networks (SNNs) based on backpropagation. We develop a multi-core neuromorphic architecture with Feedforward-Propagation, Back-Propagation, and Weight-Gradient engines in each core, supporting high efficient parallel computing at both the engine and core levels. It combines various data flows and sparse computation optimization by fully leveraging the sparsity in SNN training, obtaining a high energy efficiency of 1.05TFLOPS/W@ FP16 @ 28nm, 55 ~ 85% reduction of DRAM access compared to A100 GPU in SNN trainings, and a 20-core deep SNN training and a 5-worker federated learning on FPGAs. Our study develops the first multi-core neuromorphic architecture supporting the direct SNN training, facilitating the neuromorphic computing in edge-learnable applications.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spatial-temporal spiking feature pruning in spiking transformer</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10758407/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are known for brain-inspired architecture and low power consumption. Leveraging biocompatibility and self-attention mechanism, Spiking Transformers become the most promising SNN architecture with high accuracy. However, Spiking Transformers still faces the challenge of high training costs, such as a 51 network requiring 181 training hours on ImageNet. In this work, we explore feature pruning to reduce training costs and overcome two challenges: high pruning ratio and lightweight pruning methods. We first analyze the spiking features and find the potential for a high pruning ratio. The majority of information is concentrated on a part of the spiking features in spiking transformer, which suggests that we can keep this part of the tokens and prune the others. To achieve lightweight, a parameter-free spatial–temporal spiking feature pruning method is proposed, which uses only a …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Towards unifying understanding and generation in the era of vision foundation models: A survey from the autoregression perspective</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2410.22217" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Autoregression in large language models (LLMs) has shown impressive scalability by unifying all language tasks into the next token prediction paradigm. Recently, there is a growing interest in extending this success to vision foundation models. In this survey, we review the recent advances and discuss future directions for autoregressive vision foundation models. First, we present the trend for next generation of vision foundation models, i.e., unifying both understanding and generation in vision tasks. We then analyze the limitations of existing vision foundation models, and present a formal definition of autoregression with its advantages. Later, we categorize autoregressive vision foundation models from their vision tokenizers and autoregression backbones. Finally, we discuss several promising research challenges and directions. To the best of our knowledge, this is the first survey to comprehensively summarize autoregressive vision foundation models under the trend of unifying understanding and generation. A collection of related resources is available at https://github.com/EmmaSRH/ARVFM.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">RSC-SNN: Exploring the Trade-off Between Adversarial Robustness and Accuracy in Spiking Neural Networks via Randomized Smoothing Coding</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://dl.acm.org/doi/abs/10.1145/3664647.3680639" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) have received widespread attention due to their unique neuronal dynamics and low-power nature. Previous research empirically shows that SNNs with Poisson coding are more robust than Artificial Neural Networks (ANNs) on small-scale datasets. However, it is still unclear in theory how the adversarial robustness of SNNs is derived, and whether SNNs can still maintain its adversarial robustness advantage on large-scale dataset tasks. This work theoretically demonstrates that SNN&#x27;s inherent adversarial robustness stems from its Poisson coding. We reveal the conceptual equivalence of Poisson coding and randomized smoothing in defense strategies, and analyze in depth the trade-off between accuracy and adversarial robustness in SNNs via the proposed Randomized Smoothing Coding (RSC) method. Experiments demonstrate that the proposed RSC-SNNs show remarkable …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Guest Editorial: Special Issue on Advancing Machine Intelligence With Neuromorphic Computing</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10716578/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
NEUROMORPHIC computing, also known as “brain-inspired computing,” represents a novel computing paradigm utilizing neural spikes for communication between computing blocks. This technology garners significant attention due to its potential to achieve brainlike computing efficiency and human cognitive intelligence. As generally trained by spike-based learning schemes that mimic the biological neural codes, dynamics, and circuitry and built on non-von Neumann computing architecture, neuromorphic computing systems have shown very high-energy efficiency and have emerged as an exciting interdisciplinary field with great potential for building a more powerful computing paradigm of machine intelligence in the next generation. In recent years, scientists and engineers have made great breakthroughs in the field of neuromorphic computing, including neuromorphic chips (eg, TrueNorth, Loihi, and Tianjic …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Correlation-Aware Select and Merge Attention for Efficient Fine-Tuning and Context Length Extension</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2410.04211" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Modeling long sequences is crucial for various large-scale models; however, extending existing architectures to handle longer sequences presents significant technical and resource challenges. In this paper, we propose an efficient and flexible attention architecture that enables the extension of context lengths in large language models with reduced computational resources and fine-tuning time compared to other excellent methods. Specifically, we introduce correlation-aware selection and merging mechanisms to facilitate efficient sparse attention. In addition, we also propose a novel data augmentation technique involving positional encodings to enhance generalization to unseen positions. The results are as follows: First, using a single A100, we achieve fine-tuning on Llama2-7B with a sequence length of 32K, which is more efficient than other methods that rely on subsets for regression. Second, we present a comprehensive method for extending context lengths across the pre-training, fine-tuning, and inference phases. During pre-training, our attention mechanism partially breaks translation invariance during token selection, so we apply positional encodings only to the selected tokens. This approach achieves relatively high performance and significant extrapolation capabilities. For fine-tuning, we introduce Cyclic, Randomly Truncated, and Dynamically Growing NTK Positional Embedding (CRD NTK). This design allows fine-tuning with a sequence length of only 16K, enabling models such as Llama2-7B and Mistral-7B to perform inference with context lengths of up to 1M or even arbitrary lengths. Our method achieves 100\% accuracy on the passkey task with a context length of 4M and maintains stable perplexity at a 1M context length. This represents at least a 64-fold reduction in resource requirements compared to traditional full-attention mechanisms, while still achieving competitive performance.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Embedded prompt tuning: Towards enhanced calibration of pretrained models for medical images</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S136184152400183X" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Foundation models pre-trained on large-scale data have been widely witnessed to achieve success in various natural imaging downstream tasks. Parameter-efficient fine-tuning (PEFT) methods aim to adapt foundation models to new domains by updating only a small portion of parameters in order to reduce computational overhead. However, the effectiveness of these PEFT methods, especially in cross-domain few-shot scenarios, e.g., medical image analysis, has not been fully explored. In this work, we facilitate the study of the performance of PEFT when adapting foundation models to medical image classification tasks. Furthermore, to alleviate the limitations of prompt introducing ways and approximation capabilities on Transformer architectures of mainstream prompt tuning methods, we propose the Embedded Prompt Tuning (EPT) method by embedding prompt tokens into the expanded channels. We also find …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://link.springer.com/chapter/10.1007/978-3-031-73411-3_15" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Brain-inspired Spiking Neural Networks (SNNs) have bio-plausibility and low-power advantages over Artificial Neural Networks (ANNs). Applications of SNNs are currently limited to simple classification tasks because of their poor performance. In this work, we focus on bridging the performance gap between ANNs and SNNs on object detection. Our design revolves around network architecture and spiking neuron. First, the overly complex module design causes spike degradation when the YOLO series is converted to the corresponding spiking version. We design a SpikeYOLO architecture to solve this problem by simplifying the vanilla YOLO and incorporating meta SNN blocks. Second, object detection is more sensitive to quantization errors in the conversion of membrane potentials into binary spikes by spiking neurons. To address this challenge, we design a new spiking neuron that activates Integer values during …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Distance-forward learning: enhancing the forward-forward algorithm towards high-performance on-chip learning</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2408.14925" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The Forward-Forward (FF) algorithm was recently proposed as a local learning method to address the limitations of backpropagation (BP), offering biological plausibility along with memory-efficient and highly parallelized computational benefits. However, it suffers from suboptimal performance and poor generalization, largely due to inadequate theoretical support and a lack of effective learning strategies. In this work, we reformulate FF using distance metric learning and propose a distance-forward algorithm (DF) to improve FF performance in supervised vision tasks while preserving its local computational properties, making it competitive for efficient on-chip learning. To achieve this, we reinterpret FF through the lens of centroid-based metric learning and develop a goodness-based N-pair margin loss to facilitate the learning of discriminative features. Furthermore, we integrate layer-collaboration local update strategies to reduce information loss caused by greedy local parameter updates. Our method surpasses existing FF models and other advanced local learning approaches, with accuracies of 99.7\% on MNIST, 88.2\% on CIFAR-10, 59\% on CIFAR-100, 95.9\% on SVHN, and 82.5\% on ImageNette, respectively. Moreover, it achieves comparable performance with less than 40\% memory cost compared to BP training, while exhibiting stronger robustness to multiple types of hardware-related noise, demonstrating its potential for online learning and energy-efficient computation on neuromorphic chips.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Scalable autoregressive image generation with mamba</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2408.12245" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
We introduce AiM, an autoregressive (AR) image generative model based on Mamba architecture. AiM employs Mamba, a novel state-space model characterized by its exceptional performance for long-sequence modeling with linear time complexity, to supplant the commonly utilized Transformers in AR image generation models, aiming to achieve both superior generation quality and enhanced inference speed. Unlike existing methods that adapt Mamba to handle two-dimensional signals via multi-directional scan, AiM directly utilizes the next-token prediction paradigm for autoregressive image generation. This approach circumvents the need for extensive modifications to enable Mamba to learn 2D spatial representations. By implementing straightforward yet strategically targeted modifications for visual generative tasks, we preserve Mamba&#x27;s core structure, fully exploiting its efficient long-sequence modeling capabilities and scalability. We provide AiM models in various scales, with parameter counts ranging from 148M to 1.3B. On the ImageNet1K 256*256 benchmark, our best AiM model achieves a FID of 2.21, surpassing all existing AR models of comparable parameter counts and demonstrating significant competitiveness against diffusion models, with 2 to 10 times faster inference speed. Code is available at https://github.com/hp-l33/AiM
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Toward large-scale spiking neural networks: A comprehensive survey and future directions</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2409.02111" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Deep learning has revolutionized artificial intelligence (AI), achieving remarkable progress in fields such as computer vision, speech recognition, and natural language processing. Moreover, the recent success of large language models (LLMs) has fueled a surge in research on large-scale neural networks. However, the escalating demand for computing resources and energy consumption has prompted the search for energy-efficient alternatives. Inspired by the human brain, spiking neural networks (SNNs) promise energy-efficient computation with event-driven spikes. To provide future directions toward building energy-efficient large SNN models, we present a survey of existing methods for developing deep spiking neural networks, with a focus on emerging Spiking Transformers. Our main contributions are as follows: (1) an overview of learning methods for deep spiking neural networks, categorized by ANN-to-SNN conversion and direct training with surrogate gradients; (2) an overview of network architectures for deep spiking neural networks, categorized by deep convolutional neural networks (DCNNs) and Transformer architecture; and (3) a comprehensive comparison of state-of-the-art deep SNNs with a focus on emerging Spiking Transformers. We then further discuss and outline future directions toward large-scale SNNs.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Brain-inspired computing: A systematic survey and future trends</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10636118/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Brain-inspired computing (BIC) is an emerging research field that aims to build fundamental theories, models, hardware architectures, and application systems toward more general artificial intelligence (AI) by learning from the information processing mechanisms or structures/functions of biological nervous systems. It is regarded as one of the most promising research directions for future intelligent computing in the post-Moore era. In the past few years, various new schemes in this field have sprung up to explore more general AI. These works are quite divergent in the aspects of modeling/algorithm, software tool, hardware platform, and benchmark data since BIC is an interdisciplinary field that consists of many different domains, including computational neuroscience, AI, computer science, statistical physics, material science, and microelectronics. This situation greatly impedes researchers from obtaining a clear …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Network Decomposition Based Online Localization and State Recovery for False Data Injection Attacks in Smart Grid</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10664729/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Most existing countermeasure works formulate the detection of false data inject (FDI) attacks as a typical binary classification problem, which, however, could not be able to localize and further eliminate the impact of FDI attacks. To bridge this gap, in this paper, a network decomposition strategy is proposed to online localize FDI attacks and recover state in smart grid. Firstly, several non-overlapping subnets are decomposed by a large-scale power system, and then a set of lightweight networks are deployed on them. The proposed strategy could not only achieve real-time localization of FDI attacks, but also recover the attacked state value. More importantly, different from existing centralized strategies, the proposed strategy can carry out fast offline training when the measured data is updated dynamically or even has preferable privacy-preserving ability since each subnet only requires local data. Several case studies …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Network model with internal complexity bridges artificial intelligence and neuroscience</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.nature.com/articles/s43588-024-00674-9" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Artificial intelligence (AI) researchers currently believe that the main approach to building more general model problems is the big AI model, where existing neural networks are becoming deeper, larger and wider. We term this the big model with external complexity approach. In this work we argue that there is another approach called small model with internal complexity, which can be used to find a suitable path of incorporating rich properties into neurons to construct larger and more efficient AI models. We uncover that one has to increase the scale of the network externally to stimulate the same dynamical properties. To illustrate this, we build a Hodgkin–Huxley (HH) network with rich internal complexity, where each neuron is an HH model, and prove that the dynamical properties and performance of the HH network can be equivalent to a bigger leaky integrate-and-fire (LIF) network, where each neuron is a LIF …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Multi-scale full spike pattern for semantic segmentation</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608024002545" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs), as the brain-inspired neural networks, encode information in spatio-temporal dynamics. They have the potential to serve as low-power alternatives to artificial neural networks (ANNs) due to their sparse and event-driven nature. However, existing SNN-based models for pixel-level semantic segmentation tasks suffer from poor performance and high memory overhead, failing to fully exploit the computational effectiveness and efficiency of SNNs. To address these challenges, we propose the multi-scale and full spike segmentation network (MFS-Seg), which is based on the deep direct trained SNN and represents the first attempt to train a deep SNN with surrogate gradients for semantic segmentation. Specifically, we design an efficient fully-spike residual block (EFS-Res) to alleviate representation issues caused by spiking noise on different channels. EFS-Res utilizes depthwise …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Unveiling the Potential of Spiking Dynamics in Graph Representation Learning through Spatial-Temporal Normalization and Coding Strategies</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2407.20508" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In recent years, spiking neural networks (SNNs) have attracted substantial interest due to their potential to replicate the energy-efficient and event-driven processing of biological neurons. Despite this, the application of SNNs in graph representation learning, particularly for non-Euclidean data, remains underexplored, and the influence of spiking dynamics on graph learning is not yet fully understood. This work seeks to address these gaps by examining the unique properties and benefits of spiking dynamics in enhancing graph representation learning. We propose a spike-based graph neural network model that incorporates spiking dynamics, enhanced by a novel spatial-temporal feature normalization (STFN) technique, to improve training efficiency and model stability. Our detailed analysis explores the impact of rate coding and temporal coding on SNN performance, offering new insights into their advantages for deep graph networks and addressing challenges such as the oversmoothing problem. Experimental results demonstrate that our SNN models can achieve competitive performance with state-of-the-art graph neural networks (GNNs) while considerably reducing computational costs, highlighting the potential of SNNs for efficient neuromorphic computing applications in complex graph-based scenarios.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spikevoice: High-quality text-to-speech via efficient spiking neural network</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2408.00788" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Brain-inspired Spiking Neural Network (SNN) has demonstrated its effectiveness and efficiency in vision, natural language, and speech understanding tasks, indicating their capacity to &quot;see&quot;, &quot;listen&quot;, and &quot;read&quot;. In this paper, we design \textbf{SpikeVoice}, which performs high-quality Text-To-Speech (TTS) via SNN, to explore the potential of SNN to &quot;speak&quot;. A major obstacle to using SNN for such generative tasks lies in the demand for models to grasp long-term dependencies. The serial nature of spiking neurons, however, leads to the invisibility of information at future spiking time steps, limiting SNN models to capture sequence dependencies solely within the same time step. We term this phenomenon &quot;partial-time dependency&quot;. To address this issue, we introduce Spiking Temporal-Sequential Attention STSA in the SpikeVoice. To the best of our knowledge, SpikeVoice is the first TTS work in the SNN field. We perform experiments using four well-established datasets that cover both Chinese and English languages, encompassing scenarios with both single-speaker and multi-speaker configurations. The results demonstrate that SpikeVoice can achieve results comparable to Artificial Neural Networks (ANN) with only 10.5 energy consumption of ANN.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Event-based depth prediction with deep spiking neural network</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10592043/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras have gained popularity in depth estimation due to their superior features such as high-temporal resolution, low latency, and low-power consumption. Spiking neural network (SNN) is a promising approach for processing event camera inputs due to its spike-based event-driven nature. However, SNNs face performance degradation when the network becomes deeper, affecting their performance in depth estimation tasks. To address this issue, we propose a deep spiking U-Net model. Our spiking U-Net architecture leverages refined shortcuts and residual blocks to avoid performance degradation and boost task performance. We also propose a new event representation method designed for multistep SNNs to effectively utilize depth information in the temporal dimension. Our experiments on MVSEC dataset show that the proposed method improves accuracy by 18.50% and 25.18% compared to …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SpikeLLM: Scaling up spiking neural network to large language models via saliency-based spiking</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2407.04752" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Recent advancements in large language models (LLMs) with billions of parameters have improved performance in various applications, but their inference processes demand significant energy and computational resources. In contrast, the human brain, with approximately 86 billion neurons, is much more energy-efficient than LLMs with similar parameters. Inspired by this, we redesign 770 billion parameter LLMs using bio-plausible spiking mechanisms, emulating the efficient behavior of the human brain. We propose the first spiking large language model, SpikeLLM. Coupled with the proposed model, two essential approaches are proposed to improve spike training efficiency: Generalized Integrate-and-Fire (GIF) neurons to compress spike length from to bits, and an Optimal Brain Spiking framework to divide outlier channels and allocate different for GIF neurons, which further compresses spike length to approximate bits. The necessity of spike-driven LLM is proved by comparison with quantized LLMs with similar operations. In the OmniQuant pipeline, SpikeLLM reduces 11.01% WikiText2 perplexity and improves 2.55% accuracy of common scene reasoning on a LLAMA-7B W4A4 model. In the GPTQ pipeline, SpikeLLM achieves direct additive in linear layers, significantly exceeding PB-LLMs.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Connectional-style-guided contextual representation learning for brain disease diagnosis</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S089360802400220X" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Structural magnetic resonance imaging (sMRI) has shown great clinical value and has been widely used in deep learning (DL) based computer-aided brain disease diagnosis. Previous DL-based approaches focused on local shapes and textures in brain sMRI that may be significant only within a particular domain. The learned representations are likely to contain spurious information and have poor generalization ability in other diseases and datasets. To facilitate capturing meaningful and robust features, it is necessary to first comprehensively understand the intrinsic pattern of the brain that is not restricted within a single data/task domain. Considering that the brain is a complex connectome of interlinked neurons, the connectional properties in the brain have strong biological significance, which is shared across multiple domains and covers most pathological information. In this work, we propose a connectional …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spikelm: Towards general spike-driven language modeling via elastic bi-spiking mechanisms</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2406.03287" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Towards energy-efficient artificial intelligence similar to the human brain, the bio-inspired spiking neural networks (SNNs) have advantages of biological plausibility, event-driven sparsity, and binary activation. Recently, large-scale language models exhibit promising generalization capability, making it a valuable issue to explore more general spike-driven models. However, the binary spikes in existing SNNs fail to encode adequate semantic information, placing technological challenges for generalization. This work proposes the first fully spiking mechanism for general language tasks, including both discriminative and generative ones. Different from previous spikes with {0,1} levels, we propose a more general spike formulation with bi-directional, elastic amplitude, and elastic frequency encoding, while still maintaining the addition nature of SNNs. In a single time step, the spike is enhanced by direction and amplitude information; in spike frequency, a strategy to control spike firing rate is well designed. We plug this elastic bi-spiking mechanism in language modeling, named SpikeLM. It is the first time to handle general language tasks with fully spike-driven models, which achieve much higher accuracy than previously possible. SpikeLM also greatly bridges the performance gap between SNNs and ANNs in language modeling. Our code is available at https://github.com/Xingrun-Xing/SpikeLM.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Large-scale self-normalizing neural networks</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S2949855424000194" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Self-normalizing neural networks (SNN) regulate the activation and gradient flows through activation functions with the self-normalization property. As SNNs do not rely on norms computed from minibatches, they are more friendly to data parallelism, kernel fusion, and emerging architectures such as ReRAM-based accelerators. However, existing SNNs have mainly demonstrated their effectiveness on toy datasets and fall short in accuracy when dealing with large-scale tasks like ImageNet. They lack the strong normalization, regularization, and expression power required for wider, deeper models and larger-scale tasks. To enhance the normalization strength, this paper introduces a comprehensive and practical definition of the self-normalization property in terms of the stability and attractiveness of the statistical fixed points. It is comprehensive as it jointly considers all the fixed points used by existing studies: the first …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">High-Performance Temporal Reversible Spiking Neural Networks with  Training Memory and  Inference Cost</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2405.16466" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Multi-timestep simulation of brain-inspired Spiking Neural Networks (SNNs) boost memory requirements during training and increase inference energy cost. Current training methods cannot simultaneously solve both training and inference dilemmas. This work proposes a novel Temporal Reversible architecture for SNNs (T-RevSNN) to jointly address the training and inference challenges by altering the forward propagation of SNNs. We turn off the temporal dynamics of most spiking neurons and design multi-level temporal reversible interactions at temporal turn-on spiking neurons, resulting in a training memory. Combined with the temporal reversible nature, we redesign the input encoding and network organization of SNNs to achieve inference energy cost. Then, we finely adjust the internal units and residual connections of the basic SNN block to ensure the effectiveness of sparse temporal information interaction. T-RevSNN achieves excellent accuracy on ImageNet, while the memory efficiency, training time acceleration, and inference energy efficiency can be significantly improved by , , and , respectively. This work is expected to break the technical bottleneck of significantly increasing memory cost and training time for large-scale SNNs while maintaining high performance and low inference energy cost. Source code and models are available at: https://github.com/BICLab/T-RevSNN.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spike-based dynamic computing with asynchronous sensing-computing neuromorphic chip</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.nature.com/articles/s41467-024-47811-6" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
By mimicking the neurons and synapses of the human brain and employing spiking neural networks on neuromorphic chips, neuromorphic computing offers a promising energy-efficient machine intelligence. How to borrow high-level brain dynamic mechanisms to help neuromorphic computing achieve energy advantages is a fundamental issue. This work presents an application-oriented algorithm-software-hardware co-designed neuromorphic system for this issue. First, we design and fabricate an asynchronous chip called “Speck”, a sensing-computing neuromorphic system on chip. With the low processor resting power of 0.42mW, Speck can satisfy the hardware requirements of dynamic computing: no-input consumes no energy. Second, we uncover the “dynamic imbalance” in spiking neural networks and develop an attention-based framework for achieving the algorithmic requirements of dynamic computing …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Sufficient control of complex networks</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0378437124002607" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In this paper, we propose to study sufficient control of complex networks, which is to control a sufficiently large portion of the network, where only the quantity of controllable nodes matters. To the best of our knowledge, this is the first time that such a problem is investigated. We prove that the sufficient controllability problem can be converted into a minimum-cost flow problem, for which an algorithm with polynomial complexity can be devised. Further, we study the problem of minimum-cost sufficient control, which is to drive a sufficiently large subset of the network nodes to any predefined state with the minimum cost using a given number of controllers. The problem is NP-hard. We propose an “extended L 0-norm-constraint-based Projected Gradient Method”(eLPGM) algorithm, which achieves suboptimal solutions for the problems at small or medium sizes. To tackle the large-scale problems, we propose to convert the …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Network Group Partition and Core Placement Optimization for Neuromorphic Multi-Core and Multi-Chip Systems</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10487993/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Neuromorphic chips with multi-core architecture are considered to be of great potential for the next generation of artificial intelligence (AI) chips because of the avoidance of the memory wall effect. Deploying deep neural networks (DNNs) to these chips requires two stages, namely, network partition and core placement. For the network partition, existing schemes are mostly manual or only focus on single-layer, small-scale network partitions. For the core placement, to the best of our knowledge, there is still no work that has completely solved the communication deadlock problem at the clock-level which commonly exists in the applications of neuromorphic multi-core and multi-chip (NMCMC) systems. To address these issues that affect the operating and deployment efficiency of NMCMC systems, we formulate the network group partition problem as an optimization problem for the first time and propose a search …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spatio-temporal fusion graph convolutional network for traffic flow forecasting</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S1566253523005122" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In most recent research, the traffic forecasting task is typically formulated as a spatio-temporal graph modeling problem. For spatial correlation, they typically learn the shared pattern (i.e., the most salient pattern) of traffic series and measure the interdependence between traffic series based on a predefined graph. On the one hand, learning a specific traffic pattern for each node (traffic series) is crucial and essential for accurate spatial correlation learning. On the other hand, most predefined graphs cannot accurately represent the interdependence between traffic series because they are unchangeable while the prediction task changes. For temporal correlation, they usually concentrate on contiguous temporal correlation. Therefore, they are insufficient due to their lack of global temporal correlation learning. To overcome these aforementioned limitations, we propose a novel method named Spatio-Temporal Fusion …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Gated attention coding for training high-performance and efficient spiking neural networks</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/27816" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are emerging as an energy-efficient alternative to traditional artificial neural networks (ANNs) due to their unique spike-based event-driven nature. Coding is crucial in SNNs as it converts external input stimuli into spatio-temporal feature sequences. However, most existing deep SNNs rely on direct coding that generates powerless spike representation and lacks the temporal dynamics inherent in human vision. Hence, we introduce Gated Attention Coding (GAC), a plug-and-play module that leverages the multi-dimensional gated attention unit to efficiently encode inputs into powerful representations before feeding them into the SNN architecture. GAC functions as a preprocessing layer that does not disrupt the spike-driven nature of the SNN, making it amenable to efficient neuromorphic hardware implementation with minimal modifications. Through an observer model theoretical analysis, we demonstrate GAC&#x27;s attention mechanism improves temporal dynamics and coding efficiency. Experiments on CIFAR10/100 and ImageNet datasets demonstrate that GAC achieves state-of-the-art accuracy with remarkable efficiency. Notably, we improve top-1 accuracy by 3.10% on CIFAR100 with only 6-time steps and 1.07% on ImageNet while reducing energy usage to 66.9% of the previous works. To our best knowledge, it is the first time to explore the attention-based dynamic coding scheme in deep SNNs, with exceptional effectiveness and efficiency on large-scale datasets. Code is available at https://github. com/bollossom/GAC.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Hardvs: Revisiting human activity recognition with dynamic vision sensors</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/28372" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The main streams of human activity recognition (HAR) algorithms are developed based on RGB cameras which usually suffer from illumination, fast motion, privacy preservation, and large energy consumption. Meanwhile, the biologically inspired event cameras attracted great interest due to their unique features, such as high dynamic range, dense temporal but sparse spatial resolution, low latency, low power, etc. As it is a newly arising sensor, even there is no realistic large-scale dataset for HAR. Considering its great practical value, in this paper, we propose a large-scale benchmark dataset to bridge this gap, termed HARDVS, which contains 300 categories and more than 100K event sequences. We evaluate and report the performance of multiple popular HAR algorithms, which provide extensive baselines for future works to compare. More importantly, we propose a novel spatial-temporal feature learning and fusion framework, termed ESTF, for event stream based human activity recognition. It first projects the event streams into spatial and temporal embeddings using StemNet, then, encodes and fuses the dual-view representations using Transformer networks. Finally, the dual features are concatenated and fed into a classification head for activity prediction. Extensive experiments on multiple datasets fully validated the effectiveness of our model. Both the dataset and source code will be released at https://github.com/Event-AHU/HARDVS.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Target-embedding autoencoder with knowledge distillation for multi-label classification</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10477613/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In the task of multi-label classification, it is a key challenge to determine the correlation between labels. One solution to this is the Target Embedding Autoencoder (TEA), but most TEA-based frameworks have numerous parameters, large models, and high complexity, which makes it difficult to deal with the problem of large-scale learning. To address this issue, we provide a Target Embedding Autoencoder framework based on Knowledge Distillation (KD-TEA) that compresses a Teacher model with large parameters into a small Student model through knowledge distillation. Specifically, KD-TEA transfers the dark knowledge learned from the Teacher model to the Student model. The dark knowledge can provide effective regularization to alleviate the over-fitting problem in the training process, thereby enhancing the generalization ability of the Student model, and better completing the multi-label task. In order to make …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Structural Controllability of Multiplex Networks With the Minimum Number of Driver Nodes</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10458257/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In this article, we focus on the problem of structural controllability of multiplex networks. By proposing a graph-theoretic framework, we address the problem of identifying the minimum set of driver nodes to ensure the structural controllability of multiplex networks, where the driver nodes can only be located in a single layer. We rigorously prove that the problem is essentially a minimum-cost flow problem and devise an algorithm termed “minimum-cost flow-based driver-node identification,” which can achieve the optimal solution with polynomial time complexity. Extensive simulations on synthetic and real-life multiplex networks demonstrate the validity and efficiency of the proposed algorithm.
</p>

</div>
</details>


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2023</summary>

<div class="paper-card">

<h3 class="paper-title">Spike-driven transformer</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) provide an energy-efficient deep learning option due to their unique spike-based event-driven (ie, spike-driven) paradigm. In this paper, we incorporate the spike-driven paradigm into Transformer by the proposed Spike-driven Transformer with four unique properties:(1) Event-driven, no calculation is triggered when the input of Transformer is zero;(2) Binary spike communication, all matrix multiplications associated with the spike matrix can be transformed into sparse additions;(3) Self-attention with linear complexity at both token and channel dimensions;(4) The operations between spike-form Query, Key, and Value are mask and addition. Together, there are only sparse addition operations in the Spike-driven Transformer. To this end, we design a novel Spike-Driven Self-Attention (SDSA), which exploits only mask and addition operations without any multiplication, and thus having up to lower computation energy than vanilla self-attention. Especially in SDSA, the matrix multiplication between Query, Key, and Value is designed as the mask operation. In addition, we rearrange all residual connections in the vanilla Transformer before the activation functions to ensure that all neurons transmit binary spike signals. It is shown that the Spike-driven Transformer can achieve 77.1\% top-1 accuracy on ImageNet-1K, which is the state-of-the-art result in the SNN field.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Improving graph collaborative filtering via spike signal embedding perturbation</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10341208/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Nowadays, graph collaborative filtering has proven to be a highly effective method in recommendation systems. It learns user preferences through interactions between users and items. During the training process of graph collaborative filtering, introducing suitable perturbations is crucial to model training. It is commonly used to prevent overfitting and enhance model robustness. Perturbation is widely adopted as a data augmentation technique in recommendation systems and extensively used in contrastive learning. Contrastive learning involves multitask learning aimed at learning various views from diverse data augmentations. However, these tasks can sometimes interfere with each other, impacting their effectiveness. In contrast to methods that focus on learning various views in contrastive learning to achieve better embedding representations, we propose a straightforward yet highly effective approach to …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Neural network information receiving method, sending method, system, apparatus and readable storage medium</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://patents.google.com/patent/US11823030B2/en" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
(CN) method and system, and a sending method and system. The receiving method comprises: acquiring a reception initiation time for neuron information (S100); receiving rostral neuron information output by rostral neurons (S200); acquiring delay information of the rostral neuron information according to the reception initiation time, the rostral neuron information and a delay algorithm (S300); and determining (51) Int. Cl. composite information output by the rostral neurons accord-c00Sc 0S000 (202301) ing to the rostral neuron information and the delay infor-
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">NP-hardness of tensor network contraction ordering</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2310.06140" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
We study the optimal order (or sequence) of contracting a tensor network with a minimal computational cost. We conclude 2 different versions of this optimal sequence: that minimize the operation number (OMS) and that minimize the time complexity (CMS). Existing results only shows that OMS is NP-hard, but no conclusion on CMS problem. In this work, we firstly reduce CMS to CMS-0, which is a sub-problem of CMS with no free indices. Then we prove that CMS is easier than OMS, both in general and in tree cases. Last but not least, we prove that CMS is still NP-hard. Based on our results, we have built up relationships of hardness of different tensor network contraction problems.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spikingjelly: An open-source machine learning infrastructure platform for spike-based intelligence</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.science.org/doi/abs/10.1126/sciadv.adi1480" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) aim to realize brain-inspired intelligence on neuromorphic chips with high energy efficiency by introducing neural dynamics and spike properties. As the emerging spiking deep learning paradigm attracts increasing interest, traditional programming frameworks cannot meet the demands of the automatic differentiation, parallel computation acceleration, and high integration of processing neuromorphic datasets and deployment. In this work, we present the SpikingJelly framework to address the aforementioned dilemma. We contribute a full-stack toolkit for preprocessing neuromorphic datasets, building deep SNNs, optimizing their parameters, and deploying SNNs on neuromorphic chips. Compared to existing methods, the training of deep SNNs can be accelerated 11×, and the superior extensibility and flexibility of SpikingJelly enable users to accelerate custom models at low costs …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Filtered observations for model-based multi-agent reinforcement learning</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://link.springer.com/chapter/10.1007/978-3-031-43421-1_32" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Reinforcement learning (RL) pursues high sample efficiency in practical environments to avoid costly interactions. Learning to plan with a world model in a compact latent space for policy optimization significantly improves sample efficiency in single-agent RL. Although world model construction methods for single-agent can be naturally extended, existing multi-agent schemes fail to acquire world models effectively as redundant information increases rapidly with the number of agents. To address this issue, we in this paper leverage guided diffusion to filter this noisy information, which harms teamwork. Obtained purified global states are then used to build a unified world model. Based on the learned world model, we denoise each agent observation and plan for multi-agent policy optimization, facilitating efficient cooperation. We name our method UTOPIA, a model-based method for cooperative multi-agent …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spike attention coding for spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10246307/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs), an important family of neuroscience-oriented intelligent models, play an essential role in the neuromorphic computing community. Spike rate coding and temporal coding are the mainstream coding schemes in the current modeling of SNNs. However, rate coding usually suffers from limited representation resolution and long latency, while temporal coding usually suffers from under-utilization of spike activities. To this end, we propose spike attention coding (SAC) for SNNs. By introducing learnable attention coefficients for each time step, our coding scheme can naturally unify rate coding and temporal coding, and then flexibly learn optimal coefficients for better performance. Several normalization and regularization techniques are further incorporated to control the range and distribution of the learned attention coefficients. Extensive experiments on classification, generation, and …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Sparser spiking activity can be better: Feature refine-and-mask spiking neural network for event-based visual recognition</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608023003660" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event-based visual, a new visual paradigm with bio-inspired dynamic perception and μ s level temporal resolution, has prominent advantages in many specific visual scenarios and gained much research interest. Spiking neural network (SNN) is naturally suitable for dealing with event streams due to its temporal information processing capability and event-driven nature. However, existing works SNN neglect the fact that the input event streams are spatially sparse and temporally non-uniform, and just treat these variant inputs equally. This situation interferes with the effectiveness and efficiency of existing SNNs. In this paper, we propose the feature Refine-and-Mask SNN (RM-SNN), which has the ability of self-adaption to regulate the spiking response in a data-dependent way. We use the Refine-and-Mask (RM) module to refine all features and mask the unimportant features to optimize the membrane potential of …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Boosting zero-shot learning via contrastive optimization of attribute representations</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10198750/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Zero-shot learning (ZSL) aims to recognize classes that do not have samples in the training set. One representative solution is to directly learn an embedding function associating visual features with corresponding class semantics for recognizing new classes. Many methods extend upon this solution, and recent ones are especially keen on extracting rich features from images, e.g., attribute features. These attribute features are normally extracted within each individual image; however, the common traits for features across images yet belonging to the same attribute are not emphasized. In this article, we propose a new framework to boost ZSL by explicitly learning attribute prototypes beyond images and contrastively optimizing them with attribute-level features within images. Besides the novel architecture, two elements are highlighted for attribute representations: a new prototype generation module (PM) is designed …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Target controllability of multiplex networks with weighted interlayer edges</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10187617/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In this work, we focus on the target controllability of multiplex networks with weighted interlayer edges representing the cost for control, which is faced in many real world applications but remains unsolved. The main objective is to locate a minimum number of nodes with external control sources to ensure the controllability of a target subset of the nodes, concurrently guaranteeing the cost for controlling the subset reaches the minimum. To solve this problem, we first convert it into a path cover problem, and then model it as a minimum cost maximum flow problem, which can be solved efficiently using algorithms in graph theory. This method is termed as “Minimum Cost Maximum Flow based Target Path-cover” (MCMFTP). By using MCMFTP, we have studied the effect of the proportion of target nodes, the number of layers, and the distribution of weights on the controllability of multiplex networks. Simulation results …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Effects of microstate dynamic brain network disruption in different stages of schizophrenia</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10145423/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Schizophrenia is a heterogeneous mental disorder with unknown etiology or pathological characteristics. Microstate analysis of the electroencephalogram (EEG) signal has shown significant potential value for clinical research. Importantly, significant changes in microstate-specific parameters have been extensively reported; however, these studies have ignored the information interactions within the microstate network in different stages of schizophrenia. Based on recent findings, since rich information about the functional organization of the brain can be revealed by functional connectivity dynamics, we use the first-order autoregressive model to construct the functional connectivity of intra- and intermicrostate networks to identify information interactions among microstate networks. We demonstrate that, beyond abnormal parameters, disrupted organization of the microstate networks plays a crucial role in different …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Test-time training-free domain adaptation</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10096430/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Deploying deep learning models to new environments is very challenging. Domain adaptation (DA) is a promising paradigm to solve the problem by collecting and adapting to unlabeled data in new environments. Though research efforts have led to steady performance improvement over the past decade, DA algorithms are still hard to deploy, as training on unlabeled new data makes tuning difficult and not feasible for inference-only devices. To make DA practical, in this paper we study a new problem named Test-time Training-Free Domain Adaptation (TTDA), where trained models must adapt to a single input (mimicking the test-time scenario) without training. By exploiting spatial activation that was previously overlooked and simply averaged out, we propose a simple method based on Feature Statistics Transformation (FST) on-the-fly for each test example. The proposed algorithm is tested in the TTDA setting on …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Probabilistic modeling: Proving the lottery ticket hypothesis in spiking neural network</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2305.12148" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The Lottery Ticket Hypothesis (LTH) states that a randomly-initialized large neural network contains a small sub-network (i.e., winning tickets) which, when trained in isolation, can achieve comparable performance to the large network. LTH opens up a new path for network pruning. Existing proofs of LTH in Artificial Neural Networks (ANNs) are based on continuous activation functions, such as ReLU, which satisfying the Lipschitz condition. However, these theoretical methods are not applicable in Spiking Neural Networks (SNNs) due to the discontinuous of spiking function. We argue that it is possible to extend the scope of LTH by eliminating Lipschitz condition. Specifically, we propose a novel probabilistic modeling approach for spiking neurons with complicated spatio-temporal dynamics. Then we theoretically and experimentally prove that LTH holds in SNNs. According to our theorem, we conclude that pruning directly in accordance with the weight size in existing SNNs is clearly not optimal. We further design a new criterion for pruning based on our theory, which achieves better pruning results than baseline.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Semi-supervised partial label learning algorithm via reliable label propagation</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://link.springer.com/article/10.1007/s10489-022-04027-9" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Partial label learning (PLL) is a weakly supervised learning method that is able to predict one label as the correct answer from a given candidate label set. In PLL, when all possible candidate labels are as signed to real-world training examples, PLL will hava noisy labeling in its training data set. In the real world, it is unrealistic to assign candidate label to all the training examples. Because semi-supervised partial label learning combines two difficult learning conditions, partial label learning and semi-supervised learning, improving recognition accuracy is a big challenge. Some existing semi-supervised partial label learning boosts the model performance, by assigning to unlabeled data in their label propagation. However, those methods neglect the noisy label in their label propagation, which introduces contaminated data, at the same time it declines model performance. We proposed a semi-supervised partial …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Event-based semantic segmentation with posterior attention</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10058930/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In the past years, attention-based Transformers have swept across the field of computer vision, starting a new stage of backbones in semantic segmentation. Nevertheless, semantic segmentation under poor light conditions remains an open problem. Moreover, most papers about semantic segmentation work on images produced by commodity frame-based cameras with a limited framerate, hindering their deployment to auto-driving systems that require instant perception and response at milliseconds. An event camera is a new sensor that generates event data at microseconds and can work in poor light conditions with a high dynamic range. It looks promising to leverage event cameras to enable perception where commodity cameras are incompetent, but algorithms for event data are far from mature. Pioneering researchers stack event data as frames so that event-based segmentation is converted to frame-based …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spikegpt: Generative pre-trained language model with spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2302.13939" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
As the size of large language models continue to scale, so does the computational resources required to run it. Spiking Neural Networks (SNNs) have emerged as an energy-efficient approach to deep learning that leverage sparse and event-driven activations to reduce the computational overhead associated with model inference. While they have become competitive with non-spiking models on many computer vision tasks, SNNs have also proven to be more challenging to train. As a result, their performance lags behind modern deep learning, and we are yet to see the effectiveness of SNNs in language generation. In this paper, inspired by the Receptance Weighted Key Value (RWKV) language model, we successfully implement `SpikeGPT&#x27;, a generative language model with binary, event-driven spiking activation units. We train the proposed model on two model variants: 45M and 216M parameters. To the best of our knowledge, SpikeGPT is the largest backpropagation-trained SNN model to date, rendering it suitable for both the generation and comprehension of natural language. We achieve this by modifying the transformer block to replace multi-head self attention to reduce quadratic computational complexity O(N^2) to linear complexity O(N) with increasing sequence length. Input tokens are instead streamed in sequentially to our attention mechanism (as with typical SNNs). Our preliminary experiments show that SpikeGPT remains competitive with non-spiking models on tested benchmarks, while maintaining 20x fewer operations when processed on neuromorphic hardware that can leverage sparse, event-driven activations. Our code implementation is available at https://github.com/ridgerchu/SpikeGPT.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Attention spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10032591/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Brain-inspired spiking neural networks (SNNs) are becoming a promising energy-efficient alternative to traditional artificial neural networks (ANNs). However, the performance gap between SNNs and ANNs has been a significant hindrance to deploying SNNs ubiquitously. To leverage the full potential of SNNs, in this paper we study the attention mechanisms, which can help human focus on important information. We present our idea of attention in SNNs with a multi-dimensional attention module, which infers attention weights along the temporal, channel, as well as spatial dimension separately or simultaneously. Based on the existing neuroscience theories, we exploit the attention weights to optimize membrane potentials, which in turn regulate the spiking response. Extensive experimental results on event-based action recognition and image classification datasets demonstrate that attention facilitates vanilla …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Training full spike neural networks via auxiliary accumulation pathway</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2301.11929" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Due to the binary spike signals making converting the traditional high-power multiply-accumulation (MAC) into a low-power accumulation (AC) available, the brain-inspired Spiking Neural Networks (SNNs) are gaining more and more attention. However, the binary spike propagation of the Full-Spike Neural Networks (FSNN) with limited time steps is prone to significant information loss. To improve performance, several state-of-the-art SNN models trained from scratch inevitably bring many non-spike operations. The non-spike operations cause additional computational consumption and may not be deployed on some neuromorphic hardware where only spike operation is allowed. To train a large-scale FSNN with high performance, this paper proposes a novel Dual-Stream Training (DST) method which adds a detachable Auxiliary Accumulation Pathway (AAP) to the full spiking residual networks. The accumulation in AAP could compensate for the information loss during the forward and backward of full spike propagation, and facilitate the training of the FSNN. In the test phase, the AAP could be removed and only the FSNN remained. This not only keeps the lower energy consumption but also makes our model easy to deploy. Moreover, for some cases where the non-spike operations are available, the APP could also be retained in test inference and improve feature discrimination by introducing a little non-spike consumption. Extensive experiments on ImageNet, DVS Gesture, and CIFAR10-DVS datasets demonstrate the effectiveness of DST.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Self-adaptive threshold neuron information processing method, self-adaptive leakage value neuron information processing method, system computer device and readable storage medium</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://patents.google.com/patent/US11551074B2/en" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The disclosure relates to a self-adaptive leakage value neuron information processing method and system. The method includes: receiving front end pulse neuron output information; reading current pulse neuron information, wherein the current pulse neuron information includes self (Continued)
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Task-prompt generalised world model in multi-environment offline reinforcement learning</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ebooks.iospress.nl/doi/10.3233/FAIA230588" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Offline reinforcement learning (RL) circumvents costly interactions with the environment by utilising historical trajectories. Incorporating a world model into this method could substantially enhance the transfer performance of various tasks without expensive calculations from scratch. However, due to the complexity arising from different types of generalisation, previous works have focused almost exclusively on single-environment tasks. In this study, we introduce a multi-environment offline RL setting to investigate whether a generalised world model can be learned from large, diverse datasets and serve as a good surrogate for policy learning in different tasks. Inspired by the success of multi-task prompt methods, we propose the Task-prompt Generalised World Model (TGW) framework, which demonstrates notable performance in this setting. TGW comprises three modules: a task-state prompter, a generalised …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Inherent redundancy in spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Inherent_Redundancy_in_Spiking_Neural_Networks_ICCV_2023_paper.html?ref=https://coder.social" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) are well known as a promising energy-efficient alternative to conventional artificial neural networks. Subject to the preconceived impression that SNNs are sparse firing, the analysis and optimization of inherent redundancy in SNNs have been largely overlooked, thus the potential advantages of spike-based neuromorphic computing in accuracy and energy efficiency are interfered. In this work, we pose and focus on three key questions regarding the inherent redundancy in SNNs. We argue that the redundancy is induced by the spatio-temporal invariance of SNNs, which enhances the efficiency of parameter utilization but also invites lots of noise spikes. Further, we analyze the effect of spatio-temporal invariance on the spatio-temporal dynamics and spike firing of SNNs. Then, motivated by these analyses, we propose an Advance Spatial Attention (ASA) module to harness SNNs&#x27; redundancy, which can adaptively optimize their membrane potential distribution by a pair of individual spatial attention sub-modules. In this way, noise spike features are accurately regulated. Experimental results demonstrate that the proposed method can significantly drop the spike firing with better performance than state-of-the-art baselines. Our code is available in https://github. com/BICLab/ASA-SNN.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Deep directly-trained spiking neural networks for object detection</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://openaccess.thecvf.com/content/ICCV2023/html/Su_Deep_Directly-Trained_Spiking_Neural_Networks_for_Object_Detection_ICCV_2023_paper.html?ref=https://githubhelp.com" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are brain-inspired energy-efficient models that encode information in spatiotemporal dynamics. Recently, deep SNNs trained directly have shown great success in achieving high performance on classification tasks with very few time steps. However, how to design a directly-trained SNN for the regression task of object detection still remains a challenging problem. To address this problem, we propose EMS-YOLO, a novel directly-trained SNN framework for object detection, which is the first trial to train a deep SNN with surrogate gradients for object detection rather than ANN-SNN conversion strategies. Specifically, we design a full-spike residual block, EMS-ResNet, which can effectively extend the depth of the directly-trained SNN with low power consumption. Furthermore, we theoretically analyze and prove the EMS-ResNet could avoid gradient vanishing or exploding. The results demonstrate that our approach outperforms the state-of-the-art ANN-SNN conversion methods (at least 500 time steps) in extremely fewer time steps (only 4 time steps). It is shown that our model could achieve comparable performance to the ANN with the same architecture while consuming 5.83 x less energy on the frame-based COCO Dataset and the event-based Gen1 Dataset. Our code is available in https://github. com/BICLab/EMS-YOLO.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Active learning for name entity recognition with external knowledge</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://dl.acm.org/doi/abs/10.1145/3593023" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Named Entity Recognition (NER) is an important task in knowledge extraction, which targets extracting structural information from unstructured text. To fully employ the prior-knowledge of the pre-trained language models, some research works formulate the NER task into the machine reading comprehension form (MRC-form) to enhance their model generalization capability of commonsense knowledge. However, this transformation still faces the data-hungry issue with limited training data for the specific NER tasks. To address the low-resource issue in NER, we introduce a method named active multi-task-based NER (AMT-NER), which is a two-stage multi-task active learning training model. Specifically, A multi-task learning module is first introduced into AMT-NER to improve its representation capability in low-resource NER tasks. Then, a two-stage training strategy is proposed to optimize AMT-NER multi-task …
</p>

</div>
</details>

