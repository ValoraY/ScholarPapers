## 📑 Huajin Tang Papers

论文按年份分组（点击年份或空白区域可展开/折叠该年份的论文）


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2025</summary>

<div class="paper-card">

<h3 class="paper-title">Rethink the motor cortical control via the experiment-analysis-model flywheel: an overview</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://link.springer.com/article/10.1007/s44258-025-00059-1" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The ability to plan and execute movements is a critical brain function. Understanding the neural mechanisms of motor control, especially motor cortical control, has been a central focus in neuroscience research. Here we review recent research on motor cortex through the lens of the experiment-analysis-model flywheel, a virtuous cycle that promotes the development of this field. We summarize experiments that gather large-scale neural data, computational methods that analyze this data to yield new insights, and computational models that explain these insights and motivate further experiments. Each component of the flywheel drives the others, forming a self-reinforcing cycle of discovery and innovation. Additionally, we discuss efforts that leverage findings from motor cortical control to develop high-performance brain-computer interfaces. In summary, the experiment-analysis-model flywheel not only promotes the …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking neural networks with temporal attention-guided adaptive fusion for imbalanced multi-modal learning</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://dl.acm.org/doi/abs/10.1145/3746027.3755622" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Multimodal spiking neural networks (SNNs) hold significant potential for energy-efficient sensory processing but face critical challenges in modality imbalance and temporal misalignment. Current approaches suffer from uncoordinated convergence speeds across modalities and static fusion mechanisms that ignore time-varying cross-modal interactions. We propose the temporal attention-guided adaptive fusion framework for multimodal SNNs with two synergistic innovations: 1) The Temporal Attention-guided Adaptive Fusion (TAAF) module that dynamically assigns importance scores to fused spiking features at each timestep, enabling hierarchical integration of temporally heterogeneous spike-based features; 2) The temporal adaptive balanced fusion loss that modulates learning rates per modality based on the above attention scores, preventing dominant modalities from monopolizing optimization. The proposed …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">MSER: Multi-scale event representation model for enhanced spatio-temporal feature extraction</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0925231225024452" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras, designed to mimic biological perception principles, are a new type of visual sensor offering advantages such as low latency, low power consumption, and high dynamic range. However, the non-uniform, discontinuous nature of their spatially sparse yet temporally dense data presents significant challenges for extracting meaningful spatio-temporal features. Representing such novel data paradigms to provide high-quality inputs for neural network models remains a formidable task. Consequently, a key consideration is how to extract spatio-temporal features from event streams effectively. Currently, most work on event data typically performs feature extraction at either a single temporal or spatial scale, which often struggles to fully utilize the complex spatio-temporal relationships, leading to inaccurate capture of dynamic changes. Inspired by the brain’s multi-scale processing mechanisms, we propose …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">DarwinWafer: A Wafer-Scale Neuromorphic Chip</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2509.16213" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Neuromorphic computing promises brain-like efficiency, yet today's multi-chip systems scale over PCBs and incur orders-of-magnitude penalties in bandwidth, latency, and energy, undermining biological algorithms and system efficiency. We present DarwinWafer, a hyperscale system-on-wafer that replaces off-chip interconnects with wafer-scale, high-density integration of 64 Darwin3 chiplets on a 300 mm silicon interposer. A GALS NoC within each chiplet and an AER-based asynchronous wafer fabric with hierarchical time-step synchronization provide low-latency, coherent operation across the wafer. Each chiplet implements 2.35 M neurons and 0.1 B synapses, yielding 0.15 B neurons and 6.4 B synapses per wafer.At 333 MHz and 0.8 V, DarwinWafer consumes ~100 W and achieves 4.9 pJ/SOP, with 64 TSOPS peak throughput (0.64 TSOPS/W). Realization is enabled by a holistic chiplet-interposer co-design flow (including an in-house interposer-bump planner with early SI/PI and electro-thermal closure) and a warpage-tolerant assembly that fans out I/O via PCBlets and compliant pogo-pin connections, enabling robust, demountable wafer-to-board integration. Measurements confirm 10 mV supply droop and a uniform thermal profile (34-36 °C) under ~100 W. Application studies demonstrate whole-brain simulations: two zebrafish brains per chiplet with high connectivity fidelity (Spearman r = 0.896) and a mouse brain mapped across 32 chiplets (r = 0.645). To our knowledge, DarwinWafer represents a pioneering demonstration of wafer-scale neuromorphic computing, establishing a viable and scalable path toward large-scale, brain-like computation on silicon by replacing PCB-level interconnects with high-density, on-wafer integration.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Target tracking method and system of spiking neural network based on event camera</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://patents.google.com/patent/US12401885B2/en" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
A target tracking method and a target tracking system of a spiking neural network based on an event camera are provided. The method includes: acquiring a data stream of asynchronous events in a high dynamic scene of a target by an event camera as input data; dividing the data stream of the asynchronous events into synchronous event frames with millisecond time resolution; training a twin network based on a spiking neural network by a gradient substitution algorithm with a target image as a template image and a complete image as a searched image; and tracking the target by a trained twin network with interpolating a result of feature mapping to up-sample and obtaining the position of the target in an original image. The twin network includes a feature extractor and a cross-correlation calculator.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">EDyGS: Event Enhanced Dynamic 3D Radiance Fields from Blurry Monocular Video</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1781.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The task of generating novel views in dynamic scenes plays a critical role in the 3D vision domain. Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) have shown great promise in this domain but struggle with motion blur, which often arises in real-world scenarios due to camera or object motion. Existing methods address camera motion blur but fall short in dynamic scenes, where the coupling of camera and object motion complicates multi-view consistency and temporal coherence. In this work, we propose EDyGS, a model designed to reconstruct sharp novel views from event streams and monocular videos of dynamic scenes with motion blur. Our approach introduces a motion-mask 3D Gaussian model that assigns each Gaussian an additional attribute to distinguish between static and dynamic regions. By leveraging this motion mask field, we separate and optimize the static and dynamic regions independently. A progressive learning strategy is adopted, where static regions are reconstructed by jointly optimizing camera poses and learnable 3D Gaussians, while dynamic regions are modeled using an implicit deformation field alongside learnable 3D Gaussians. We conduct both quantitative and qualitative experiments on synthetic and real-world data. Experimental results demonstrate that EDyGS effectively handles blurry inputs in dynamic scenes.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Task-structured Modularity Emerges in Artificial Networks and Aligns with Brain Architecture</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.researchsquare.com/article/rs-6987147/latest" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Understanding how neural systems develop modular organization is fundamental to both neuroscience and artificial intelligence. Although modular architectures can improve adaptability and cognitive performance, the processes leading to their emergence are poorly understood. Here we demonstrate that multitask and incremental learning enhance modularity in recurrent neural networks (RNNs) compared to single-task learning, revealing how functional demands influence the structural organization of neural networks. We trained RNNs on cognitive tasks under three distinct learning paradigms: single-task, simultaneous multitask, and incremental multitask learning. Our results suggest that networks adopting multitask learning show an enhanced degree of modularity compared to single-task training, especially when the task load exceeds the network's representation capacity. Those trained with incremental …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">NeuroSimWorm: A multisensory framework for modeling and simulating neural circuits of Caenorhabditis elegans</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0925231225007271" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Biological behaviors emerge from the dynamic interplay among the inner neurodynamic system, embodied mechanical structure, and external environmental inputs. Nonetheless, existing approaches simply consider the static brain model that cannot fully exploit the potential of continuous interaction and feedback from the body and the environment. To address these problems, we introduce NeuroSimWorm, a multisensory closed-loop neural circuit simulation approach of the widely studied organism, Caenorhabditis elegans (C. elegans). The full closed-loop simulation platform integrates four key subcomponents including Environment, Neural Computing, Biomechanical Model, and Visualization modules. We initially define multiple sensory environments for chemical, mechanical, and thermal stimuli. Subsequently, we construct four types of neural circuits including Locomotion, Chemosensation, Thermosensation …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Rapid Memory Encoding in a Spiking Hippocampus Circuit Model</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://direct.mit.edu/neco/article/doi/10.1162/neco_a_01762/131056" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Memory is a complex process in the brain that involves the encoding, consolidation, and retrieval of previously experienced stimuli. The brain is capable of rapidly forming memories of sensory input. However, applying the memory system to real-world data poses challenges in practical implementation. This article demonstrates that through the integration of sparse spike pattern encoding scheme population tempotron, and various spike-timing-dependent plasticity (STDP) learning rules, supported by bounded weights and biological mechanisms, it is possible to rapidly form stable neural assemblies of external sensory inputs in a spiking neural circuit model inspired by the hippocampal structure. The model employs neural ensemble module and competitive learning strategies that mimic the pattern separation mechanism of the hippocampal dentate gyrus (DG) area to achieve nonoverlapping sparse coding. It …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">HSRL: A Hierarchical Control System Based on Spiking Deep Reinforcement Learning for Robot Navigation</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11128063/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Reinforcement Learning (RL) has shown promise in robotic navigation tasks, yet applying it to real-world environments remains challenging due to dynamic complexities and the need for dynamically feasible actions. We propose a hierarchical control framework based on Spiking Deep Reinforcement Learning (SDRL) for robust robot navigation in real environments. Our approach utilizes a two-layer architecture: a high-level decision layer powered by a Spiking GRU network for handling partially observable environments, and a low-level executive layer employing Continuous Attractor Neural Networks (CANNs) to ensure precise and continuous actions. This hierarchical structure allows real-time decisionmaking that respects the physical constraints of the robot. Experimental results show that our method adapts effectively to new environments without fine-tuning and surpasses existing methods in performance. We …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Brain-Inspired Spatial Continuous State Encoding for Efficient Spiking-Based Navigation</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11128824/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) show great potential in mapless navigation tasks due to their low power consumption, but the continuous representation of spatial information poses a challenge to SNN training. Neuroscience findings reveal that spatial cognition cells encode spatial information through population spike patterns. Inspired by this, we propose a navigation method based on SNNs, leveraging spatial cognition cells, which include grid cells (GCs), head direction cells (HDCs), and boundary vector cells (BVCs). Our method integrates spike-based information to achieve precise navigation goal encoding and egocentric environment perception, significantly improving SNN navigation capabilities in complex environments. Simulation and real-world experiments demonstrate that our method achieves significant improvements in navigation success rate and energy efficiency, showcasing superior adaptability …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">ASRC-SNN: Adaptive Skip Recurrent Connection Spiking Neural Network</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2505.11455" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In recent years, Recurrent Spiking Neural Networks (RSNNs) have shown promising potential in long-term temporal modeling. Many studies focus on improving neuron models and also integrate recurrent structures, leveraging their synergistic effects to improve the long-term temporal modeling capabilities of Spiking Neural Networks (SNNs). However, these studies often place an excessive emphasis on the role of neurons, overlooking the importance of analyzing neurons and recurrent structures as an integrated framework. In this work, we consider neurons and recurrent structures as an integrated system and conduct a systematic analysis of gradient propagation along the temporal dimension, revealing a challenging gradient vanishing problem. To address this issue, we propose the Skip Recurrent Connection (SRC) as a replacement for the vanilla recurrent structure, effectively mitigating the gradient vanishing problem and enhancing long-term temporal modeling performance. Additionally, we propose the Adaptive Skip Recurrent Connection (ASRC), a method that can learn the skip span of skip recurrent connection in each layer of the network. Experiments show that replacing the vanilla recurrent structure in RSNN with SRC significantly improves the model's performance on temporal benchmark datasets. Moreover, ASRC-SNN outperforms SRC-SNN in terms of temporal modeling capabilities and robustness.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Hybrid Spiking Vision Transformer for Object Detection with Event Cameras</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2505.07715" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event-based object detection has gained increasing attention due to its advantages such as high temporal resolution, wide dynamic range, and asynchronous address-event representation. Leveraging these advantages, Spiking Neural Networks (SNNs) have emerged as a promising approach, offering low energy consumption and rich spatiotemporal dynamics. To further enhance the performance of event-based object detection, this study proposes a novel hybrid spike vision Transformer (HsVT) model. The HsVT model integrates a spatial feature extraction module to capture local and global features, and a temporal feature extraction module to model time dependencies and long-term patterns in event sequences. This combination enables HsVT to capture spatiotemporal features, improving its capability to handle complex event-based object detection tasks. To support research in this area, we developed and publicly released The Fall Detection Dataset as a benchmark for event-based object detection tasks. This dataset, captured using an event-based camera, ensures facial privacy protection and reduces memory usage due to the event representation format. We evaluated the HsVT model on GEN1 and Fall Detection datasets across various model sizes. Experimental results demonstrate that HsVT achieves significant performance improvements in event detection with fewer parameters.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">The roles of internal dynamics and proprioceptive feedback in motor cortex during movement execution</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0925231225002231" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The motor cortex controls arm movements by sending commands to lower motor centers. The synaptic connections within the motor cortex (internal dynamics) are vital to generate motor commands during movement execution. However, recent studies suggest that proprioception also contribute to motor cortex processing. To investigate the contributions of internal dynamics and proprioceptive feedback to movement execution, we built a recurrent neural network model of the motor cortex; the model receives proprioceptive feedback from a virtual arm performing a delayed-reach task. We found that both internal dynamics and proprioceptive feedback contribute to the resemblance to the real motor cortex data. We then dissected their contributions by disrupting them separately. Internal dynamics dominate the generation of neural population activity, while proprioceptive feedback modulates neural responses and controls movement deceleration. Additionally, proprioceptive feedback improves the network’s robustness in noisy initial conditions. We further investigated the relative importance of the components in proprioceptive feedback and found that hand velocity is most important. Our findings could inform the development of neural prosthetics that can replicate the sensorimotor control of the biological system.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Ebad-gaussian: Event-driven bundle adjusted deblur gaussian splatting</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2504.10012" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
While 3D Gaussian Splatting (3D-GS) achieves photorealistic novel view synthesis, its performance degrades with motion blur. In scenarios with rapid motion or low-light conditions, existing RGB-based deblurring methods struggle to model camera pose and radiance changes during exposure, reducing reconstruction accuracy. Event cameras, capturing continuous brightness changes during exposure, can effectively assist in modeling motion blur and improving reconstruction quality. Therefore, we propose Event-driven Bundle Adjusted Deblur Gaussian Splatting (EBAD-Gaussian), which reconstructs sharp 3D Gaussians from event streams and severely blurred images. This method jointly learns the parameters of these Gaussians while recovering camera motion trajectories during exposure time. Specifically, we first construct a blur loss function by synthesizing multiple latent sharp images during the exposure time, minimizing the difference between real and synthesized blurred images. Then we use event stream to supervise the light intensity changes between latent sharp images at any time within the exposure period, supplementing the light intensity dynamic changes lost in RGB images. Furthermore, we optimize the latent sharp images at intermediate exposure times based on the event-based double integral (EDI) prior, applying consistency constraints to enhance the details and texture information of the reconstructed images. Extensive experiments on synthetic and real-world datasets show that EBAD-Gaussian can achieve high-quality 3D scene reconstruction under the condition of blurred images and event stream inputs.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">EvHDR-GS: Event-guided HDR Video Reconstruction with 3D Gaussian Splatting</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/32237" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
High Dynamic Range (HDR) video reconstruction seeks to accurately restore the extensive dynamic range present in real-world scenes and is widely employed in downstream applications. Existing methods typically operate on one or a small number of consecutive frames, which often leads to inconsistent brightness across the video due to their limited perspective on the video sequence. Moreover, supervised learning-based approaches are susceptible to data bias, resulting in reduced effectiveness when confronted with test inputs exhibiting a domain gap relative to the training data. To address these limitations, we present an event-guided HDR video reconstruction method through building 3D Gaussian Splatting (3DGS), to ensure consistent brightness imposed by 3D consistency. We introduce HDR 3D Gaussians capable of simultaneously representing HDR and low-dynamic-range (LDR) colors. Furthermore, we incorporate a learnable HDR-to-LDR transformation optimized by input event streams and LDR frames to eliminate the data bias. Experimental results on both synthetic and real-world datasets demonstrate that the proposed method achieves state-of-the-art performance.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">EvHDR-NeRF: Building High Dynamic Range Radiance Fields with Single Exposure Images and Events</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/32238" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
We present EvHDR-NeRF to recover a High Dynamic Range (HDR) radiance field from event streams and a set of Low Dynamic Range (LDR) views with single exposures. Using the EvHDR-NeRF, we can generate both novel HDR views and novel LDR views under different exposures. The key to our method is to model the new relationship between events streams and LDR images, which considers both the Camera Response Function (CRF) and exposure time. Based on this relationship, we categorize events into inter-frame events and intra-exposure. The former is utilized for building HDR radiance field and the latter is used to deblur potentially blurred images. Compared to existing methods, this method can effectively reconstruct the HDR radiance field even when the input images are degraded. Experimental results demonstrate that our method achieves state-of-the-art HDR reconstruction, providing a more adaptable and accurate solution for complex imaging applications.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">EvSTVSR: Event Guided Space-Time Video Super-Resolution</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/32983" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In the domain of space-time video super-resolution, it is typically challenging to handle complex motions (including large and nonlinear motions) and varying illumination scenes due to the lack of inter-frame information. Leveraging the dense temporal information provided by event signals offers a promising solution. Traditional event-based methods typically rely on multiple images, using motion estimation and compensation, which can introduce errors. Accumulated errors from multiple frames often lead to artifacts and blurriness in the output. To mitigate these issues, we propose EvSTVSR, a method that uses fewer adjacent frames and integrates dense temporal information from events to guide alignment. Additionally, we introduce a coordinate-based feature fusion upsampling module to achieve spatial super-resolution. Experimental results demonstrate that our method not only outperforms existing RGB-based approaches but also excels in handling large motion scenarios.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">ALADE-SNN: Adaptive Logit Alignment in Dynamically Expandable Spiking Neural Networks for Class Incremental Learning</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/34171" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Inspired by the human brain's ability to adapt to new tasks without erasing prior knowledge, we develop spiking neural networks (SNNs) with dynamic structures for Class Incremental Learning (CIL). Our analytical experiments reveal that limited datasets introduce biases in logits distributions among tasks. Fixed features from frozen past-task extractors can cause overfitting and hinder the learning of new tasks. To address these challenges, we propose the ALADE-SNN framework, which includes adaptive logit alignment for balanced feature representation and OtoN suppression to manage weights mapping frozen old features to new classes during training, releasing them during fine-tuning. This approach dynamically adjusts the network architecture based on analytical observations, improving feature extraction and balancing performance between new and old tasks. Experiment results show that ALADE-SNN achieves an average incremental accuracy of 75.42±0.74% on the CIFAR100-B0 dataset over 10 incremental steps. ALADE-SNN not only matches the performance of DNN-based methods but also surpasses state-of-the-art SNN-based continual learning algorithms. This advancement enhances continual learning in neuromorphic computing, offering a brain-inspired, energy-efficient solution for real-time data processing.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/32139" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are widely applied in various fields due to their energy-efficient and fast-inference capabilities. Applying SNNs to reinforcement learning (RL) can significantly reduce the computational resource requirements for agents and improve the algorithm's performance under resource-constrained conditions. However, in current spiking reinforcement learning (SRL) algorithms, the simulation results of multiple time steps can only correspond to a single-step decision in RL. This is quite different from the real temporal dynamics in the brain and also fails to fully exploit the capacity of SNNs to process temporal data. In order to address this temporal mismatch issue and further take advantage of the inherent temporal dynamics of spiking neurons, we propose a novel temporal alignment paradigm (TAP) that leverages the single-step update of spiking neurons to accumulate historical state information in RL and introduces gated units to enhance the memory capacity of spiking neurons. Experimental results show that our method can solve partially observable Markov decision processes (POMDPs) and multi-agent cooperation problems with similar performance as recurrent neural networks (RNNs) but with about 50\% power consumption.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Two-Stream Spiking Neural Network for Event-based Action Recognition</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10889468/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are increasingly applied to event-based data generated by event cameras due to their asynchronous and sparse properties. Event cameras can inherently respond to the changes in the scene, which is a quite desirable property for action recognition tasks. However, existing works of SNNs for event-based action recognition are still limited. To capture the rich dynamics embedded in event streams, we propose the two-stream SNN that consists of spatial spiking stream and motion spiking stream to address event-based action recognition. To effectively build the two-stream SNN, we present a motion feature aggregation strategy and an attention-based two-stream fusion method. The motion feature aggregation strategy accumulates motion information and groups it into distinct channels for input into the SNN, which can alleviate the dilemma of information loss caused by compact …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Adaptive Gradient-Based Timesurface for Event-based Detection</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10888665/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The advantages of high temporal resolution and high dynamic range provided by event cameras are particularly suitable for moving object detection, especially in scenarios with motion blur and extreme lighting conditions. Current popular methods predominantly focus on designing powerful network architectures to extract event features, often neglecting the rationality of event representation design which has been proven to impact significantly on downstream tasks. In particular, current event representations typically rely on fixed hyperparameters, without considering variations in relative motion speed, a key factor in motion-rich scenes captured by event cameras. To tackle this challenge, we propose a gradient-based scaled Timesurface (STS) motivated by the observation of the relationship between motion speeds and the gradient strength, which adaptively rescales the decay factor at different spatial positions …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Temporal spiking generative adversarial networks for heading direction decoding</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608024009043" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The spike-based neuronal responses within the ventral intraparietal area (VIP) exhibit intricate spatial and temporal dynamics in the posterior parietal cortex, presenting decoding challenges such as limited data availability at the biological population level. The practical difficulty in collecting VIP neuronal response data hinders the application of sophisticated decoding models. To address this challenge, we propose a unified spike-based decoding framework leveraging spiking neural networks (SNNs) for both generative and decoding purposes, for their energy efficiency and suitability for neural decoding tasks. We propose the Temporal Spiking Generative Adversarial Networks (T-SGAN), a model based on a spiking transformer, to generate synthetic time-series data reflecting the neuronal response of VIP neurons. T-SGAN incorporates temporal segmentation to reduce the temporal dimension length, while spatial …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Context gating in spiking neural networks: Achieving lifelong learning through integration of local and global plasticity</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0950705125000474" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Humans learn multiple tasks in succession with minimal mutual interference, through the context gating mechanism in the prefrontal cortex (PFC). The brain-inspired models of spiking neural networks (SNN) have drawn massive attention for their energy efficiency and biological plausibility. To overcome catastrophic forgetting when learning multiple tasks in sequence, current SNN models for lifelong learning focus on memory reserving or regularization-based modification, while lacking SNN to replicate human experimental behavior. Inspired by biological context-dependent gating mechanisms found in PFC, we propose SNN with context gating trained by the local plasticity rule (CG-SNN) for lifelong learning. The iterative training between global and local plasticity for task units is designed to strengthen the connections between task neurons and hidden neurons and preserve the multi-task relevant information. The …
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

<h3 class="paper-title">2025 New Year Message From the Editor-in-Chief</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10877686/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
As a journal dedicated to interdisciplinary research, TCDS has continuously encompassed emerging scientific frontiers. Building on its core areas such as cognitive systems, computational intelligence, cognitive robotics, and computational neuroscience, TCDS has embraced more cutting-edge interdisciplinary topics. These include brain-inspired artificial intelligence (AI), neuromorphic computing, neurocognitive robotics, and multiagent systems. These expansions have deepened academic inquiry, enhanced diversity and inclusivity, and established TCDS as a premier platform for interdisciplinary research in AI, neuroscience, robotics, etc.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Bio-plausible reconfigurable spiking neuron for neuromorphic computing</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://www.science.org/doi/abs/10.1126/sciadv.adr6733" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Biological neurons use diverse temporal expressions of spikes to achieve efficient communication and modulation of neural activities. Nonetheless, existing neuromorphic computing systems mainly use simplified neuron models with limited spiking behaviors due to high cost of emulating these biological spike patterns. Here, we propose a compact reconfigurable neuron design using the intrinsic dynamics of a NbO2-based spiking unit and excellent tunability in an electrochemical memory (ECRAM) to emulate the fast-slow dynamics in a bio-plausible neuron. The resistance of the ECRAM was effective in tuning the temporal dynamics of the membrane potential, contributing to flexible reconfiguration of various bio-plausible firing modes, such as phasic and burst spiking, and exhibiting adaptive spiking behaviors in changing environment. We used the bio-plausible neuron model to build spiking neural networks with …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Object recognition method and apparatus</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://patentimages.storage.googleapis.com/8e/7a/66/b9dd62ed704deb/US12217477.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In an object recognition method, an object recognition device obtains AER data of a to-be-recognized object, wherein the AER data includes a plurality of AER events of the to-be-recognized object, each AER event comprising a timestamp and address information. The object recognition device extracts a plurality of feature maps of the AER data. Each feature map including partial spatial information and partial temporal information of the to-be-recognized object, and the partial spatial information and the partial temporal information are obtained based on the timestamp and the address information of each AER event. The object recognition device then recognizes the to-be-recognized object based on the plurality of feature maps of the AER data.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">STSF: Spiking Time Sparse Feedback Learning for Spiking Neural Networks</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10847298/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are biologically plausible models known for their computational efficiency. A significant advantage of SNNs lies in the binary information transmission through spike trains, eliminating the need for multiplication operations. However, due to the spatio-temporal nature of SNNs, direct application of traditional backpropagation (BP) training still results in significant computational costs. Meanwhile, learning methods based on unsupervised synaptic plasticity provide an alternative for training SNNs but often yield suboptimal results. Thus, efficiently training high-accuracy SNNs remains a challenge. In this article, we propose a highly efficient and biologically plausible spiking time sparse feedback (STSF) learning method. This algorithm modifies synaptic weights by incorporating a neuromodulator for global supervised learning using sparse direct feedback alignment (DFA) and local …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">CIS Publication Spotlight [Publication Spotlight]</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/iel8/10207/10844565/10844372.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
This exposition discusses continuous-time reinforcement learning (CT-RL) for the control of affine nonlinear systems. We review four seminal methods that are the centerpieces of the most recent results on CT-RL control. We survey the theoretical results of the four methods, highlighting their fundamental importance and successes by including discussions on problem formulation, key assumptions, algorithm procedures, and theoretical guarantees. Subsequently, we evaluate the performance of the control designs to provide analyses and insights on the feasibility of these design methods for applications from a control designer's point of view. Through systematic evaluations, we point out when theory diverges from practical controller synthesis. We, furthermore, introduce a new quantitative analytical framework to diagnose the observed discrepancies. Based on the analyses and the insights gained through …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">E-NeMF: Event-based Neural Motion Field for Novel Space-time View Synthesis of Dynamic Scenes</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://openaccess.thecvf.com/content/ICCV2025/html/Liu_E-NeMF_Event-based_Neural_Motion_Field_for_Novel_Space-time_View_Synthesis_ICCV_2025_paper.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Synthesizing novel space-time views from a monocular video is a highly ill-posed problem, and its effectiveness relies on accurately reconstructing motion and appearance of the dynamic scene. Frame-based methods for novel space-time view synthesis in dynamic scenes rely on simplistic motion assumptions due to the absence of inter-frame cues, which makes them fall in complex motion. Event camera captures inter-frame cues with high temporal resolution, which makes it hold the promising potential to handle high speed and complex motion. However, it is still difficult due to the event noise and sparsity. To mitigate the impact caused by event noise and sparsity, we propose E-NeMF, which alleviates the impact of event noise with Parametric Motion Representation and mitigates the event sparsity with Flow Prediction Module. Experiments on multiple real-world datasets demonstrate our superior performance in handling high-speed and complex motion.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">DarwinSync: An adaptive time step execution framework for large‐scale neuromorphic systems</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/ell2.70153" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The time step functions as a crucial temporal unit for simulating neuronal dynamics within spiking neural networks, which play a significant role in neuromorphic computing systems. Efficient management of these time steps is vital to ensure model accuracy while optimizing overall system performance. As system scale increases, variations in hardware across subsystems and their asynchronous operations create challenges in achieving effective time step control. To address this issue, this paper proposes an innovative framework for managing time steps in large‐scale neuromorphic systems. This framework allows subsystems to dynamically adjust their time step lengths according to computational loads and to perform look‐ahead computations. Such a strategy effectively reduces the overhead related to time step synchronization, enhancing system efficiency. Additionally, the paper introduces a safeguard …
</p>

</div>
</details>


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2024</summary>

<div class="paper-card">

<h3 class="paper-title">Darkit: A User-Friendly Software Toolkit for Spiking Large Language Model</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2412.15634" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Large language models (LLMs) have been widely applied in various practical applications, typically comprising billions of parameters, with inference processes requiring substantial energy and computational resources. In contrast, the human brain, employing bio-plausible spiking mechanisms, can accomplish the same tasks while significantly reducing energy consumption, even with a similar number of parameters. Based on this, several pioneering researchers have proposed and implemented various large language models that leverage spiking neural networks. They have demonstrated the feasibility of these models, validated their performance, and open-sourced their frameworks and partial source code. To accelerate the adoption of brain-inspired large language models and facilitate secondary development for researchers, we are releasing a software toolkit named DarwinKit (Darkit). The toolkit is designed specifically for learners, researchers, and developers working on spiking large models, offering a suite of highly user-friendly features that greatly simplify the learning, deployment, and development processes.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">FEEL-SNN: Robust spiking neural networks with frequency encoding and evolutionary leak factor</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/a73474c359ed523e6cd3174ed29a4d56-Abstract-Conference.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Currently, researchers think that the inherent robustness of spiking neural networks (SNNs) stems from their biologically plausible spiking neurons, and are dedicated to developing more bio-inspired models to defend attacks. However, most work relies solely on experimental analysis and lacks theoretical support, and the direct-encoding method and fixed membrane potential leak factor they used in spiking neurons are simplified simulations of those in the biological nervous system, which makes it difficult to ensure generalizability across all datasets and networks. Contrarily, the biological nervous system can stay reliable even in a highly complex noise environment, one of the reasons is selective visual attention and non-fixed membrane potential leaks in biological neurons. This biological finding has inspired us to design a highly robust SNN model that closely mimics the biological nervous system. In our study, we first present a unified theoretical framework for SNN robustness constraint, which suggests that improving the encoding method and evolution of the membrane potential leak factor in spiking neurons can improve SNN robustness. Subsequently, we propose a robust SNN (FEEL-SNN) with Frequency Encoding (FE) and Evolutionary Leak factor (EL) to defend against different noises, mimicking the selective visual attention mechanism and non-fixed leak observed in biological systems. Experimental results confirm the efficacy of both our FE, EL, and FEEL methods, either in isolation or in conjunction with established robust enhancement algorithms, for enhancing the robustness of SNNs.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Solving Expensive Dynamic Multi-objective Problem via Cross-Problem Knowledge Transfer</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://link.springer.com/chapter/10.1007/978-981-96-7036-9_18" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Expensive Dynamic Multi-Objective Optimization Problems (EXDMOPs) pose significant challenges due to their dynamic and costly nature, as both the objective and constraint functions evolve over time with limited fitness evaluations. Traditional methods treat EXDMOPs as multiple independent and static expensive multi-objective problems, which often ignore the experiences obtained in solving the EXDMOPs or reuse experiences within a single EXDMOP, resulting in inefficient optimization performance. Taking this cue, in this paper, we introduce a novel approach leveraging cross-problem knowledge to enhance the ability to solve EXDMOPs. Unlike existing knowledge transfer methods within a single dynamic problem, our proposed method leverages archived solutions from well-optimized tasks to construct an effective initial population through cross-problem task selection and knowledge transfer. Specifically …
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

<h3 class="paper-title">Enhancing snn-based spatio-temporal learning: A benchmark dataset and cross-modality attention model</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608024006014" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs), renowned for their low power consumption, brain-inspired architecture, and spatio-temporal representation capabilities, have garnered considerable attention in recent years. Similar to Artificial Neural Networks (ANNs), high-quality benchmark datasets are of great importance to the advances of SNNs. However, our analysis indicates that many prevalent neuromorphic datasets lack strong temporal correlation, preventing SNNs from fully exploiting their spatio-temporal representation capabilities. Meanwhile, the integration of event and frame modalities offers more comprehensive visual spatio-temporal information. Yet, the SNN-based cross-modality fusion remains underexplored.In this work, we present a neuromorphic dataset called DVS-SLR that can better exploit the inherent spatio-temporal properties of SNNs. Compared to existing datasets, it offers advantages in terms of …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Video-Based Respiratory Monitoring System for Inactive Non-Human Primates</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10920540/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Non-human primates (NHPs) are critical in neuroscience research, where researchers implant electrodes into their brains to observe neuronal activity. However, the high post-operative mortality rate of NHPs remains a significant concern, primarily due to inadequate post-surgical monitoring. Thus, it is essential to develop a monitoring system for assessing their physiological status. Conventional wearable monitoring devices are often removed by NHPs, making it challenging to monitor their well-being. To address this issue, we propose a video-based monitoring system that utilizes Eulerian video magnification techniques and principal component analysis to detect and estimate the respiratory rate remotely. The results show that our method has a difference of less than 1 breathing rate per minute compared to a commercial monitor and works well in experiments involving different species, demonstrating the …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Event-ID: Intrinsic Decomposition Using an Event Camera</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://dl.acm.org/doi/abs/10.1145/3664647.3681133" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Reconstructing 3D scenes from multi-view images is challenging, especially under extreme scenarios. We propose Event-ID, an event-based intrinsic decomposition framework that leverages events and images for stable decomposition under extreme scenarios. Our method is based on two observations: event cameras maintain good imaging quality under blurry or poorly exposed scenarios, and event signals from different viewpoints exhibit similarity in diffuse regions while varying in specular regions. We establish an event-based reflectance model and introduce an event-based warping method to extract specular clues. Our two-stage framework constructs a radiance field and decomposes the scene into normal, material, and lighting. Experimental results demonstrate superior performance compared to state-of-the-art methods. Our project can be found at https://zehaoc.github.io/EventID.github.io/
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Casrl: Collision avoidance with spiking reinforcement learning among dynamic, decision-making agents</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10802416/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Developing an efficient collision avoidance policy with Spiking Reinforcement Learning for dynamic, decision-making agents remains challenging. Moreover, the implementation of energy-efficient collision avoidance is important for mobile robots that operate with limited on-board computing resources. Most existing energy-efficient methods via spiking reinforcement learning are predominately concerned with the navigational capabilities of a single agent, and are unable to handle a large, and possibly varying number of agents. To overcome these limitations, we propose a model called collision avoidance with spiking reinforcement learning (CASRL), based on proximal policy optimization algorithms. This proposed model consists of an actor with spiking neural networks (SNNs) and a critic with deep neural networks (DNNs). Our spiking reinforcement learning algorithm is advantageous to handle an arbitrary …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking gs: Towards high-accuracy and low-cost surface reconstruction via spiking neuron-based gaussian splatting</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2410.07266" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
3D Gaussian Splatting is capable of reconstructing 3D scenes in minutes. Despite recent advances in improving surface reconstruction accuracy, the reconstructed results still exhibit bias and suffer from inefficiency in storage and training. This paper provides a different observation on the cause of the inefficiency and the reconstruction bias, which is attributed to the integration of the low-opacity parts (LOPs) of the generated Gaussians. We show that LOPs consist of Gaussians with overall low-opacity (LOGs) and the low-opacity tails (LOTs) of Gaussians. We propose Spiking GS to reduce such two types of LOPs by integrating spiking neurons into the Gaussian Splatting pipeline. Specifically, we introduce global and local full-precision integrate-and-fire spiking neurons to the opacity and representation function of flattened 3D Gaussians, respectively. Furthermore, we enhance the density control strategy with spiking neurons' thresholds and a new criterion on the scale of Gaussians. Our method can represent more accurate reconstructed surfaces at a lower cost. The supplementary material and code are available at https://github.com/zju-bmi-lab/SpikingGS.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Understanding and bridging the gap between neuromorphic computing and machine learning, volume II</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1455530/full" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Pursuing intelligence is a long-term goal of the human, towards which two routes have been paved on the road: neuromorphic computing driven by neuroscience and machine learning driven by computer science (Pei et al., 2019). Spiking neural networks (SNNs) and neuromorphic chips (Basu et al., 2022;Christensen et al., 2022) dominate the neuromorphic computing domain, while artificial neural networks (ANNs) and machine learning accelerators (Deng et al., 2020) dominate the machine learning domain. Neuromorphic computing with efficient models and hardware has shown energy efficiency superiority (Renner et al., 2021), however, still lies in its infant stage and presents a gap in terms of accuracy and applications compared to the mature machine learning ecosystem.To this end, we proposed a Research Topic, named "Understanding and Bridging the Gap between Neuromorphic Computing and Machine Learning", in Frontiers in Neuroscience and Frontiers in Computational Neuroscience in 2019, and have successfully published 14 papers on neuromorphic computing and machine learning (Deng et al., 2021). Encouraged by such positive impetus for the neuromorphic computing community, we relaunched the Research Topic in 2022.This time, we have accepted 11 submissions in the end. The scope of these works covers neuromorphic models and algorithms, hardware implementation, and programming frameworks.SNNs encode information in spike events and process information using neural dynamics, which differ from ANNs. Due to the complicated spatiotemporal dynamics and non-differentiable spike activities, the SNN …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Point-of-interest recommendation method and system based on brain-inspired spatiotemporal perceptual representation</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://patents.google.com/patent/US20240330690A1/en" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
A POI recommendation method and system based on brain-inspired spatiotemporal perceptual representation is provided. The method includes: constructing a POI context graph structure based on a POI check-in dataset; sampling a check-in sequence context graph, and training a POI check-in sequence embedding model in a brain-inspired spatiotemporal perceptual embedding model by unsupervised learning; sampling a spatial context graph and a spatiotemporal context graph to train a spatiotemporal embedding model in a brain-inspired spatiotemporal perceptual embedding model; combining a POI sequence representation vector and a POI spatiotemporal union representation vector into a POI spatiotemporal perceptual representation vector; training a recurrent neural network recommender based on the POI spatiotemporal perceptual representation vector; and recommending a next POI through the …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Reliable object tracking by multimodal hybrid feature extraction and transformer-based fusion</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608024004179" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Visual object tracking, which is primarily based on visible light image sequences, encounters numerous challenges in complicated scenarios, such as low light conditions, high dynamic ranges, and background clutter. To address these challenges, incorporating the advantages of multiple visual modalities is a promising solution for achieving reliable object tracking. However, the existing approaches usually integrate multimodal inputs through adaptive local feature interactions, which cannot leverage the full potential of visual cues, thus resulting in insufficient feature modeling. In this study, we propose a novel multimodal hybrid tracker (MMHT) that utilizes frame-event-based data for reliable single object tracking. The MMHT model employs a hybrid backbone consisting of an artificial neural network (ANN) and a spiking neural network (SNN) to extract dominant features from different visual modalities and then uses …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Eas-snn: End-to-end adaptive sampling and representation for event-based detection with recurrent spiking neural networks</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://link.springer.com/chapter/10.1007/978-3-031-73027-6_18" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras, with their high dynamic range and temporal resolution, are ideally suited for object detection in scenarios with motion blur and challenging lighting conditions. However, while most existing approaches prioritize optimizing spatiotemporal representations with advanced detection backbones and early aggregation functions, the crucial issue of adaptive event sampling remains largely unaddressed. Spiking Neural Networks (SNNs), operating on an event-driven paradigm, align closely with the behavior of an ideal temporal event sampler. Motivated by this, we propose a novel adaptive sampling module that leverages recurrent convolutional SNNs enhanced with temporal memory, facilitating a fully end-to-end learnable framework for event-based detection. Additionally, we introduce Residual Potential Dropout (RPD) and Spike-Aware Training (SAT) to regulate potential distribution and address …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Neuromorphic auditory perception by neural spiketrum</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10683976/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Neuromorphic computing holds the promise to achieve the energy efficiency and robust learning performance of biological neural systems. To realize the promised brain-like intelligence, it needs to solve the challenges of the neuromorphic hardware architecture design of biological neural substrate and the hardware amicable algorithms with spike-based encoding and learning. Here we introduce a neural spike coding model termed spiketrum, to characterize and transform the time-varying analog signals, typically auditory signals, into computationally efficient spatiotemporal spike patterns. It minimizes the information loss occurring at the analog-to-spike transformation and possesses informational robustness to neural fluctuations and spike losses. The model provides a sparse and efficient coding scheme with precisely controllable spike rate that facilitates training of spiking neural networks in various auditory …
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

<h3 class="paper-title">CLFR-M: Continual learning framework for robots via human feedback and dynamic memory</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10672832/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Robots working in dynamic real-world environments need continual learning to adapt to changing situations and challenges. Traditional robotics learning methods lack robustness and transferability and thus are inefficient for complex open-ended robot tasks in dynamic environments. In recent years, large language models(LLMs) have become one of the most promising schemes for robot planning tasks thanks to their generalization and convenience in different tasks. However, LLMs-planners are not good at continuously learning from long-term experience. To address this problem, we present a continual learning framework for robots using LLMs via human feedback and dynamic memory (CLFR-M), which continuously improves robots’ behavior without additional training or intricate finetuning. We have built a persistent, dynamic embedding memory and practical knowledge structure to record and organize …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Emergence and reconfiguration of modular structure for artificial neural networks during continual familiarity detection</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.science.org/doi/abs/10.1126/sciadv.adm8430" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Advances in artificial intelligence enable neural networks to learn a wide variety of tasks, yet our understanding of the learning dynamics of these networks remains limited. Here, we study the temporal dynamics during learning of Hebbian feedforward neural networks in tasks of continual familiarity detection. Drawing inspiration from network neuroscience, we examine the network’s dynamic reconfiguration, focusing on how network modules evolve throughout learning. Through a comprehensive assessment involving metrics like network accuracy, modular flexibility, and distribution entropy across diverse learning modes, our approach reveals various previously unknown patterns of network reconfiguration. We find that the emergence of network modularity is a salient predictor of performance and that modularization strengthens with increasing flexibility throughout learning. These insights not only elucidate the …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking transfer learning from rgb image to neuromorphic event stream</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10608063/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Recent advances in bio-inspired vision with event cameras and associated spiking neural networks (SNNs) have provided promising solutions for low-power consumption neuromorphic tasks. However, as the research of event cameras is still in its infancy, the amount of labeled event stream data is much less than that of the RGB database. The traditional method of converting static images into event streams by simulation to increase the sample size cannot simulate the characteristics of event cameras such as high temporal resolution. To take advantage of both the rich knowledge in labeled RGB images and the features of the event camera, we propose a transfer learning method from the RGB to the event domain in this paper. Specifically, we first introduce a transfer learning framework named R2ETL (RGB to Event Transfer Learning), including a novel encoding alignment module and a feature alignment module …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">The balanced multi-modal spiking neural networks with online loss adjustment and time alignment</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10687515/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Optimizing multi-modal learning of SNNs has the advantages of energy efficiency and performance improvements. However, modality imbalance in multi-modal SNNs results in performance decline due to heterogeneity of modalities and temporal inconsistencies across different SNNs branches. In this paper, we propose the Balanced Multi-modal SNNs (BM-SNNs) model, equipped with a novel online loss adjustment (LA) algorithm and time alignment (TA) modules, ultimately achieving balanced training across multiple modalities. LA supervises the learning of uni-modal feature extractors by adding unimodal loss components without additional classifier. Moreover, the modulation factors enable the adaptive adjustment of unimodal learning rate. Furthermore, the proposed TA adopts the optimal timestep for different modalities to avoid information redundancy. Experimental results reveal that BM-SNNs model …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spike Neural Network of Motor Cortex Model for Arm Reaching Control</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10781802/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Motor cortex modeling is crucial for understanding movement planning and execution. While interconnected recurrent neural networks have successfully described the dynamics of neural population activity, most existing methods utilize continuous signal-based neural networks, which do not reflect the biological spike neural signal. To address this limitation, we propose a recurrent spike neural network to simulate motor cortical activity during an arm-reaching task. Specifically, our model is built upon integrate-and-fire spiking neurons with conductance-based synapses. We carefully designed the interconnections of neurons with two different firing time scales - "fast" and "slow" neurons. Experimental results demonstrate the effectiveness of our method, with the model's neuronal activity in good agreement with monkey's motor cortex data at both single-cell and population levels. Quantitative analysis reveals a …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SpikingMiniLM: Energy-efficient Spiking Transformer for Natural Language Understanding</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="http://scis.scichina.com/en/2024/200406.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In the era of large-scale pretrained models, artificial neural networks (ANNs) have excelled in natural language understanding (NLU) tasks. However, their success often necessitates substantial computational resources and energy consumption. To address this, we explore the potential of spiking neural networks (SNNs) in NLU—a promising avenue with demonstrated advantages, including reduced power consumption and improved efficiency due to their event-driven characteristics. We propose the SpikingMiniLM, a novel spiking Transformer model tailored for natural language understanding. We first introduce a multistep encoding method to convert text embeddings into spike trains. Subsequently, we redesign the attention mechanism and residual connections to make our model operate on the pure spike-based paradigm without any normalization technique. To facilitate stable and fast convergence, we propose a general parameter initialization method grounded in the stable firing rate principle. Furthermore, we apply an ANN-to-SNN knowledge distillation to overcome the challenges of pretraining SNNs. Our approach achieves a macro-average score of 75.5 on the dev sets of the GLUE benchmark, retaining 98% of the performance exhibited by the teacher model MiniLMv2. Our smaller model also achieves similar performance to BERTMINI with fewer parameters and much lower energy consumption, underscoring its competitiveness and resource efficiency in NLU tasks.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Adaptive Multi-Level Firing for Direct Training Deep Spiking Neural Networks</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10650059/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) with bio-inspired spatio-temporal dynamics, have increasingly manifested their superiority in energy efficiency. However, the non-differential spiking activity hinders the implementation of the efficient error backpropagation in training SNNs. The existing solution circumvents this problem by a relaxation on the gradient calculation using a continuous function, which is referred to as surrogate gradient learning. Nevertheless, such a solution leads to a prominent gradient mismatch problem due to the low precision of spikes, which limits the performance of directly trained SNNs on deeper architectures. To tackle this issue, we propose the adaptive multilevel firing (AMLF) method incorporating the spiking dormant-suppressed residual network (spiking DS-ResNet) to get well-behaved SNNs. The AMLF method can not only adaptively implement incremental expression ability of spiking …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">An Event-based Feature Representation Method for Event Stream Classification using Deep Spiking Neural Networks</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10650426/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event streams output by event cameras have low data redundancy and retain accurate temporal information in the form of Address Event Representation (AER) which are different from the outputs of traditional frame-based cameras. Spiking Neural Networks (SNNs) are considered an effective tool for handling event-based scenarios due to their inherent temporal properties. However, most existing SNNs directly convert an event stream to several static frames with temporal relationships by channel-wise accumulation of events. These serial frames lose temporal characteristics in some extent and potentially affect the capacity of the SNNs to learn and recognize event streams. In this work, we proposed a novel event-based feature descriptor called time interval correlation time-surface (TICTS) for SNNs and introduced this event-based feature extraction method into SNNs. The TICTS can capture more precise …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Multi-scale Harmonic Mean Time Surfaces for Event-based Object Classification</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10650679/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras have attracted increasing attention in the field of computer vision due to their advantages in terms of high temporal resolution, high dynamic range and low power consumption. However, the output of event cameras is a sparse and discrete event stream, with each individual event in the event streams containing only little information. Therefore, extracting more effective features from the available information in the event streams is currently a major challenge in event-based object classification. In this paper, we propose a novel event-based feature representation to encode the spatiotemporal features of event streams. The Harmonic Mean Time Surfaces (HMTS) representation makes efficient use of information about past events, which enhances the spatiotemporal relationship between events, thus establishing robust representation. Furthermore, we propose a multi-scale feature extraction model that …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SFedCA: Credit Assignment-Based Active Client Selection Strategy for Spiking Federated Learning</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2406.12200" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking federated learning is an emerging distributed learning paradigm that allows resource-constrained devices to train collaboratively at low power consumption without exchanging local data. It takes advantage of both the privacy computation property in federated learning (FL) and the energy efficiency in spiking neural networks (SNN). Thus, it is highly promising to revolutionize the efficient processing of multimedia data. However, existing spiking federated learning methods employ a random selection approach for client aggregation, assuming unbiased client participation. This neglect of statistical heterogeneity affects the convergence and accuracy of the global model significantly. In our work, we propose a credit assignment-based active client selection strategy, the SFedCA, to judiciously aggregate clients that contribute to the global sample distribution balance. Specifically, the client credits are assigned by the firing intensity state before and after local model training, which reflects the local data distribution difference from the global model. Comprehensive experiments are conducted on various non-identical and independent distribution (non-IID) scenarios. The experimental results demonstrate that the SFedCA outperforms the existing state-of-the-art spiking federated learning methods, and requires fewer communication rounds.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Robust sensory information reconstruction and classification with augmented spikes</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10547380/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Sensory information recognition is primarily processed through the ventral and dorsal visual pathways in the primate brain visual system, which exhibits layered feature representations bearing a strong resemblance to convolutional neural networks (CNNs), encompassing reconstruction and classification. However, existing studies often treat these pathways as distinct entities, focusing individually on pattern reconstruction or classification tasks, overlooking a key feature of biological neurons, the fundamental units for neural computation of visual sensory information. Addressing these limitations, we introduce a unified framework for sensory information recognition with augmented spikes. By integrating pattern reconstruction and classification within a single framework, our approach not only accurately reconstructs multimodal sensory information but also provides precise classification through definitive labeling …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Application based Evaluation of an Efficient Spike-Encoder," Spiketrum"</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2405.15927" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spike-based encoders represent information as sequences of spikes or pulses, which are transmitted between neurons. A prevailing consensus suggests that spike-based approaches demonstrate exceptional capabilities in capturing the temporal dynamics of neural activity and have the potential to provide energy-efficient solutions for low-power applications. The Spiketrum encoder efficiently compresses input data using spike trains or code sets (for non-spiking applications) and is adaptable to both hardware and software implementations, with lossless signal reconstruction capability. The paper proposes and assesses Spiketrum's hardware, evaluating its output under varying spike rates and its classification performance with popular spiking and non-spiking classifiers, and also assessing the quality of information compression and hardware resource utilization. The paper extensively benchmarks both Spiketrum hardware and its software counterpart against state-of-the-art, biologically-plausible encoders. The evaluations encompass benchmarking criteria, including classification accuracy, training speed, and sparsity when using encoder outputs in pattern recognition and classification with both spiking and non-spiking classifiers. Additionally, they consider encoded output entropy and hardware resource utilization and power consumption of the hardware version of the encoders. Results demonstrate Spiketrum's superiority in most benchmarking criteria, making it a promising choice for various applications. It efficiently utilizes hardware resources with low power consumption, achieving high classification accuracy. This work also emphasizes the potential of encoders in spike-based processing to improve the efficiency and performance of neural computing systems.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Darwin3: a large-scale neuromorphic chip with a novel ISA and on-chip learning</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://academic.oup.com/nsr/advance-article/doi/10.1093/nsr/nwae102/7631347" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are gaining increasing attention for their biological plausibility and potential for improved computational efficiency. To match the high spatial-temporal dynamics in SNNs, neuromorphic chips are highly desired to execute SNNs in hardware-based neuron and synapse circuits directly. This paper presents a large-scale neuromorphic chip named Darwin3 with a novel instruction set architecture, which comprises 10 primary instructions and a few extended instructions. It supports flexible neuron model programming and local learning rule designs. The Darwin3 chip architecture is designed in a mesh of computing nodes with an innovative routing algorithm. We used a compression mechanism to represent synaptic connections, significantly reducing memory usage. The Darwin3 chip supports up to 2.35 million neurons, making it the largest of its kind on the neuron scale. The …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Revealing the mechanisms of semantic satiation with deep learning models</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.nature.com/articles/s42003-024-06162-0" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The phenomenon of semantic satiation, which refers to the loss of meaning of a word or phrase after being repeated many times, is a well-known psychological phenomenon. However, the microscopic neural computational principles responsible for these mechanisms remain unknown. In this study, we use a deep learning model of continuous coupled neural networks to investigate the mechanism underlying semantic satiation and precisely describe this process with neuronal components. Our results suggest that, from a mesoscopic perspective, semantic satiation may be a bottom-up process. Unlike existing macroscopic psychological studies that suggest that semantic satiation is a top-down process, our simulations use a similar experimental paradigm as classical psychology experiments and observe similar results. Satiation of semantic objectives, similar to the learning process of our network model used for …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Trainable spiking-yolo for low-latency and high-performance object detection</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608023007530" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are considered an attractive option for edge-side applications due to their sparse, asynchronous and event-driven characteristics. However, the application of SNNs to object detection tasks faces challenges in achieving good detection accuracy and high detection speed. To overcome the aforementioned challenges, we propose an end-to-end Trainable Spiking-YOLO (Tr-Spiking-YOLO) for low-latency and high-performance object detection. We evaluate our model on not only frame-based PASCAL VOC dataset but also event-based GEN1 Automotive Detection dataset, and investigate the impacts of different decoding methods on detection performance. The experimental results show that our model achieves competitive/better performance in terms of accuracy, latency and energy consumption compared to similar artificial neural network (ANN) and conversion-based SNN object …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A high-precision LiDAR-inertial odometry via invariant extended Kalman filtering and efficient surfel mapping</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10484977/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Simultaneous localization and mapping (SLAM) via light detection and ranging (LiDAR)-inertial odometry is a crucial technology in many automated applications. However, constructing a consistent state estimator with an efficient mapping method still remains a challenge for LiDAR-inertial odometry (LIO) systems. In this article, we propose a tightly coupled LIO system via invariant extended Kalman filter (InEKF) and efficient surfel mapping. First, based on the InEKF theory, we build a consistent state estimator for a tightly coupled LIO system. Second, we propose a novel LIO system by combining the InEKF state estimator with a surfel-based map, named SuIn-LIO, which not only enables the accuracy of state estimation and mapping but also enables real-time registration of a new LiDAR scan. Extensive experiments on different public benchmark datasets demonstrate that SuIn-LIO can achieve comparable …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Efficient spiking neural networks with sparse selective activation for continual learning</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/27817" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The next generation of machine intelligence requires the capability of continual learning to acquire new knowledge without forgetting the old one while conserving limited computing resources.  Spiking neural networks (SNNs), compared to artificial neural networks (ANNs), have more characteristics that align with biological neurons, which may be helpful as a potential gating function for knowledge maintenance in neural networks. Inspired by the selective sparse activation principle of context gating in biological systems, we present a novel SNN model with selective activation to achieve continual learning. The trace-based K-Winner-Take-All (K-WTA) and variable threshold components are designed to form the sparsity in selective activation in spatial and temporal dimensions of spiking neurons, which promotes the subpopulation of neuron activation to perform specific tasks. As a result, continual learning can be maintained by routing different tasks via different populations of neurons in the network. The experiments are conducted on MNIST and CIFAR10 datasets under the class incremental setting. The results show that the proposed SNN model achieves competitive performance similar to and even surpasses the other regularization-based methods deployed under traditional ANNs.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Successive POI recommendation via brain-inspired spatiotemporal aware representation</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/27813" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Existing approaches usually perform spatiotemporal representation in the spatial and temporal dimensions, respectively, which isolates the spatial and temporal natures of the target and leads to sub-optimal embeddings. Neuroscience research has shown that the mammalian brain entorhinal-hippocampal system provides efficient graph representations for general knowledge. Moreover, entorhinal grid cells present concise spatial representations, while hippocampal place cells represent perception conjunctions effectively. Thus, the entorhinal-hippocampal system provides a novel angle for spatiotemporal representation, which inspires us to propose the SpatioTemporal aware Embedding framework (STE) and apply it to POIs (STEP). STEP considers two types of POI-specific representations: sequential representation and spatiotemporal conjunctive representation, learned using sparse unlabeled data based on the proposed graph-building policies. Notably, STEP jointly represents the spatiotemporal natures of POIs using both observations and contextual information from integrated spatiotemporal dimensions by constructing a spatiotemporal context graph. Furthermore, we introduce a user privacy secure successive POI recommendation method using STEP, and it achieves the state-of-the-art performance on two benchmarks. In addition, we demonstrate the excellent performance of the STE representation approach in other spatiotemporal representation-centered tasks through a case study of traffic flow prediction problem. Therefore, this work provides a novel solution to spatiotemporal aware representation and paves a new way for …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking neural network for ultralow-latency and high-accurate object detection</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10472977/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) have attracted significant attention for their energy-efficient and brain-inspired event-driven properties. Recent advancements, notably Spiking-YOLO, have enabled SNNs to undertake advanced object detection tasks. Nevertheless, these methods often suffer from increased latency and diminished detection accuracy, rendering them less suitable for latency-sensitive mobile platforms. Additionally, the conversion of artificial neural networks (ANNs) to SNNs frequently compromises the integrity of the ANNs’ structure, resulting in poor feature representation and heightened conversion errors. To address the issues of high latency and low detection accuracy, we introduce two solutions: timestep compression and spike-time-dependent integrated (STDI) coding. Timestep compression effectively reduces the number of timesteps required in the ANN-to-SNN conversion by condensing …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Im-lif: Improved neuronal dynamics with attention mechanism for direct training deep spiking neural network</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10433858/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are increasingly applied to deep architectures. Recent works are developed to apply spatio-temporal backpropagation to directly train deep SNNs. But the binary and non-differentiable properties of spike activities force directly trained SNNs to suffer from serious gradient vanishing. In this paper, we first analyze the cause of the gradient vanishing problem and identify that the gradients mostly backpropagate along the synaptic currents. Based on that, we modify the synaptic current equation of leaky-integrate-fire neuron model and propose the improved LIF (IM-LIF) neuron model on the basis of the temporal-wise attention mechanism. We utilize the temporal-wise attention mechanism to selectively establish the connection between the current and historical response values, which can empirically enable the neuronal states to update resilient to the gradient vanishing problem …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spike Neural Network of Motor Cortex Model for Arm Reaching Controlling</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.biorxiv.org/content/10.1101/2024.02.07.579412.abstract" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Motor cortex modeling is crucial for understanding movement planning and execution. While interconnected recurrent neural networks have successfully described the dynamics of neural population activity, most existing methods utilize continuous signal-based neural networks, which do not reflect the biological spike neural signal. To address this limitation, we propose a recurrent spike neural network to simulate motor cortical activity during an arm-reaching task. Specifically, our model is built upon integrate-and-fire spiking neurons with conductance-based synapses. We carefully designed the interconnections of neurons with two different firing time scales - “fast” and “slow” neurons. Experimental results demonstrate the effectiveness of our method, with the model’s neuronal activity in good agreement with monkey’s motor cortex data at both single-cell and population levels. Quantitative analysis reveals a correlation coefficient 0.89 between the model’s and real data. These results suggest the possibility of multiple timescales in motor cortical control.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Editorial IEEE Transactions on Cognitive and Developmental Systems</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10419123/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
As we usher into the new year of 2024, in my capacity as the Editor-in-Chief of the IEEE Transactions on Cognitive and Developmental Systems (TCDS), I am happy to extend to you a tapestry of New Year greetings, may this year be filled with prosperity, success, and groundbreaking achievements in our shared fields.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Computing platform, method, and apparatus for spiking neural network learning and simulation</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://patents.google.com/patent/US20240013035A1/en" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
2023-08-26 Assigned to ZHEJIANG UNIVERSITY, Zhejiang Lab reassignment ZHEJIANG UNIVERSITY ASSIGNMENT OF ASSIGNORS INTEREST (SEE DOCUMENT FOR DETAILS). Assignors: HONG, Chaofei, LU, Yujing, PAN, GANG, TANG, Huajin, WANG, XIAO, YUAN, Mengwen, ZHANG, Mengxiao, ZHAO, WENYI
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SPAIC: a spike-based artificial intelligence computing framework</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10384520/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Neuromorphic computing is an emerging research field that aims to develop new intelligent systems by integrating theories and technologies from multiple disciplines, such as neuroscience, deep learning and microelectronics. Various software frameworks have been developed for related fields, but an efficient framework dedicated to spike-based computing models and algorithms is lacking. In this work, we present a Python-based spiking neural network (SNN) simulation and training framework, named SPAIC, that aims to support brain-inspired model and algorithm research integrated with features from both deep learning and neuroscience. To integrate different methodologies from multiple disciplines and balance flexibility and efficiency, SPAIC is designed with a neuroscience-style frontend and a deep learning-based backend. Various types of examples are provided to demonstrate the wide usability of the …
</p>

</div>
</details>


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2023</summary>

<div class="paper-card">

<h3 class="paper-title">Deep Pulse-Coupled Neural Networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2401.08649" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) capture the information processing mechanism of the brain by taking advantage of spiking neurons, such as the Leaky Integrate-and-Fire (LIF) model neuron, which incorporates temporal dynamics and transmits information via discrete and asynchronous spikes. However, the simplified biological properties of LIF ignore the neuronal coupling and dendritic structure of real neurons, which limits the spatio-temporal dynamics of neurons and thus reduce the expressive power of the resulting SNNs. In this work, we leverage a more biologically plausible neural model with complex dynamics, i.e., a pulse-coupled neural network (PCNN), to improve the expressiveness and recognition performance of SNNs for vision tasks. The PCNN is a type of cortical model capable of emulating the complex neuronal activities in the primary visual cortex. We construct deep pulse-coupled neural networks (DPCNNs) by replacing commonly used LIF neurons in SNNs with PCNN neurons. The intra-coupling in existing PCNN models limits the coupling between neurons only within channels. To address this limitation, we propose inter-channel coupling, which allows neurons in different feature maps to interact with each other. Experimental results show that inter-channel coupling can efficiently boost performance with fewer neurons, synapses, and less training time compared to widening the networks. For instance, compared to the LIF-based SNN with wide VGG9, DPCNN with VGG9 uses only 50%, 53%, and 73% of neurons, synapses, and training time, respectively. Furthermore, we propose receptive field and time dependent batch normalization (RFTD-BN) to speed up the convergence and performance of DPCNNs.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Toward high-accuracy and low-latency spiking neural networks with two-stage optimization</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10361844/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) operating with asynchronous discrete events show higher energy efficiency with sparse computation. A popular approach for implementing deep SNNs is artificial neural network (ANN)–SNN conversion combining both efficient training of ANNs and efficient inference of SNNs. However, the accuracy loss is usually nonnegligible, especially under few time steps, which restricts the applications of SNN on latency-sensitive edge devices greatly. In this article, we first identify that such performance degradation stems from the misrepresentation of the negative or overflow residual membrane potential in SNNs. Inspired by this, we decompose the conversion error into three parts: quantization error, clipping error, and residual membrane potential representation error. With such insights, we propose a two-stage conversion algorithm to minimize those errors, respectively. In addition, we show …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Enhancing adaptive history reserving by spiking convolutional block attention module in recurrent neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/b8734840bf65c8facd619f5105c6acd0-Abstract-Conference.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) serve as one type of efficient model to process spatio-temporal patterns in time series, such as the Address-Event Representation data collected from Dynamic Vision Sensor (DVS). Although convolutional SNNs have achieved remarkable performance on these AER datasets, benefiting from the predominant spatial feature extraction ability of convolutional structure, they ignore temporal features related to sequential time points. In this paper, we develop a recurrent spiking neural network (RSNN) model embedded with an advanced spiking convolutional block attention module (SCBAM) component to combine both spatial and temporal features of spatio-temporal patterns. It invokes the history information in spatial and temporal channels adaptively through SCBAM, which brings the advantages of efficient memory calling and history redundancy elimination. The performance of our model was evaluated in DVS128-Gesture dataset and other time-series datasets. The experimental results show that the proposed SRNN-SCBAM model makes better use of the history information in spatial and temporal dimensions with less memory space, and achieves higher accuracy compared to other models.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Temporal conditioning spiking latent variable models of the neural response to natural visual scenes</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/0bcf9cf6ffe26bba3af99e18be0e1d8d-Abstract-Conference.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Developing computational models of neural response is crucial for understanding sensory processing and neural computations. Current state-of-the-art neural network methods use temporal filters to handle temporal dependencies, resulting in an unrealistic and inflexible processing paradigm. Meanwhile, these methods target trial-averaged firing rates and fail to capture important features in spike trains. This work presents the temporal conditioning spiking latent variable models (TeCoS-LVM) to simulate the neural response to natural visual stimuli. We use spiking neurons to produce spike outputs that directly match the recorded trains. This approach helps to avoid losing information embedded in the original spike trains. We exclude the temporal dimension from the model parameter space and introduce a temporal conditioning operation to allow the model to adaptively explore and exploit temporal dependencies in stimuli sequences in a natural paradigm. We show that TeCoS-LVM models can produce more realistic spike activities and accurately fit spike statistics than powerful alternatives. Additionally, learned TeCoS-LVM models can generalize well to longer time scales. Overall, while remaining computationally tractable, our model effectively captures key features of neural coding systems. It thus provides a useful tool for building accurate predictive computational accounts for various sensory perception circuits.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Cmci: A robust multimodal fusion method for spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://link.springer.com/chapter/10.1007/978-981-99-8067-3_12" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Human understand the external world through a variety of perceptual processes such as sight, sound, touch and smell. Simulating such biological multi-sensory fusion decisions using a computational model is important for both computer and neuroscience research. Spiking Neural Networks (SNNs) mimic the neural dynamics of the brain, which are expected to reveal the biological multimodal perception mechanism. However, existing works of multimodal SNNs are still limited, and most of them only focus on audiovisual fusion and lack systematic comparison of the performance and robustness of the models. In this paper, we propose a novel fusion module called Cross-modality Current Integration (CMCI) for multimodal SNNs and systematically compare it with other fusion methods on visual, auditory and olfactory fusion recognition tasks. Besides, a regularization technique called Modality-wise Dropout (ModDrop …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Emergence and reconfiguration of modular structure for synaptic neural networks during continual familiarity detection</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2311.05862" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
While advances in artificial intelligence and neuroscience have enabled the emergence of neural networks capable of learning a wide variety of tasks, our understanding of the temporal dynamics of these networks remains limited. Here, we study the temporal dynamics during learning of Hebbian Feedforward (HebbFF) neural networks in tasks of continual familiarity detection. Drawing inspiration from the field of network neuroscience, we examine the network's dynamic reconfiguration, focusing on how network modules evolve throughout learning. Through a comprehensive assessment involving metrics like network accuracy, modular flexibility, and distribution entropy across diverse learning modes, our approach reveals various previously unknown patterns of network reconfiguration. In particular, we find that the emergence of network modularity is a salient predictor of performance, and that modularization strengthens with increasing flexibility throughout learning. These insights not only elucidate the nuanced interplay of network modularity, accuracy, and learning dynamics but also bridge our understanding of learning in artificial and biological realms.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Grid cell modeling with mapping representation of self-motion for path integration</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://link.springer.com/article/10.1007/s00521-021-06039-x" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The representation of grid cells in the medial entorhinal cortex region is crucial for path integration. In this paper, we proposed a grid cell modeling mechanism by mapping the agent’s self-motion in Euclidean space to the neuronal activity of grid cells. Our representational model can achieve multi-scale hexagonal patterns of grid cells from recurrent neural network (RNN) and enables path integration for 1D, 2D and 3D spaces. Different from the existing works which need to learn weights of RNN to get the vector representation of grid cells, our method can obtain weights by direct matrix operations. Moreover, compared with the classical models based on continuous attractor network, our model avoids the connection matrix’s symmetry limitation and spatial representation redundancy problems. In this paper, we also discuss the connection pattern between grid cells and place cells to demonstrate grid cells …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Esvae: An efficient spiking variational autoencoder with reparameterizable poisson spiking sampling</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2310.14839" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In recent years, studies on image generation models of spiking neural networks (SNNs) have gained the attention of many researchers. Variational autoencoders (VAEs), as one of the most popular image generation models, have attracted a lot of work exploring their SNN implementation. Due to the constrained binary representation in SNNs, existing SNN VAE methods implicitly construct the latent space by an elaborated autoregressive network and use the network outputs as the sampling variables. However, this unspecified implicit representation of the latent space will increase the difficulty of generating high-quality images and introduces additional network parameters. In this paper, we propose an efficient spiking variational autoencoder (ESVAE) that constructs an interpretable latent space distribution and design a reparameterizable spiking sampling method. Specifically, we construct the prior and posterior of the latent space as a Poisson distribution using the firing rate of the spiking neurons. Subsequently, we propose a reparameterizable Poisson spiking sampling method, which is free from the additional network. Comprehensive experiments have been conducted, and the experimental results show that the proposed ESVAE outperforms previous SNN VAE methods in reconstructed & generated images quality. In addition, experiments demonstrate that ESVAE's encoder is able to retain the original image information more efficiently, and the decoder is more robust. The source code is available at https://github.com/QgZhan/ESVAE.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking reinforcement learning with memory ability for mapless navigation</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10341738/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Our study focuses on mapless navigation in robotics, which involves navigating without an established obstacle map of the environment. Spiking Neural Networks (SNNs) have recently been applied to this task using Deep Reinforcement Learning (DRL), but face challenges in dynamic and partially observable environments, as well as inaccuracies in transmitted data. To overcome these issues, we propose a Multi-Critic DDPG with Spiking Memory (MC-DDPGSM) framework. Our approach introduces a spiking Gate Recurrent Unit layer (Spiking-GRU) to provide memory function and evaluates the state-action value with multi-critic networks. The experimental results demonstrate that our method achieves better performance (success rate, navigation distance, navigation time spent, and power consumption) in complex navigation tasks compared to the state-of-the-art approaches. Furthermore, our model can be …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Exploiting Noise as a Resource for Computation and Learning in Spiking Neural Networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.cell.com/patterns/fulltext/S2666-3899(23)00200-3?ref=https://githubhelp.com" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Networks of spiking neurons underpin the extraordinary information-processing capabilities of the brain and have become pillar models in neuromorphic artificial intelligence. Despite extensive research on spiking neural networks (SNNs), most studies are established on deterministic models, overlooking the inherent non-deterministic, noisy nature of neural computations. This study introduces the noisy SNN (NSNN) and the noise-driven learning (NDL) rule by incorporating noisy neuronal dynamics to exploit the computational advantages of noisy neural processing. The NSNN provides a theoretical framework that yields scalable, flexible, and reliable computation and learning. We demonstrate that this framework leads to spiking neural models with competitive performance, improved robustness against challenging perturbations compared with deterministic SNNs, and better reproducing probabilistic computation …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Dual memory model for experience-once task-incremental lifelong learning</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0893608023003672" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Experience replay (ER) is a widely-adopted neuroscience-inspired method to perform lifelong learning. Nonetheless, existing ER-based approaches consider very coarse memory modules with simple memory and rehearsal mechanisms that cannot fully exploit the potential of memory replay. Evidence from neuroscience has provided fine-grained memory and rehearsal mechanisms, such as the dual-store memory system consisting of PFC-HC circuits. However, the computational abstraction of these processes is still very challenging. To address these problems, we introduce the Dual-Memory (Dual-MEM) model emulating the memorization, consolidation, and rehearsal process in the PFC-HC dual-store memory circuit. Dual-MEM maintains an incrementally updated short-term memory to benefit current-task learning. At the end of the current task, short-term memories will be consolidated into long-term ones for …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Sequential sparse autoencoder for dynamic heading representation in ventral intraparietal area</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0010482523005796" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
To navigate in space, it is important to predict headings in real-time from neural responses in the brain to vestibular and visual signals, and the ventral intraparietal area (VIP) is one of the critical brain areas. However, it remains unexplored in the population level how the heading perception is represented in VIP. And there are no commonly used methods suitable for decoding the headings from the population responses in VIP, given the large spatiotemporal dynamics and heterogeneity in the neural responses. Here, responses were recorded from 210 VIP neurons in three rhesus monkeys when they were performing a heading perception task. And by specifically and separately modelling the both dynamics with sparse representation, we built a sequential sparse autoencoder (SSAE) to do the population decoding on the recorded dataset and tried to maximize the decoding performance. The SSAE relies on a three …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Event-driven spiking learning algorithm using aggregated labels</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10236547/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Traditional spiking learning algorithm aims to train neurons to spike at a specific time or on a particular frequency, which requires precise time and frequency labels in the training process. While in reality, usually only aggregated labels of sequential patterns are provided. The aggregate-label (AL) learning is proposed to discover these predictive features in distracting background streams only by aggregated spikes. It has achieved much success recently, but it is still computationally intensive and has limited use in deep networks. To address these issues, we propose an event-driven spiking aggregate learning algorithm (SALA) in this article. Specifically, to reduce the computational complexity, we improve the conventional spike-threshold-surface (STS) calculation in AL learning by analytical calculating voltage peak values in spiking neurons. Then we derive the algorithm to multilayers by event-driven strategy …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Unleashing the potential of spiking neural networks for sequential modeling with contextual embedding</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2308.15150" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The human brain exhibits remarkable abilities in integrating temporally distant sensory inputs for decision-making. However, existing brain-inspired spiking neural networks (SNNs) have struggled to match their biological counterpart in modeling long-term temporal relationships. To address this problem, this paper presents a novel Contextual Embedding Leaky Integrate-and-Fire (CE-LIF) spiking neuron model. Specifically, the CE-LIF model incorporates a meticulously designed contextual embedding component into the adaptive neuronal firing threshold, thereby enhancing the memory storage of spiking neurons and facilitating effective sequential modeling. Additionally, theoretical analysis is provided to elucidate how the CE-LIF model enables long-term temporal credit assignment. Remarkably, when compared to state-of-the-art recurrent SNNs, feedforward SNNs comprising the proposed CE-LIF neurons demonstrate superior performance across extensive sequential modeling tasks in terms of classification accuracy, network convergence speed, and memory capacity.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A Review of Image Reconstruction Based on Event Cameras</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://jeit.ac.cn/en/article/doi/10.11999/JEIT221456?viewType=HTML" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras are bio-inspired sensors that outputs a stream of events when the brightness change of pixels exceeds the threshold. This type of visual sensor asynchronously outputs events that encode the time, location and sign of the brightness changes. Hence, event cameras offer attractive properties, such as high temporal resolution, very high dynamic range, low latency, low power consumption, and high pixel bandwidth. It can capture information in high-speed motion and high-dynamic scenes, which can be used to reconstruct high-dynamic range and high-speed motion scenes. Brightness images obtained by image reconstruction can be interpreted as a representation, and be used for recognition, segmentation, tracking and optical flow estimation, which is one of the important research directions in the field of vision. This survey first briefly introduces event cameras from their working principle …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Spiking Neural Network Recognition Method Based on Dynamic Visual Motion Features</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://jeit.ac.cn/en/article/doi/10.11999/JEIT221478" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Considering the shortcomings of the low recognition accuracy and poor real-time performance of existing Spiking Neural Networks (SNN) for dynamic visual event streams, a SNN recognition method based on dynamic visual motion features is proposed in this paper. First, the dynamic motion features in the event stream are extracted using the event-based motion history information representation and gradient direction calculation. Then, the spatiotemporal pooling operation is introduced to eliminate the redundancy of events in the temporal and spatial domain, further retaining the significant motion features. Finally, the feature event streams are fed into the SNN for learning and recognition. Experiments conducted on benchmark dynamic visual datasets show that dynamic visual motion features can significantly improve the recognition accuracy and computational speed of SNN for event streams.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Learnable Surrogate Gradient for Direct Training Spiking Neural Networks.</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.ijcai.org/proceedings/2023/0335.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) have increasingly drawn massive research attention due to biological interpretability and efficient computation. Recent achievements are devoted to utilizing the surrogate gradient (SG) method to avoid the dilemma of non-differentiability of spiking activity to directly train SNNs by backpropagation. However, the fixed width of the SG leads to gradient vanishing and mismatch problems, thus limiting the performance of directly trained SNNs. In this work, we propose a novel perspective to unlock the width limitation of SG, called the learnable surrogate gradient (LSG) method. The LSG method modulates the width of SG according to the change of the distribution of the membrane potentials, which is identified to be related to the decay factors based on our theoretical analysis. Then we introduce the trainable decay factors to implement the LSG method, which can optimize the width of SG automatically during training to avoid the gradient vanishing and mismatch problems caused by the limited width of SG. We evaluate the proposed LSG method on both image and neuromorphic datasets. Experimental results show that the LSG method can effectively alleviate the blocking of gradient propagation caused by the limited width of SG when training deep SNNs directly. Meanwhile, the LSG method can help SNNs achieve competitive performance on both latency and accuracy.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Adaptive smoothing gradient learning for spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://proceedings.mlr.press/v202/wang23j.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) with biologically inspired spatio-temporal dynamics demonstrate superior energy efficiency on neuromorphic architectures. Error backpropagation in SNNs is prohibited by the all-or-none nature of spikes. The existing solution circumvents this problem by a relaxation on the gradient calculation using a continuous function with a constant relaxation de-gree, so-called surrogate gradient learning. Nevertheless, such a solution introduces additional smoothing error on spike firing which leads to the gradients being estimated inaccurately. Thus, how to adaptively adjust the relaxation degree and eliminate smoothing error progressively is crucial. Here, we propose a methodology such that training a prototype neural network will evolve into training an SNN gradually by fusing the learnable relaxation degree into the network with random spike noise. In this way, the network learns adaptively the accurate gradients of loss landscape in SNNs. The theoretical analysis further shows optimization on such a noisy network could be evolved into optimization on the embedded SNN with shared weights progressively. Moreover, The experiments on static images, dynamic event streams, speech, and instrumental sounds show the proposed method achieves state-of-the-art performance across all the datasets with remarkable robustness on different relaxation degrees.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Esl-snns: An evolutionary structure learning strategy for spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ojs.aaai.org/index.php/AAAI/article/view/25079" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) have manifested remarkable advantages in power consumption and event-driven property during the inference process. To take full advantage of low power consumption and improve the efficiency of these models further, the pruning methods have been explored to find sparse SNNs without redundancy connections after training. However, parameter redundancy still hinders the efficiency of SNNs during training. In the human brain, the rewiring process of neural networks is highly dynamic, while synaptic connections maintain relatively sparse during brain development. Inspired by this, here we propose an efficient evolutionary structure learning (ESL) framework for SNNs, named ESL-SNNs, to implement the sparse SNN training from scratch. The pruning and regeneration of synaptic connections in SNNs evolve dynamically during learning, yet keep the structural sparsity at a certain level. As a result, the ESL-SNNs can search for optimal sparse connectivity by exploring all possible parameters across time. Our experiments show that the proposed ESL-SNNs framework is able to learn SNNs with sparse structures effectively while reducing the limited accuracy. The ESL-SNNs achieve merely 0.28% accuracy loss with 10% connection density on the DVS-Cifar10 dataset. Our work presents a brand-new approach for sparse training of SNNs from scratch with biologically plausible evolutionary mechanisms, closing the gap in the expressibility between sparse training and dense training. Hence, it has great potential for SNN lightweight training and inference with low power consumption and small memory usage.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Mitigating communication costs in neural networks: The role of dendritic nonlinearity</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2306.11950" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Our comprehension of biological neuronal networks has profoundly influenced the evolution of artificial neural networks (ANNs). However, the neurons employed in ANNs exhibit remarkable deviations from their biological analogs, mainly due to the absence of complex dendritic trees encompassing local nonlinearity. Despite such disparities, previous investigations have demonstrated that point neurons can functionally substitute dendritic neurons in executing computational tasks. In this study, we scrutinized the importance of nonlinear dendrites within neural networks. By employing machine-learning methodologies, we assessed the impact of dendritic structure nonlinearity on neural network performance. Our findings reveal that integrating dendritic structures can substantially enhance model capacity and performance while keeping signal communication costs effectively restrained. This investigation offers pivotal insights that hold considerable implications for the development of future neural network accelerators.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Bipolar population threshold encoding for audio recognition with deep spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10191235/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) have been in-creasingly investigated for audio recognition due to the low power consumption on neuromorphic hardware by mimicking biological neural systems. Since the SNNs are learned from spikes, a critical step lies in the efficient neural encoding of real-valued sound signals to represent complex temporal patterns in speech and environmental sounds. In this paper, we propose a novel Bipolar Population Threshold (BPT) encoding model that effectively captures the trajectory information of time-series speech data by combining temporal and spatial dimensions. The bipolar encoding technique uses positive and negative neurons to capture the dynamic changes in the audio signal, while the threshold intervals allow for a sparse representation that focuses on encoding significant changes, resulting in an efficient and simplified recognition process. Extensively experimenting on …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Event-driven spiking neural networks with spike-based learning</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://link.springer.com/article/10.1007/s12293-023-00391-2" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) use spikes to communicate between neurons, leading to biological plausible implementation. Considering spikes as events, SNNs are inherently suitable for processing address event representation (AER) data. Despite the progress in event-driven methods for AER data, there is little study on the relationship between time-driven and event-driven algorithms, that is required to gain insight into the understanding of SNNs. In this paper, an in-depth analysis of time-driven and event-driven algorithms was given. A same-timestamp problem in event-driven simulation, which may lead to an error spike, is found and solved in a simple efficacious way. An event-driven learning algorithm was proposed, which is efficient and compatible with a multitude of spike-based plasticity mechanisms. Leaky integrate-and-fire neurons with precise spike driven synaptic plasticity was used to demonstrate …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Symmetric-threshold ReLU for fast and nearly lossless ANN-SNN conversion</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://link.springer.com/article/10.1007/s11633-022-1388-2" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The artificial neural network-spiking neural network (ANN-SNN) conversion, as an efficient algorithm for deep SNNs training, promotes the performance of shallow SNNs, and expands the application in various tasks. However, the existing conversion methods still face the problem of large conversion error within low conversion time steps. In this paper, a heuristic symmetric-threshold rectified linear unit (stReLU) activation function for ANNs is proposed, based on the intrinsically different responses between the integrate-and-fire (IF) neurons in SNNs and the activation functions in ANNs. The negative threshold in stReLU can guarantee the conversion of negative activations, and the symmetric thresholds enable positive error to offset negative error between activation value and spike firing rate, thus reducing the conversion error from ANNs to SNNs. The lossless conversion from ANNs with stReLU to SNNs is …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Diverse effects of gaze direction on heading perception in humans</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://academic.oup.com/cercor/article-abstract/33/11/6772/7024719" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Gaze change can misalign spatial reference frames encoding visual and vestibular signals in cortex, which may affect the heading discrimination. Here, by systematically manipulating the eye-in-head and head-on-body positions to change the gaze direction of subjects, the performance of heading discrimination was tested with visual, vestibular, and combined stimuli in a reaction-time task in which the reaction time is under the control of subjects. We found the gaze change induced substantial biases in perceived heading, increased the threshold of discrimination and reaction time of subjects in all stimulus conditions. For the visual stimulus, the gaze effects were induced by changing the eye-in-world position, and the perceived heading was biased in the opposite direction of gaze. In contrast, the vestibular gaze effects were induced by changing the eye-in-head position, and the perceived heading was …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Corrigendum: SCTN: event-based object tracking with energy-efficient deep convolutional spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1204334/full" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras are asynchronous and neuromorphically inspired visual sensors, which have shown great potential in object tracking because they can easily detect moving objects. Since event cameras output discrete events, they are inherently suitable to coordinate with Spiking Neural Network (SNN), which has aunique event-driven computation characteristic and energy-e□cient computing. In this paper, we tackle the problem of event-based object tracking by a novel architecture with a discriminatively trained SNN, called the Spiking Convolutional Tracking Network (SCTN). Taking a segment of events as input, SCTN notonly better exploits implicit associations among events rather than event-wise processing, but also fully utilizes precise temporal information and maintains the sparse representation in segments instead of frames. To make SCTN more suitable for object tracking, we propose a new loss function …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Biologically inspired structure learning with reverse knowledge distillation for spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2304.09500" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) have superb characteristics in sensory information recognition tasks due to their biological plausibility. However, the performance of some current spiking-based models is limited by their structures which means either fully connected or too-deep structures bring too much redundancy. This redundancy from both connection and neurons is one of the key factors hindering the practical application of SNNs. Although Some pruning methods were proposed to tackle this problem, they normally ignored the fact the neural topology in the human brain could be adjusted dynamically. Inspired by this, this paper proposed an evolutionary-based structure construction method for constructing more reasonable SNNs. By integrating the knowledge distillation and connection pruning method, the synaptic connections in SNNs can be optimized dynamically to reach an optimal state. As a result, the structure of SNNs could not only absorb knowledge from the teacher model but also search for deep but sparse network topology. Experimental results on CIFAR100 and DVS-Gesture show that the proposed structure learning method can get pretty well performance while reducing the connection redundancy. The proposed method explores a novel dynamical way for structure learning from scratch in SNNs which could build a bridge to close the gap between deep learning and bio-inspired neural dynamics.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Effective active learning method for spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10081108/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
A large quantity of labeled data is required to train high-performance deep spiking neural networks (SNNs), but obtaining labeled data is expensive. Active learning is proposed to reduce the quantity of labeled data required by deep learning models. However, conventional active learning methods in SNNs are not as effective as that in conventional artificial neural networks (ANNs) because of the difference in feature representation and information transmission. To address this issue, we propose an effective active learning method for a deep SNN model in this article. Specifically, a loss prediction module ActiveLossNet is proposed to extract features and select valuable samples for deep SNNs. Then, we derive the corresponding active learning algorithm for deep SNN models. Comprehensive experiments are conducted on CIFAR-10, MNIST, Fashion-MNIST, and SVHN on different SNN frameworks, including …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SCTN: Event-based object tracking with energy-efficient deep convolutional spiking neural networks</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.frontiersin.org/articles/10.3389/fnins.2023.1123698/full" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Event cameras are asynchronous and neuromorphically inspired visual sensors, which have shown great potential in object tracking because they can easily detect moving objects. Since event cameras output discrete events, they are inherently suitable to coordinate with Spiking Neural Network (SNN), which has a unique event-driven computation characteristic and energy-efficient computing. In this paper, we tackle the problem of event-based object tracking by a novel architecture with a discriminatively trained SNN, called the Spiking Convolutional Tracking Network (SCTN). Taking a segment of events as input, SCTN not only better exploits implicit associations among events rather than event-wise processing, but also fully utilizes precise temporal information and maintains the sparse representation in segments instead of frames. To make SCTN more suitable for object tracking, we propose a new loss function that introduces an exponential Intersection over Union (IoU) in the voltage domain. To the best of our knowledge, this is the first tracking network directly trained with SNN. Besides, we present a new event-based tracking dataset, dubbed DVSOT21. In contrast to other competing trackers, experimental results on DVSOT21 demonstrate that our method achieves competitive performance with very low energy consumption compared to ANN based trackers with very low energy consumption compared to ANN based trackers. With lower energy consumption, tracking on neuromorphic hardware will reveal its advantage.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Attention-based deep spiking neural networks for temporal credit assignment problems</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10038509/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The temporal credit assignment (TCA) problem, which aims to detect predictive features hidden in distracting background streams, remains a core challenge in biological and machine learning. Aggregate-label (AL) learning is proposed by researchers to resolve this problem by matching spikes with delayed feedback. However, the existing AL learning algorithms only consider the information of a single timestep, which is inconsistent with the real situation. Meanwhile, there is no quantitative evaluation method for TCA problems. To address these limitations, we propose a novel attention-based TCA (ATCA) algorithm and a minimum editing distance (MED)-based quantitative evaluation method. Specifically, we define a loss function based on the attention mechanism to deal with the information contained within the spike clusters and use MED to evaluate the similarity between the spike train and the target clue flow …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A spatial cognition approach based on grid cell group representation for embodied intelligence</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Abstract unavailable. This publication does not provide a summary using scholarly.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Constructing deep spiking neural networks from artificial neural networks with knowledge distillation</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="http://openaccess.thecvf.com/content/CVPR2023/html/Xu_Constructing_Deep_Spiking_Neural_Networks_From_Artificial_Neural_Networks_With_CVPR_2023_paper.html" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking neural networks (SNNs) are well known as the brain-inspired models with high computing efficiency, due to a key component that they utilize spikes as information units, close to the biological neural systems. Although spiking based models are energy efficient by taking advantage of discrete spike signals, their performance is limited by current network structures and their training methods. As discrete signals, typical SNNs cannot apply the gradient descent rules directly into parameters adjustment as artificial neural networks (ANNs). Aiming at this limitation, here we propose a novel method of constructing deep SNN models with knowledge distillation (KD) that uses ANN as teacher model and SNN as student model. Through ANN-SNN joint training algorithm, the student SNN model can learn rich feature information from the teacher ANN model through the KD method, yet it avoids training SNN from scratch when communicating with non-differentiable spikes. Our method can not only build a more efficient deep spiking structure feasibly and reasonably, but use few time steps to train whole model compared to direct training or ANN to SNN methods. More importantly, it has a superb ability of noise immunity for various types of artificial noises and natural signals. The proposed novel method provides efficient ways to improve the performance of SNN through constructing deeper structures in a high-throughput fashion, with potential usage for light and efficient brain-inspired computing of practical scenarios.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A low latency adaptive coding spiking framework for deep reinforcement learning</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2211.11760" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In recent years, spiking neural networks (SNNs) have been used in reinforcement learning (RL) due to their low power consumption and event-driven features. However, spiking reinforcement learning (SRL), which suffers from fixed coding methods, still faces the problems of high latency and poor versatility. In this paper, we use learnable matrix multiplication to encode and decode spikes, improving the flexibility of the coders and thus reducing latency. Meanwhile, we train the SNNs using the direct training method and use two different structures for online and offline RL algorithms, which gives our model a wider range of applications. Extensive experiments have revealed that our method achieves optimal performance with ultra-low latency (as low as 0.8% of other SRL methods) and excellent energy efficiency (up to 5X the DNNs) in different algorithms and different environments.
</p>

</div>
</details>

