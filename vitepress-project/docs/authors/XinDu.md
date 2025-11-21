## 📑 Xin Du Papers

论文按年份分组（点击年份或空白区域可展开/折叠该年份的论文）


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2025</summary>

<div class="paper-card">

<h3 class="paper-title">SeMi: When Imbalanced Semi-Supervised Learning Meets Mining Hard Examples</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://dl.acm.org/doi/abs/10.1145/3746027.3755450" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Semi-Supervised Learning (SSL) can leverage abundant unlabeled data to boost model performance. However, the class-imbalanced data distribution in real-world scenarios poses great challenges to SSL, resulting in performance degradation. Existing class-imbalanced semi-supervised learning (CISSL) methods mainly focus on rebalancing datasets but ignore the potential of using hard examples to enhance performance, making it difficult to fully harness the power of unlabeled data even with sophisticated algorithms. To address this issue, we propose a method that enhances the performance of Imbalanced Semi-Supervised Learning by Mining Hard Examples (SeMi). This method distinguishes the entropy differences among logits of hard and easy examples, thereby identifying hard examples and increasing the utility of unlabeled data, better addressing the imbalance problem in CISSL. In addition, we …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">DepAsync: An Asynchronous SNN Accelerator Based on Core-Dependency</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11216144/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) are widely used in brain-inspired computing and neuroscience research. Several many-core accelerators have been built to improve the running speed and energy efficiency of SNNs. However, current accelerators generally need explicit synchronization among all cores after each timestep of SNNs, which poses a challenge to overall efficiency. This paper proposes DepAsync, an asynchronous architecture that eliminates inter-core synchronization, facilitating fast and energy-efficient SNN inference with commendable scalability. The main idea is to exploit the dependency of neuromorphic cores predetermined at compile time. We design a DepAsync scheduler for each core to trace the running state of its dependencies and control the core to safely forward to the next timestep without waiting for other cores to complete their tasks. This approach prevents the necessity for global …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SAFA-SNN: Sparsity-Aware On-Device Few-Shot Class-Incremental Learning with Fast-Adaptive Structure of Spiking Neural Network</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2510.03648" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Continuous learning of novel classes is crucial for edge devices to preserve data privacy and maintain reliable performance in dynamic environments. However, the scenario becomes particularly challenging when data samples are insufficient, requiring on-device few-shot class-incremental learning (FSCIL) to maintain consistent model performance. Although existing work has explored parameter-efficient FSCIL frameworks based on artificial neural networks (ANNs), their deployment is still fundamentally constrained by limited device resources. Inspired by neural mechanisms, Spiking neural networks (SNNs) process spatiotemporal information efficiently, offering lower energy consumption, greater biological plausibility, and compatibility with neuromorphic hardware than ANNs. In this work, we present an SNN-based method for On-Device FSCIL, i.e., Sparsity-Aware and Fast Adaptive SNN (SAFA-SNN). We first propose sparsity-conditioned neuronal dynamics, in which most neurons remain stable while a subset stays active, thereby mitigating catastrophic forgetting. To further cope with spike non-differentiability in gradient estimation, we employ zeroth-order optimization. Moreover, during incremental learning sessions, we enhance the discriminability of new classes through subspace projection, which alleviates overfitting to novel classes. Extensive experiments conducted on two standard benchmark datasets (CIFAR100 and Mini-ImageNet) and three neuromorphic datasets (CIFAR-10-DVS, DVS128gesture, and N-Caltech101) demonstrate that SAFA-SNN outperforms baseline methods, specifically achieving at least 4.01% improvement at …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Biologically Plausible Learning via Bidirectional Spike-Based Distillation</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2509.20284" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Developing biologically plausible learning algorithms that can achieve performance comparable to error backpropagation remains a longstanding challenge. Existing approaches often compromise biological plausibility by entirely avoiding the use of spikes for error propagation or relying on both positive and negative learning signals, while the question of how spikes can represent negative values remains unresolved. To address these limitations, we introduce Bidirectional Spike-based Distillation (BSD), a novel learning algorithm that jointly trains a feedforward and a backward spiking network. We formulate learning as a transformation between two spiking representations (i.e., stimulus encoding and concept encoding) so that the feedforward network implements perception and decision-making by mapping stimuli to actions, while the backward network supports memory recall by reconstructing stimuli from concept representations. Extensive experiments on diverse benchmarks, including image recognition, image generation, and sequential regression, show that BSD achieves performance comparable to networks trained with classical error backpropagation. These findings represent a significant step toward biologically grounded, spike-driven learning in neural networks.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">DarwinWafer: A Wafer-Scale Neuromorphic Chip</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2509.16213" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Neuromorphic computing promises brain-like efficiency, yet today&#x27;s multi-chip systems scale over PCBs and incur orders-of-magnitude penalties in bandwidth, latency, and energy, undermining biological algorithms and system efficiency. We present DarwinWafer, a hyperscale system-on-wafer that replaces off-chip interconnects with wafer-scale, high-density integration of 64 Darwin3 chiplets on a 300 mm silicon interposer. A GALS NoC within each chiplet and an AER-based asynchronous wafer fabric with hierarchical time-step synchronization provide low-latency, coherent operation across the wafer. Each chiplet implements 2.35 M neurons and 0.1 B synapses, yielding 0.15 B neurons and 6.4 B synapses per wafer.At 333 MHz and 0.8 V, DarwinWafer consumes ~100 W and achieves 4.9 pJ/SOP, with 64 TSOPS peak throughput (0.64 TSOPS/W). Realization is enabled by a holistic chiplet-interposer co-design flow (including an in-house interposer-bump planner with early SI/PI and electro-thermal closure) and a warpage-tolerant assembly that fans out I/O via PCBlets and compliant pogo-pin connections, enabling robust, demountable wafer-to-board integration. Measurements confirm 10 mV supply droop and a uniform thermal profile (34-36 {\deg}C) under ~100 W. Application studies demonstrate whole-brain simulations: two zebrafish brains per chiplet with high connectivity fidelity (Spearman r = 0.896) and a mouse brain mapped across 32 chiplets (r = 0.645). To our knowledge, DarwinWafer represents a pioneering demonstration of wafer-scale neuromorphic computing, establishing a viable and scalable path toward large-scale, brain …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Mapping Large-Scale Spiking Neural Network on Arbitrary Meshed Neuromorphic Hardware</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11134841/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Neuromorphic hardware systems—designed as 2D-mesh structures with parallel neurosynaptic cores—have proven highly efficient at executing large-scale spiking neural networks (SNNs). A critical challenge, however, lies in mapping neurons efficiently to these cores. While existing approaches work well with regular, fully functional mesh structures, they falter in real-world scenarios where hardware has irregular shapes or non-functional cores caused by defects or resource fragmentation. To address these limitations, we propose a novel mapping method based on an innovative space-filling curve: the Adaptive Locality-Preserving (ALP) curve. Using a unique divide-and-conquer construction algorithm, the ALP curve ensures adaptability to meshes of any shape while maintaining crucial locality properties—essential for efficient mapping. Our method demonstrates exceptional computational efficiency, making it …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Universal Backdoor Defense via Label Consistency in Vertical Federated Learning</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4900.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Backdoor attacks in vertical federated learning (VFL) are particularly concerning as they can covertly compromise VFL decision-making, posing a severe threat to critical applications of VFL. Existing defense mechanisms typically involve either label obfuscation during training or model pruning during inference. However, the inherent limitations on the defender’s access to the global model and complete training data in VFL environments fundamentally constrain the effectiveness of these conventional methods. To address these limitations, we propose the Universal Backdoor Defense (UBD) framework. UBD leverages Label Consistent Clustering (LCC) to synthesize plausible latent triggers associated with the backdoor class. This synthesized information is then utilized for mitigating backdoor threats through Linear Probing (LP), guided by a constraint on Batch Normalization (BN) statistics. Positioned within a unified VFL backdoor defense paradigm, UBD offers a generalized framework for both detection and mitigation that critically does not necessitate access to the entire model or dataset. Extensive experiments across multiple datasets rigorously demonstrate the efficacy of the UBD framework, achieving state-of-the-art performance against diverse backdoor attack types in VFL, including both dirty-label and clean-label variants.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Cost-Effective On-Device Sequential Recommendation with Spiking Neural Networks</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/3150.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
On-device sequential recommendation (SR) systems are designed to make local inferences using real-time features, thereby alleviating the communication burden on server-based recommenders when handling concurrent requests from millions of users. However, the resource constraints of edge devices, including limited memory and computational capacity, pose significant challenges to deploying efficient SR models. Inspired by the energy-efficient and sparse computing properties of deep Spiking Neural Networks (SNNs), we propose a cost-effective on-device SR model named SSR, which encodes dense embedding representations into sparse spike-wise representations and integrates novel spiking filter modules to extract temporal patterns and critical features from item sequences, optimizing computational and memory efficiency without sacrificing recommendation accuracy. Extensive experiments on real-world datasets demonstrate the superiority of SSR. Compared to other SR baselines, SSR achieves comparable recommendation performance while reducing energy consumption by an average of 59.43%. In addition, SSR significantly lowers memory usage, making it particularly well-suited for deployment on resource-constrained edge devices.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">MetricEmbedding: Accelerate Metric Nearness by Tropical Inner Product</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Abstract unavailable. This publication does not provide a summary using scholarly.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">SNN-IoT: Efficient Partitioning and Enabling of Deep Spiking Neural Networks in IoT Services</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11095670/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs), due to their inherent biological plausibility and energy-saving characteristics, naturally align with the requirements of IoT services. However, current SNNs require a multi-layer structure to achieve effective applications across various fields. The multi-layer deep SNNs with massive model parameters demand computational resources, rendering them incompatible with resource-constrained IoT devices. To address this problem, in this work, a deep SNN partitioning framework called SNN-IoT is proposed to run complex SNN models on IoT devices. The SNN-IoT first partitions a full deep SNN model into smaller sub-models, leveraging the event-driven sparsity of SNNs and channel-level firing patterns to distribute filters with lower levels of spike activity onto devices with more constrained resources. The SNN model partitioning and deployment is formulated as an optimization problem …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Efficient Joint Communication and Computation Placement for Large-scale SNN Simulation on Supercomputers</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11183793/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Network (SNN) simulation involves emulating the activation and firing of spiking neurons on hardware platforms. This is a highly time-sensitive task, requiring the simulation of billions of neurons and their intercommunication within a few milliseconds. Each neuron performs a complex, interdependent multi-stage communication and computation task. We consider the task placement of SNN on supercomputers to accelerate SNN simulation. Existing task placement methods for SNN simulations have two major limitations. First, they lack the capability to handle large-scale SNNs with billions of neurons. Second, they focus primarily on optimizing communication delay, while neglecting multi-stage computation delays in SNN simulations. In this paper, we formalize the SNN Joint Multi-stage Communication and Computation Placement (SJCCP) problem. We demonstrate that SJCCP can be solved using an …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Edge Intelligence with Spiking Neural Networks</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2507.14069" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
eeded
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">ECC-SNN: Cost-Effective Edge-Cloud Collaboration for Spiking Neural Networks</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2505.20835" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Most edge-cloud collaboration frameworks rely on the substantial computational and storage capabilities of cloud-based artificial neural networks (ANNs). However, this reliance results in significant communication overhead between edge devices and the cloud and high computational energy consumption, especially when applied to resource-constrained edge devices. To address these challenges, we propose ECC-SNN, a novel edge-cloud collaboration framework incorporating energy-efficient spiking neural networks (SNNs) to offload more computational workload from the cloud to the edge, thereby improving cost-effectiveness and reducing reliance on the cloud. ECC-SNN employs a joint training approach that integrates ANN and SNN models, enabling edge devices to leverage knowledge from cloud models for enhanced performance while reducing energy consumption and processing latency. Furthermore, ECC-SNN features an on-device incremental learning algorithm that enables edge models to continuously adapt to dynamic environments, reducing the communication overhead and resource consumption associated with frequent cloud update requests. Extensive experimental results on four datasets demonstrate that ECC-SNN improves accuracy by 4.15%, reduces average energy consumption by 79.4%, and lowers average processing latency by 39.1%.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A Data Replication Placement Strategy for the Distributed Storage System in Cloud-Edge-Terminal Orchestrated Computing Environments</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/11016095/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Cloud-edge-terminal orchestrated computing, as an expansion of cloud computing, has sunk resources to the edge nodes and terminal equipment, which can provide high-quality services for delay-sensitive applications and reduce the cost of network communication. Due to the high volume of data generated by Internet of Things (IoT) devices and the limited storage capacities of edge nodes, a significant number of terminal devices are now being considered for utilization as storage nodes. However, because of the heterogeneous storage capacity and reliability of these hardware devices and the different data requirements of user services, the performance and storage reliability of applications deployed in cloud-edge-terminal orchestrated computing environments have become urgent problems to be solved. Especially, for a distributed storage system in these environments, it is required to ensure reliable storage …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Concept Enhancement Engineering: A Lightweight and Efficient Robust Defense Against Jailbreak Attacks in Embodied AI</h3>

<div class="paper-meta">📄 2025</div>

<a class="paper-link" href="https://arxiv.org/abs/2504.13201" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Embodied Intelligence (EI) systems integrated with large language models (LLMs) face significant security risks, particularly from jailbreak attacks that manipulate models into generating harmful outputs or executing unsafe physical actions. Traditional defense strategies, such as input filtering and output monitoring, often introduce high computational overhead or interfere with task performance in real-time embodied scenarios. To address these challenges, we propose Concept Enhancement Engineering (CEE), a novel defense framework that leverages representation engineering to enhance the safety of embodied LLMs by dynamically steering their internal activations. CEE operates by (1) extracting multilingual safety patterns from model activations, (2) constructing control directions based on safety-aligned concept subspaces, and (3) applying subspace concept rotation to reinforce safe behavior during inference. Our experiments demonstrate that CEE effectively mitigates jailbreak attacks while maintaining task performance, outperforming existing defense methods in both robustness and efficiency. This work contributes a scalable and interpretable safety mechanism for embodied AI, bridging the gap between theoretical representation engineering and practical security applications. Our findings highlight the potential of latent-space interventions as a viable defense paradigm against emerging adversarial threats in physically grounded AI systems.
</p>

</div>
</details>


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2024</summary>

<div class="paper-card">

<h3 class="paper-title">Exploiting label skewness for spiking neural networks in federated learning</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2412.17305" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The energy efficiency of deep spiking neural networks (SNNs) aligns with the constraints of resource-limited edge devices, positioning SNNs as a promising foundation for intelligent applications leveraging the extensive data collected by these devices. To address data privacy concerns when deploying SNNs on edge devices, federated learning (FL) facilitates collaborative model training by leveraging data distributed across edge devices without transmitting local data to a central server. However, existing FL approaches struggle with label-skewed data across devices, which leads to drift in local SNN models and degrades the performance of the global SNN model. In this paper, we propose a novel framework called FedLEC, which incorporates intra-client label weight calibration to balance the learning intensity across local labels and inter-client knowledge distillation to mitigate local SNN model bias caused by label absence. Extensive experiments with three different structured SNNs across five datasets (i.e., three non-neuromorphic and two neuromorphic datasets) demonstrate the efficiency of FedLEC. Compared to eight state-of-the-art FL algorithms, FedLEC achieves an average accuracy improvement of approximately 11.59% for the global SNN model under various label skew distribution settings.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Darkit: A User-Friendly Software Toolkit for Spiking Large Language Model</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2412.15634" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Large language models (LLMs) have been widely applied in various practical applications, typically comprising billions of parameters, with inference processes requiring substantial energy and computational resources. In contrast, the human brain, employing bio-plausible spiking mechanisms, can accomplish the same tasks while significantly reducing energy consumption, even with a similar number of parameters. Based on this, several pioneering researchers have proposed and implemented various large language models that leverage spiking neural networks. They have demonstrated the feasibility of these models, validated their performance, and open-sourced their frameworks and partial source code. To accelerate the adoption of brain-inspired large language models and facilitate secondary development for researchers, we are releasing a software toolkit named DarwinKit (Darkit). The toolkit is designed specifically for learners, researchers, and developers working on spiking large models, offering a suite of highly user-friendly features that greatly simplify the learning, deployment, and development processes.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">FedLEC: Effective Federated Learning Algorithm with Spiking Neural Networks Under Label Skews</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ui.adsabs.harvard.edu/abs/2024arXiv241217305Y/abstract" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
With the advancement of neuromorphic chips, implementing Federated Learning (FL) with Spiking Neural Networks (SNNs) potentially offers a more energy-efficient schema for collaborative learning across various resource-constrained edge devices. However, one significant challenge in the FL systems is that the data from different clients are often non-independently and identically distributed (non-IID), with label skews presenting substantial difficulties in various federated SNN learning tasks. In this study, we propose a practical post-hoc framework named FedLEC to address the challenge. This framework penalizes the corresponding local logits for locally missing labels to enhance each local model&#x27;s generalization ability. Additionally, it leverages the pertinent label distribution information distilled from the global model to mitigate label bias. Extensive experiments with three different structured SNNs across five …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Simulation and assimilation of the digital human brain</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.nature.com/articles/s43588-024-00731-3" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Here we present the Digital Brain (DB)—a platform for simulating spiking neuronal networks at the large neuron scale of the human brain on the basis of personalized magnetic resonance imaging data and biological constraints. The DB aims to reproduce both the resting state and certain aspects of the action of the human brain. An architecture with up to 86 billion neurons and 14,012 GPUs—including a two-level routing scheme between GPUs to accelerate spike transmission in up to 47.8 trillion neuronal synapses—was implemented as part of the simulations. We show that the DB can reproduce blood-oxygen-level-dependent signals of the resting state of the human brain with a high correlation coefficient, as well as interact with its perceptual input, as demonstrated in a visual task. These results indicate the feasibility of implementing a digital representation of the human brain, which can open the door to a broad …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Mitigating critical nodes in brain simulations via edge removal</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S1389128624006923" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Brain simulation holds promise for advancing our comprehension of brain mechanisms, brain-inspired intelligence, and addressing brain-related disorders. However, during brain simulations on high-performance computing platforms, the sparse and irregular communication patterns within the brain can lead to the emergence of critical nodes in the simulated network, which in turn become bottlenecks for inter-process communication. Therefore, effective moderation of critical nodes is crucial for the smooth conducting of brain simulation. In this paper, we formulate the routing communication problem commonly encountered in brain simulation networks running on supercomputers. To address this issue, we firstly propose the Node-Edge Centrality Addressing Algorithm (NCA) for identifying critical nodes and edges, based on an enhanced closeness centrality metric. Furthermore, drawing on the homology of spikes …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Backdoor attack on vertical federated graph neural network learning</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2410.11290" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Federated Graph Neural Network (FedGNN) integrate federated learning (FL) with graph neural networks (GNNs) to enable privacy-preserving training on distributed graph data. Vertical Federated Graph Neural Network (VFGNN), a key branch of FedGNN, handles scenarios where data features and labels are distributed among participants. Despite the robust privacy-preserving design of VFGNN, we have found that it still faces the risk of backdoor attacks, even in situations where labels are inaccessible. This paper proposes BVG, a novel backdoor attack method that leverages multi-hop triggers and backdoor retention, requiring only four target-class nodes to execute effective attacks. Experimental results demonstrate that BVG achieves nearly 100% attack success rates across three commonly used datasets and three GNN models, with minimal impact on the main task accuracy. We also evaluated various defense methods, and the BVG method maintained high attack effectiveness even under existing defenses. This finding highlights the need for advanced defense mechanisms to counter sophisticated backdoor attacks in practical VFGNN applications.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">CoLLaRS: A cloud–edge–terminal collaborative lifelong learning framework for AIoT</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0167739X24001870" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
AIoT applications often encounter challenges such as terminal resource constraints, data drift, and data heterogeneity in real world, leading to problems such as catastrophic forgetting, low generalization ability, and low accuracy during model training. To address these challenges, we proposed CoLLaRS, a cloud–edge–terminal collaborative lifelong learning framework for AIoT applications. In the CoLLaRS framework, we alleviate the problem of terminal resource constraints by uploading terminal tasks at the edge. CoLLaRS uses continuous training at the edge to achieve lifelong learning training of the model and solve the problem of catastrophic forgetting. CoLLaRS employs federated optimization in the cloud to perform personalized aggregation of different edge models and solve the problem of weak model generalization ability. Finally, the model is fine-tuned at the terminal to further optimize its accuracy in …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">EC-SNN: Splitting Deep Spiking Neural Networks for Edge Devices</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.ijcai.org/proceedings/2024/0596.pdf" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Deep Spiking Neural Networks (SNNs), as an advanced form of SNNs characterized by their multilayered structure, have recently achieved significant breakthroughs in performance across various domains. The biological plausibility and energy efficiency of SNNs naturally align with the requisites of edge computing (EC) scenarios, thereby prompting increased interest among researchers to explore the migration of these deep SNN models onto edge devices such as sensors and smartphones. However, the progress of migration work has been notably challenging due to the influence of the substantial increase in model parameters and the demanding computational requirements in practical applications. In this work, we propose a deep SNN splitting framework named EC-SNN to run the intricate SNN models on edge devices. We first partition the full SNN models into smaller sub-models to allocate their model parameters on multiple edge devices. Then, we provide a channel-wise pruning method to reduce the size of each sub-model, thereby further reducing the computational load. We design extensive experiments on six datasets (ie, four non-neuromorphic and two neuromorphic datasets) to substantiate that our approach can significantly diminish the inference execution latency on edge devices and reduce the overall energy consumption per deployed device with an average reduction of 60.7% and 27.7% respectively while keeping the effectiveness of the accuracy.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Motico: an attentional mechanism network model for smart aging disease risk prediction based on image data classification</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0010482524008485" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The current disease risk prediction model with many parameters is complex to run smoothly on mobile terminals such as tablets and mobile phones in imaginative elderly care application scenarios. In order to further reduce the number of parameters in the model and enable the disease risk prediction model to run smoothly on mobile terminals, we designed a model called Motico (An Attention Mechanism Network Model for Image Data Classification). During the implementation of the Motico model, in order to protect image features, we designed an image data preprocessing method and an attention mechanism network model for image data classification. The Motico model parameter size is only 5.26 MB, and the memory only takes up 135.69 MB. In the experiment, the accuracy of disease risk prediction was 96 %, the precision rate was 97 %, the recall rate was 93 %, the specificity was 98 %, the F1 score was 95 …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">An Asynchronous Multi-core Accelerator for SNN inference</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://arxiv.org/abs/2407.20947" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Spiking Neural Networks (SNNs) are extensively utilized in brain-inspired computing and neuroscience research. To enhance the speed and energy efficiency of SNNs, several many-core accelerators have been developed. However, maintaining the accuracy of SNNs often necessitates frequent explicit synchronization among all cores, which presents a challenge to overall efficiency. In this paper, we propose an asynchronous architecture for Spiking Neural Networks (SNNs) that eliminates the need for inter-core synchronization, thus enhancing speed and energy efficiency. This approach leverages the pre-determined dependencies of neuromorphic cores established during compilation. Each core is equipped with a scheduler that monitors the status of its dependencies, allowing it to safely advance to the next timestep without waiting for other cores. This eliminates the necessity for global synchronization and minimizes core waiting time despite inherent workload imbalances. Comprehensive evaluations using five different SNN workloads show that our architecture achieves a 1.86x speedup and a 1.55x increase in energy efficiency compared to state-of-the-art synchronization architectures.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A Hierarchical Neural Task Scheduling Algorithm in the Operating System of Neuromorphic Computers</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://link.springer.com/chapter/10.1007/978-981-97-5501-1_11" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Bionic computing, increasingly favored for its sophisticated approach to knowledge applications, is experiencing a revolution bolstered by neuromorphic hardware, which delivers versatile solutions in a multitude of scenarios. In this context, we introduce DarwinOS Scheduler-a hierarchical distributed operating system framework optimized for configurable multi-core neuromorphic chips with synchronous communication. This scheduler adeptly manages neuronal computation tasks influenced by data streams, promoting dynamic spiking neural network (SNN) operations where process states switch responsively to activity events. To bolster efficiency, DarwinOS incorporates general-purpose executors for task-related data handling, both prior to and after core processing. Alongside, we propose a neuromorphic task scheduling method, Hierarchical Distributed Scheduling for Neuromorphic tasks (HDSN), that …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Towards transferable adversarial attacks on vision transformers for image classification</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S1383762124000924" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The deployment of high-performance Vision Transformer (ViT) models has garnered attention from both industry and academia. However, their vulnerability to adversarial examples highlights security risks for scenarios such as intelligent surveillance, autonomous driving, and fintech regulation. As a black-box attack technique, transfer attacks leverage a surrogate model to generate transferable adversarial examples to attack a target victim model, which mainly focuses on a forward (input diversification) and a backward (gradient modification) approach. However, both approaches are currently implemented straightforwardly and limit the transferability of surrogate models. In this paper, we propose a Forward-Backward Transferable Adversarial Attack framework (FBTA) that can generate highly transferable adversarial examples against different models by fully leveraging ViT’s distinctive intermediate layer structures …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A balanced and reliable data replica placement scheme based on reinforcement learning in edge–cloud environments</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0167739X24000499" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
With the rapid development of edge–cloud computing, distributing resources to edge nodes and terminal devices to provide high-quality services for latency-sensitive applications and reduce network communication costs has become increasingly important. However, the complexity and heterogeneity of the edge–cloud environment pose significant challenges to the reliability of data storage and device load balancing. To address these issues, in this paper, we propose a Deep Reinforcement Learning (DRL)-based data replica placement scheme, BRPS. This scheme considers the different geographic locations of hardware devices, the heterogeneous storage capacity and reliability, and the different data requirements of varying user services in edge–cloud environments. Firstly, we constructed a model for data replica placement in the edge–cloud environment, addressing key factors such as latency, reliability, and …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">The digital twin of the human brain: Simulation and assimilation</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.researchsquare.com/article/rs-4321313/latest" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
The Digital Twin Brain (DTB) is a simulation of the human brain at the large scale, with up to 86 billion neurons and 47.8 trillion neural synapses. The DTB mimics both the resting state and aspects of the action of its biological counterpart. A novel architectural assignment of neurons to 14,012 GPUs (with 57.39 million cores) and a two-level routing scheme between GPUs was implemented in the simulations. This was combined with development of a hierarchical mesoscale data assimilation method, that is capable of constructing trillions of parameters from estimated hyper-parameters. The constructed DTB reproduces blood-oxygen-level-dependent (BOLD) signals of the resting-state of its living biological counterpart with high correlation (&gt; 0.9). We enabled the DTB to interact with its environment, as demonstrated in a visual task. The result indicates the feasibility of implementing a digital twin of the human brain …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">LAECIPS: Large Vision Model Assisted Adaptive Edge-Cloud Collaboration for IoT-based Embodied Intelligence System</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S2452414X25001785" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Embodied intelligence (EI) enables manufacturing systems to flexibly perceive, reason, adapt, and operate within dynamic shop floor environments. In smart manufacturing, a representative EI scenario is robotic visual inspection, where industrial robots must accurately inspect components on rapidly changing, heterogeneous production lines. This task requires both high inference accuracy—especially for uncommon defects—and low latency to match production speeds, despite evolving lighting, part geometries, and surface conditions. To meet these needs, we propose LAECIPS, a large vision model-assisted adaptive edge-cloud collaboration framework for IoT-based embodied intelligence systems. LAECIPS decouples large vision models in the cloud from lightweight models on the edge, enabling flexible model deployment and continual learning (automated model updates). Through identifying complex …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Laecips: Large vision model assisted adaptive edge-cloud collaboration for iot-based perception system</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ui.adsabs.harvard.edu/abs/2024arXiv240410498H/abstract" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Recent large vision models (eg, SAM) enjoy great potential to facilitate intelligent perception with high accuracy. Yet, the resource constraints in the IoT environment tend to limit such large vision models to be locally deployed, incurring considerable inference latency thereby making it difficult to support real-time applications, such as autonomous driving and robotics. Edge-cloud collaboration with large-small model co-inference offers a promising approach to achieving high inference accuracy and low latency. However, existing edge-cloud collaboration methods are tightly coupled with the model architecture and cannot adapt to the dynamic data drifts in heterogeneous IoT environments. To address the issues, we propose LAECIPS, a new edge-cloud collaboration framework. In LAECIPS, both the large vision model on the cloud and the lightweight model on the edge are plug-and-play. We design an edge-cloud …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">HRCM: A Hierarchical Regularizing Mechanism for Sparse and Imbalanced Communication in Whole Human Brain Simulations</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10496893/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Brain simulation is one of the most important measures to understand how information is represented and processed in the brain, which usually needs to be realized in supercomputers with a large number of interconnected graphical processing units (GPUs). For the whole human brain simulation, tens of thousands of GPUs are utilized to simulate tens of billions of neurons and tens of trillions of synapses for the living brain to reveal functional connectivity patterns. However, as an application of the irregular spares communication problem on a large-scale system, the sparse and imbalanced communication patterns of the human brain make it particularly challenging to design a communication system for supporting large-scale brain simulations. To face this challenge, this paper proposes a hierarchical regularized communication mechanism, HRCM. The HRCM maintains a hierarchical virtual communication …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A lightweight neural network model for disease risk prediction in edge intelligent computing architecture</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.mdpi.com/1999-5903/16/3/75" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In the current field of disease risk prediction research, there are many methods of using servers for centralized computing to train and infer prediction models. However, this centralized computing method increases storage space, the load on network bandwidth, and the computing pressure on the central server. In this article, we design an image preprocessing method and propose a lightweight neural network model called Linge (Lightweight Neural Network Models for the Edge). We propose a distributed intelligent edge computing technology based on the federated learning algorithm for disease risk prediction. The intelligent edge computing method we proposed for disease risk prediction directly performs prediction model training and inference at the edge without increasing storage space. It also reduces the load on network bandwidth and reduces the computing pressure on the server. The lightweight neural network model we designed has only 7.63 MB of parameters and only takes up 155.28 MB of memory. In the experiment with the Linge model compared with the EfficientNetV2 model, the accuracy and precision increased by 2%, the recall rate increased by 1%, the specificity increased by 4%, the F1 score increased by 3%, and the AUC (Area Under the Curve) value increased by 2%.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Universal adversarial backdoor attacks to fool vertical federated learning</h3>

<div class="paper-meta">📄 2024</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S0167404823005114" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Vertical federated learning (VFL) is a privacy-preserving distribution learning paradigm that enables participants, owning different features of the same sample space to train a machine learning model collaboratively while retaining their data locally. This paradigm facilitates improved efficiency and security for participants such as financial or medical fields, making VFL an essential component of data-driven Artificial Intelligence systems. Nevertheless, the partitioned structure of VFL can be exploited by adversaries to inject a backdoor, enabling them to manipulate the VFL predictions. In this paper, we aim to investigate the vulnerability of VFL in the context of binary classification tasks. To this end, we define a threat model for backdoor attacks in VFL and introduce a universal adversarial backdoor (UAB) attack to poison the predictions of VFL. The UAB attack, consisting of universal trigger generation and clean-label …
</p>

</div>
</details>


<details class="year-block" open>
<summary class="year-summary"><span class="icon">📅</span>2023</summary>

<div class="paper-card">

<h3 class="paper-title">Fidan: a predictive service demand model for assisting nursing home health-care robots</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.tandfonline.com/doi/abs/10.1080/09540091.2023.2267791" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
While population aging has sharply increased the demand for nursing staff, it has also increased the workload of nursing staff. Although some nursing homes use robots to perform part of the work, such robots are the type of robots that perform set tasks. The requirements in actual application scenarios often change, so robots that perform set tasks cannot effectively reduce the workload of nursing staff. In order to provide practical help to nursing staff in nursing homes, we innovatively combine the LightGBM algorithm with the machine learning interpretation framework SHAP (Shapley Additive exPlanations) and use comprehensive data analysis methods to propose a service demand prediction model Fidan (Forecast service demand model). This model analyzes and predicts the demand for elderly services in nursing homes based on relevant health management data (including physiological and sleep data), ward …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">ACSarF: a DRL-based adaptive consortium blockchain sharding framework for supply chain finance</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.sciencedirect.com/science/article/pii/S2352864823001724" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Blockchain technologies have been used to facilitate Web 3.0 and FinTech applications. However, conventional blockchain technologies suffer from long transaction delays and low transaction success rates in some Web 3.0 and FinTech applications such as Supply Chain Finance (SCF). Blockchain sharding has been proposed to improve blockchain performance. However, the existing sharding methods either use a static sharding strategy, which lacks the adaptability for the dynamic SCF environment, or are designed for public chains, which are not applicable to consortium blockchain-based SCF. To address these issues, we propose an adaptive consortium blockchain sharding framework named ACSarF, which is based on the deep reinforcement learning algorithm. The proposed framework can improve consortium blockchain sharding to effectively reduce transaction delay and adaptively adjust the sharding …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Improved Random Forest Based Anomaly Detection for Urban Rail Transits</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10349185/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
As the construction of urban rail transit is rapidly expanding, the maintenance costs associated with it are also increasing. However, traditional anomaly detection methods have limitations as they require setting thresholds for various indicators to monitor rail transit status. To address this issue, there is a pressing need for utilizing machine learning methods to detect possible anomalies in trains. This paper proposes an anomaly detection method based on random forest, which can estimate the abnormality score of the train&#x27;s traction motor based on low-dimensional sensor data and insufficient labels collected by the motor sensor. This approach can identify potential anomalies of the motor and help maintenance personnel develop specific plans to address them. In addition to the proposed anomaly detection method, we have designed a train traction motor anomaly alarm system that effectively detects potential …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Digital Twin Brain: a simulation and assimilation platform for whole human brain</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2308.01241" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
In this work, we present a computing platform named digital twin brain (DTB) that can simulate spiking neuronal networks of the whole human brain scale and more importantly, a personalized biological brain structure. In comparison to most brain simulations with a homogeneous global structure, we highlight that the sparseness, couplingness and heterogeneity in the sMRI, DTI and PET data of the brain has an essential impact on the efficiency of brain simulation, which is proved from the scaling experiments that the DTB of human brain simulation is communication-intensive and memory-access intensive computing systems rather than computation-intensive. We utilize a number of optimization techniques to balance and integrate the computation loads and communication traffics from the heterogeneous biological structure to the general GPU-based HPC and achieve leading simulation performance for the whole human brain-scaled spiking neuronal networks. On the other hand, the biological structure, equipped with a mesoscopic data assimilation, enables the DTB to investigate brain cognitive function by a reverse-engineering method, which is demonstrated by a digital experiment of visual evaluation on the DTB. Furthermore, we believe that the developing DTB will be a promising powerful platform for a large of research orients including brain-inspiredintelligence, rain disease medicine and brain-machine interface.
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">HSFL: Efficient and privacy-preserving offloading for split and federated learning in IoT services</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10248280/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Distributed machine learning methods like Federated Learning (FL) and Split Learning (SL) meet the growing demands of processing large-scale datasets under privacy restrictions. Recently, FL and SL are combined in hybrid SLFL (SFL) frameworks to exploit both methods’ advantages to facilitate ubiquitous intelligence in the Internet of Things (IoT), for example, smart finance. Despite its significant impact on the performance and costs of SFL, model decomposition that splits an ML model into the client-server pair has not been sufficiently studied, especially for SFL in a large-scale dynamic IoT environment. In this paper, we propose a new SFL framework HSFL with a lightweight model decomposition method to offload a part of model training to the edge server. Specifically, we develop a method for estimating the training latency of HSFL and designed a metric for measuring privacy leakage in HSFL, based on …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">A blockchain-assisted intelligent edge cooperation system for iot environments with multi-infrastructure providers</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://ieeexplore.ieee.org/abstract/document/10144289/" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
While edge computing has the potential to offer low-latency services and overcome the limitations of traditional cloud computing, it presents new challenges in terms of trust, security, and privacy (TSP) in Internet of Things environments. Cooperative edge computing (CEC) has emerged as a solution to address these challenges through resource sharing among edge nodes. However, for multi-infrastructure providers, incentive and trust mechanisms among edge nodes are crucial technical issues that must be addressed alongside system latency and reliability to meet performance requirements. In this article, we propose a blockchain-assisted intelligent edge cooperation system (BIECS) to systematically solve these issues. By leveraging blockchain technology, we construct trust among edge nodes and employ an incentive mechanism for resource sharing among multi-infrastructure providers. We formulate the …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Universal adversarial backdoor attacks to fool vertical federated learning in cloud-edge collaboration</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://arxiv.org/abs/2304.11432" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
Vertical federated learning (VFL) is a cloud-edge collaboration paradigm that enables edge nodes, comprising resource-constrained Internet of Things (IoT) devices, to cooperatively train artificial intelligence (AI) models while retaining their data locally. This paradigm facilitates improved privacy and security for edges and IoT devices, making VFL an essential component of Artificial Intelligence of Things (AIoT) systems. Nevertheless, the partitioned structure of VFL can be exploited by adversaries to inject a backdoor, enabling them to manipulate the VFL predictions. In this paper, we aim to investigate the vulnerability of VFL in the context of binary classification tasks. To this end, we define a threat model for backdoor attacks in VFL and introduce a universal adversarial backdoor (UAB) attack to poison the predictions of VFL. The UAB attack, consisting of universal trigger generation and clean-label backdoor injection, is incorporated during the VFL training at specific iterations. This is achieved by alternately optimizing the universal trigger and model parameters of VFL sub-problems. Our work distinguishes itself from existing studies on designing backdoor attacks for VFL, as those require the knowledge of auxiliary information not accessible within the split VFL architecture. In contrast, our approach does not necessitate any additional data to execute the attack. On the LendingClub and Zhongyuan datasets, our approach surpasses existing state-of-the-art methods, achieving up to 100\% backdoor task performance while maintaining the main task performance. Our results in this paper make a major advance to revealing the hidden backdoor risks …
</p>

</div>

<div class="paper-card">

<h3 class="paper-title">Lidom: a disease risk prediction model based on LightGBM applied to nursing homes</h3>

<div class="paper-meta">📄 2023</div>

<a class="paper-link" href="https://www.mdpi.com/2079-9292/12/4/1009" target="_blank">🔗 Read Paper</a>

<p class="paper-abstract">
With the innovation of technologies such as sensors and artificial intelligence, some nursing homes use wearable devices to monitor the movement and physiological indicators of the elderly and provide prompts for any health risks. Nevertheless, this kind of risk warning is a decision based on a particular physiological indicator. Therefore, such decisions cannot effectively predict health risks. To achieve this goal, we propose a model Lidom (A LightGBM-based Disease Prediction Model) based on the combination of the LightGBM algorithm, InterpretML framework, and sequence confrontation network (SeqGAN). The Lidom model first solves the problem of uneven samples based on the sequence confrontation network (SeqGAN), then trains the model based on the LightGBM algorithm, uses the InterpretML framework for analysis, and finally obtains the best model. This paper uses the public dataset MIMIC-III, subject data, and the early diabetes risk prediction dataset in UCI as sample data. The experimental results show that the Lidom model has an accuracy rate of 93.46% for disease risk prediction and an accuracy rate of 99.8% for early diabetes risk prediction. The results show that the Lidom model can provide adequate support for the prediction of the health risks of the elderly.
</p>

</div>
</details>

