## 读过的一些论文

***Transformer*** 抛弃RNN，使用纯注意力

***ViT*** 将Transformer引入视觉领域，将图片划分为patch再输入

***BERT*** 使用Transformer的编码器，通过’完型填空’自监督预训练模型

***MAE*** CV版的BERT，掩蔽75%的patch尝试还原

***GPT*** 使用Transformer的解码器，对下游不同的任务只需要调整输入的形式即可

***SAM*** 分割领域基础模型，将分割的各种任务转换为提示词工程再给SAM做

***MSA*** 在SAM的结构上新增几个adapter以适应医学图像，同时处理了3D数据

***Swim Transformer*** 通过层级采样and在每一个patch内做自主注意力来处理不同尺寸的特征以及减少计算复杂度，通过shifted window变相达到全局注意力

***CLIP*** 多模态图片分类任务，可以不局限于训练中的categorical label

***Bidirectional LSTM-CRF Models for Sequence Tagging*** baidu在2015年首次将LSTM+CRF用于序列标注

***Text Segmentation as a Supervised Learning Task*** 将文章视作句子序列，再在句子之间添加分隔符

***Sentence BERT*** 使用siamese和triplet预训练BERT，使得BERT产生更好的sentence embedding

***Parameter-Efficient Transfer Learning for NLP*** 微调BERT，在模型中添加adapter，只训练adapter

***Prefix-Tuning:Optimizing Continuous Prompts for Generation*** 在输入模型token之前添加一段可学习的prefix向量而整个模型无需任何改变

***The Power of Scale for Parameter-Efficient Prompt Tuning*** 微调T5，prefix tuning的简化版，去掉了输入前的MLP，同时发现模型越大效果越好

***GPT2*** 参数扩大到1.5B，同时聚焦模型的zero-shot能力，即无需微调，转而使用预训练+prompt

***GPT3*** 参数扩大到175B，主要使用few-shot（这里给的标记数据<也就是例子>也是给在prompt中）

***P-tuning*** 同时支持GPT和BERT，将NLU的任务都转换为MLM任务（也就是完形填空），离散可微的prompt位置不固定

***P-tuning V2*** 让prompt微调在不同尺寸不同任务上匹敌full tuning，深度提示调优

***AlphaFlod2*** 给定氨基酸序列预测蛋白质三维结构

***Neural Machine Reading Comprehension: Methods and Trends*** 机器阅读理解综述

***PubMedBERT*** 在PubMed数据上进行从头训练（而不是先前的在通用语料上预训练再微调），还提出了新的benchmark-BLURE

***CMGN*** 分子生成模型，逆向药物设计

***BART*** BERT+GPT，在MLM模型的基础上，允许任何的对原始文本的破环方案，重点关注训练和推理时的不同

***FP-GNN*** 结合分子指纹和分子图预测分子性质

***MINN-DTI*** 使用蛋白质的成对距离图（表示3D信息）和药物分子图作为输入

***AdaptFormer*** 视觉adapter微调方法

***Deep-DTA*** 使用卷积处理蛋白质和分子的序列，预测药物和靶点的亲和力得分（回归任务）

***Towards Segment Anything Model (SAM) for Medical Image Segmentation*** 综述，将SAM运用到医学图像分割领域

***An Exploration of 2D and 3D Deep Learning Techniques for Cardiac MR Image Segmentation*** 使用UNet处理ACDC数据集

***UNet++*** UNet改进版本，来自编码器的特征图在连接之前融合了更多信息

***InstrucrGPT*** 使用基于人类反馈的强化学习将GPT3训练成更加符合人类意图（显式和隐式）的对话模型

***LLaMA*** 只使用公开数据集，更小的模型尺寸也可以达到好的性能

***TRPO*** 信任域策略优化

***UniLM*** BERT结构模型，通过不同的掩码策略训练（三种：单项、双向、序列到序列）完成理解和生成任务

***Self Instruct*** 使用模型生成指令微调数据，迭代训练

***TransE*** 平移模型，将关系嵌入为和实体同纬度的向量，三元组关系成立时认为 h+r = t

***TransH*** 平移模型，将h和t投影到超平面上进行平移

***TransR*** 平移模型，实体和关系分别在不同的空间中嵌入，计算时将实体投影到关系空间中进行平移

***TransD*** 平移模型，为实体和关系分别定义了两个向量，第一个向量表示实体或关系的意义；另一个向量表示将实体嵌入向量投影到关系向量空间中

***RotatE*** 将关系定义为复数向量空间中从头实体到尾实体的旋转

***RESCAL*** 三维张量分解

***DistMult*** 限制关系矩阵为对角矩阵以减少参数量

***HolE*** 使用循环相关算子计算交互

***ComplEx*** 复数版本的DistMult

***SimplE*** 为实体和关系分别学习两个向量并通过评分函数进行联系

***ANALOGY*** 建模类比结构

***R-GCN*** 在GCN的基础上，为每个关系（包括in/out关系）分开建模

***ConvE*** 使用图像卷积的思路学习嵌入表示

***CompGCN*** 同时建模实体和关系，扩展逆向关系和自连接关系

***OpenKE*** 知识图谱嵌入工具包

***TypeConstrained*** 类型限制方法，在替换实体仅仅替换那些在KG 中与该关系同时出现过的实体

**Bert-WWM** 使用全词掩码（whole word mask）训练BERT模型，并使用LTP的分词作为最小的掩盖单元

**Relation Classification via Convolutional Deep Neural Network** 将关系抽取考虑为分类任务，同时基于词嵌入设计实体词特征和整体的句特征，最后拼接做分类

***DeepKE*** 知识抽取工具包

***GPT-NER*** 将序列标注的任务转换为生成任务，即要求LLM使用特殊字符将实体包裹起来，其余部分的输出与输入完全相同，还提出了样本检索的策略

***Large Language Models for Generative Information Extraction: A Survey*** 大模型信息抽取综述，从信息抽取子任务和学习范式两个不同的维度

***Empirical Study of Zero-Shot NER with ChatGPT*** 提出了三种增强 LLM 在 NER 任务上的 zero-shot 能力

***Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models*** 提出了在严格的zero shot的情况下，无训练而可以自我提升的框架

***Self-Consistency Improves Chain of Thought Reasoning in Language Models*** 用思维链提示中的自洽解码替代简单的贪婪解码

***GPT-RE*** 优化使用大模型做关系抽取时示例的检索策略

***QA4RE*** 将关系抽取作为多项选择题，即为实体对选择正确的关系

***A Survey of Chain of Thought Reasoning: Advances, Frontiers and Future*** 广义思维链综述

***ChatIE*** 通过两阶段的框架（ChatIE）将zero-shot的IE任务转换为多轮问答问题

***GraphRAG*** 使用LLM构建知识图谱，解决传统RAG不能回答文档高层次的问题

***AgentRE*** 使用Anget进行信息抽取

***ART** / **ReAct*** 将大模型的推理（CoT）和决策（Action 行动）结合

***GOT*** 编码器-解码器架构的纯OCR模型，拒绝使用视觉大模型做OCR（因其本质任务还是VQA） ，编码器使用 VitDet（local attention） 解码器使用Qwen-0.5B

***Cross-Domain Document Object Detection: Benchmark Suite and Method*** 跨领域文档目标检测，使用source domain labled data 和 target domain unlabled data 训练

***DERT***  使用transformer进行端到端目标检测

***GAN*** 生成模型和自然模型对抗以最好的捕获数据分布

***Unsupervised Domain Adaptation by Backpropagation*** 类似GAN的思想训练一个分类器来实现domain adaptation，同时提出了梯度反转的trick

***Multiscal Domain Adaptive YOLO for Cross-Ddmain Object Detection*** 域适应的yolov4

***PrimeKG*** 聚焦疾病的生物医学知识图谱

***SAC-KG*** 使用生成器、验证器、修剪器使用大模型从一个实体开始不断迭代生成三元组

***A novel prompting method for few-shot NER via LLMs*** 提出了三种增强promot用以进行NER的方法

**DeepSeek-R1** 纯RL复现o1推理模型

***DDPM*** 去噪扩散模型

***DiT*** 基于Transformer的扩散模型，替换传统的UNet网络

***DITAM*** 一个统一的DTI、DTA和MoA预测框架
