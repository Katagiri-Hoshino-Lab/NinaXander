# 00 · 概述

[English](../00-overview.md) | [日本語](../ja/00-overview.md) | 简体中文

## 目标

**关于名称。** NinaXander有意取自《钢之炼金术师》中的奇美拉——Nina Tucker与她的狗Alexander
被融合成一个会说话的生物。选择这一名称是因为这种对应关系，而不只是“奇美拉 = 融合”：
融合成功了，那个奇美拉也确实*能运行*（会说话），却不及融合前的存在。这正是本项目的结论：
组合模型可以运行，但没有一个达到更强的父模型。

**NinaXander**是本项目所生成的一族奇美拉语言模型的名称：这些组合模型由不同预训练架构族的模块组成，
模块保持**冻结**并通过同一个学习得到的共享潜空间适配器相连。单个NinaXander模型由其跨族路径与切换层指定
（例如`NinaXander-BA@4`，即L=4的Pythia→RWKV；见下文命名）。当前研究在四个固定位置评估
一个模型族边界；它**没有**报告对任意逐层选择的搜索。复现contract只包含当前32层实验。

不同模型族的内部表征位于不兼容的空间中（宽度不同、基不同）。适配器负责把一个模型族空间中的
残差转换到另一个模型族空间，使外来模块能够接着运行。

## 难点

两个冻结模型使用不兼容的残差空间。一个有用的接口必须保留足够信息，使另一模型族能够继续推理；
同时又必须足够小，不能只是在两个父模型之间另学一个新模型。

本项目使用**共享潜空间适配器**重新研究这个问题：两个模型族编码到、解码自同一个潜空间，
并在所有层上共同训练。研究以功能性方式询问组合能否工作，而不只看对齐R²。

## 四路径命名

公开记法是2×2执行路径矩阵：**A = RWKV**，**B = Tulu-Pythia**；
第一个字母表示prefix模型族，第二个表示suffix/read-out模型族。

| prefix ↓ / suffix → | A（RWKV） | B（Pythia） |
|---|---|---|
| A（RWKV） | `NinaXander-AA@L`（同族对照） | `NinaXander-AB@L`（跨族奇美拉） |
| B（Pythia） | `NinaXander-BA@L`（跨族奇美拉） | `NinaXander-BB@L`（同族对照） |

`@L`表示prefix运行到第L块，read-out/suffix从第L+1块开始。CSV metadata使用
`A_to_A`、`A_to_B`、`B_to_B`和`B_to_A`。
对角单元`NinaXander-AA@L`与`NinaXander-BB@L`只是沿用命名方式的同族对照，
并不是上文所定义的NinaXander模型（后者总是跨越模型族边界）。

## 当前、如实的论点

1. 两个模型族都能编码到的共享潜空间**确实存在**，并且可以在所有层上共同学习；
   但质量必须用**centered** alignment衡量，不能用raw correlation。由于使用non-affine
   LayerNorm，raw corr只是loss的代数重述；optimizer可以通过不携带信息的token无关common mode
   免费满足它。详见[02-findings.md](02-findings.md)。
2. 借助共享潜空间，**单边界奇美拉可以运行**——虽有退化，但没有崩溃。这与此前evolutionary
   experiment的灾难性结果不同。
3. 残差接口是**有损的**，但当前7B对照不能确定translation是唯一代价。观察到的差距包含
   同族重构（encode–decode往返）项和cross-family-substitution项；后者混合了两个prefix父模型的能力差
   与translation error。在7B上，配对read-out满足
   **A→B 0.559 < B→B 0.711**和**B→A 0.590 < A→A 0.695**，且在每个测量层都成立。
   因此即使在较高的潜空间对齐（ρ_ctr 0.901）下，同族重构本身就已不完美，而cross read-out还低于它。
   该误差来自encoder、潜空间归一化、decoder还是训练时noise，尚未区分；结合第5项的domain shift，
   障碍在于不完美的对齐、重构误差与domain shift。限制self-map的因素仍未解决：潜空间与残差同宽，
   non-affine LN每个token只丢弃均值与范数两个自由度，因此宽度或秩不太可能是强约束；
   noise σ是否设定了B→B 0.711的ceiling尚不清楚，可通过σ sweep检验。
   适配器在任一跨族路径上都**不是近线性的**：
   最佳affine imitation解释AB输出的86.6%和BA输出的88.0%，direct affine map达到
   0.487 / 0.535。把这个联合训练的适配器的residual-block branch设为α=0，会使两条read-out跌到负R²，
   两条路径的下游准确率也随之崩溃；但这并不说明非线性普遍必要，而且在BA准确率上，单独拟合的affine对照
   与适配器的差距很小（−3.4至+2.0分）。
4. 在完整测试集上，任一跨族路径都未超过**更强的**Pythia父模型。AB@4得到
   SciQ 70.9 / ARC 59.6，只在SciQ上超过RWKV，在ARC上更差。BA@4得到69.2 / 62.2，
   在两个任务上都与RWKV统计持平。深层BA切换明显强于同层AB
   （L=16/24时高8.9至18.8个百分点），说明深度效应依赖路径。同族重构对照（AA/BB）
   隔离的是encode–decode往返造成的损失，该损失在BB的每个切换层、两个任务上，以及AA在L=16与L=24的
   两个任务上都显著体现在下游准确率中；但它们与跨族路径的差距混合了前半父模型能力与跨族translation，
   不能作为纯translation error估计。
5. 实测最佳内存点是`NinaXander-BA@4`：5个Transformer块、**削减84.375% KV**、
   ARC 62.2、SciQ 69.2。双任务均值65.70只比AB@4的65.25高0.45个百分点，而且是在相同报告测试上
   从2条跨族路径 × 4个切换点中post-hoc选出的，没有独立deployment-selection split。
   物理剪枝BA模型占14.11 GiB，并精确复现未剪枝路径。不过纯RWKV仍为零KV，
   匹配的Pythia提前退出权重更少，因此这种组合是一种权衡，而不是普遍Pareto最优架构。
   domain shift也依然存在：BA@4把WikiText context-2048 perplexity从AB@4的83.4改善到34.8，
   但更好的父模型仍为8.27。

## 目标模型对（已构建；适配器已到平台；奇美拉已测量）

早期使用的base 130M模型本身无法对话，因此无法回答“奇美拉是否保留*对话*能力”。
本项目为此选择了共享GPT-NeoX tokenizer的纯非Transformer **instruct**模型对：

| | RWKV-4-Raven-7B | open-instruct-pythia-6.9b-tulu |
|---|---|---|
| 架构 | 纯RNN（零attention） | Transformer |
| 块数 | 32 | 32 |
| 宽度 | 4096 | 4096 |
| tokenizer | GPT-NeoX-20B | GPT-NeoX-20B |
| 微调 | instruct（Alpaca/ShareGPT） | instruct（AllenAI Tulu SFT） |

原计划的Dolly-v2-7b已从HF删除，因此以
`allenai/open-instruct-pythia-6.9b-tulu`（Pythia-6.9b + Tulu SFT）
替代为instruct Transformer。两个模型均以Alpaca指令文本对齐。

相同块数和tokenizer是共享潜空间方法的**必要条件**：残差需要在匹配token位置成对，
所以两个模型必须产生完全相同的tokenization；block *j* ↔ block *j*还要求深度相同。
多数跨架构instruct模型使用不兼容tokenizer，因此很难组成这样的模型对。
RWKV-Raven上游以BlinkDL `.pth`提供，本项目用未收录的一次性helper转换为HF格式，
并针对transformers 5.x进行了patch。详见[03-roadmap.md](03-roadmap.md)。
