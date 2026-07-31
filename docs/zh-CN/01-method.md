# 01 · 方法

[English](../01-method.md) | [日本語](../ja/01-method.md) | 简体中文

## 残差对（数据）

向共享tokenizer的两个冻结模型A、B输入**相同文本**。在token位置*t*和block索引*j*，
分别取两个模型第*j*块的输出，形成一个**索引匹配、位置匹配的残差对**。

残差通过**每个块的forward hook**捕获，而不是从`hidden_states`读出。不同模型族的
`hidden_states`并不代表统一的量，其最后一项甚至不一定是块输出：

- RNN/SSM模型（RWKV、Mamba）在块循环*内部*append，因此`hidden_states[j]`是第*j*块输出，
  `ln_out`作为单独的最后一项追加。
- Transformer（GPTNeoX）先放入embedding，因此第*j*块位于`hidden_states[j+1]`，
  **最后一项为`final_layer_norm(block_{L-1})`**。最后一个块的原始输出从未暴露，
  而且norm丢弃每个token的均值和scale，所以不可逆。

因此报告适配器使用forward-hook捕获：两个模型族的所有层都对应同一种量，包含layer 0，
且32个块都与原始块输出配对。hook必须处理两个风险，并通过**gate验证**而非假定：
RWKV在块返回*之后*每隔`config.rescale_every`个块将stream减半（hook必须复现该操作）；
`RwkvBlock`返回tuple，而`GPTNeoXLayer`返回裸tensor。

当前extractor从相同prompt/token/layer索引构建两份cache，并将hook捕获值与
Hugging Face forward输出进行gate检查。

## 共享潜空间适配器

每个模型各有一个encoder和decoder；两个encoder映射到由**所有层共享的同一个潜空间**
`z ∈ ℝ^d_z`（没有layer conditioning——同一个E_A同时服务layer 1和layer 23）：

```
z_A = LN(E_A(r_A)),   z_B = LN(E_B(r_B))          LN = non-affine LayerNorm
r̂_A = D_A(z_A + ξ),  r̂_B = D_B(z_B + ξ)          ξ ~ N(0, σ² I)
L = ‖r̂_A − r_A‖² + ‖r̂_B − r_B‖² + λ · ‖z_A − z_B‖²        λ = 1
```

每一项都是逐元素均值（`F.mse_loss`），而不是平方范数。残差进入适配器前按维度标准化。

**两条cross-map都没有对跨族target拟合。** 对
`A→B := D_B(E_A(r_A))`而言，`r_A`不出现在B的重构loss中；对
`B→A := D_A(E_B(r_B))`也完全对称。它们没有被直接训练。但这**不是**“unsupervised”：
alignment项`λ‖z_A − z_B‖²`使用了与supervised bridge完全相同的位置对齐跨族样本，
并提供配对关系。准确且更窄的表述是：**两条路径都不需要cross-reconstruction target**。

### 两个关键设计选择

- **对z使用non-affine LayerNorm。** 如果没有它，align项存在scale退化：
  optimizer缩小`z`、decoder再放大，即使没有任何对齐，align也会趋近0。LN固定`‖z‖²`。
- **解码前加入noise ξ（σ > 0）。** LN固定z的*范数*，但不固定token无关common mode与
  token变化signal之间的分配。报告的32层模型固定σ=0.4。目前没有匹配的σ sweep，
  因此本工作不声称0.4最优，也不把最终重构ceiling归因于noise。

可选的`lam_cross > 0`会增加supervised cross项
`‖D_B(z_A) − r_B‖² + ‖D_A(z_B) − r_A‖²`。报告checkpoint未使用它；
当前结果采用`lam_cross = 0`。

## 四条执行路径

全文采用**A = RWKV**、**B = Tulu-Pythia**。路径名按prefix、suffix/read-out顺序写作
`AA`、`AB`、`BB`或`BA`。公开模型名为`NinaXander-XY@L`，结构化数据使用`X_to_Y`。

两条跨族路径都在layer *L***只切换一次**模型族，并被直接执行：

| 路径 | prefix块 | 一次转换 | suffix块 + head | 保留Pythia块 |
|---|---|---|---|---:|
| `NinaXander-AB@L` | RWKV `0..L` | A空间 → B空间 | Pythia `L+1..31` | `31−L` |
| `NinaXander-BA@L` | Pythia `0..L` | B空间 → A空间 | RWKV `L+1..31` | `L+1` |

```
h_A = A.hidden_states[L]                                  # RWKV第L块输出（A已运行0..L，共L+1块）
h_B = unstd_B( D_B( E_A( std_A(h_A) ) ) )                # A空间 → B空间，仅一次转换
inject h_B as the input to B.block_{L+1}                  # B运行31-L个块 + head

h_B = B.hidden_states[L+1]                                # Pythia第L块输出
h_A = unstd_A( D_A( E_B( std_B(h_B) ) ) )                # B空间 → A空间，仅一次转换
inject h_A as the input to A.block_{L+1}                  # RWKV运行31-L个块 + head
```

划分是**L+1 / 31−L**，而不是L / 32−L：两个prefix都已经执行了*L+1*个块。
因此AB@16为17个RWKV + 15个Pythia块，BA@16为17个Pythia + 15个RWKV块。
AB余下的Pythia块通过带pre-hook、替换注入层输入的**HF自身forward**运行，
不手工重写GPTNeoX的partial rotary和causal mask。

BA同样直接评估，而不是从表征R²推断。三个独立gate保护这条较特殊的路径：
通过HF pre-hook重新注入RWKV真实post-L残差必须恢复纯RWKV logits；
无需prefix的direct RWKV suffix必须恢复相同logits；物理剪枝的Pythia-front/RWKV-back实现
必须与未剪枝hook路径一致。每个gate都必须低于`1e-4`。
full QA、generation和long-context CE使用direct suffix，避免每个arm重新计算已丢弃的RWKV prefix。
未剪枝serving benchmark特意测量完整re-forward实现，剪枝benchmark则测量可部署的截块路径。

QA答案选项始终采用标准RWKV batch size 1。拼接adapter、affine和A→A-control arm会使logits产生
相对`1.8e-3`变化，因而被equivalence gate拒绝。独立batch-one arm只有在所有切换点都通过
sequential/concurrent gate后，才可放到不同CUDA stream运行；这样保留batch-one kernel路径，
不会产生数值不同的batched结果。

双向checkpoint本身不变。仅运行BA时，不可达的E_A/D_B模块可以留在CPU上；
QA保留E_A用于A→A decoder-loss对照，而D_B在所有BA arm中都不可达。
每项定量BA比较都固定为通过held-out A→B选定的同一公开step-415,000 checkpoint，
不会查看B→A或BA QA后重新选择checkpoint。

其余两个单元`NinaXander-AA@L`与`NinaXander-BB@L`是同族重构对照。
它们在同一边界使用相同encoder/decoder接口，但不改变模型族。
四个单元共同报告在`paper_four_path_qa_accuracy.csv`中。

当前构造只有一个模型族边界，因此只有一次转换。多边界组合不属于报告实验。

## 指标

- cross-map `A→B`、`B→A`和self-map `A→A`、`B→B`的**held-out逐维R²**。
  held-out为不相交的半份语料，当前extractor固定该划分。
- **centered alignment** ρ和**token-varying fraction**
  f = tr(Cov_t(z)) / E‖z‖²。`align = 2·f·(1 − ρ)`。
  应报告ρ与f，**绝不报告raw corr(z_A, z_B)**，因为它在构造上等于`1 − align/2`。
- 用于功能测试的**奇美拉CE / perplexity**和确定性greedy generation。
  四个prompt逐字以英文提供；解码使用temperature 0（argmax）、最多40个新token，
  只在输出EOS时提前停止。

## 不应报告的内容（本轮已落实的修正）

- 将raw `corr(z_A, z_B)`作为共享空间证据——它只是训练loss的重述，零信息时也能达到最优。
- 将`A→B`或`B→A`称为“emergent”或“unsupervised”——应称
  “两条路径都不需要cross-reconstruction target”。
- 单一“supervised ceiling百分比”——报告checkpoint没有匹配的多seed supervised对照。
- 把`A→A`/`B→B`当作`B→A`/`A→B`的*ceiling*，或把任一差距称为“sharing代价”——
  target位于不同空间；真正的参考ceiling是直接supervise的逐层bridge。

## 为什么适配器必须轻量（关键约束）

NinaXander检验的假设是：异构模型的匹配层可在一个*简单*变换后共享语义空间。
因此适配器容量不仅关系overfitting，也是方法约束。重型适配器（深/宽MLP或大型ResNet）
可以近似两个空间之间的任意映射，所以用重型适配器得到高AB/BA read-out**不能**证明共享空间。
只有对齐两个空间的**轻量、低容量**映射支持该假设。最清晰的容量对照是无法拟合任意对应关系的
*线性*映射；它仍达到AB 0.487和BA 0.535。这些匹配的线性可达性测量承载共享空间论点。

报告潜空间不压缩（`z = d = 4096`），所以“轻量”指*容量*而非宽度。
适配器是在两个线性映射之间放置一个零初始化residual block；它从严格线性映射开始，
只增加实际需要的非线性。268.5M参数是14.2B冻结权重的1.9%。
两个cross-readout必须连同参数量与架构一并报告；只有映射受到相应约束时，任一方向才构成证据。

训练*后*还剩多少近线性成分可以测量，而结果撤回了“near-linear”表述：
在完全相同的真实残差行上，最佳affine imitation解释AB适配器输出的**86.6%**、
BA输出的**88.0%**，故13.4% / 12.0%为不可约非线性。移除同一个已训练residual-MLP branch
会使两条cross read-out崩溃（AB 0.56→−0.41，BA 0.58→−0.30）。
因此适配器应称为低*容量*，绝不能称为“near-linear”。
