# 02 · 当前发现

[English](../02-findings.md) | [日本語](../ja/02-findings.md) | 简体中文

本文所有数值都属于当前公开实验：32层RWKV-4-Raven-7B /
Tulu-Pythia-6.9B模型对、step-415,000的forward-hook适配器，以及通过
`./reproduce.sh tables`重建的结构化产物。不包含该contract之外的结果。

## 结果1 · 报告适配器：32层、forward hook、到达平台

报告适配器采用无压缩潜空间（`z = d = 4096`），公平配对全部32个块。

配置为`z = d = 4096`（无压缩）、一个零初始化ResBlock（hidden 4096）、σ=0.4、
N/layer=228,000、4张GPU上batch 4096、lr 1e-3并按plateau衰减至1e-5。
残差通过**全部32个块的forward hook**捕获，而非`hidden_states`。
训练跨越两个walltime周期（作业1832 → 1837），运行至**416,000步**时因计算资源额度结束而停止；所有报告数值均使用held-out A→B最优的**step 415,000**检查点。
268.5M适配器参数占14.2B冻结权重的1.9%。最深点（step 415,000）的held-out结果：

| | 数值 |
|---|---|
| A→A（self） | 0.6950 |
| A→B（cross，held-out） | **0.5588** |
| B→B（self，reconstruction） | **0.7112** |
| B→A（cross，B-prefix） | 0.5900 |
| ρ_ctr（centered alignment） | **0.9006** |
| f（token-varying energy fraction） | 0.3501 |
| ρ_raw | 0.9654 |

关键发现是成对的次序：**ρ_ctr 0.901 > B→B 0.711 > A→B 0.559**，
以及**ρ_ctr 0.901 > A→A 0.695 > B→A 0.590**。在全部32层中，
AB均低于使用同一B decoder的BB self-map，BA均低于使用同一A decoder的AA self-map。
**稀缺的不是对齐，而是解码重构。**

**限制两种self-map的因素仍未解决。** decoder被训练为从z+ξ、ξ~N(0,σ²)重构，
因此最佳系数缩小为1/(1+σ²)；但1/(1+σ²)≈0.862是该系数在*noisy*输入上评估的R²，
而本研究在**clean z（ξ=0）**上评估。此时scalar-linear ceiling为
1−(σ²/(1+σ²))² ≈ **0.98**，并已通过数值确认。两种self-map都远低于noise ceiling，
所以**尚未证明noise是binding constraint**；需要匹配的σ sweep
{0,0.1,0.2,0.4,0.8}。B→A（0.590）略高于A→B（0.559），说明从Transformer潜空间
重构RNN侧残差略微更容易。

**negative control（`four_path_layer_metrics.py`，forward hook，400个window）。**
为排除ρ_ctr 0.90和两种cross-readout只是分布重叠而非逐token对应的可能性，
保持A的行顺序，对B行应用一次固定随机置换后重新计算。结果**完全崩溃**：
ρ_ctr 0.9007 → +0.0000、A→B 0.5601 → −0.5909、B→A 0.5911 → −0.6083
（A→A/B→B self-map按构造不变）。重评也复现了报告均值，因此两种cross-readout确实依赖真实token对应。

训练signal为Alpaca的2,075个window × 112个token（约232k个独立token位置），
在32层上汇集成7.44M个残差对，循环约228次。

每项metric都在**每一层的全部token上单独计算**（centered / SST统计在层内移除），
再对32层取平均，而不是跨层pool。诚实限制：仅一个seed；在step 408k–415k间，
四种read-out变化均小于0.001（AB 0.5584→0.5588、BA 0.5899→0.5900、
AA 0.6941→0.6950、BB 0.7106→0.7112），因此这是**渐近平台而非严格收敛**，
报告最深点。

## 结果2 · 两条执行路径带来的收益：实测KV cache

实测而非假定（`kvcache_measure.py`，作业1809）：Tulu-Pythia每token每层缓存
**16.0 KiB**，32层共512 KiB/token，与2·L·d·2字节完全一致。
替代它的RWKV recurrent state为**每层40 KiB，且不随序列长度变化**。

AB保留`31−L`个Transformer suffix块，BA保留`L+1`个Transformer prefix块。
因此实测内存/准确率表从一开始就包含两条路径：

| 配置 | Transformer层数 | KV/token | KV @ 4k ctx | 节省 | ARC / SciQ（acc_norm） |
|---|---|---|---|---|---|
| pure Tulu-Pythia | 32 | 512 KiB | 2.00 GiB | — | 65.3 / 81.0 |
| pure RWKV-Raven | 0 | 0 | 0 | 100% | 61.8 / 67.0 |
| AB@4 | 27 | 432 KiB | 1.69 GiB | 15.6% | 59.6 / 70.9 |
| **BA@4** | **5** | **80 KiB** | **0.31 GiB** | **84.4%** | **62.2 / 69.2** |
| AB@8 | 23 | 368 KiB | 1.44 GiB | 28.1% | 59.0 / 67.7 |
| BA@8 | 9 | 144 KiB | 0.56 GiB | 71.9% | 57.3 / 64.6 |
| AB@16 | 15 | 240 KiB | 0.94 GiB | 53.1% | 48.6 / 54.5 |
| BA@16 | 17 | 272 KiB | 1.06 GiB | 46.9% | 57.5 / 68.0 |
| AB@24 | 7 | 112 KiB | 0.44 GiB | 78.1% | 47.0 / 51.1 |
| BA@24 | 25 | 400 KiB | 1.56 GiB | 21.9% | 59.6 / 69.9 |

QA使用**完整测试集**（ARC-Easy 2376 / SciQ 1000，acc_norm，temperature 0）
和配对McNemar/bootstrap比较（结果5）。SciQ合并后的non-dominated序列为
Pythia → AB@4 → BA@24 → BA@4 → RWKV；这是post-hoc单任务frontier，不是普遍排序。
BA@4用5个而非27个Transformer块，仍保持接近AB@4的双任务均值（65.70对65.25）。

纯RWKV完全没有KV，且不显著差于BA@4，因此新点是一种权衡，而不是支配性架构。
此前“没有其他点能在该内存达到该准确率”的表述是错误的。随后完成了等KV的Pythia提前退出比较
（结果8）：它与浅层切换持平、输给深层切换，但resident weight更小。该曲线尚未与KV量化、
sliding-window attention或专门训练的高效Transformer比较。

代价也并非为零。在本实现中，HF RWKV顺序执行recurrence，因此当前reference runtime在
ctx 2048剪枝prefill上**约慢60倍**（22.6 s对0.378 s），在未剪枝cacheless decode上
**约慢64倍**（0.148对9.49 token/s）。这是实现特性，不是架构极限，但对发布的reference path确实存在。

**serving microbench（同时测量`a_to_b_*`和`b_to_a_*`）。**
未剪枝reference实现完整加载两个父模型，并在完整父模型forward内覆盖边界，所以没有实现
resident-weight节省。物理剪枝AB/BA路径删除未使用块，并精确复现未剪枝reference；
resident weight分别为13.93–14.55 GiB和13.49–14.11 GiB。
速度仍受HF RWKV顺序执行和cacheless reference decoder支配。这些是实现特性而非架构极限，
但它们限制了当前内存轴的实际收益。

结果8提供该物理剪枝路径，并与未剪枝reference进行精确gate。它实现resident-weight缩减，
而上述未剪枝时间仍作为reference-runtime测量，不声称是优化后的throughput。

## 结果3 · 两条跨族路径都不是“近线性”

在报告32层适配器的五个probe layer {4, 10, 16, 22, 28}上，用真实held-out残差测量
（`artifacts/metrics/raw/cross_family_linearity_L32.json`、
`experiments/linearity.py`；每层50,400个fit行 /
100,800个不相交eval行，fp64 normal equation，
ridge 1e-3·tr(XᵀX)/(d+1) ≈ 50）：

| 路径 | adapter R² | affine-imitation R² | 不可约非线性 | imitation cross-R² | 最佳direct affine R² |
|---|---:|---:|---:|---:|---:|
| AB | 0.558 | 0.866 | **13.4%** | 0.490 | 0.487 |
| BA | 0.576 | 0.880 | **12.0%** | 0.516 | 0.535 |

由此得到两点。（1）“适配器从严格线性映射开始”只是*初始化*性质
（ResBlock第二个矩阵为zero-init）；415k步后的映射明显非线性，因此**撤回**“near-linear”表述。
（2）最佳*direct* affine map仍达到AB 0.487、BA 0.535。它们是无法拟合任意对应关系的
低容量映射，所以共享空间论点由这些direct-affine baseline承担，而不是重型适配器。
与适配器的差距为AB +0.071、BA +0.041。

在downstream中，单独拟合的affine map在AB上比适配器低8–23个百分点；
BA差异较小，有时甚至改变符号。这些跨映射比较只是描述性的。结果7使用同一适配器的alpha干预，
为两条路径提供因果检验。

## 结果4 · translation具有domain-specific性：退化来自domain shift，而非上下文长度

适配器在Alpaca（指令文本）上训练。为区分“长上下文使其失效”和“域外文本使其失效”，
匹配的`a_to_b_longctx_ce_mp.py`、`b_to_a_longctx_ce_mp.py`运行在
**域内Alpaca**与**域外WikiText-103**的ctx 512 / 1024 / 2048上测量next-token CE。
每条路径还测量最佳affine map，但会在每个domain前40,000个token上**分别重新拟合**，
并在不相交suffix上评估。因此WikiText affine arm是domain-adapted supervised oracle，
不是把Alpaca拟合map转移到域外。

- **两条路径的关键轴都是domain，而非长度。** ctx 512时，AB@4的Alpaca/WikiText
  perplexity为6.40/93.87，BA@4为3.85/45.00；两个父模型的退化小得多。
  BA缓解了shift，但其WikiText perplexity仍远高于父模型。
- **增加上下文长度不会使任一奇美拉崩溃。** 从ctx 512 → 2048，
  WikiText perplexity在AB@4从93.87改善为83.40，在BA@4从45.00改善为34.78。
  更好的父模型在ctx 2048仍为8.27，因此问题不是上下文长度限制，BA也未解决domain问题。
- **域内affine对照是描述性的，不是干净的非线性ablation。** WikiText lin@8 CE从
  ctx 512 → 2048的5.65升至6.28，而adapter@8从4.74变为4.72；但两个映射的objective和
  training data不同，不能据此把原因归于适配器非线性。结果7中匹配的同一适配器α干预，
  才是AB和BA上的干净证据。

这为两个方向的内存收益（结果2）划定范围：KV节省真实存在，但只适用于与训练domain相似的输入。

## 结果5 · 带配对检验的对称完整QA

AB和BA评估器以完全相同的题目顺序运行**完整**ARC-Easy（2376）与SciQ（1000）测试集。
它们包含两个同族arm：BB把B残差送入`D_B(E_B(·))`，AA把A残差送入`D_A(E_A(·))`。
`paired_stats.py`使用McNemar精确检验和paired bootstrap，把每个切换点分别与两个父模型比较，
并在相同题目上比较AB与BA。完整acc_norm：

| 任务 / 层 | AA | AB | BB | BA |
|---|---:|---:|---:|---:|
| ARC-Easy @4 | 62.8 | 59.6 | 62.2 | 62.2 |
| ARC-Easy @8 | 59.8 | 59.0 | 62.7 | 57.3 |
| ARC-Easy @16 | 54.8 | 48.6 | 57.4 | 57.5 |
| ARC-Easy @24 | 52.2 | 47.0 | 59.4 | 59.6 |
| SciQ @4 | 68.5 | 70.9 | 74.8 | 69.2 |
| SciQ @8 | 66.7 | 67.7 | 74.8 | 64.6 |
| SciQ @16 | 59.8 | 54.5 | 71.1 | 68.0 |
| SciQ @24 | 57.9 | 51.1 | 70.1 | 69.9 |

- **每个AB和BA切换点在两个任务上都显著低于更强的Pythia父模型。**
- **与较弱父模型的关系依赖路径和任务。** AB@4在SciQ上超过RWKV（+3.9，p=0.019），
  但在ARC上更差（−2.2，p=0.042）。没有BA切换点显著超过RWKV；BA@4在
  ARC（+0.4，p=0.66）和SciQ（+2.2，p=0.14）上都持平。
- **同族arm不是普遍可加的“decoder cost”。** 对B suffix，BB低于Pythia，
  AB又比BB低3–19个百分点。对A suffix，深层BA反而高于AA：L=16/24时，
  ARC高+2.7/+7.4，SciQ高+8.2/+12.0个百分点。AA/BB隔离decoder通路，
  但跨族差异同时改变prefix父模型，因此不能解释为纯translation error。
- **深度效应依赖路径。** AB随L单调下降，而BA在L=16/24恢复，
  比同层AB高8.9–18.8个百分点。仅凭local cross-readout R²不能预测端到端排序。

标准结构化结果为`paper_four_path_qa_accuracy.csv`；两个方向的所有父模型、affine、
等内存与干预行位于`paper_cross_family_qa_accuracy.csv`。

## 结果6 · 同编号层并非天然特殊——对齐是学习得到的

`all_layer_corr.py`计算A的layer j与B的layer k之间无需适配器的32×32对应关系
（linear CKA和in-sample best-linear R²）。按**linear CKA**，原始跨族表征只具有很弱的相似性
（对角均值0.049，非对角0.035），在A的32个layer中，最相似的B layer恰好同编号的只有**2/32**行
（argmax通常为B的最后一层）。best-linear R²在各处都很高（约0.85），但这是in-sample inflation，
没有信息价值。

因此ρ_ctr=0.90是适配器在所选同层对上**学习得到的**对齐，而不是内在逐层对应。
这与将论点从“共享*语义*空间”收窄为“在匹配条件下可恢复的token级表征对应关系”一致。

## 结果7 · 同一个nonlinear branch支撑AB与BA read-out

结果3的linear-map比较使用了*单独拟合*的线性映射（objective/data/layer-set均不同），
可能存在混杂。`cross_family_mlp_intervention.py`和两个标准QA评估器消除了这一问题：
它们用α缩放**同一个已训练适配器**唯一的非线性——ResBlock branch
fc2(GELU(fc1(·)))。α=1为完整模型；α=0使用*完全相同的weight/layer/row*，只禁用该branch。
表征sweep与两条完整、逐题配对QA路径共同存储：

| 路径 / α | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---:|---:|---:|---:|---:|
| mean AB R² | **−0.41** | −0.75 | 0.35 | 0.52 | **0.56** |
| mean BA R² | **−0.30** | −0.66 | 0.36 | 0.53 | **0.58** |

禁用branch（α=0）会使AB跌至**−0.41**、BA跌至**−0.30**。
完整规范化QA在两条执行路径上直接显示同一效应
（每格为α=1 → α=0准确率百分比；随机水平为25）：

| L | ARC-Easy AB | ARC-Easy BA | SciQ AB | SciQ BA |
|---:|---:|---:|---:|---:|
| 4 | 59.6 → 31.9 | 62.2 → 31.8 | 70.9 → 31.3 | 69.2 → 32.6 |
| 8 | 59.0 → 29.9 | 57.3 → 35.5 | 67.7 → 31.4 | 64.6 → 37.0 |
| 16 | 48.6 → 28.5 | 57.5 → 43.7 | 54.5 → 31.7 | 68.0 → 47.3 |
| 24 | 47.0 → 34.8 | 59.6 → 48.4 | 51.1 → 40.0 | 69.9 → 49.1 |

branch在**AB贡献+11.1至+39.6个百分点**，在**BA贡献+11.2至+36.6个百分点**；
全部16个逐题McNemar检验均有p<2e-9。由此得到两点：（1）两条α sweep都**非单调**
（0.25比0更差），说明branch magnitude是关键，而非小修正。（2）每个适配器自身的裸linear skeleton
（AB −0.41、BA −0.30）比单独拟合的direct affine map（AB 0.487、BA 0.535）更差：
外围Linear是为配合ResBlock训练的，移除它会使两条路径崩溃。

## 结果8 · 等KV Transformer baseline与真正剪枝的serving

配对的`a_to_b_serving_pruned.py` / `b_to_a_serving_pruned.py`路径物理释放未使用块，
并以rel=0复现未剪枝奇美拉。`equal_mem_baseline.py`提供与每个AB或BA切换点
完全相同Transformer深度/KV的Pythia提前退出。

- **两条路径的剪枝权重约为一半。** AB占13.93–14.55 GiB，BA占13.49–14.11 GiB，
  而两个完整父模型约27 GiB。匹配的提前退出Pythia baseline仍更轻，
  因为奇美拉还需携带RWKV块与适配器。
- **Transformer深度较低时，非Transformer部分会增加真实能力。** 对AB，
  @16比exit@15在ARC/SciQ高+11.2/+12.8，AB@24比exit@7高+18.5/+18.7
  （全部McNemar p<1e-10）；浅层切换持平（AB@4对exit@27相差±1点，p>0.5）。
  BA与5/9/17/25块exit相比，增益分别为ARC +35.4/+27.6/+16.9/+2.1、
  SciQ +38.7/+31.6/+17.9/+0.5；前三项显著，25块比较持平。
  因而在保留Transformer很短时，RWKV prefix或suffix都能增加实际能力。
- **诚实的frontier包含两条路径及两个父模型。** 纯RWKV为零KV，
  匹配Pythia exit的resident weight更小。尚未比较KV量化和sliding-window attention。
