# 指标数据

[English](README.md) | [日本語](README.ja.md) | 简体中文

指标流水线分为两层。

`raw/`由实验程序写入。含嵌套输出的实验同时写入`name.json`和矩形`name.csv`；
天然为表格形式的实验直接写CSV。逐题正确性保留在JSON中，以便复现配对检验。

`raw/training_curve.csv`是公开训练轨迹，采用与当前训练相同的只追加schema。
最初417个历史点曾从权威调度器日志迁移一次，因此公开checkout不再需要这些本地日志。

`tables/`按确定性方式重建：

```bash
./reproduce.sh tables
```

`tables/MANIFEST.csv`描述每个面向论文的表，并记录行数和schema。
`tools/validate_results.py`检查JSON可读性、CSV header及非空性、manifest覆盖、
预期行数，以及最终表征表与Hugging Face发布metadata的一致性。AB和BA验证要求显式路径metadata、
完整ARC-Easy/SciQ题数，以及相同的18种QA配置：两个父模型、四条同族路径、四条跨族路径、
四种同一适配器alpha-zero干预和四种affine对照。两条路径都要求逐题数组对齐，
且路径专属边界gate低于`1e-4`；BA还记录direct-suffix和concurrent-arm gate，
因为其RWKV suffix使用自定义batch-1 runtime。两个纯父模型的逐题结果数组还必须在AB和BA QA运行间
完全一致，以防数据集顺序或评分失去对称性。等内存对照为AB使用27/23/15/7个Pythia块，
为BA使用5/9/17/25个，并在两条路径中保留完整逐题数组。长上下文window数、serving配置集合、
生成gate、实测KV层数和削减比例也会检查。serving单元只接受有限正值或显式结构化runtime error。
checkpoint step、线性拟合行数与ridge、上下文window数、并发cutoff、serving重复次数、KV probe长度、
最终检查点与生成的一致性都属于semantic contract，而非非正式日志metadata。

长上下文affine对照会在各评估domain互不重叠的前40,000个token上重新拟合。因此WikiText affine arm
是经过domain适配的oracle，而不是将在Alpaca上拟合的map直接用于域外测试。四个历史AB长上下文/
serving JSON产生于结构化run metadata之前；其测量单元保持不变，从标准调用恢复的字段带有明确的
`legacy_metadata_enrichment`记录，未保存的runtime gate值只列为缺失，不进行重建。

`cross_family_runtime_smoke`记录AB和BA在全部四个切换点的父模型复现gate、有限logit和输出token检查。
由于BA的RWKV suffix对batch敏感，它还记录direct-vs-hook、独立CUDA stream和fused batch-3诊断。
fused接受标志必须与实测`1e-4`阈值一致，但拒绝是合法结果：报告的QA从不使用fused选项，
其标准RWKV路径始终采用独立batch-1调用。

配对QA、MLP干预、等内存以及同层AB/BA比较会合并到
`tables/paper_cross_family_paired_statistics.csv`。AB与BA QA都将四个切换点分别与两个父模型比较，
不会在检验前挑选任务专属的最佳切换点。所有producer和table builder均使用
`src/ninaxander/paired.py`，其精确二项分布尾概率在log空间计算；验证会拒绝下溢为
`mcnemar_p=0`的结果。

`raw/cross_family_mlp_representation.json`为AB和BA各包含25个匹配干预单元。
任务级`raw/cross_family_mlp_intervention_{arc,sciq}.json` bundle同样对称：
它将每条路径的sweep与八种alpha=0/1 QA配置及逐题结果连接起来。
`tools/complete_cross_family_mlp_bundle.py`在标准AB、BA QA评估器完成后进行无损join。
已被取代的单向`linearity_L32.*`和`mlp_ablation_*.*` bundle会被验证拒绝，不属于结果contract。
历史`mlp_items_*`及`paired_statistics_mlp.csv`中间文件只会被合并进标准QA/任务bundle一次，
此后同样被拒绝。

标准`paper_four_path_qa_accuracy.csv`表严格包含
2个任务 × 4个切换层 × 4条路径（AA、AB、BB、BA）。raw产物中的`a_to_b_`和`b_to_a_`
producer前缀仅保留为来源信息。公开表将`producer_path`与真实`execution_path`分开，
去除重复父模型行，并交错排列匹配的AB/BA条件。尤其是Pythia单独提前退出应归为
`B_early_exit`，不能仅因跨路径评估器请求了它就归入AB/BA或BB适配器路径。

每一对方向CSV还必须具有完全相同的列schema；normalizer在验证前通过共享AB/BA代码路径
重写旧generation、KV、长上下文和serving sidecar。`paper_result_coverage.csv`审计所有公开结果系列，
验证会拒绝任何没有匹配BA的AB-only系列。paired表在每个标准切换点加入同题BA对AB比较，
不会把两条路径当作独立样本。

`raw/four_path_layer_metrics.{json,csv}`同样在一个bundle中记录32层全部AA、AB、BB和BA。
该名称特意保持路径中立，旧BA catch-up命名已停用。
