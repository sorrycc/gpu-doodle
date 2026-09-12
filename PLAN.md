# gpu-doodle 实施计划

目标：在浏览器里用手写 WGSL 跑一个几万参数的涂鸦分类器，边画边猜。数据来自 Google Quick Draw（CC BY 4.0），配方沿用 gpu-time。

## 已定决策

| 项     | 决定                                                                                        |
| ------ | ------------------------------------------------------------------------------------------- |
| 输入   | 笔画点序列（stroke-3：Δx、Δy、抬笔），不是位图                                              |
| 类别   | 第一版 30 类，见下表，可增删                                                                |
| 推理   | 手写 WGSL，零依赖，权重内联                                                                 |
| 目录   | ~/Projects/gpu-doodle，pnpm monorepo                                                        |
| Python | uv 钉 3.13，与 gpu-time 一致                                                                |
| 数据   | full/simplified/<class>.ndjson 前 20 MB 前缀（阶段 3 决定；每类 3 万到 7 万张，非随机样本） |

## 目录结构

```
gpu-doodle/
├── AGENTS.md                      规则（改自 gpu-time）
├── README.md / MODEL_CARD.md      含 Google Quick Draw 署名
├── package.json / pnpm-workspace.yaml / .gitignore
├── .github/workflows/ci.yml       build、test、size:gate
├── packages/core/                 可发布包 gpu-doodle
│   ├── src/
│   │   ├── index.ts               classify(strokes, {topK}) / defineClassifier({backend})
│   │   ├── preprocess.ts          浏览器与训练共用：对齐、缩放到 255、RDP ε=2、stroke-3
│   │   ├── labels.ts              30 个类别名 + 中文
│   │   └── model/
│   │       ├── kernel.wgsl        重写：序列分类，无逐 token 输出头
│   │       ├── cpu.ts             标量参考实现，与 kernel 逐位对齐
│   │       ├── gpu.ts             从 gpu-time 搬，改 buffer 布局和读回
│   │       ├── decode.ts          原样搬（int6 字符串解码）
│   │       ├── shader-source.ts   原样搬，去掉 BOUNDARY_THRESHOLD / COMPACT_FEATURES
│   │       └── weights.gen.ts     由 export.py 生成
│   ├── scripts/build.ts           原样搬，改 entryPoints 与 define 名
│   └── test/                      preprocess 一致性、CPU 与 PyTorch logits 一致性、browser 一致性
├── packages/training/
│   ├── pyproject.toml             torch、numpy
│   ├── torch/
│   │   ├── fetch.py               Range 下载 simplified ndjson 前缀，校验 sha256，写 data/manifest.json
│   │   ├── dataset.py             recognized 过滤 → key_id 哈希切分 → toStroke3 同款差分 → bucket 64/128
│   │   ├── model.py               DoodleTagger，改自 TimeTagger
│   │   ├── train.py               改自 gpu-time：QAT、余弦退火、源码快照
│   │   ├── export.py              改自 gpu-time：int6 编码、导出门禁、parity fixture
│   │   └── evaluate.py            按类准确率、混淆对、前 k 笔准确率
│   ├── data/                      simplified/ 与 synth/ 忽略；manifest.json、classes.json 跟踪
│   ├── active/                    export-report.json、parity.* fixture
│   └── runs/                      report.json 与 source/ 快照跟踪，.pt 忽略
└── apps/site/                     Vite + TS，canvas 画板，边画边猜
```

## 从 gpu-time 直接复用与需要重写的部分

| 文件                                                                | 处理                                                               |
| ------------------------------------------------------------------- | ------------------------------------------------------------------ |
| decode.ts、shader-source.ts、build.ts、gpu.ts 的设备管理与恢复逻辑  | 直接搬，改名字                                                     |
| export.py 的 int6 编码、lineage、源码快照、门禁框架                 | 直接搬，把 reserved/bare 两个语料换成 test split 与「前 3 笔」子集 |
| train.py 的优化器、余弦退火、QAT 切换、best.pt 选择、report.json    | 直接搬，损失换成单个交叉熵                                         |
| model.py 的仿射扫描、quantize、half_storage                         | 直接搬                                                             |
| tokenizer.ts、labels.ts 的角色体系、compile/resolve/rrule、calendar | 全部不要                                                           |
| kernel.wgsl                                                         | 重写，结构见下                                                     |
| featurize.ts                                                        | 不要，ndjson 在 dataset.py 里差分编码，几何不进 Python             |

## 模型设计

输入每个点 8 个连续特征：Δx、Δy（除以 255）、抬笔标志、落笔标志、绝对 x、y（除以 255）、笔画序号（除以 16 截断）、是否为序列末点。

```
input 8 → linear → 64
depthwise conv width 5 + tanh              （抄 gpu-time）
gate / candidate 64×64，仿射扫描正向 + 反向  （抄 gpu-time）
combine 128 → 64，tanh
mean pool 64 + max pool 64 → 128
head 128 → 64 tanh → 30 logits
```

参数量约 30K，int6 后权重约 20 KB Brotli，整包应在 30 KB 以内，门禁设 50,000 字节。

和 gpu-time 的差别：没有逐 token 头，没有边界头，没有 hash embedding（改为线性投影），加了 max pool。hidden 从 32 提到 64 是因为 30 类比 40 个角色更依赖全局形状。

## WGSL kernel 结构

一个 workgroup 处理一张涂鸦，workgroup_size 64 对应 hidden 64。阶段：

1. 每 lane 算自己通道的 input projection，写 state[0]。
2. 卷积 + tanh，写 state[1]。
3. gate、candidate，正向扫描写 state[0]，反向扫描写 state[3]。gpu-time 的顺序扫描写法保留，一次 dispatch 内一个 workgroup 顺序遍历点，300 个点以内足够快。
4. combine，同时累加 mean 与 max 到 workgroup 共享数组。
5. head 与输出，lane 0 写 logits[stream * 30 + i] 到 storage buffer。

去掉 packedLabels、scores、BOUNDARY_THRESHOLD。读回只有 logits，softmax 和 top-k 在 CPU 做。

## 预处理一致性（最容易出错的地方）

Quick Draw 官方的 `full/simplified` 数据是四步流程的产物：对齐左上、最长边缩到 255、1 像素重采样、RDP ε=2。浏览器的 pointer 事件必须走完全相同的流程，否则模型看到的分布不一样。

做法：

1. `preprocess.ts` 实现上述四步，输出 stroke-3。已完成（阶段 2）。
2. Python `dataset.py` 只做几何之后的归一化（除以 255、bucket、pad），不再碰几何。
3. 一致性测试：`packages/core/test/fixtures/quickdraw-cat-parity.json` 保存 120 对同 key_id 的 raw 与 simplified 记录（`pnpm --filter gpu-doodle fixture:parity` 重新抓取），TS 预处理 raw 后与 simplified 逐点比较。已完成：120 对中 116 对点数一致、104 对逐点相同、109 对偏差不超过 1 像素；在 1,433 对上为 98%、90%、93%，剩余差异是 RDP 与重采样的浮点临界情况。

**阶段 2 的发现，影响阶段 3 的数据选择。** 阶段 1 下载的 `sketchrnn/<class>.npz` 不是 simplified 数据的 stroke-3 版本。用 raw 数据核对（400 条中 221 条命中），npz 的几何是：原始设备坐标、不对齐、不缩放、不重采样，直接在原始尺度上做 RDP ε=2，再丢掉第一个点、以第一个点为原点做差分。它的坐标范围随采集设备变化（5000 条 cat 的最大跨度中位数 341，最大 1459），与浏览器管线不一致，也没有尺度不变性。阶段 3 二选一：

- （推荐）换用 `full/simplified/<class>.ndjson` 作为训练数据（每类约 50 到 80 MB，可只取文件前缀），Python 侧只做 `toStroke3` 同款差分，几何完全由 `preprocess.ts` 定义，parity 测试才有意义。需要自己按 key_id 哈希切分，并过滤 `recognized`。
- 保留 npz，则浏览器管线要改成「原始尺度 RDP + 每张归一化」，且归一化必须在 Python 侧重复实现，违反「几何只在 TS」的规则。

## 数据

30 类候选（视觉区分度高，中文可翻译）：

apple, banana, bicycle, bird, book, butterfly, cat, car, clock, cloud, cup, dog, elephant, eye, fish, flower, guitar, house, key, lightning, moon, mountain, pizza, rabbit, star, sun, tree, umbrella, airplane, t-shirt

阶段 3 实际数据（`full/simplified` 每类前 20 MB，共 572 MB，git 忽略，`fetch.py` 记录 sha256）：

| 项                   | 值                                                   |
| -------------------- | ---------------------------------------------------- |
| 记录数               | 1,333,226，其中 recognized 1,248,415                 |
| 每类记录             | 30,408 到 70,650，不均衡                             |
| 丢弃                 | 未识别 84,811；超过 128 点 3,520                     |
| train / valid / test | 1,120,607 / 62,248 / 62,040（按 key_id 哈希 90/5/5） |
| 平均点数             | 39.6，bucket 只留 64 / 128                           |

simplified 记录比预想小得多，20 MB 前缀给了每类 3 万到 7 万张，而不是 1.5 万。每 epoch 默认用全部 train，`--samples` 可以封顶。类别不均衡（最多 2.5 倍）第一版没有做重采样。

## 评测

| 指标                                  | 用途                                |
| ------------------------------------- | ----------------------------------- |
| test top-1 / top-3                    | 主指标，导出门禁看 top-1            |
| 按类准确率 + 混淆对                   | 决定下一版换哪些类                  |
| 前 k 笔准确率（k = 1、2、3、全部）    | demo 的真实数字，「边画边猜」的体验 |
| CPU 与 PyTorch 512 条 logits 最大误差 | parity fixture，跟踪进 active/      |
| WebGPU 与 CPU 10,000 条 argmax 一致   | test:browser                        |

导出门禁：候选 test top-1 严格高于已发布权重，每类准确率在两比例 z=1.96 容忍内不回退，前 3 笔准确率不回退。`--force` 记录覆盖。

## 阶段 4 结果

`export.py` 把 `runs/stage3-baseline/best.pt` 导出为 int6：27,742 参数，`weights.gen.ts` 29,544 字节，Brotli 16,269 字节。解码后的权重在 test 上 top-1 93.6%、前 3 笔 81.0%，与 f32 checkpoint 一致。首次导出没有已发布基线，用 `--force` 通过门禁，报告里记录了被覆盖的失败项。

`cpu.ts` 是标量参考实现，`featurize` 与 `Dataset.features_of` 对齐。512 条 test 序列的 CPU 与 PyTorch parity：argmax 全部一致，logits 最大绝对误差 4.8e-6，每张约 0.6 ms。`model-parity.test.ts` 同时校验 `weights.gen.ts` 的 sha256 与 `export-report.json` 一致。

导出源码快照在 `exports/0d4f8a779637…/source/`。评估在 CPU 上用顺序扫描跑全部 valid 与 test，一次导出约 150 秒。

## 站点

Vite + 原生 TS，一个 canvas，pointer 事件收集笔画，每次抬笔调用一次 classify，显示 top-3 和概率条，类别名中英双语。加一个「随机题目」按钮：给一个词让用户画，猜中就下一题，这是原版 Quick Draw 的传播形态。页脚署名 Google Quick Draw 数据集。

## 分阶段

1. 骨架：monorepo、AGENTS.md、gitignore、CI 占位、fetch.py 下载 30 类。
2. preprocess.ts 与一致性测试。已完成。
3. model.py、train.py、dataset.py，跑通一次 5 epoch 看曲线。已完成，结果见 `packages/training/runs/stage3-baseline/report.json`。
4. export.py、cpu.ts、decode.ts，CPU 与 PyTorch parity。已完成，见下。
5. kernel.wgsl、gpu.ts、test:browser。
6. 站点 demo。
7. size:gate、CI、README、MODEL_CARD。

每阶段结束都能独立验证，第 3 阶段结束就能知道这个模型规模够不够。
