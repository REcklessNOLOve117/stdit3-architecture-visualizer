# STDiT3 架构学习可视化

这是一个纯静态、离线可用的中文学习页面，用来逐步比较：

- Open-Sora `opensora/v1.3` 的原生 `STDiT3-XL/2`
- WMPO `main` 中 action-conditioned STDiT3 世界模型

目录包含七个互相链接的页面：

- `system-loop.html`：从系统层解释 VLA Policy、World Model、Reward Model 如何交替生成 imagined trajectory、评分并通过 GRPO 更新 Policy，同时区分 Policy Behavior Alignment 对 World Model 的更新。
- `index.html`：八步 forward 教学版。
- `overview.html`：完整大架构总览；28 对主干以第 1–3、26–28 对和中间省略块表示。
- `end-to-end.html`：一张可拖拽缩放的大画布，从 RGB/VAE 外部上下文和噪声 latent 输入开始，通过放大箭头依次展开 28 对主干、单个 Spatial/Temporal Pair、Attention QKV、MLP、Final Layer、unpatchify 与 latent 输出。
- `block-detail.html`：以经典 Transformer 架构图的粒度，纵向展开每个 Norm、AdaLN、Attention、gate、residual add 与 MLP；分别查看 WMPO Spatial、WMPO Temporal 和原生 Open-Sora Block。
- `adaln-toy.html`：用 hidden size=2 数值手算 ActionEncoder、两处 Action-AdaLN、gate、Temporal 输入和 28 层复用。
- `time-alignment.html`：用统一时间轴解释 4 个历史 RGB/latent、8 个 action transition 与 8 个未来帧的严格对应，并可切换 video Hz 与 action Hz 查看重采样关系。

## 打开方式

直接双击 `index.html`，或用 Edge / Chrome 打开。页面不需要安装依赖、启动服务或访问网络。

## 操作

- 在“原生 Open-Sora / WMPO / 并排对比”之间切换。
- 使用“上一步 / 下一步”按 forward 顺序学习，也可以按键盘左右方向键。
- “自动播放”每 4.5 秒前进一步，到第 8 步后停止。
- 顶部八个步骤按钮可直接跳转。

### WMPO 三模型闭环

- 打开 `system-loop.html` 从系统层理解 VLA Policy、World Model 和 Reward Model 的职责边界。
- 六个阶段按钮可分别高亮完整闭环、Policy 动作、World Model 想象、Reward 评分、GRPO 更新和 Policy Behavior Alignment。
- 主闭环中 GRPO 只更新 Policy；World Model 与 Reward Model 作为环境和评估器。行为对齐是另一条单独更新 World Model 的辅助闭环。

### 完整架构总览

- 打开 `overview.html` 查看 2700×1700 的完整架构画布。
- 在画布空白处拖拽，使用滚轮或顶部按钮缩放；按 `0` 可适合窗口。
- 左侧章节可以聚焦输入、条件、Spatial/Temporal Block、28 对主干和输出。
- 点击任意模块或任意一层 Block，右侧会显示张量形状、关键计算和源码位置。
- “原生 / WMPO / 对比”用于突出或弱化对应模型；“学习 / 结构 / 源码”控制信息密度。
- 缩放不仅改变尺寸，也会在“总览 / 结构 / 计算”三个语义层级之间自动切换。
- 点击模块会高亮完整上下游路径；Dimension Trace 会指出 Spatial 的 `B·T / axis=S` 与 Temporal 的 `B·S / axis=T`。

### Action-AdaLN 数值手算

- 从 `overview.html` 顶部的“Action-AdaLN 数值手算”进入，或直接打开 `adaln-toy.html`。
- 页面包含 9 个步骤，可用按钮或键盘左右方向键切换。
- “同时显示真实模型维度”用于在 2 维手算和真实 1152 维形状之间对照。

### Block 内部精细架构

- 从 `overview.html` 顶部的“Block 内部精细图”进入，或直接打开 `block-detail.html`。
- 使用三个标签切换 WMPO Spatial Block、WMPO Temporal Block 和原生 Open-Sora Spatial Block。
- 读图方向统一为真实 forward 的从上到下；右侧长线是 residual skip，左侧虚线是 condition 参数注入。
- 点击任一彩色模块，可以查看该模块的张量形状、公式、作用和源码位置。
- WMPO Spatial 图明确展示 Action-AdaLN 直接进入 Self-Attention 前和 MLP 前，不存在独立的 Action Layer。
- WMPO 中关闭的 Text Cross-Attention 作为侧边 ghost branch 展示，不再占据主计算流。
- `Forward / Backward` 可切换到从 Loss 返回 gate、AdaLN、t_block 与 ActionEncoder 的梯度主链。

### 输入到输出精细架构

- 从 `overview.html` 顶部的“输入到输出精细图”进入，或直接打开 `end-to-end.html`。
- 所有模块都位于同一张画布，不再拆成多个纵向章节；编号和节点内“详见”标签表示从主干模块到内部实现的逐级展开关系。
- 左侧主干从 RGB/VAE 外部上下文开始，明确标出 STDiT3 真正的输入是噪声 latent `x_t`；中间展开一个完整 Spatial/Temporal Pair；右侧展开 Attention Q/K/V 和 MLP；底部展开 Final Layer 与 unpatchify。
- 蓝色实线单独标出 video hidden 主数据流：`latent → PatchEmbed3D → x₀=tokens+PE → Pair ℓ 的 Spatial 输入 → x₂ → Temporal`；绿色虚线只表示 Action-AdaLN 六组参数的侧向路由。
- 在画布空白处拖拽，使用滚轮或顶部按钮缩放；聚焦按钮可快速定位输入、条件、单层 Block、Attention/MLP 和输出。
- Attention 面板可在 Spatial 与 Temporal causal 两种形状之间切换。
- 点击 `gate_msa`、`gate_mlp` 或 `x₂` 会聚焦对应完整路径；`x₂` Inspector 会显示 produced by、consumed by 和 action influence。

### Video / Action 时间对齐

- 打开 `time-alignment.html`，在同一时间轴查看 RGB frame、物理 action 区间、逐帧 VAE latent、STDiT condition position 和原始控制采样。
- 选择任意 `a_t … a_t+7`，页面会高亮它控制的帧间转移以及它最终进入的 future latent 条件位置。
- 可切换 `30/30 Hz`、`30/60 Hz`、`30/10 Hz`，观察一对一、控制更快需要聚合、控制更慢需要保持三种情况。

### 阅读舒适度

- 所有页面右上角提供“阅读设置”：Comfort Dark、90%/100%/115% 字号、网格开关与动画开关。
- 设置保存在浏览器本地；页面仍可完全离线运行。

## 源码依据

- `C:\Users\YXWANG\toys\worldmodel_factory\Open-Sora\opensora\models\stdit\stdit3.py`
- `C:\Users\YXWANG\toys\worldmodel_factory\WMPO\dependencies\opensora\opensora\models\stdit\stdit3.py`
- `C:\Users\YXWANG\toys\worldmodel_factory\WMPO\dependencies\opensora\opensora\datasets\simplevla_webdataset.py`

页面中的 WMPO 示例采用源码默认 `To=4`、`Ta=8`、`action_dim=7`；符号形状仍适用于其他帧数和 action 维度。

## Bib
https://github.com/WM-PO/WMPO/


[https://github.com/hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.3)
