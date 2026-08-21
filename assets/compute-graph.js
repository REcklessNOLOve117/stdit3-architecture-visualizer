(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const canvas = { width: 3600, height: 3160 };
  const dims = Object.freeze({
    B: 1,
    rgbChannels: 3,
    latentChannels: 4,
    history: 4,
    future: 8,
    T: 12,
    image: 256,
    latent: 32,
    patchT: 1,
    patchH: 2,
    patchW: 2,
    H: 1152,
    heads: 16,
    outChannels: 8,
    steps: 30,
  });
  const derived = Object.freeze({
    grid: dims.latent / dims.patchH,
    S: (dims.latent / dims.patchH) * (dims.latent / dims.patchW),
    totalTokens: dims.T * (dims.latent / dims.patchH) * (dims.latent / dims.patchW),
    headDim: dims.H / dims.heads,
    qkv: 3 * dims.H,
    mlp: 4 * dims.H,
    sixH: 6 * dims.H,
    finalProjection: dims.patchT * dims.patchH * dims.patchW * dims.outChannels,
  });

  window.computeGraphShapeRegistry = Object.freeze({ ...dims, ...derived });

  const sourceRoot = "WMPO/dependencies/opensora/opensora";
  const sources = {
    rollout: "robwm_rollout.py:354–403",
    rflow: `${sourceRoot}/schedulers/rf/__init__.py:190–246`,
    transform: `${sourceRoot}/schedulers/rf/rectified_flow.py:10–38`,
    stdit: `${sourceRoot}/models/stdit/stdit3.py:37–72, 412–527`,
    block: `${sourceRoot}/models/stdit/stdit3.py:141–206`,
    attention: `${sourceRoot}/models/layers/blocks.py:137–229`,
    final: `${sourceRoot}/models/layers/blocks.py:728–763`,
    vae: `${sourceRoot}/models/vae/vae.py:55–71`,
  };

  const svg = document.getElementById("compute-graph");
  const viewport = document.getElementById("compute-graph-viewport");
  if (!svg || !viewport) return;

  const make = (tag, attrs = {}, text = "") => {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    if (text) element.textContent = text;
    return element;
  };
  const pathFromPoints = (points) => points.map((point, index) => `${index ? "L" : "M"} ${point[0]} ${point[1]}`).join(" ");

  const shape = (concrete, symbolic) => ({ concrete, symbolic });
  const io = (concrete, symbolic = concrete) => ({ concrete, symbolic });

  const panels = [
    { id: "rollout", x: 40, y: 60, w: 650, h: 2100, cls: "", headerBandX: 40, headerWidth: 650, headerHeight: 104, showNote: true, label: "OUTSIDE STDiT", title: "Rollout wrapper", note: "首轮单帧编码；后续复用上一 chunk latent；concat 只发生一次" },
    { id: "sampler", x: 720, y: 60, w: 2840, h: 2580, cls: "sampler", headerBandX: 1980, headerWidth: 1000, headerHeight: 70, label: "EXECUTED SAMPLER", title: "Rectified Flow sampling loop ×30", note: "每一步更新完整 12-slot latent；同一个 STDiT3 被重复调用" },
    { id: "model", x: 780, y: 460, w: 2700, h: 1770, cls: "model", headerBandX: 1745, headerWidth: 1640, headerHeight: 70, label: "ONE STDiT3 FORWARD", title: "STDiT3-XL/2 · 28 Spatial → Temporal Pair", note: "仅展开 Pair #1；蓝色是 hidden 主链，条件参数从独立显式端口进入" },
    { id: "action", x: 805, y: 665, w: 720, h: 1450, cls: "", headerBandX: 805, headerWidth: 410, headerHeight: 70, label: "CONDITION BRANCH", title: "ActionEncoder + Spatial 6H", note: "Action 不拼接为 token；它只改变 Spatial modulation" },
    { id: "spatial", x: 1550, y: 665, w: 650, h: 1450, cls: "spatial", headerBandX: 1550, headerWidth: 270, headerHeight: 62, label: "PAIR #1 · SPACE", title: "Spatial Block", note: "每帧内部的 256 个空间 token 做 Self-Attention" },
    { id: "temporal", x: 2225, y: 665, w: 700, h: 1450, cls: "temporal", headerBandX: 2225, headerWidth: 340, headerHeight: 62, label: "PAIR #1 · TIME", title: "Temporal Block", note: "每个空间位置沿 12 个时间槽做 causal attention + RoPE" },
    { id: "masklane", x: 2950, y: 560, w: 480, h: 1555, cls: "", headerBandX: 2950, headerWidth: 480, headerHeight: 88, showNote: true, label: "ACTIVE SELECTOR", title: "x_mask / t₀=15", note: "选择 modulation；不是 Attention Mask" },
    { id: "output", x: 720, y: 2690, w: 2840, h: 410, cls: "", headerBandX: 960, headerWidth: 1380, headerHeight: 50, label: "AFTER 30 STEPS", title: "Future latent → VAE Decoder → 8 future RGB", note: "scheduler 返回完整 12 槽，wrapper 先取最后 8 个 temporal slots" },
  ];

  const nodes = [
    // Rollout wrapper.
    {
      id: "rgbCurrent", x: 90, y: 165, w: 250, h: 104, kind: "video", kindLabel: "FIRST CHUNK INPUT",
      title: "当前单张 RGB", note: "不是 4 张独立历史图", shapes: shape("[1,3,256,256]", "[B,3,Himg,Wimg]"),
      input: io("camera observation"), description: "当前 rollout 首个 chunk 只编码一张当前观测。代码随后增加 T=1 维度。", formula: "rgb.unsqueeze(2)\n[1,3,256,256] → [1,3,1,256,256]", source: sources.rollout, paths: ["video"],
    },
    {
      id: "vaeEncode", x: 90, y: 310, w: 250, h: 104, kind: "video", kindLabel: "VAE ENCODER",
      title: "SDXL 2D VAE", note: "只在首轮历史初始化", shapes: shape("[1,4,1,32,32]", "[B,Cz,1,h,w]"),
      input: io("[1,3,1,256,256]", "[B,3,1,Himg,Wimg]"), description: "将单张 RGB 压缩为一个 latent 时间槽。后续 chunk 的历史 latent 不再重新编码。", formula: "空间压缩率 = 8\n256×256 → 32×32", source: sources.rollout, paths: ["video"],
    },
    {
      id: "repeatHistory", x: 90, y: 455, w: 250, h: 104, kind: "video", kindLabel: "FIRST CHUNK ONLY",
      title: "同一 latent repeat ×4", note: "初始化 history queue", shapes: shape("[1,4,4,32,32]", "[B,Cz,Tₕ,h,w]"),
      input: io("[1,4,1,32,32]", "[B,Cz,1,h,w]"), description: "首轮把同一个当前观测的 latent 复制四次，形成长度为4的 condition queue；不是四帧相机流。", formula: "latent.repeat(1,1,queue_len,1,1)\nqueue_len = 4", source: sources.rollout, paths: ["video"],
    },
    {
      id: "previousLatent", x: 390, y: 455, w: 250, h: 104, kind: "video", kindLabel: "LATER CHUNKS",
      title: "上一 chunk 最后4槽", note: "直接复用 latent", shapes: shape("[1,4,4,32,32]", "[B,Cz,Tₕ,h,w]"),
      input: io("previous generated latent"), description: "第二轮及以后，历史条件来自上一 chunk 生成结果的最后4个 latent，不再经过 VAE Encoder。", formula: "z_next[:, :, -4:]", source: sources.rollout, paths: ["video"],
    },
    {
      id: "historyQueue", x: 235, y: 620, w: 270, h: 104, kind: "video", kindLabel: "CONDITION SLOTS 0…3",
      title: "history latent queue", note: "首轮 / 后续二选一", shapes: shape("[1,4,4,32,32]", "[B,Cz,Tₕ,h,w]"),
      input: io("repeat result OR previous chunk"), description: "统一的4槽历史条件张量。它会与8槽高斯噪声在 scheduler 调用前拼接一次。", formula: "Tₕ = 4", source: sources.rollout, paths: ["video"],
    },
    {
      id: "futureNoise", x: 390, y: 780, w: 250, h: 104, kind: "video", kindLabel: "GENERATION SLOTS 4…11",
      title: "future Gaussian noise", note: "8 个待生成槽", shapes: shape("[1,4,8,32,32]", "[B,Cz,Tf,h,w]"),
      input: io("N(0,I)"), description: "Rectified Flow 的未来部分从高斯噪声开始。历史4槽保持为 condition，未来8槽被迭代更新。", formula: "Tf = 8", source: sources.rollout, paths: ["video"],
    },
    {
      id: "concatOnce", x: 225, y: 945, w: 290, h: 112, kind: "video", kindLabel: "SOURCE OPERATOR · ONCE",
      title: "concat(history, noise)", note: "不在30步循环中重复", shapes: shape("[1,4,12,32,32]", "[B,Cz,Tₕ+Tf,h,w]"),
      input: io("[1,4,4,32,32] + [1,4,8,32,32]", "[B,Cz,Tₕ,h,w] + [B,Cz,Tf,h,w]"), description: "源码在进入 scheduler.sample 之前只执行一次 concat。循环里直接维护完整12槽 z。", formula: "concat(dim=2)\nT = 4 + 8 = 12", source: sources.rollout, paths: ["video"],
    },
    {
      id: "maskSpec", x: 225, y: 1115, w: 290, h: 112, kind: "mask", kindLabel: "CONDITION-FRAME SELECTOR",
      title: "frame mask", note: "0=history · 1=future", shapes: shape("[1,12] = [0×4 | 1×8]", "[B,T]"),
      input: io("Tₕ=4, Tf=8"), description: "这个布尔布局告诉 scheduler 哪些槽固定、并告诉 STDiT Block 哪些位置选择 t0 modulation。它不是 self-attention mask。", formula: "history=False\nfuture=True", source: sources.rollout, paths: ["mask"],
    },
    {
      id: "actionChunk", x: 225, y: 1310, w: 290, h: 112, kind: "action", kindLabel: "POLICY ACTION CHUNK",
      title: "8 个 7D action", note: "每个 action 控制一次 transition", shapes: shape("[1,8,7]", "[B,Ta,Da]"),
      input: io("policy output"), description: "动作 chunk 与未来8个 transition 按 control step 编号对应；网络内部再补齐4个 NONE 槽，使 condition tensor 的时间长度为12。", formula: "Ta=8, Da=7\na_s → o_{s+1}", source: `${sources.rollout}; ${sources.stdit}`, paths: ["action"],
    },

    // RFlow loop controls.
    {
      id: "zCurrent", x: 815, y: 160, w: 310, h: 112, kind: "video", kindLabel: "RFLOW STATE · STEP k",
      title: "z_current", note: "[condition slots | future zₜ]", shapes: shape("[1,4,12,32,32]", "[B,Cz,T,h,w]"),
      input: io("initial concat OR previous update"), description: "每个采样步都把完整12槽状态交给同一个 STDiT3。这里表达语义组成，不表示再次执行 concat。", formula: "zₖ = [fixed history | current future]", source: sources.rflow, paths: ["video"],
    },
    {
      id: "timestepRaw", x: 1250, y: 145, w: 255, h: 104, kind: "time", kindLabel: "SAMPLING STEP",
      title: "timestep tₖ", note: "第 k 个采样时刻", shapes: shape("[1]", "[B]"),
      input: io("sampling schedule"), description: "30个采样时刻在循环前构造；每一步取出当前 timestep。30来自当前 inference 配置，不是 RFLOW 类默认值。", formula: "k = 1…30", source: sources.rflow, paths: ["time"],
    },
    {
      id: "timestepTransform", x: 1250, y: 290, w: 255, h: 112, kind: "time", kindLabel: "ACTIVE CONFIG",
      title: "timestep_transform", note: "shape 不变", shapes: shape("[1]", "[B]"),
      input: io("[1] + height,width,num_frames", "[B] + Himg,Wimg,T"), description: "根据分辨率与视频长度变换采样时间。当前函数不使用 fps，视频 temporal factor 在实现中固定为4。", formula: "τ=t/N\nτ′=ratio·τ / (1+(ratio−1)·τ)\nt′=N·τ′; shape [B]→[B]", source: sources.transform, paths: ["time"],
    },
    {
      id: "xMask", x: 1665, y: 160, w: 285, h: 112, kind: "mask", kindLabel: "PER-STEP SELECTOR",
      title: "x_mask", note: "未来=True · 历史=False", shapes: shape("[1,12]", "[B,T]"),
      input: io("frame mask + current timestep"), description: "RFlow 每一步根据 frame mask 和当前 timestep 构造 x_mask。它在 Block/Final Layer 中选择 normal 或 t0 modulation。", formula: "True → normal t/action params\nFalse → t₀=15 params", source: sources.rflow, paths: ["mask"],
    },
    {
      id: "stepContract", x: 2130, y: 145, w: 440, h: 120, kind: "time", kindLabel: "LOOP BODY ×30",
      title: "同一个 STDiT3 forward", note: "predict flow → update z → restore history", shapes: shape("30 × one forward", "Nstep × one forward"),
      input: io("z_current, tₖ, x_mask, actions"), description: "循环不是模型内部层数。每个采样步重新调用同一组模型权重，并用预测 flow 更新完整 latent 状态。", formula: "for k in sampling_steps:\n  pred = model(z, tₖ, …)\n  z ← z + pred·dt", source: sources.rflow, paths: ["video", "time", "mask"],
    },
    {
      id: "cfgGhost", x: 3030, y: 145, w: 300, h: 126, kind: "disabled", kindLabel: "CONFIGURED · NOT EXECUTED",
      title: "CFG scale = 7", note: "cond/uncond combine 已注释", shapes: shape("no active tensor", "no active tensor"),
      input: io("configuration only"), description: "当前 RFLOW 执行路径没有 batch 复制、cond/uncond 拆分或 CFG combine。活跃结果是 v_pred = pred。", formula: "# cond/uncond code commented\nv_pred = pred", source: sources.rflow, paths: [],
    },

    // Video tokenization.
    {
      id: "patchConv", x: 820, y: 505, w: 260, h: 110, kind: "video", kindLabel: "PATCHEMBED3D INTERNAL",
      title: "Conv3d patchify", note: "kernel / stride = (1,2,2)", shapes: shape("[1,1152,12,16,16]", "[B,H,T,h/p,w/p]"),
      input: io("[1,4,12,32,32]", "[B,Cz,T,h,w]"), description: "PatchEmbed3D 内部 Conv3d 将每个 2×2 latent patch 投影到1152维。这个5D张量是模块内部中间结果。", formula: "32÷2=16\nchannels: 4 → 1152", source: `${sourceRoot}/models/layers/blocks.py:104–129`, paths: ["video"],
    },
    {
      id: "patchFlatten", x: 1110, y: 505, w: 260, h: 110, kind: "video", kindLabel: "PATCHEMBED3D RETURN",
      title: "flatten patches", note: "公开输出已展平", shapes: shape("[1,3072,1152]", "[B,T·S,H]"),
      input: io("[1,1152,12,16,16]", "[B,H,T,h/p,w/p]"), description: "PatchEmbed3D 实际返回 flatten+transpose 后的3D token tensor，不直接返回内部 Conv3d 的5D形状。", formula: "T·S = 12×16×16 = 3072", source: `${sourceRoot}/models/layers/blocks.py:121–129`, paths: ["video"],
    },
    {
      id: "tokenGrid", x: 1400, y: 505, w: 260, h: 110, kind: "video", kindLabel: "GRID RESTORE + POSITION",
      title: "rearrange + 2D pos", note: "2D位置编码广播到12帧", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("[1,3072,1152]", "[B,T·S,H]"), description: "STDiT 恢复时间×空间网格并加2D spatial position embedding。Temporal RoPE 是后面另一种位置编码。", formula: "S=16×16=256\nx₀ = reshape(tokens)+pos₂ᴅ", source: `${sources.stdit}; ${sourceRoot}/models/layers/blocks.py:944–993`, paths: ["video"],
    },

    // ActionEncoder and Spatial 6H pipeline.
    {
      id: "actionFlatten", x: 835, y: 750, w: 250, h: 94, kind: "action", kindLabel: "RESHAPE",
      title: "flatten B×Ta", note: "保留每个7D动作", shapes: shape("[8,7]", "[B·Ta,Da]"),
      input: io("[1,8,7]", "[B,Ta,Da]"), description: "ActionEncoder 先合并 batch 与 action 时间维，逐 action 使用同一个 MLP。", formula: "rearrange('B T D → (B T) D')", source: sources.stdit, paths: ["action"],
    },
    {
      id: "actionFc1", x: 835, y: 865, w: 250, h: 94, kind: "action", kindLabel: "LINEAR",
      title: "7 → 4608", note: "4H expansion", shapes: shape("[8,4608]", "[B·Ta,4H]"),
      input: io("[8,7]", "[B·Ta,Da]"), description: "动作向量经第一层线性层扩展到4倍 hidden size。", formula: "Da=7, 4H=4×1152=4608", source: sources.stdit, paths: ["action"],
    },
    {
      id: "actionSilu", x: 835, y: 980, w: 250, h: 94, kind: "action", kindLabel: "ACTIVATION",
      title: "SiLU", note: "x·sigmoid(x)", shapes: shape("[8,4608]", "[B·Ta,4H]"),
      input: io("[8,4608]", "[B·Ta,4H]"), description: "ActionEncoder 中唯一的非线性激活。它位于两个 Linear 之间，shape 不变。", formula: "SiLU(x)=x·σ(x)", source: sources.stdit, paths: ["action"],
    },
    {
      id: "actionFc2", x: 835, y: 1095, w: 250, h: 94, kind: "action", kindLabel: "LINEAR",
      title: "4608 → 1152", note: "投影回 H", shapes: shape("[8,1152]", "[B·Ta,H]"),
      input: io("[8,4608]", "[B·Ta,4H]"), description: "第二层线性层将动作特征投影回 Transformer hidden size。", formula: "4H → H", source: sources.stdit, paths: ["action"],
    },
    {
      id: "actionEight", x: 835, y: 1210, w: 250, h: 94, kind: "action", kindLabel: "RESHAPE",
      title: "8 action embeddings", note: "未来8槽的normal条件", shapes: shape("[1,8,1152]", "[B,Ta,H]"),
      input: io("[8,1152]", "[B·Ta,H]"), description: "恢复 batch 和 action 时间维。每个 action embedding 对应一个未来 temporal slot 的 normal modulation 分支。", formula: "reshape → [B,Ta,H]", source: sources.stdit, paths: ["action"],
    },
    {
      id: "noneParam", x: 1120, y: 910, w: 250, h: 104, kind: "action", kindLabel: "ONE LEARNED PARAMETER",
      title: "none_action", note: "只有1个可学习向量", shapes: shape("[1,1152]", "[1,H]"),
      input: io("Embedding(1,H).weight"), description: "源码只有一个 learned NONE vector，不是4个互不相同的可学习向量。", formula: "nn.Embedding(1,H)", source: sources.stdit, paths: ["action"],
    },
    {
      id: "noneRepeat", x: 1120, y: 1040, w: 250, h: 104, kind: "action", kindLabel: "TENSOR REPEAT",
      title: "repeat NONE ×4", note: "只为补齐时间长度", shapes: shape("[1,4,1152]", "[B,Tₕ,H]"),
      input: io("[1,1152]", "[1,H]"), description: "同一个 NONE embedding 被复制到4个历史槽。masked inference 中这些历史位置最终选择 t0=15 分支，因此不要理解为 NONE 实际调制了历史特征。", formula: "repeat(B, pad_action_num=4, 1)", source: sources.stdit, paths: ["action"],
    },
    {
      id: "actionConcat", x: 1120, y: 1210, w: 250, h: 104, kind: "action", kindLabel: "TIME ALIGNMENT",
      title: "concat NONE + actions", note: "tensor-level length = 12", shapes: shape("[1,12,1152]", "[B,T,H]"),
      input: io("[1,4,1152] + [1,8,1152]", "[B,Tₕ,H] + [B,Ta,H]"), description: "人为把 Action condition 扩展到与12个 video slots 相同长度。这是 tensor-level alignment，不代表12个物理 action 与12帧天然一一对应。", formula: "[NONE×4 | E(a_s)…E(a_{s+7})]", source: sources.stdit, paths: ["action"],
    },
    {
      id: "actionFlat12", x: 1120, y: 1340, w: 250, h: 104, kind: "action", kindLabel: "ACTIONENCODER RETURN",
      title: "flatten condition", note: "准备与 repeat timestep 相加", shapes: shape("[12,1152]", "[B·T,H]"),
      input: io("[1,12,1152]", "[B,T,H]"), description: "ActionEncoder forward 的最终公开输出再次把 batch 与12槽展平。", formula: "rearrange('B T H → (B T) H')", source: sources.stdit, paths: ["action"],
    },
    {
      id: "tEmbed", x: 1245, y: 660, w: 245, h: 104, kind: "time", kindLabel: "TIMESTEP EMBEDDER",
      title: "Eₜ(tₖ)", note: "sinusoidal → MLP", shapes: shape("[1,1152]", "[B,H]"),
      input: io("[1]", "[B]"), description: "当前 sampling timestep 经过 sinusoidal frequency embedding 与两层 MLP 得到全局 timestep embedding。", formula: "[B] → [B,256] → [B,H]", source: `${sourceRoot}/models/layers/blocks.py:788–827`, paths: ["time"],
    },
    {
      id: "tRepeat", x: 1245, y: 790, w: 245, h: 104, kind: "time", kindLabel: "SPATIAL ONLY",
      title: "repeat timestep ×12", note: "每个frame slot一份", shapes: shape("[12,1152]", "[B·T,H]"),
      input: io("[1,1152]", "[B,H]"), description: "Spatial condition 需要逐 temporal slot 的参数，因此将全局 timestep embedding 复制12份。", formula: "repeat(t, c=T=12)", source: sources.stdit, paths: ["time"],
    },
    {
      id: "spatialSum", x: 1245, y: 1470, w: 245, h: 108, kind: "action", kindLabel: "CONDITION FUSION",
      title: "cₛ = Eₜ + Eₐ", note: "不是加到video token", shapes: shape("[12,1152]", "[B·T,H]"),
      input: io("[12,1152] + [12,1152]", "[B·T,H] + [B·T,H]"), description: "Action branch 与 video latent branch 在这里并不直接融合。Action 只与 timestep embedding 相加，随后生成 Spatial AdaLN/gate 参数。", formula: "cₛᵢ = Eₜ(t)+Eₐ(action_slotᵢ)", source: sources.stdit, paths: ["action", "time"],
    },
    {
      id: "tBlockSpatial", x: 1245, y: 1600, w: 245, h: 104, kind: "action", kindLabel: "SHARED t_block",
      title: "Spatial t_block", note: "H → 6H", shapes: shape("[12,6912]", "[B·T,6H]"),
      input: io("[12,1152]", "[B·T,H]"), description: "共享的 t_block 将每个 temporal slot 的 condition 映射为六组 modulation/gate 参数的打包向量。", formula: "6H = 6×1152 = 6912", source: sources.stdit, paths: ["action", "time"],
    },
    {
      id: "spatialSixH", x: 1245, y: 1730, w: 245, h: 126, kind: "action", kindLabel: "STRICT BLOCK SHAPE",
      title: "reshape + table", note: "加每个Block自己的table", shapes: shape("[1,12,6,1152]", "[B,T,6,H]"),
      input: io("[12,6912] + [6,1152]", "[B·T,6H] + [6,H]"), description: "进入当前 Spatial Block 后先 reshape，再加该 Block 自己的 scale_shift_table。", formula: "[B·T,6H] → [B,T,6,H]\n+ scale_shift_table[6,H]", source: sources.block, paths: ["action", "time"],
    },
    {
      id: "spatialRearrange", x: 1245, y: 1880, w: 245, h: 94, kind: "action", kindLabel: "REARRANGE",
      title: "pack 6H", note: "保留T与broadcast轴", shapes: shape("[1,12,1,6912]", "[B,T,1,6H]"),
      input: io("[1,12,6,1152]", "[B,T,6,H]"), description: "把六组 hidden 参数重新打包到最后一维，并保留一个空间 token 广播轴。", formula: "B T n H → B T 1 (n H)", source: sources.block, paths: ["action", "time"],
    },
    {
      id: "spatialChunk", x: 1245, y: 1990, w: 245, h: 104, kind: "action", kindLabel: "CHUNK(6)",
      title: "6 组 Spatial 参数", note: "shift / scale / gate ×2", shapes: shape("6 × [1,12,1,1152]", "6 × [B,T,1,H]"),
      input: io("[1,12,1,6912]", "[B,T,1,6H]"), description: "依次得到 shift_msa、scale_msa、gate_msa、shift_mlp、scale_mlp、gate_mlp。四条绿色路由只进入 Spatial 两处 AdaLN 和两个 gate。", formula: "chunk(6, dim=-1)", source: sources.block, paths: ["action", "time"],
    },

    // Spatial block main chain.
    {
      id: "spInput", x: 1840, y: 730, w: 275, h: 94, kind: "video", kindLabel: "SPATIAL INPUT",
      title: "x₀", note: "每帧256个token", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("token grid + pos₂ᴅ"), description: "Spatial Block 接收视频 hidden state。Action 参数通过旁路调制，不作为 token 拼进 x。", formula: "x₀ ∈ R^{B×T×S×H}", source: sources.block, paths: ["video"],
    },
    {
      id: "spNorm1", x: 1840, y: 842, w: 275, h: 84, kind: "video", kindLabel: "PRE-NORM",
      title: "LayerNorm 1", note: "affine=False", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("[1,12,256,1152]", "[B,T,S,H]"), description: "先归一化 hidden；固定 affine 被关闭，随后由 condition 动态提供 scale/shift。", formula: "x̂ = LN(x)", source: sources.block, paths: ["video"],
    },
    {
      id: "spAda1", x: 1840, y: 944, w: 275, h: 96, kind: "action", kindLabel: "AdaLN + torch.where",
      title: "Spatial AdaLN 1", note: "normal/t₀按slot选择", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("LN(x), shift_msa, scale_msa, t₀, x_mask"), description: "future槽选 normal timestep+action 参数；history槽选 t0=15 参数。历史 NONE normal 参数在当前 masked inference 中不会成为最终选择。", formula: "xₘ=(1+scale)·LN(x)+shift\nparams=torch.where(x_mask,normal,t₀)", source: sources.block, paths: ["video", "action", "mask"],
    },
    {
      id: "spBT", x: 1840, y: 1058, w: 275, h: 84, kind: "video", kindLabel: "REARRANGE",
      title: "B·T batch", note: "逐帧做空间注意力", shapes: shape("[12,256,1152]", "[B·T,S,H]"),
      input: io("[1,12,256,1152]", "[B,T,S,H]"), description: "把12个 frame slot 视为独立 batch，每帧内部让256个 spatial tokens 互相注意。", formula: "B T S H → (B T) S H", source: sources.block, paths: ["video"],
    },
    {
      id: "spQkv", x: 1840, y: 1160, w: 275, h: 96, kind: "video", kindLabel: "QKV LINEAR",
      title: "1152 → 3456", note: "16 heads · head_dim 72", shapes: shape("Q/K/V [12,16,256,72]", "Q/K/V [B·T,A,S,Dh]"),
      input: io("[12,256,1152]", "[B·T,S,H]"), description: "一次线性投影得到 QKV，再拆成16个注意力头，每头72维。", formula: "QKV=3H=3456\nDh=H/heads=72", source: sources.attention, paths: ["video"],
    },
    {
      id: "spScores", x: 1840, y: 1274, w: 275, h: 104, kind: "video", kindLabel: "LOGICAL SHAPE",
      title: "Spatial attention scores", note: "FlashAttention可不显式物化", shapes: shape("[12,16,256,256]", "[B·T,A,S,S]"),
      input: io("Q·Kᵀ"), description: "这是数学上的 score shape。当前示例 Spatial 通常满足 FlashAttention 条件，因此完整 score tensor 可能不会在显存中显式生成。", formula: "scores = QKᵀ / √72", source: sources.attention, paths: ["video"],
    },
    {
      id: "spAttention", x: 1840, y: 1396, w: 275, h: 92, kind: "video", kindLabel: "SELF-ATTENTION",
      title: "Attention output", note: "restore heads + projection", shapes: shape("[12,256,1152]", "[B·T,S,H]"),
      input: io("scores, V"), description: "空间 self-attention 聚合每帧内部的256个 token，再合并16个 heads 并做输出投影。", formula: "softmax(scores)·V → H", source: sources.attention, paths: ["video"],
    },
    {
      id: "spGate1", x: 1840, y: 1506, w: 275, h: 96, kind: "action", kindLabel: "GATE + RESIDUAL",
      title: "gate_msa · attn + x₀", note: "gate也按mask选择", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("attention, gate_msa, t₀ gate, residual"), description: "Attention 输出恢复到4D网格，乘逐slot gate_msa 后加回原 residual。gate 同样通过 x_mask 在 normal 与 t0 间选择。", formula: "x₁=x₀+gate_msa·Attn(AdaLN₁(x₀))", source: sources.block, paths: ["video", "action", "mask"],
    },
    {
      id: "spNorm2", x: 1840, y: 1620, w: 275, h: 84, kind: "video", kindLabel: "PRE-NORM",
      title: "LayerNorm 2", note: "进入MLP支路", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("x₁"), description: "第二条 residual 支路先做 LayerNorm。", formula: "LN(x₁)", source: sources.block, paths: ["video"],
    },
    {
      id: "spAda2", x: 1840, y: 1722, w: 275, h: 96, kind: "action", kindLabel: "AdaLN + torch.where",
      title: "Spatial AdaLN 2", note: "shift_mlp / scale_mlp", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("LN(x₁), MLP modulation, t₀, x_mask"), description: "第二组 shift/scale 在 MLP 前调制 normalized hidden，仍逐 temporal slot 广播到全部 spatial tokens。", formula: "(1+scale_mlp)·LN(x₁)+shift_mlp", source: sources.block, paths: ["video", "action", "mask"],
    },
    {
      id: "spMlp", x: 1840, y: 1836, w: 275, h: 96, kind: "video", kindLabel: "FEED-FORWARD",
      title: "MLP 1152→4608→1152", note: "逐token处理", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("[1,12,256,1152]", "[B,T,S,H]"), description: "Transformer MLP 将每个 token 扩展到4H，再投影回H；这里与 ActionEncoder 的 SiLU 位置不是同一个模块。", formula: "H → 4H → H\n4H=4608", source: sources.block, paths: ["video"],
    },
    {
      id: "spX2", x: 1840, y: 1950, w: 275, h: 116, kind: "output", kindLabel: "GATE + RESIDUAL · HANDOFF",
      title: "x₂ · Spatial 输出", note: "shape不变 · 数值已含action影响", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("x₁, MLP output, gate_mlp"), description: "MLP 输出乘 gate_mlp 后加回 residual。x₂ 是普通 hidden tensor，它直接进入 Temporal Block；不是“action广播张量”。", formula: "x₂=x₁+gate_mlp·MLP(AdaLN₂(x₁))", source: sources.block, paths: ["video", "action", "mask"],
    },

    // Temporal block condition and main chain.
    {
      id: "tBlockTemporal", x: 2250, y: 730, w: 205, h: 94, kind: "time", kindLabel: "TEMPORAL CONDITION",
      title: "t_block(Eₜ)", note: "无直接action", shapes: shape("[1,6912]", "[B,6H]"),
      input: io("[1,1152]", "[B,H]"), description: "Temporal Block 只使用全局 diffusion timestep condition，不直接加 Action embedding。它通过输入 x₂ 间接接收 action 影响。", formula: "Eₜ(t) → t_block → 6H", source: sources.stdit, paths: ["time"],
    },
    {
      id: "temporalSixH", x: 2250, y: 842, w: 205, h: 104, kind: "time", kindLabel: "STRICT SHAPE",
      title: "reshape + table", note: "T维为broadcast 1", shapes: shape("[1,1,6,1152]", "[B,1,6,H]"),
      input: io("[1,6912] + [6,1152]", "[B,6H] + [6,H]"), description: "Temporal condition 在 Block 内 reshape 后加该 Block 自己的 scale_shift_table。", formula: "[B,6H] → [B,1,6,H]", source: sources.block, paths: ["time"],
    },
    {
      id: "temporalPack", x: 2250, y: 964, w: 205, h: 94, kind: "time", kindLabel: "REARRANGE",
      title: "pack 6H", note: "跨T和S广播", shapes: shape("[1,1,1,6912]", "[B,1,1,6H]"),
      input: io("[1,1,6,1152]", "[B,1,6,H]"), description: "生成能够广播到全部时间和空间 token 的参数布局。", formula: "B T n H → B T 1 (nH)", source: sources.block, paths: ["time"],
    },
    {
      id: "temporalChunk", x: 2250, y: 1076, w: 205, h: 104, kind: "time", kindLabel: "CHUNK(6)",
      title: "6组Temporal参数", note: "shift/scale/gate ×2", shapes: shape("6 × [1,1,1,1152]", "6 × [B,1,1,H]"),
      input: io("[1,1,1,6912]", "[B,1,1,6H]"), description: "六组 timestep-only 参数分别送入 Temporal 两处 AdaLN 与两个 gate。无绿色 action edge。", formula: "chunk(6, dim=-1)", source: sources.block, paths: ["time"],
    },
    {
      id: "tmInput", x: 2605, y: 730, w: 280, h: 104, kind: "video", kindLabel: "DIRECT HIDDEN HANDOFF",
      title: "Temporal input = x₂", note: "蓝色实线 · 原样传递", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("Spatial x₂"), description: "Spatial 输出作为普通 hidden state 原样进入 Temporal；shape不变，数值已经携带 action 对 Spatial 计算的影响。", formula: "x_temporal_in := x₂", source: sources.block, paths: ["video"],
    },
    {
      id: "tmBS", x: 2605, y: 980, w: 280, h: 92, kind: "video", kindLabel: "REARRANGE",
      title: "B·S batch", note: "每个空间位置看12个时间槽", shapes: shape("[256,12,1152]", "[B·S,T,H]"),
      input: io("AdaLN output [1,12,256,1152]", "AdaLN output [B,T,S,H]"), description: "AdaLN完成后，才把256个空间位置视为独立 batch，每个位置沿12个 temporal slots 做注意力。", formula: "B T S H → (B S) T H", source: sources.block, paths: ["video"],
    },
    {
      id: "tmAda1", x: 2605, y: 858, w: 280, h: 104, kind: "time", kindLabel: "LN + AdaLN + torch.where",
      title: "Timestep-AdaLN 1", note: "先调制4D网格，再rearrange", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("x₂, Temporal params, t₀, x_mask"), description: "Temporal modulation只依赖 timestep，但 condition/history槽仍由 x_mask 选择 t0=15 参数。", formula: "params=torch.where(x_mask,normal,t₀)\nAdaLN(LN(x₂), params)", source: sources.block, paths: ["video", "time", "mask"],
    },
    {
      id: "tmQkv", x: 2605, y: 1096, w: 280, h: 98, kind: "video", kindLabel: "QKV LINEAR",
      title: "Temporal Q / K / V", note: "16 heads · head_dim 72", shapes: shape("Q/K/V [256,16,12,72]", "Q/K/V [B·S,A,T,Dh]"),
      input: io("[256,12,1152]", "[B·S,T,H]"), description: "沿时间轴为每个空间位置生成 QKV。Temporal 直接输入仍是被 action 影响过的 x₂。", formula: "H → 3H; split 16 heads", source: sources.attention, paths: ["video"],
    },
    {
      id: "tmScores", x: 2605, y: 1212, w: 280, h: 108, kind: "video", kindLabel: "ATTENTION SCORES",
      title: "Temporal logical scores", note: "当前示例通常实际物化", shapes: shape("[256,16,12,12]", "[B·S,A,T,T]"),
      input: io("Q·Kᵀ + causal mask"), description: "数学 score shape 为[256,16,12,12]。当前运行条件下 Temporal 通常不走 FlashAttention，因此这张较小矩阵会实际物化。", formula: "scores = QKᵀ/√72\ncausal: j>i → −∞", source: sources.attention, paths: ["video"],
    },
    {
      id: "tmAttention", x: 2605, y: 1338, w: 280, h: 108, kind: "video", kindLabel: "CAUSAL SELF-ATTENTION",
      title: "Temporal Attention + RoPE", note: "causal是Attention属性", shapes: shape("[256,12,1152]", "[B·S,T,H]"),
      input: io("Q/K/V + causal + RoPE"), description: "RoPE 表达时间顺序，causal 由 Temporal Attention 的 is_causal 属性启用；它不是 STDiT 外部的独立算子。", formula: "RoPE(Q,K) → causal attention", source: `${sources.attention}; ${sources.stdit}`, paths: ["video"],
    },
    {
      id: "tmGate1", x: 2605, y: 1464, w: 280, h: 96, kind: "time", kindLabel: "GATE + RESIDUAL",
      title: "gate_msa(t) + residual", note: "mask选择normal/t₀ gate", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("temporal attention, gate_msa(t), residual"), description: "Temporal attention 恢复网格后乘 timestep gate 并加回 x₂ residual。", formula: "z₁=x₂+gate_msa(t)·TemporalAttn", source: sources.block, paths: ["video", "time", "mask"],
    },
    {
      id: "tmAda2", x: 2605, y: 1578, w: 280, h: 104, kind: "time", kindLabel: "LN + AdaLN + torch.where",
      title: "Timestep-AdaLN 2", note: "MLP前调制", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("z₁, Temporal MLP params, t₀, x_mask"), description: "Temporal 第二处 LayerNorm/AdaLN，同样不直接接 Action 参数。", formula: "AdaLN(LN(z₁), shift_mlp(t), scale_mlp(t))", source: sources.block, paths: ["video", "time", "mask"],
    },
    {
      id: "tmMlp", x: 2605, y: 1700, w: 280, h: 96, kind: "video", kindLabel: "FEED-FORWARD",
      title: "Temporal MLP", note: "1152→4608→1152", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("Temporal AdaLN 2 output"), description: "逐 token 的 feed-forward 网络，shape 保持不变。", formula: "H → 4H → H", source: sources.block, paths: ["video"],
    },
    {
      id: "pairOutput", x: 2605, y: 1814, w: 280, h: 112, kind: "output", kindLabel: "GATE + RESIDUAL",
      title: "Pair #1 output", note: "进入下一Spatial Pair", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("z₁, MLP output, gate_mlp(t)"), description: "Temporal MLP 输出乘 gate_mlp(t) 后加回 residual，完成一个 Spatial→Temporal Pair。", formula: "z₂=z₁+gate_mlp(t)·MLP(AdaLN₂(z₁))", source: sources.block, paths: ["video", "time", "mask"],
    },

    // t0 / mask lane.
    {
      id: "t0Constant", x: 3055, y: 680, w: 270, h: 96, kind: "mask", kindLabel: "FIXED CONDITION TIME",
      title: "t₀ timestep = 15", note: "仅在 x_mask 存在时构造", shapes: shape("[1]", "[B]"),
      input: io("ones_like(timestep) × 15"), description: "STDiT 为 condition/history slots 构造一个固定 timestep t0=15，用来生成替代 modulation。", formula: "t₀ = 15·ones_like(t)", source: sources.stdit, paths: ["mask"],
    },
    {
      id: "t0Embed", x: 3055, y: 800, w: 270, h: 96, kind: "mask", kindLabel: "TIMESTEP EMBEDDER",
      title: "Eₜ(t₀)", note: "与normal共享embedder", shapes: shape("[1,1152]", "[B,H]"),
      input: io("[1]", "[B]"), description: "t0 使用同一个 TimestepEmbedder。", formula: "t₀ → TimestepEmbedder", source: sources.stdit, paths: ["mask"],
    },
    {
      id: "t0Block", x: 3055, y: 920, w: 270, h: 104, kind: "mask", kindLabel: "SHARED t_block",
      title: "t₀_mlp", note: "condition-frame 6H参数", shapes: shape("[1,6912]", "[B,6H]"),
      input: io("[1,1152]", "[B,H]"), description: "t0 embedding 经过同一个 t_block 生成历史/condition 槽使用的六组参数。每个 Block 再与自己的 table 结合。", formula: "Eₜ(t₀) → t_block → 6H", source: sources.stdit, paths: ["mask"],
    },
    {
      id: "maskMeaning", x: 3030, y: 1090, w: 320, h: 150, kind: "mask", kindLabel: "FOUR SELECTORS PER BLOCK",
      title: "torch.where ×4", note: "AdaLN₁ · gate_msa · AdaLN₂ · gate_mlp", shapes: shape("broadcast over S=256", "broadcast over spatial tokens"),
      input: io("normal params, t₀ params, x_mask"), description: "每个 Spatial/Temporal Block 实际在四个位置独立选择参数。图中的紫色端口准确落到这四类操作，而不是连到整个Block边框。", formula: "where(mask, normal, t₀)\nmask shape [B,T] → broadcast", source: sources.block, paths: ["mask"],
    },
    {
      id: "noneCaveat", x: 3030, y: 1285, w: 320, h: 172, kind: "mask", kindLabel: "IMPORTANT INFERENCE SEMANTICS",
      title: "历史 NONE ≠ 最终调制", note: "NONE先补齐；t₀后覆盖normal分支", shapes: shape("slots 0…3 select t₀", "history slots select t₀"),
      input: io("NONE normal params + t₀ selector"), description: "ActionEncoder 仍为历史4槽生成重复 NONE normal 参数，但 x_mask=False 使这些位置最终选择 t0=15 modulation。", formula: "history: where(False, NONE-normal, t₀) = t₀\nfuture: where(True, action-normal, t₀) = action-normal", source: `${sources.stdit}; ${sources.block}`, paths: ["action", "mask"],
    },
    {
      id: "posEncodingNote", x: 3030, y: 1510, w: 320, h: 156, kind: "video", kindLabel: "TWO POSITION MECHANISMS",
      title: "2D Pos ≠ Temporal RoPE", note: "分别描述空间与时间", shapes: shape("pos₂ᴅ [1,256,1152]", "pos₂ᴅ [1,S,H]"),
      input: io("spatial grid / temporal Q,K"), description: "2D Spatial Position Embedding 直接加到 token 主干；RoPE 只在 Temporal Attention 内旋转 Q/K。", formula: "token += pos₂ᴅ\nQ,K = RoPE(Q,K)", source: `${sources.stdit}; ${sources.attention}`, paths: ["video"],
    },

    // Collapsed remaining backbone and final layer.
    {
      id: "remainingPairs", x: 2140, y: 2140, w: 360, h: 112, kind: "video", kindLabel: "BACKBONE ×28 TOTAL",
      title: "Pair #2 … Pair #28", note: "上方只展开 Pair #1", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("Pair #1 output + reused conditions"), description: "真实 STDiT3-XL/2 按顺序执行28组 Spatial→Temporal。上方展示的是其中第一组，其余27组在此折叠；同一份 t_s_mlp、t_t_mlp 与 t0/x_mask 会复用于每一组，而每个Block拥有自己的 scale_shift_table。", formula: "for spatial, temporal in zip(28,28):\n  x=spatial(x,t_s,t₀,mask)\n  x=temporal(x,t_t,t₀,mask)", source: sources.stdit, paths: ["video", "action", "time", "mask"],
    },
    {
      id: "finalAda", x: 2540, y: 2140, w: 265, h: 112, kind: "time", kindLabel: "FINAL LN / AdaLN",
      title: "Final modulation", note: "也使用 x_mask 选择 t/t₀", shapes: shape("[1,12,256,1152]", "[B,T,S,H]"),
      input: io("hidden, t, t₀, x_mask"), description: "Final Layer 先归一化并调制 hidden；它同样为history/future槽选择t0/normal参数，不能漏掉 mask 支路。", formula: "shift,scale=where(x_mask, t, t₀)\ny=(1+scale)·LN(x)+shift", source: sources.final, paths: ["video", "time", "mask"],
    },
    {
      id: "finalLinear", x: 2845, y: 2140, w: 250, h: 112, kind: "video", kindLabel: "LINEAR PROJECTION",
      title: "1152 → 32", note: "1×2×2×8 = 32", shapes: shape("[1,12,256,32]", "[B,T,S,pₜpₕp_wCout]"),
      input: io("[1,12,256,1152]", "[B,T,S,H]"), description: "每个 patch token 输出 patch volume×8 channels，共32个数。", formula: "final_dim=1×2×2×8=32", source: `${sources.final}; ${sources.stdit}`, paths: ["video"],
    },
    {
      id: "unpatchify", x: 3135, y: 2140, w: 250, h: 112, kind: "video", kindLabel: "UNPATCHIFY",
      title: "restore latent grid", note: "STDiT 8-channel output", shapes: shape("[1,8,12,32,32]", "[B,2Cz,T,h,w]"),
      input: io("[1,12,256,32]", "[B,T,S,patch·2Cz]"), description: "将每个2×2 patch的8通道预测恢复成完整 latent 网格。后4通道不在页面中做未经代码确认的物理解释。", formula: "[B,T,S,32] → [B,8,T,32,32]", source: sources.stdit, paths: ["video"],
    },

    // RFlow update and final decode.
    {
      id: "firstHalf", x: 3025, y: 2340, w: 300, h: 112, kind: "video", kindLabel: "SCHEDULER CONSUMES",
      title: "take first channel half", note: "不是 CFG chunk", shapes: shape("[1,4,12,32,32]", "[B,Cz,T,h,w]"),
      input: io("[1,8,12,32,32]", "[B,2Cz,T,h,w]"), description: "活跃代码对通道维执行 chunk(2)[0]，取前4通道作为 flow prediction。这一步与被注释的CFG无关。", formula: "pred = model(...).chunk(2,dim=1)[0]", source: sources.rflow, paths: ["video"],
    },
    {
      id: "rflowUpdate", x: 2645, y: 2340, w: 320, h: 112, kind: "video", kindLabel: "RECTIFIED FLOW UPDATE",
      title: "z ← z + v_pred · dt", note: "更新完整12槽", shapes: shape("[1,4,12,32,32]", "[B,Cz,T,h,w]"),
      input: io("z_current, flow, dt"), description: "用当前 flow prediction 做一次 Euler-style 更新。此时condition槽也暂时被计算，下一节点再恢复。", formula: "z_next = z + v_pred·dt", source: sources.rflow, paths: ["video", "time"],
    },
    {
      id: "restoreCondition", x: 2250, y: 2340, w: 335, h: 112, kind: "mask", kindLabel: "PRESERVE CONDITION SLOTS",
      title: "torch.where restore history", note: "前4槽固定", shapes: shape("[1,4,12,32,32]", "[B,Cz,T,h,w]"),
      input: io("updated z, pre-update x₀, mask"), description: "源码在本步更新前保存 x0=z.clone()；随后用它的前4个history槽恢复condition位置，后8槽保留更新结果。整个12槽状态再进入下一 sampling step。", formula: "x₀ = z.clone() before update\nz = where(mask, updated_z, x₀)", source: sources.rflow, paths: ["video", "mask"],
    },
    {
      id: "futureSlice", x: 1020, y: 2790, w: 320, h: 112, kind: "output", kindLabel: "AFTER STEP 30",
      title: "take last 8 temporal slots", note: "丢弃condition slots", shapes: shape("[1,4,8,32,32]", "[B,Cz,Tf,h,w]"),
      input: io("[1,4,12,32,32]", "[B,Cz,T,h,w]"), description: "30步结束后 scheduler 返回完整12槽。rollout wrapper 明确取最后8槽作为未来 latent，再送入 VAE Decoder。", formula: "future = z[:, :, -8:]", source: sources.rollout, paths: ["video"],
    },
    {
      id: "vaeDecode", x: 1600, y: 2790, w: 320, h: 112, kind: "output", kindLabel: "VAE DECODER",
      title: "decode future latent", note: "空间放大×8", shapes: shape("[1,3,8,256,256]", "[B,3,Tf,Himg,Wimg]"),
      input: io("[1,4,8,32,32]", "[B,Cz,Tf,h,w]"), description: "VAE 将8个future latent slots解码为8帧RGB，返回布局是[B,C,T,H,W]。", formula: "[B,4,8,32,32] → [B,3,8,256,256]", source: sources.vae, paths: ["video"],
    },
    {
      id: "futureRgb", x: 2180, y: 2790, w: 360, h: 112, kind: "output", kindLabel: "DISPLAY LAYOUT",
      title: "8 future RGB frames", note: "T移到第2维，channels最后", shapes: shape("[1,8,256,256,3]", "[B,Tf,Himg,Wimg,3]"),
      input: io("[1,3,8,256,256]", "[B,3,Tf,Himg,Wimg]"), description: "rollout 最后 permute 为便于显示/保存的视频布局。", formula: "permute(0,2,3,4,1)", source: sources.rollout, paths: ["video"],
    },
  ];

  const nodeById = new Map(nodes.map((node) => [node.id, node]));

  const edges = [
    // Rollout setup.
    { id: "e-rgb-vae", from: "rgbCurrent", to: "vaeEncode", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-vae-repeat", from: "vaeEncode", to: "repeatHistory", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-repeat-queue", from: "repeatHistory", to: "historyQueue", type: "video", fromSide: "bottom", toSide: "left", via: [[215, 590], [215, 672]], paths: ["video"] },
    { id: "e-prev-queue", from: "previousLatent", to: "historyQueue", type: "video", fromSide: "bottom", toSide: "right", via: [[515, 590], [515, 672]], label: "后续chunk", labelAt: [530, 598], paths: ["video"] },
    { id: "e-queue-concat", from: "historyQueue", to: "concatOnce", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-noise-concat", from: "futureNoise", to: "concatOnce", type: "video", fromSide: "bottom", toSide: "right", via: [[515, 915], [550, 1001]], paths: ["video"] },
    { id: "e-concat-z", from: "concatOnce", to: "zCurrent", type: "video", fromSide: "right", toSide: "left", via: [[675, 1001], [675, 216], [770, 216]], label: "仅一次真实 concat", labelAt: [590, 190], paths: ["video"] },
    { id: "e-mask-xmask", from: "maskSpec", to: "xMask", type: "mask", fromSide: "right", toSide: "left", via: [[650, 1171], [650, 420], [1600, 420], [1600, 216]], paths: ["mask"] },
    { id: "e-action-input", from: "actionChunk", to: "actionFlatten", type: "action", fromSide: "right", toSide: "left", via: [[680, 1366], [680, 797], [790, 797]], paths: ["action"] },

    // Sampler / model input.
    { id: "e-z-patch", from: "zCurrent", to: "patchConv", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-t-transform", from: "timestepRaw", to: "timestepTransform", type: "time", fromSide: "bottom", toSide: "top", paths: ["time"] },
    { id: "e-transform-embed", from: "timestepTransform", to: "tEmbed", type: "time", fromSide: "bottom", toSide: "top", via: [[1378, 440], [1378, 620]], paths: ["time"] },
    { id: "e-transform-xmask", from: "timestepTransform", to: "xMask", type: "time", fromSide: "right", toSide: "left", via: [[1560, 346], [1560, 216], [1620, 216]], label: "current transformed t", labelAt: [1570, 198], paths: ["time", "mask"] },
    { id: "e-mask-t0", from: "xMask", to: "t0Constant", type: "mask", fromSide: "right", toSide: "top", via: [[2995, 216], [2995, 635], [3190, 635]], paths: ["mask"] },
    { id: "e-xmask-meaning", from: "xMask", to: "maskMeaning", type: "mask", fromSide: "right", toSide: "top", via: [[2990, 216], [2990, 1060], [3190, 1060]], paths: ["mask"] },
    { id: "e-patch-flat", from: "patchConv", to: "patchFlatten", type: "video", fromSide: "right", toSide: "left", paths: ["video"] },
    { id: "e-flat-grid", from: "patchFlatten", to: "tokenGrid", type: "video", fromSide: "right", toSide: "left", paths: ["video"] },
    { id: "e-grid-sp", from: "tokenGrid", to: "spInput", type: "video", fromSide: "right", toSide: "top", via: [[1700, 560], [1978, 560], [1978, 690]], paths: ["video"] },

    // Action and timestep conditions.
    { id: "e-aflat-fc1", from: "actionFlatten", to: "actionFc1", type: "action", fromSide: "bottom", toSide: "top", paths: ["action"] },
    { id: "e-fc1-silu", from: "actionFc1", to: "actionSilu", type: "action", fromSide: "bottom", toSide: "top", paths: ["action"] },
    { id: "e-silu-fc2", from: "actionSilu", to: "actionFc2", type: "action", fromSide: "bottom", toSide: "top", paths: ["action"] },
    { id: "e-fc2-a8", from: "actionFc2", to: "actionEight", type: "action", fromSide: "bottom", toSide: "top", paths: ["action"] },
    { id: "e-none-repeat", from: "noneParam", to: "noneRepeat", type: "action", fromSide: "bottom", toSide: "top", paths: ["action"] },
    { id: "e-a8-concat", from: "actionEight", to: "actionConcat", type: "action", fromSide: "right", toSide: "left", paths: ["action"] },
    { id: "e-none-concat", from: "noneRepeat", to: "actionConcat", type: "action", fromSide: "bottom", toSide: "top", paths: ["action"] },
    { id: "e-concat-aflat", from: "actionConcat", to: "actionFlat12", type: "action", fromSide: "bottom", toSide: "top", paths: ["action"] },
    { id: "e-tembed-repeat", from: "tEmbed", to: "tRepeat", type: "time", fromSide: "bottom", toSide: "top", paths: ["time"] },
    { id: "e-repeat-sum", from: "tRepeat", to: "spatialSum", type: "time", fromSide: "bottom", toSide: "top", via: [[1480, 915], [1510, 915], [1510, 1420], [1368, 1420]], paths: ["time"] },
    { id: "e-aflat-sum", from: "actionFlat12", to: "spatialSum", type: "action", fromSide: "bottom", toSide: "left", via: [[1245, 1450], [1210, 1450], [1210, 1524]], paths: ["action"] },
    { id: "e-sum-tblock", from: "spatialSum", to: "tBlockSpatial", type: "action", fromSide: "bottom", toSide: "top", paths: ["action", "time"] },
    { id: "e-tblock-six", from: "tBlockSpatial", to: "spatialSixH", type: "action", fromSide: "bottom", toSide: "top", paths: ["action", "time"] },
    { id: "e-six-pack", from: "spatialSixH", to: "spatialRearrange", type: "action", fromSide: "bottom", toSide: "top", paths: ["action", "time"] },
    { id: "e-pack-chunk", from: "spatialRearrange", to: "spatialChunk", type: "action", fromSide: "bottom", toSide: "top", paths: ["action", "time"] },

    // Spatial data main chain.
    { id: "e-spinput-n1", from: "spInput", to: "spNorm1", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spn1-ada", from: "spNorm1", to: "spAda1", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spada-bt", from: "spAda1", to: "spBT", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spbt-qkv", from: "spBT", to: "spQkv", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spqkv-score", from: "spQkv", to: "spScores", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spscore-attn", from: "spScores", to: "spAttention", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spattn-gate", from: "spAttention", to: "spGate1", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spgate-n2", from: "spGate1", to: "spNorm2", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spn2-ada2", from: "spNorm2", to: "spAda2", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spada2-mlp", from: "spAda2", to: "spMlp", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-spmlp-x2", from: "spMlp", to: "spX2", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },

    // Green routes terminate at explicit operator ports.
    { id: "e-sch-ada1", from: "spatialChunk", to: "spAda1", type: "action", fromSide: "right", fromRatio: .18, toSide: "left", toRatio: .42, via: [[1530, 2009], [1530, 984], [1800, 984]], label: "shift_msa · scale_msa", labelAt: [1655, 970], paths: ["action"] },
    { id: "e-sch-gate1", from: "spatialChunk", to: "spGate1", type: "action", fromSide: "right", fromRatio: .38, toSide: "left", toRatio: .42, via: [[1550, 2030], [1550, 1546], [1800, 1546]], label: "gate_msa", labelAt: [1710, 1532], paths: ["action"] },
    { id: "e-sch-ada2", from: "spatialChunk", to: "spAda2", type: "action", fromSide: "right", fromRatio: .62, toSide: "left", toRatio: .42, via: [[1570, 2054], [1570, 1762], [1800, 1762]], label: "shift_mlp · scale_mlp", labelAt: [1690, 1748], paths: ["action"] },
    { id: "e-sch-x2", from: "spatialChunk", to: "spX2", type: "action", fromSide: "right", fromRatio: .82, toSide: "left", toRatio: .44, via: [[1595, 2075], [1595, 2001], [1800, 2001]], label: "gate_mlp", labelAt: [1705, 1987], paths: ["action"] },

    // Explicit x2 handoff and Temporal condition pipeline.
    { id: "e-x2-tmin", from: "spX2", to: "tmInput", type: "video", fromSide: "right", toSide: "left", via: [[2180, 2008], [2180, 850], [2570, 850], [2570, 782]], label: "x₂ 原样传递 · shape不变", labelAt: [2380, 834], paths: ["video"] },
    { id: "e-tembed-tblockt", from: "tEmbed", to: "tBlockTemporal", type: "time", fromSide: "right", toSide: "left", via: [[1530, 712], [2205, 712], [2205, 777]], label: "Temporal 无直接action", labelAt: [1950, 694], paths: ["time"] },
    { id: "e-tbt-six", from: "tBlockTemporal", to: "temporalSixH", type: "time", fromSide: "bottom", toSide: "top", paths: ["time"] },
    { id: "e-tsix-pack", from: "temporalSixH", to: "temporalPack", type: "time", fromSide: "bottom", toSide: "top", paths: ["time"] },
    { id: "e-tpack-chunk", from: "temporalPack", to: "temporalChunk", type: "time", fromSide: "bottom", toSide: "top", paths: ["time"] },
    { id: "e-tmin-ada1", from: "tmInput", to: "tmAda1", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmada1-bs", from: "tmAda1", to: "tmBS", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmbs-qkv", from: "tmBS", to: "tmQkv", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmqkv-scores", from: "tmQkv", to: "tmScores", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmscores-attn", from: "tmScores", to: "tmAttention", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmattn-gate", from: "tmAttention", to: "tmGate1", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmgate-ada2", from: "tmGate1", to: "tmAda2", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmada2-mlp", from: "tmAda2", to: "tmMlp", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tmmlp-out", from: "tmMlp", to: "pairOutput", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-tch-ada1", from: "temporalChunk", to: "tmAda1", type: "time", fromSide: "right", fromRatio: .25, toSide: "left", toRatio: .35, via: [[2480, 1102], [2480, 894], [2560, 894]], paths: ["time"] },
    { id: "e-tch-gate1", from: "temporalChunk", to: "tmGate1", type: "time", fromSide: "right", fromRatio: .45, toSide: "left", toRatio: .38, via: [[2495, 1123], [2495, 1500], [2560, 1500]], paths: ["time"] },
    { id: "e-tch-ada2", from: "temporalChunk", to: "tmAda2", type: "time", fromSide: "right", fromRatio: .65, toSide: "left", toRatio: .38, via: [[2510, 1144], [2510, 1618], [2560, 1618]], paths: ["time"] },
    { id: "e-tch-out", from: "temporalChunk", to: "pairOutput", type: "time", fromSide: "right", fromRatio: .82, toSide: "left", toRatio: .4, via: [[2525, 1162], [2525, 1859], [2560, 1859]], paths: ["time"] },

    // t0 lane and explicit selection ports.
    { id: "e-t0-embed", from: "t0Constant", to: "t0Embed", type: "mask", fromSide: "bottom", toSide: "top", paths: ["mask"] },
    { id: "e-t0-block", from: "t0Embed", to: "t0Block", type: "mask", fromSide: "bottom", toSide: "top", paths: ["mask"] },
    { id: "e-t0-selectors", from: "t0Block", to: "maskMeaning", type: "mask", fromSide: "bottom", fromRatio: .35, toSide: "top", toRatio: .35, paths: ["mask"] },
    { id: "e-t0-caveat", from: "t0Block", to: "noneCaveat", type: "mask", fromSide: "bottom", toSide: "top", via: [[3190, 1050], [3190, 1250], [3190, 1250]], paths: ["mask"] },
    { id: "e-mask-spada1", from: "maskMeaning", to: "spAda1", type: "mask", fromSide: "left", fromRatio: .18, toSide: "right", toRatio: .65, via: [[2990, 1117], [2990, 630], [2170, 630], [2170, 1007], [2155, 1007]], paths: ["mask"] },
    { id: "e-mask-spgate1", from: "maskMeaning", to: "spGate1", type: "mask", fromSide: "left", fromRatio: .38, toSide: "right", toRatio: .62, via: [[2975, 1147], [2975, 1540], [2155, 1540]], paths: ["mask"] },
    { id: "e-mask-spada2", from: "maskMeaning", to: "spAda2", type: "mask", fromSide: "left", fromRatio: .62, toSide: "right", toRatio: .62, via: [[2960, 1183], [2960, 1782], [2155, 1782]], paths: ["mask"] },
    { id: "e-mask-spx2", from: "maskMeaning", to: "spX2", type: "mask", fromSide: "left", fromRatio: .82, toSide: "right", toRatio: .62, via: [[2945, 1213], [2945, 2080], [2180, 2080], [2180, 2022], [2155, 2022]], paths: ["mask"] },
    { id: "e-mask-tmada1", from: "maskMeaning", to: "tmAda1", type: "mask", fromSide: "left", fromRatio: .22, toSide: "right", toRatio: .7, via: [[2925, 1123], [2925, 931]], paths: ["mask"] },
    { id: "e-mask-tmgate1", from: "maskMeaning", to: "tmGate1", type: "mask", fromSide: "left", fromRatio: .43, toSide: "right", toRatio: .7, via: [[2915, 1155], [2915, 1532]], paths: ["mask"] },
    { id: "e-mask-tmada2", from: "maskMeaning", to: "tmAda2", type: "mask", fromSide: "left", fromRatio: .66, toSide: "right", toRatio: .7, via: [[2905, 1189], [2905, 1651]], paths: ["mask"] },
    { id: "e-mask-tmout", from: "maskMeaning", to: "pairOutput", type: "mask", fromSide: "left", fromRatio: .88, toSide: "right", toRatio: .7, via: [[2895, 1222], [2895, 1892]], paths: ["mask"] },

    // Remaining pairs and final layer.
    { id: "e-pair-remain", from: "pairOutput", to: "remainingPairs", type: "video", fromSide: "bottom", toSide: "top", via: [[2745, 2075], [2320, 2075], [2320, 2100]], label: "×28总数；其余27对折叠", labelAt: [2490, 2092], paths: ["video"] },
    { id: "e-sch-remaining", from: "spatialChunk", to: "remainingPairs", type: "action", fromSide: "right", fromRatio: .52, toSide: "left", toRatio: .35, via: [[1630, 2044], [1630, 2179], [2100, 2179]], label: "同一 t_s_mlp → 每个 Spatial", labelAt: [1835, 2163], paths: ["action"] },
    { id: "e-tch-remaining", from: "temporalChunk", to: "remainingPairs", type: "time", fromSide: "left", fromRatio: .52, toSide: "top", toRatio: .5, via: [[2210, 1130], [2210, 2100], [2320, 2100]], label: "t_t_mlp → 每个 Temporal", labelAt: [2200, 2058], paths: ["time"] },
    { id: "e-mask-remaining", from: "maskMeaning", to: "remainingPairs", type: "mask", fromSide: "bottom", fromRatio: .62, toSide: "right", toRatio: .68, via: [[3228, 2110], [2540, 2110], [2540, 2216]], label: "t₀ + x_mask → every Block", labelAt: [2860, 2093], paths: ["mask"] },
    { id: "e-remain-final", from: "remainingPairs", to: "finalAda", type: "video", fromSide: "right", toSide: "left", paths: ["video"] },
    { id: "e-final-linear", from: "finalAda", to: "finalLinear", type: "video", fromSide: "right", toSide: "left", paths: ["video"] },
    { id: "e-linear-unpatch", from: "finalLinear", to: "unpatchify", type: "video", fromSide: "right", toSide: "left", paths: ["video"] },
    { id: "e-t-final", from: "tEmbed", to: "finalAda", type: "time", fromSide: "right", fromRatio: .28, toSide: "top", toRatio: .35, via: [[1520, 689], [1520, 430], [3460, 430], [3460, 2065], [2633, 2065], [2633, 2100]], paths: ["time"] },
    { id: "e-t0-final", from: "t0Embed", to: "finalAda", type: "mask", fromSide: "right", fromRatio: .35, toSide: "top", toRatio: .52, via: [[3380, 834], [3380, 2075], [2678, 2075], [2678, 2100]], paths: ["mask"] },
    { id: "e-mask-final", from: "maskMeaning", to: "finalAda", type: "mask", fromSide: "bottom", toSide: "top", toRatio: .7, via: [[3190, 1270], [3420, 1270], [3420, 2100], [2726, 2100]], paths: ["mask"] },

    // RFlow update loop and final decode.
    { id: "e-unpatch-half", from: "unpatchify", to: "firstHalf", type: "video", fromSide: "bottom", toSide: "top", paths: ["video"] },
    { id: "e-half-update", from: "firstHalf", to: "rflowUpdate", type: "video", fromSide: "left", toSide: "right", paths: ["video"] },
    { id: "e-z-update", from: "zCurrent", to: "rflowUpdate", type: "video", fromSide: "left", fromRatio: .68, toSide: "left", toRatio: .3, via: [[795, 236], [795, 2305], [2600, 2305], [2600, 2374]], label: "residual z_current", labelAt: [1830, 2290], paths: ["video"] },
    { id: "e-update-restore", from: "rflowUpdate", to: "restoreCondition", type: "video", fromSide: "left", toSide: "right", paths: ["video"] },
    { id: "e-z-restore", from: "zCurrent", to: "restoreCondition", type: "video", fromSide: "left", fromRatio: .32, toSide: "top", toRatio: .3, via: [[775, 196], [775, 2270], [2350, 2270], [2350, 2300]], label: "original condition z", labelAt: [1600, 2255], paths: ["video"] },
    { id: "e-xmask-restore", from: "xMask", to: "restoreCondition", type: "mask", fromSide: "bottom", toSide: "top", via: [[1808, 330], [1808, 2290], [2418, 2290]], paths: ["mask"] },
    { id: "e-loop-back", from: "restoreCondition", to: "zCurrent", type: "loop", fromSide: "left", toSide: "top", via: [[760, 2396], [760, 115], [970, 115]], label: "step k+1 · 完整12槽回送", labelAt: [950, 2380], paths: ["video"] },
    { id: "e-restore-slice", from: "restoreCondition", to: "futureSlice", type: "video", fromSide: "bottom", toSide: "top", via: [[2418, 2515], [1180, 2515], [1180, 2745]], label: "仅在第30步后", labelAt: [1780, 2500], paths: ["video"] },
    { id: "e-slice-decode", from: "futureSlice", to: "vaeDecode", type: "video", fromSide: "right", toSide: "left", paths: ["video"] },
    { id: "e-decode-rgb", from: "vaeDecode", to: "futureRgb", type: "video", fromSide: "right", toSide: "left", paths: ["video"] },
  ];

  // A fan-out is a visual routing primitive, not a model operator. Semantic
  // dependencies stay in `edges`; these descriptors only replace repeated
  // source-to-target paths with one trunk, explicit junctions, and short taps.
  const fanouts = [
    {
      id: "fanout-spatial-mod", from: "spatialChunk", type: "action", fromSide: "right", fromRatio: .5, paths: ["action", "time"],
      trunk: [[1640, 2042]], segments: [{ points: [[1640, 984], [1640, 2179]] }],
      junctions: [[1640, 2042, 13], [1640, 984, 6], [1640, 1546, 6], [1640, 1762, 6], [1640, 2001, 6], [1640, 2179, 6]],
      label: "Spatial modulation · 6H", labelAt: [1650, 930],
      branches: [
        { edgeId: "e-sch-ada1", points: [[1640, 984]], label: "shift / scale msa", labelAt: [1740, 968] },
        { edgeId: "e-sch-gate1", points: [[1640, 1546]], label: "gate msa", labelAt: [1735, 1530] },
        { edgeId: "e-sch-ada2", points: [[1640, 1762]], label: "shift / scale mlp", labelAt: [1740, 1746] },
        { edgeId: "e-sch-x2", points: [[1640, 2001]], label: "gate mlp", labelAt: [1735, 1985] },
        { edgeId: "e-sch-remaining", points: [[1640, 2179]], label: "same params → every Spatial", labelAt: [1860, 2163] },
      ],
    },
    {
      id: "fanout-temporal-mod", from: "temporalChunk", type: "time", fromSide: "right", fromRatio: .5, paths: ["time"],
      trunk: [[2520, 1128]], segments: [{ points: [[2520, 894], [2520, 2100]] }],
      junctions: [[2520, 1128, 13], [2520, 894, 6], [2520, 1500, 6], [2520, 1618, 6], [2520, 1859, 6], [2520, 2100, 6]],
      label: "Temporal modulation · 6H", labelAt: [2710, 835],
      branches: [
        { edgeId: "e-tch-ada1", points: [[2520, 894]] },
        { edgeId: "e-tch-gate1", points: [[2520, 1500]] },
        { edgeId: "e-tch-ada2", points: [[2520, 1618]] },
        { edgeId: "e-tch-out", points: [[2520, 1859]] },
        { edgeId: "e-tch-remaining", points: [[2520, 2100], [2320, 2100]], label: "same params → every Temporal", labelAt: [2410, 2084], toSide: "top", toRatio: .5 },
      ],
    },
    {
      id: "fanout-mask-selectors", from: "maskMeaning", type: "mask", fromSide: "left", fromRatio: .5, paths: ["mask"],
      trunk: [[2985, 1165]],
      segments: [
        { points: [[2985, 1165], [2985, 630], [2170, 630], [2170, 2022]] },
        { points: [[2985, 1165], [2920, 1165], [2920, 1892]] },
        { points: [[2985, 1165], [2985, 1280], [3420, 1280], [3420, 2100], [2320, 2100]] },
      ],
      junctions: [
        [2985, 1165, 13], [2170, 630, 9], [2920, 1165, 9], [3420, 1165, 9],
        [2170, 1007, 6], [2170, 1540, 6], [2170, 1782, 6], [2170, 2022, 6],
        [2920, 931, 6], [2920, 1532, 6], [2920, 1651, 6], [2920, 1892, 6],
        [2399, 2100, 6], [2726, 2100, 6],
      ],
      labels: [
        { text: "Spatial selectors ×4", at: [2520, 614] },
        { text: "Temporal selectors ×4", at: [2850, 1060] },
        { text: "remaining Pairs + Final", at: [3030, 2084] },
      ],
      branches: [
        { edgeId: "e-mask-spada1", points: [[2170, 1007]] },
        { edgeId: "e-mask-spgate1", points: [[2170, 1540]] },
        { edgeId: "e-mask-spada2", points: [[2170, 1782]] },
        { edgeId: "e-mask-spx2", points: [[2170, 2022]] },
        { edgeId: "e-mask-tmada1", points: [[2920, 931]] },
        { edgeId: "e-mask-tmgate1", points: [[2920, 1532]] },
        { edgeId: "e-mask-tmada2", points: [[2920, 1651]] },
        { edgeId: "e-mask-tmout", points: [[2920, 1892]] },
        { edgeId: "e-mask-remaining", points: [[2399, 2100]], toSide: "top", toRatio: .72 },
        { edgeId: "e-mask-final", points: [[2726, 2100]], toSide: "top", toRatio: .7 },
      ],
    },
    {
      id: "fanout-t-embed", from: "tEmbed", type: "time", fromSide: "right", fromRatio: .5, paths: ["time"],
      trunk: [[1530, 712], [1530, 640]], junctions: [[1530, 640, 13]], label: "shared E_t(t)", labelAt: [1650, 622],
      branches: [
        { edgeId: "e-tembed-repeat", points: [[1530, 640], [1530, 780], [1368, 780]], toSide: "top", toRatio: .5 },
        { edgeId: "e-tembed-tblockt", points: [[1530, 640], [2205, 640], [2205, 777]], label: "Temporal has no direct action", labelAt: [1950, 622] },
        { edgeId: "e-t-final", points: [[1530, 640], [1700, 640], [1700, 430], [3460, 430], [3460, 2065], [2633, 2065]], toSide: "top", toRatio: .35 },
      ],
    },
    {
      id: "fanout-t-transform", from: "timestepTransform", type: "time", fromSide: "right", fromRatio: .5, paths: ["time", "mask"],
      trunk: [[1560, 346]], junctions: [[1560, 346, 13]], label: "same transformed t", labelAt: [1580, 326],
      branches: [
        { edgeId: "e-transform-embed", points: [[1560, 346], [1385, 346], [1385, 620], [1368, 620]], toSide: "top", toRatio: .5 },
        { edgeId: "e-transform-xmask", points: [[1560, 346], [1560, 216]] },
      ],
    },
    {
      id: "fanout-x-mask", from: "xMask", type: "mask", fromSide: "right", fromRatio: .5, paths: ["mask"],
      trunk: [[2010, 216]], junctions: [[2010, 216, 13]], label: "same x_mask", labelAt: [2140, 198],
      branches: [
        { edgeId: "e-mask-t0", points: [[2010, 216], [2010, 300], [2995, 300], [2995, 635], [3190, 635]], toSide: "top", toRatio: .5 },
        { edgeId: "e-xmask-meaning", points: [[2010, 216], [2010, 300], [2990, 300], [2990, 1060], [3190, 1060]], toSide: "top", toRatio: .5 },
        { edgeId: "e-xmask-restore", points: [[2010, 216], [2010, 300], [3500, 300], [3500, 2290], [2418, 2290]], toSide: "top", toRatio: .5 },
      ],
    },
    {
      id: "fanout-z-current", from: "zCurrent", type: "video", fromSide: "left", fromRatio: .5, paths: ["video"],
      trunk: [[760, 216]], segments: [{ points: [[760, 216], [760, 2305]] }],
      junctions: [[760, 216, 13], [760, 450, 6], [760, 2270, 6], [760, 2305, 6]],
      label: "z_current fan-out", labelAt: [845, 434],
      branches: [
        { edgeId: "e-z-patch", points: [[760, 450], [950, 450]], toSide: "top", toRatio: .5 },
        { edgeId: "e-z-restore", points: [[760, 2270], [2350, 2270], [2350, 2300]], label: "pre-update condition z", labelAt: [1600, 2255] },
        { edgeId: "e-z-update", points: [[760, 2305], [2600, 2305], [2600, 2374]], label: "residual z_current", labelAt: [1830, 2290] },
      ],
    },
    {
      id: "fanout-t0-embed", from: "t0Embed", type: "mask", fromSide: "right", fromRatio: .5, paths: ["mask"],
      trunk: [[3380, 848]], junctions: [[3380, 848, 13]], label: "shared E_t(15)", labelAt: [3390, 828],
      branches: [
        { edgeId: "e-t0-block", points: [[3380, 848], [3380, 900], [3190, 900]], toSide: "top", toRatio: .5 },
        { edgeId: "e-t0-final", points: [[3380, 848], [3380, 2075], [2678, 2075]], toSide: "top", toRatio: .52 },
      ],
    },
    {
      id: "fanout-t0-block", from: "t0Block", type: "mask", fromSide: "bottom", fromRatio: .5, paths: ["mask"],
      trunk: [[3190, 1050]], junctions: [[3190, 1050, 13]],
      branches: [
        { edgeId: "e-t0-selectors", points: [[3190, 1050]], toSide: "top", toRatio: .35 },
        { edgeId: "e-t0-caveat", points: [[3190, 1050], [3360, 1050], [3360, 1250], [3190, 1250]], toSide: "top", toRatio: .5 },
      ],
    },
  ];

  const defs = make("defs");
  [
    ["video", "#64aef5"], ["action", "#62e0ad"], ["time", "#f2a766"],
    ["mask", "#bd94f4"], ["disabled", "#758792"], ["loop", "#f2a766"],
  ].forEach(([id, color]) => {
    const marker = make("marker", { id: `arrow-${id}`, viewBox: "0 0 12 12", refX: 10, refY: 6, markerWidth: 18, markerHeight: 18, markerUnits: "userSpaceOnUse", orient: "auto-start-reverse" });
    marker.appendChild(make("path", { d: "M 1 1 L 11 6 L 1 11 z", fill: color }));
    defs.appendChild(marker);
  });
  const detailFocusClip = make("clipPath", { id: "detail-focus-clip", clipPathUnits: "userSpaceOnUse" });
  const detailFocusClipRect = make("rect", { x: 0, y: 0, width: canvas.width, height: canvas.height });
  detailFocusClip.appendChild(detailFocusClipRect);
  defs.appendChild(detailFocusClip);
  svg.appendChild(defs);

  const overviewLayer = make("g", { class: "overview-layer" });
  const overviewEdgeLayer = make("g", { class: "overview-edge-layer" });
  const overviewNodeLayer = make("g", { class: "overview-node-layer" });
  const overviewLabelLayer = make("g", { class: "overview-label-layer" });
  const overviewPortLayer = make("g", { class: "overview-port-layer" });
  overviewLayer.append(overviewEdgeLayer, overviewNodeLayer, overviewLabelLayer, overviewPortLayer);

  const panelLayer = make("g", { class: "panel-layer" });
  const edgeLayer = make("g", { class: "edge-layer" });
  const panelHeaderLayer = make("g", { class: "panel-header-layer" });
  const nodeLayer = make("g", { class: "node-layer" });
  const labelLayer = make("g", { class: "label-layer" });
  const portLayer = make("g", { class: "port-layer" });
  const annotationLayer = make("g", { class: "annotation-layer" });
  svg.append(overviewLayer, panelLayer, edgeLayer, panelHeaderLayer, nodeLayer, labelLayer, portLayer, annotationLayer);
  const detailLayers = [panelLayer, edgeLayer, panelHeaderLayer, nodeLayer, labelLayer, portLayer, annotationLayer];

  overviewLayer.insertBefore(make("rect", { x: 40, y: 45, width: 2620, height: 1545, rx: 30, class: "overview-backdrop" }), overviewEdgeLayer);
  overviewLayer.insertBefore(make("text", { x: 90, y: 115, class: "overview-heading" }, "一次看清：输入 → 30步采样 → STDiT3 → 8帧未来视频"), overviewEdgeLayer);
  overviewLayer.insertBefore(make("text", { x: 90, y: 157, class: "overview-subtitle" }, "点击模块查看公式；点击聚焦按钮进入对应的算子级计算图。"), overviewEdgeLayer);
  overviewLayer.insertBefore(make("text", { x: 90, y: 220, class: "overview-lane-label" }, "VIDEO / HIDDEN 主数据流"), overviewEdgeLayer);
  overviewLayer.insertBefore(make("text", { x: 90, y: 845, class: "overview-lane-label" }, "CONDITION 条件支路"), overviewEdgeLayer);

  const overviewNodes = [
    { id: "overview-input", target: "historyQueue", x: 90, y: 260, w: 460, h: 160, kind: "video", step: "01 · ROLLOUT", title: "History + future noise", concrete: "[1,4,4,32²] + [1,4,8,32²]", symbolic: "[B,Cz,Tₕ,32²] + [B,Cz,Tf,32²]", note: "首轮单帧latent repeat×4", paths: ["video"] },
    { id: "overview-concat", target: "concatOnce", x: 650, y: 260, w: 400, h: 160, kind: "video", step: "02 · ONCE", title: "concat 一次", concrete: "[1,4,12,32,32]", symbolic: "[B,Cz,T,h,w]", note: "循环内不重新concat", paths: ["video"] },
    { id: "overview-rflow", target: "zCurrent", x: 1150, y: 260, w: 420, h: 160, kind: "video", step: "03 · RFLOW ×30", title: "z_current", concrete: "[1,4,12,32,32]", symbolic: "[B,Cz,T,h,w]", note: "4槽固定 · 8槽迭代更新", paths: ["video", "time", "mask"] },
    { id: "overview-video", target: "tokenGrid", x: 1670, y: 260, w: 490, h: 160, kind: "video", step: "04 · STDiT INPUT", title: "Video patch tokens", concrete: "[1,12,256,1152]", symbolic: "[B,T,S,H]", note: "PatchEmbed3D + 2D position", paths: ["video"] },
    { id: "overview-pair", target: "spInput", x: 90, y: 570, w: 620, h: 180, kind: "video", step: "05 · EXPANDED PAIR #1", title: "Spatial → Temporal", concrete: "Action-AdaLN → x₂ → causal time", symbolic: "Spatial(cₛ) → Temporal(cₜ)", note: "Action只直连Spatial；Temporal接收x₂", paths: ["video", "action", "time", "mask"] },
    { id: "overview-backbone", target: "remainingPairs", x: 820, y: 580, w: 440, h: 160, kind: "video", step: "06 · BACKBONE", title: "同构 Pair ×28", concrete: "[1,12,256,1152]", symbolic: "[B,T,S,H]", note: "只精细展开其中1个", paths: ["video", "action", "time", "mask"] },
    { id: "overview-final", target: "rflowUpdate", x: 1370, y: 570, w: 600, h: 180, kind: "video", step: "07 · PREDICT + UPDATE", title: "Final → flow → update", concrete: "8ch → first 4ch → z + v·dt", symbolic: "2Cz → Cz → zₖ₊₁", note: "随后restore前4个condition slots", paths: ["video", "time", "mask"] },
    { id: "overview-output", target: "futureRgb", x: 2080, y: 580, w: 490, h: 160, kind: "output", step: "AFTER STEP 30", title: "VAE Decoder → 8 RGB", concrete: "[1,8,256,256,3]", symbolic: "[B,Tf,Himg,Wimg,3]", note: "只解码最后8个future slots", paths: ["video"] },
    { id: "overview-action", target: "actionChunk", x: 90, y: 890, w: 400, h: 150, kind: "action", step: "ACTION INPUT", title: "8 × 7D actions", concrete: "[1,8,7]", symbolic: "[B,Ta,Da]", note: "Policy action chunk", paths: ["action"] },
    { id: "overview-action-condition", target: "spatialChunk", x: 590, y: 880, w: 580, h: 170, kind: "action", step: "ACTION CONDITION", title: "ActionEncoder → Spatial 6H", concrete: "[1,12,6,1152] → chunk(6)", symbolic: "[B,T,6,H] → 6×[B,T,1,H]", note: "参数进入Spatial两处AdaLN和gate", paths: ["action", "time"] },
    { id: "overview-time", target: "timestepTransform", x: 90, y: 1160, w: 400, h: 150, kind: "time", step: "DIFFUSION TIME", title: "timestep transform", concrete: "[1] → Eₜ [1,1152]", symbolic: "[B] → Eₜ [B,H]", note: "Spatial + Temporal + Final共享", paths: ["time"] },
    { id: "overview-mask", target: "maskMeaning", x: 590, y: 1150, w: 580, h: 170, kind: "mask", step: "CONDITION SELECTOR", title: "x_mask + t₀=15", concrete: "[0×4 | 1×8] → torch.where", symbolic: "[history False | future True]", note: "选择modulation；不是Attention Mask", paths: ["mask"] },
    { id: "overview-cfg", target: "cfgGhost", x: 2070, y: 1030, w: 500, h: 170, kind: "disabled", step: "NOT EXECUTED", title: "CFG scale = 7", concrete: "当前主链：v_pred = pred", symbolic: "no active CFG tensor", note: "cond/uncond组合代码被注释", paths: [] },
  ];

  const overviewNodeById = new Map(overviewNodes.map((node) => [node.id, node]));
  const overviewShapeTextElements = new Map();
  const overviewNodeElements = new Map();
  const renderOverviewNode = (node) => {
    const group = make("g", { id: node.id, class: `overview-node kind-${node.kind}`, tabindex: "0", role: "button", "aria-label": `${node.title}，${node.concrete}`, "data-target": node.target, "data-paths": node.paths.join(" ") });
    group.appendChild(make("rect", { x: node.x, y: node.y, width: node.w, height: node.h, rx: 18 }));
    group.appendChild(make("text", { x: node.x + 18, y: node.y + 30, class: "overview-step" }, node.step));
    group.appendChild(make("text", { x: node.x + 18, y: node.y + 70, class: "overview-title" }, node.title));
    const shapeText = make("text", { x: node.x + 18, y: node.y + 105, class: "overview-shape" }, node.concrete);
    group.appendChild(shapeText);
    group.appendChild(make("text", { x: node.x + 18, y: node.y + node.h - 10, class: "overview-note" }, node.note));
    group.addEventListener("click", () => selectNode(node.target));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(node.target); }
    });
    overviewNodeLayer.appendChild(group);
    overviewShapeTextElements.set(node.id, shapeText);
    overviewNodeElements.set(node.id, group);
  };
  overviewNodes.forEach(renderOverviewNode);

  const overviewEdges = [
    { id: "overview-edge-input-concat", from: "overview-input", to: "overview-concat", type: "video", paths: ["video"] },
    { id: "overview-edge-concat-rflow", from: "overview-concat", to: "overview-rflow", type: "video", paths: ["video"] },
    { id: "overview-edge-rflow-video", from: "overview-rflow", to: "overview-video", type: "video", paths: ["video"] },
    { id: "overview-edge-video-pair", from: "overview-video", to: "overview-pair", type: "video", fromSide: "bottom", toSide: "top", fromRatio: .72, toRatio: .5, via: [[2023, 505], [400, 505]], label: "STDiT3 forward", labelAt: [1210, 488], paths: ["video"] },
    { id: "overview-edge-pair-backbone", from: "overview-pair", to: "overview-backbone", type: "video", paths: ["video"] },
    { id: "overview-edge-backbone-final", from: "overview-backbone", to: "overview-final", type: "video", paths: ["video"] },
    { id: "overview-edge-final-output", from: "overview-final", to: "overview-output", type: "video", paths: ["video"] },
    { id: "overview-edge-action-build", from: "overview-action", to: "overview-action-condition", type: "action", paths: ["action"] },
    { id: "overview-edge-action-pair", from: "overview-action-condition", to: "overview-pair", type: "action", fromSide: "top", toSide: "bottom", fromRatio: .35, toRatio: .28, via: [[793, 815], [264, 815]], paths: ["action"] },
    { id: "overview-edge-time-pair", from: "overview-time", to: "overview-pair", type: "time", fromSide: "right", toSide: "bottom", toRatio: .52, via: [[540, 1235], [540, 800], [412, 800]], paths: ["time"] },
    { id: "overview-edge-loop", from: "overview-final", to: "overview-rflow", type: "loop", fromSide: "bottom", toSide: "bottom", fromRatio: .72, toRatio: .5, via: [[1802, 1480], [1360, 1480]], label: "restore history → 完整12槽进入 step k+1 · 共30步", labelAt: [1580, 1455], paths: ["video", "time", "mask"] },
  ];

  const overviewFanouts = [
    {
      id: "overview-fanout-mask", from: "overview-mask", type: "mask", fromSide: "top", fromRatio: .5, paths: ["mask"],
      trunk: [[880, 810]], segments: [{ points: [[574, 810], [1502, 810]] }],
      junctions: [[880, 810, 13], [574, 810, 7], [1502, 810, 7]],
      label: "same x_mask / t₀ selector", labelAt: [1110, 790],
      branches: [
        { id: "overview-edge-mask-pair", to: "overview-pair", points: [[574, 810]], toSide: "bottom", toRatio: .78 },
        { id: "overview-edge-mask-final", to: "overview-final", points: [[1502, 810]], toSide: "bottom", toRatio: .22 },
      ],
    },
  ];

  const overviewPoint = (node, side = "right", ratio = .5) => {
    if (side === "left") return [node.x, node.y + node.h * ratio];
    if (side === "right") return [node.x + node.w, node.y + node.h * ratio];
    if (side === "top") return [node.x + node.w * ratio, node.y];
    return [node.x + node.w * ratio, node.y + node.h];
  };
  const overviewEdgeElements = new Map();
  overviewEdges.forEach((edge) => {
    const fromNode = overviewNodeById.get(edge.from);
    const toNode = overviewNodeById.get(edge.to);
    const start = overviewPoint(fromNode, edge.fromSide || "right", edge.fromRatio ?? .5);
    const end = overviewPoint(toNode, edge.toSide || "left", edge.toRatio ?? .5);
    const points = [start, ...(edge.via || []), end];
    const d = points.map((point, index) => `${index ? "L" : "M"} ${point[0]} ${point[1]}`).join(" ");
    const path = make("path", { id: edge.id, d, class: `overview-edge ${edge.type}`, "data-paths": edge.paths.join(" "), "marker-end": `url(#arrow-${edge.type})`, "vector-effect": "non-scaling-stroke" });
    overviewEdgeLayer.appendChild(path);
    overviewEdgeElements.set(edge.id, path);
    [start, end].forEach((point) => overviewPortLayer.appendChild(make("circle", { cx: point[0], cy: point[1], r: 10, class: `overview-port ${edge.type === "loop" ? "time" : edge.type}`, "data-paths": edge.paths.join(" ") })));
    if (edge.label) overviewLabelLayer.appendChild(make("text", { x: edge.labelAt[0], y: edge.labelAt[1], class: "overview-edge-label", "data-paths": edge.paths.join(" ") }, edge.label));
  });

  overviewFanouts.forEach((fanout) => {
    const source = overviewNodeById.get(fanout.from);
    const sourcePoint = overviewPoint(source, fanout.fromSide || "right", fanout.fromRatio ?? .5);
    const trunk = make("path", {
      id: `${fanout.id}-trunk`, d: pathFromPoints([sourcePoint, ...fanout.trunk]), class: `overview-edge ${fanout.type} fanout-trunk`,
      "data-paths": fanout.paths.join(" "), "vector-effect": "non-scaling-stroke",
    });
    overviewEdgeLayer.appendChild(trunk);
    overviewEdgeElements.set(`${fanout.id}-trunk`, trunk);
    fanout.segments.forEach((segment, index) => {
      const rail = make("path", {
        id: `${fanout.id}-rail-${index}`, d: pathFromPoints(segment.points), class: `overview-edge ${fanout.type} fanout-rail`,
        "data-paths": fanout.paths.join(" "), "vector-effect": "non-scaling-stroke",
      });
      overviewEdgeLayer.appendChild(rail);
      overviewEdgeElements.set(`${fanout.id}-rail-${index}`, rail);
    });
    overviewPortLayer.appendChild(make("circle", { cx: sourcePoint[0], cy: sourcePoint[1], r: 10, class: `overview-port ${fanout.type}`, "data-fanout-id": fanout.id, "data-paths": fanout.paths.join(" ") }));
    fanout.junctions.forEach(([x, y, radius]) => overviewPortLayer.appendChild(make("circle", {
      cx: x, cy: y, r: radius, class: `overview-port ${fanout.type} ${radius >= 9 ? "fanout-junction" : "fanout-tap"}`,
      "data-fanout-id": fanout.id, "data-paths": fanout.paths.join(" "),
    })));
    fanout.branches.forEach((branch) => {
      const target = overviewNodeById.get(branch.to);
      const end = overviewPoint(target, branch.toSide || "left", branch.toRatio ?? .5);
      const path = make("path", {
        id: branch.id, d: pathFromPoints([...branch.points, end]), class: `overview-edge ${fanout.type} fanout-branch`,
        "data-fanout-id": fanout.id, "data-paths": fanout.paths.join(" "), "marker-end": `url(#arrow-${fanout.type})`, "vector-effect": "non-scaling-stroke",
      });
      overviewEdgeLayer.appendChild(path);
      overviewEdgeElements.set(branch.id, path);
      overviewPortLayer.appendChild(make("circle", { cx: end[0], cy: end[1], r: 10, class: `overview-port ${fanout.type}`, "data-fanout-id": fanout.id, "data-paths": fanout.paths.join(" ") }));
    });
    overviewLabelLayer.appendChild(make("text", { x: fanout.labelAt[0], y: fanout.labelAt[1], class: "overview-edge-label", "data-paths": fanout.paths.join(" ") }, fanout.label));
  });

  panels.forEach((panel) => {
    const boxGroup = make("g", { "data-panel-box": panel.id });
    boxGroup.appendChild(make("rect", { x: panel.x, y: panel.y, width: panel.w, height: panel.h, rx: 24, class: `panel-box ${panel.cls}`.trim() }));
    panelLayer.appendChild(boxGroup);

    const headerBandX = panel.headerBandX ?? panel.x;
    const headerWidth = panel.headerWidth ?? panel.w;
    const headerHeight = panel.headerHeight ?? 70;
    const headerX = headerBandX + 20;
    const showNote = panel.showNote === true;
    const headerGroup = make("g", { "data-panel": panel.id });
    headerGroup.appendChild(make("rect", { x: headerBandX, y: panel.y, width: headerWidth, height: headerHeight, rx: 16, class: `panel-header-band ${panel.cls}`.trim() }));
    headerGroup.appendChild(make("line", { x1: headerBandX + 12, y1: panel.y + headerHeight, x2: headerBandX + headerWidth - 12, y2: panel.y + headerHeight, class: "panel-header-rule" }));
    headerGroup.appendChild(make("text", { x: headerX, y: panel.y + 22, class: "panel-label" }, panel.label));
    headerGroup.appendChild(make("text", { x: headerX, y: panel.y + 50, class: "panel-title" }, panel.title));
    if (showNote) headerGroup.appendChild(make("text", { x: headerX, y: panel.y + 76, class: "panel-note" }, panel.note));
    panelHeaderLayer.appendChild(headerGroup);
  });

  const pointFor = (node, side = "right", ratio = 0.5) => {
    if (side === "left") return [node.x, node.y + node.h * ratio];
    if (side === "right") return [node.x + node.w, node.y + node.h * ratio];
    if (side === "top") return [node.x + node.w * ratio, node.y];
    return [node.x + node.w * ratio, node.y + node.h];
  };

  const edgeById = new Map(edges.map((edge) => [edge.id, edge]));
  const fanoutEdgeIds = new Set(fanouts.flatMap((fanout) => fanout.branches.map((branch) => branch.edgeId)));

  const linePath = (edge) => {
    const fromNode = nodeById.get(edge.from);
    const toNode = nodeById.get(edge.to);
    const start = pointFor(fromNode, edge.fromSide || "right", edge.fromRatio ?? 0.5);
    const end = pointFor(toNode, edge.toSide || "left", edge.toRatio ?? 0.5);
    const points = [start, ...(edge.via || []), end];
    if (!edge.via || edge.via.length === 0) {
      if ((edge.fromSide === "bottom" || edge.fromSide === "top") && (edge.toSide === "top" || edge.toSide === "bottom")) {
        const midY = (start[1] + end[1]) / 2;
        return { d: `M ${start[0]} ${start[1]} C ${start[0]} ${midY}, ${end[0]} ${midY}, ${end[0]} ${end[1]}`, start, end };
      }
      const midX = (start[0] + end[0]) / 2;
      return { d: `M ${start[0]} ${start[1]} C ${midX} ${start[1]}, ${midX} ${end[1]}, ${end[0]} ${end[1]}`, start, end };
    }
    return { d: points.map((point, index) => `${index ? "L" : "M"} ${point[0]} ${point[1]}`).join(" "), start, end };
  };

  const edgeElements = new Map();
  const routedSegments = [];
  const rememberRoute = (id, points, ignore = []) => {
    if (points.length > 1) routedSegments.push({ id, points, ignore: new Set(ignore) });
  };
  const inflateRect = (rect, padding) => ({
    x: rect.x - padding, y: rect.y - padding,
    w: rect.w + padding * 2, h: rect.h + padding * 2,
  });
  const rectsOverlap = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  const overlapArea = (a, b) => {
    const width = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const height = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    return width * height;
  };
  const panelHeaderRects = panels.map((panel) => ({
    id: `header:${panel.id}`,
    x: panel.headerBandX ?? panel.x,
    y: panel.y,
    w: panel.headerWidth ?? panel.w,
    h: panel.headerHeight ?? 70,
  }));
  const fixedLabelObstacles = [
    ...nodes.map((node) => ({ id: `node:${node.id}`, ...inflateRect(node, 10) })),
    ...panelHeaderRects.map((rect) => ({ ...rect, ...inflateRect(rect, 8) })),
  ];
  const placedLabelRects = [];
  const labelLayoutIssues = [];
  const labelOffsets = [[0, 0]];
  [36, 72, 108, 144, 180, 216, 252].forEach((distance) => {
    labelOffsets.push(
      [0, -distance], [0, distance], [-distance, 0], [distance, 0],
      [-distance, -distance], [distance, -distance], [-distance, distance], [distance, distance],
    );
  });
  const placeEdgeLabel = (text, at, width) => {
    const makeBox = ([dx, dy]) => ({ x: at[0] + dx - width / 2, y: at[1] + dy - 19, w: width, h: 28, dx, dy });
    const obstacles = [...fixedLabelObstacles, ...placedLabelRects];
    const withinCanvas = (box) => box.x >= 12 && box.y >= 12 && box.x + box.w <= canvas.width - 12 && box.y + box.h <= canvas.height - 12;
    let best = null;
    labelOffsets.forEach((offset) => {
      const box = makeBox(offset);
      if (!withinCanvas(box)) return;
      const score = obstacles.reduce((sum, obstacle) => sum + overlapArea(box, obstacle), 0);
      const distance = Math.abs(offset[0]) + Math.abs(offset[1]);
      if (!best || score < best.score || (score === best.score && distance < best.distance)) best = { ...box, score, distance };
    });
    best ||= makeBox([0, 0]);
    if (best.score > 0) labelLayoutIssues.push(`label:${text}`);
    placedLabelRects.push({ id: `label:${text}`, x: best.x, y: best.y, w: best.w, h: best.h });
    return { x: best.x + width / 2, y: best.y + 19, dx: best.dx, dy: best.dy, box: best };
  };
  const renderEdgeLabel = (text, at, paths, extraClass = "") => {
    if (!text || !at) return;
    const width = Math.max(120, text.length * 13 + 28);
    const placement = placeEdgeLabel(text, at, width);
    const shifted = placement.dx !== 0 || placement.dy !== 0;
    const group = make("g", {
      class: `edge-label-group ${extraClass} ${shifted ? "is-shifted" : ""}`.trim(),
      "data-paths": (paths || []).join(" "), "data-origin-x": at[0], "data-origin-y": at[1],
    });
    group.appendChild(make("rect", { x: placement.x - width / 2, y: placement.y - 19, width, height: 28, rx: 8, class: "edge-label-bg" }));
    group.appendChild(make("text", { x: placement.x, y: placement.y, class: "edge-label" }, text));
    labelLayer.appendChild(group);
  };

  edges.forEach((edge) => {
    if (fanoutEdgeIds.has(edge.id)) return;
    const geometry = linePath(edge);
    const path = make("path", {
      id: edge.id,
      class: `edge ${edge.type}`,
      d: geometry.d,
      "data-paths": (edge.paths || []).join(" "),
      "marker-end": `url(#arrow-${edge.type})`,
      "vector-effect": "non-scaling-stroke",
    });
    edgeLayer.appendChild(path);
    edgeElements.set(edge.id, path);
    if (edge.via?.length) rememberRoute(edge.id, [geometry.start, ...edge.via, geometry.end], [edge.from, edge.to]);
    [geometry.start, geometry.end].forEach((point) => {
      portLayer.appendChild(make("circle", { cx: point[0], cy: point[1], r: 9, class: `port ${edge.type === "loop" ? "time" : edge.type}`, "data-paths": (edge.paths || []).join(" "), "vector-effect": "non-scaling-stroke" }));
    });
    renderEdgeLabel(edge.label, edge.labelAt, edge.paths);
  });

  fanouts.forEach((fanout) => {
    const source = nodeById.get(fanout.from);
    if (!source) return;
    const groupPaths = fanout.paths || [];
    const sourcePoint = pointFor(source, fanout.fromSide || "right", fanout.fromRatio ?? .5);
    const trunkPoints = [sourcePoint, ...(fanout.trunk || [])];
    if (trunkPoints.length > 1) {
      const trunk = make("path", {
        id: `${fanout.id}-trunk`, class: `edge ${fanout.type} fanout-trunk`, d: pathFromPoints(trunkPoints),
        "data-paths": groupPaths.join(" "), "vector-effect": "non-scaling-stroke",
      });
      edgeLayer.appendChild(trunk);
      edgeElements.set(`${fanout.id}-trunk`, trunk);
      rememberRoute(`${fanout.id}-trunk`, trunkPoints, [fanout.from]);
    }
    (fanout.segments || []).forEach((segment, index) => {
      const rail = make("path", {
        id: `${fanout.id}-rail-${index}`, class: `edge ${fanout.type} fanout-rail`, d: pathFromPoints(segment.points),
        "data-paths": (segment.paths || groupPaths).join(" "), "vector-effect": "non-scaling-stroke",
      });
      edgeLayer.appendChild(rail);
      edgeElements.set(`${fanout.id}-rail-${index}`, rail);
      rememberRoute(`${fanout.id}-rail-${index}`, segment.points);
    });
    portLayer.appendChild(make("circle", {
      cx: sourcePoint[0], cy: sourcePoint[1], r: 9, class: `port ${fanout.type} fanout-source`,
      "data-fanout-id": fanout.id, "data-paths": groupPaths.join(" "), "vector-effect": "non-scaling-stroke",
    }));
    (fanout.junctions || []).forEach(([x, y, radius = 6], index) => {
      const isRoot = radius >= 9;
      portLayer.appendChild(make("circle", {
        cx: x, cy: y, r: radius, class: `port ${fanout.type} ${isRoot ? "fanout-junction" : "fanout-tap"}`,
        "data-fanout-id": fanout.id, "data-junction-index": index, "data-paths": groupPaths.join(" "), "vector-effect": "non-scaling-stroke",
      }));
    });
    renderEdgeLabel(fanout.label, fanout.labelAt, groupPaths, "fanout-label");
    (fanout.labels || []).forEach((label) => renderEdgeLabel(label.text, label.at, label.paths || groupPaths, "fanout-label"));

    fanout.branches.forEach((branch) => {
      const semanticEdge = edgeById.get(branch.edgeId);
      if (!semanticEdge) return;
      const target = nodeById.get(semanticEdge.to);
      if (!target) return;
      const end = pointFor(target, branch.toSide || semanticEdge.toSide || "left", branch.toRatio ?? semanticEdge.toRatio ?? .5);
      const branchPoints = [...branch.points, end];
      const branchPaths = [...new Set([...groupPaths, ...(semanticEdge.paths || [])])];
      const branchPath = make("path", {
        id: semanticEdge.id, class: `edge ${semanticEdge.type} fanout-branch`, d: pathFromPoints(branchPoints),
        "data-fanout-id": fanout.id, "data-paths": branchPaths.join(" "),
        "marker-end": `url(#arrow-${semanticEdge.type})`, "vector-effect": "non-scaling-stroke",
      });
      edgeLayer.appendChild(branchPath);
      edgeElements.set(semanticEdge.id, branchPath);
      rememberRoute(semanticEdge.id, branchPoints, [fanout.from, semanticEdge.to]);
      portLayer.appendChild(make("circle", {
        cx: end[0], cy: end[1], r: 9, class: `port ${semanticEdge.type} fanout-target`,
        "data-fanout-id": fanout.id, "data-paths": branchPaths.join(" "), "vector-effect": "non-scaling-stroke",
      }));
      renderEdgeLabel(branch.label, branch.labelAt, branchPaths, "fanout-branch-label");
    });
  });

  let shapeMode = "concrete";
  svg.dataset.shapeMode = shapeMode;
  const nodeElements = new Map();
  const shapeTextElements = new Map();

  const renderNode = (node) => {
    const group = make("g", {
      id: `node-${node.id}`,
      class: `cg-node kind-${node.kind}`,
      tabindex: "0",
      role: "button",
      "aria-label": `${node.title}，${node.shapes.concrete}`,
      "data-node-id": node.id,
      "data-paths": (node.paths || []).join(" "),
    });
    group.appendChild(make("rect", { x: node.x, y: node.y, width: node.w, height: node.h, rx: 15, "vector-effect": "non-scaling-stroke" }));
    group.appendChild(make("text", { x: node.x + 15, y: node.y + 20, class: "node-kind" }, node.kindLabel));
    group.appendChild(make("text", { x: node.x + 15, y: node.y + 46, class: "node-title" }, node.title));
    const showNodeNote = Boolean(node.note && node.h >= 96);
    if (showNodeNote) group.appendChild(make("text", { x: node.x + 15, y: node.y + 67, class: "node-note" }, node.note));
    const shapeText = make("text", { x: node.x + 15, y: node.y + node.h - 12, class: "node-shape" }, node.shapes[shapeMode]);
    group.appendChild(shapeText);
    group.addEventListener("click", () => selectNode(node.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectNode(node.id);
      }
    });
    nodeLayer.appendChild(group);
    nodeElements.set(node.id, group);
    shapeTextElements.set(node.id, shapeText);
  };
  nodes.forEach(renderNode);

  const addAnnotation = (x, y, w, h, title, lines) => {
    const group = make("g");
    group.appendChild(make("rect", { x, y, width: w, height: h, rx: 12, class: "annotation-box" }));
    group.appendChild(make("text", { x: x + 14, y: y + 24, class: "annotation-title" }, title));
    lines.forEach((line, index) => group.appendChild(make("text", { x: x + 14, y: y + 49 + index * 22, class: "annotation-text" }, line)));
    annotationLayer.appendChild(group);
  };
  addAnnotation(65, 1510, 600, 165, "首轮与后续轮次的边界", [
    "首轮：单张RGB → VAE → latent repeat×4",
    "后续：直接取上一chunk最后4个生成latent",
    "两条路径都汇入同一个 history latent queue。",
    "真实4帧RGB可用于训练窗口，但不是当前首轮rollout主链。",
  ]);
  addAnnotation(825, 2280, 1090, 155, "循环内没有第二个 concat 算子", [
    "z_current 始终是完整的 [history 4 slots | current future 8 slots]。",
    "update 后先恢复 history，再把整个 12-slot tensor 回送到 step k+1。",
    "第30步完成后才取最后8槽进行VAE解码。",
  ]);
  addAnnotation(3030, 1720, 320, 145, "CFG 灰色说明", [
    "cfg_scale=7 来自当前配置。",
    "cond/uncond代码当前被注释；",
    "主链不连接CFG结果。",
  ]);

  const inspector = document.getElementById("compute-inspector");
  const openInspectorButton = document.getElementById("open-inspector");
  const inspectorKind = document.getElementById("inspector-kind");
  const inspectorTitle = document.getElementById("inspector-title");
  const inspectorInput = document.getElementById("inspector-input");
  const inspectorOutput = document.getElementById("inspector-output");
  const inspectorDescription = document.getElementById("inspector-description");
  const inspectorFormula = document.getElementById("inspector-formula");
  const inspectorSource = document.getElementById("inspector-source");
  const graphStage = viewport.closest(".graph-stage");
  let selectedNodeId = null;

  const selectNode = (nodeId) => {
    const node = nodeById.get(nodeId);
    if (!node) return;
    selectedNodeId = nodeId;
    nodeElements.forEach((element, id) => element.classList.toggle("is-selected", id === nodeId));
    overviewNodeElements.forEach((element) => element.classList.toggle("is-selected", element.dataset.target === nodeId));
    inspectorKind.textContent = node.kindLabel;
    inspectorTitle.textContent = node.title;
    inspectorInput.textContent = node.input?.[shapeMode] || "—";
    inspectorOutput.textContent = node.shapes[shapeMode];
    inspectorDescription.textContent = node.description;
    inspectorFormula.textContent = node.formula;
    inspectorSource.textContent = `源码：${node.source}`;
    inspector.classList.add("is-open");
    graphStage?.classList.add("has-inspector");
    openInspectorButton.hidden = true;
    requestAnimationFrame(syncViewportHeight);
  };

  document.getElementById("close-inspector")?.addEventListener("click", () => {
    inspector.classList.remove("is-open");
    graphStage?.classList.remove("has-inspector");
    openInspectorButton.hidden = false;
    requestAnimationFrame(syncViewportHeight);
  });
  openInspectorButton?.addEventListener("click", () => {
    inspector.classList.add("is-open");
    graphStage?.classList.add("has-inspector");
    openInspectorButton.hidden = true;
    requestAnimationFrame(syncViewportHeight);
  });

  document.querySelectorAll("[data-shape-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      shapeMode = button.dataset.shapeMode;
      svg.dataset.shapeMode = shapeMode;
      document.querySelectorAll("[data-shape-mode]").forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("is-active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      nodes.forEach((node) => {
        shapeTextElements.get(node.id).textContent = node.shapes[shapeMode];
        nodeElements.get(node.id).setAttribute("aria-label", `${node.title}，${node.shapes[shapeMode]}`);
      });
      overviewNodes.forEach((node) => {
        overviewShapeTextElements.get(node.id).textContent = node[shapeMode];
        overviewNodeElements.get(node.id).setAttribute("aria-label", `${node.title}，${node[shapeMode]}`);
      });
      if (selectedNodeId) selectNode(selectedNodeId);
    });
  });

  let activePath = null;
  const applyPathFocus = (pathName) => {
    activePath = pathName === "clear" ? null : pathName;
    document.querySelectorAll("[data-path]").forEach((button) => {
      const active = activePath && button.dataset.path === activePath;
      button.classList.toggle("is-active", Boolean(active));
      button.setAttribute("aria-pressed", String(Boolean(active)));
    });
    const matches = (element) => !activePath || (element.getAttribute("data-paths") || "").split(" ").includes(activePath);
    nodeElements.forEach((element) => element.classList.toggle("is-dimmed", !matches(element)));
    edgeElements.forEach((element) => {
      element.classList.toggle("is-dimmed", !matches(element));
      element.classList.toggle("is-highlighted", Boolean(activePath && matches(element)));
    });
    labelLayer.querySelectorAll(".edge-label-group").forEach((element) => element.classList.toggle("is-dimmed", !matches(element)));
    portLayer.querySelectorAll(".port").forEach((element) => {
      element.classList.toggle("is-dimmed", !matches(element));
      element.classList.toggle("is-highlighted", Boolean(activePath && matches(element)));
    });
    overviewNodeElements.forEach((element) => element.classList.toggle("is-dimmed", !matches(element)));
    overviewEdgeElements.forEach((element) => {
      element.classList.toggle("is-dimmed", !matches(element));
      element.classList.toggle("is-highlighted", Boolean(activePath && matches(element)));
    });
    overviewPortLayer.querySelectorAll(".overview-port").forEach((element) => {
      element.classList.toggle("is-dimmed", !matches(element));
      element.classList.toggle("is-highlighted", Boolean(activePath && matches(element)));
    });
    overviewLabelLayer.querySelectorAll(".overview-edge-label").forEach((element) => element.classList.toggle("is-dimmed", !matches(element)));
  };
  document.querySelectorAll("[data-path]").forEach((button) => button.addEventListener("click", () => applyPathFocus(button.dataset.path)));

  const views = {
    overview: { x: 0, y: 0, w: 2700, h: 1650 },
    all: { x: 0, y: 0, w: canvas.width, h: canvas.height },
    rollout: { x: 15, y: 35, w: 700, h: 2160 },
    sampler: { x: 740, y: 35, w: 1910, h: 410 },
    action: { x: 780, y: 640, w: 760, h: 1490 },
    spatial: { x: 1535, y: 650, w: 680, h: 1470 },
    temporal: { x: 2210, y: 650, w: 735, h: 1470 },
    mask: { x: 2900, y: 540, w: 630, h: 1590 },
    output: { x: 2080, y: 2075, w: 1370, h: 505 },
    decode: { x: 950, y: 2680, w: 1660, h: 420 },
  };
  let densityMode = "overview";
  let view = { ...views.overview };
  let activeFocusName = null;

  const syncViewportHeight = () => {
    if (window.innerWidth <= 720) {
      viewport.style.removeProperty("height");
      return;
    }
    const availableWidth = Math.max(640, viewport.clientWidth || 1200);
    const idealHeight = availableWidth * (view.h / view.w);
    const maximum = activeFocusName ? 1700 : 1080;
    const minimum = activeFocusName && view.w / view.h > 2 ? 440 : 680;
    viewport.style.height = `${Math.round(Math.max(minimum, Math.min(maximum, idealHeight)))}px`;
  };

  const updateFocusClip = () => {
    const clipped = densityMode === "detail" && Boolean(activeFocusName);
    detailLayers.forEach((layer) => {
      if (clipped) layer.setAttribute("clip-path", "url(#detail-focus-clip)");
      else layer.removeAttribute("clip-path");
    });
    detailFocusClipRect.setAttribute("x", view.x);
    detailFocusClipRect.setAttribute("y", view.y);
    detailFocusClipRect.setAttribute("width", view.w);
    detailFocusClipRect.setAttribute("height", view.h);
    svg.classList.toggle("is-map-view", densityMode === "detail" && !activeFocusName);
    svg.classList.toggle("is-focus-view", clipped);
    viewport.classList.toggle("is-landscape-focus", clipped && view.w / view.h > 2);
  };

  const updateViewBox = () => {
    view.w = Math.max(420, Math.min(canvas.width * 2.3, view.w));
    view.h = Math.max(360, Math.min(canvas.height * 2.3, view.h));
    view.x = Math.max(-canvas.width * .25, Math.min(canvas.width * 1.25 - view.w, view.x));
    view.y = Math.max(-canvas.height * .25, Math.min(canvas.height * 1.25 - view.h, view.y));
    svg.setAttribute("preserveAspectRatio", window.innerWidth <= 720 ? "xMinYMin slice" : "xMidYMid meet");
    svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`);
    const baseWidth = densityMode === "overview" ? views.overview.w : canvas.width;
    const percent = Math.round((baseWidth / view.w) * 100);
    document.getElementById("zoom-value").textContent = `${percent}%`;
    updateFocusClip();
  };

  const setView = (nextView, focusName = null) => {
    activeFocusName = focusName;
    view = { ...nextView };
    updateViewBox();
    syncViewportHeight();
  };

  const markActiveFocus = (focusName) => {
    document.querySelectorAll("[data-focus]").forEach((candidate) => candidate.classList.toggle("is-active", candidate.dataset.focus === focusName));
  };
  const setDensityMode = (mode, resetView = true) => {
    densityMode = mode;
    svg.classList.toggle("mode-overview", mode === "overview");
    svg.classList.toggle("mode-detail", mode === "detail");
    document.querySelectorAll("[data-density]").forEach((button) => {
      const active = button.dataset.density === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (resetView) {
      markActiveFocus(null);
      setView(mode === "overview" ? views.overview : views.all, null);
    }
  };
  document.querySelectorAll("[data-density]").forEach((button) => button.addEventListener("click", () => setDensityMode(button.dataset.density)));
  document.querySelectorAll("[data-focus]").forEach((button) => {
    button.addEventListener("click", () => {
      setDensityMode("detail", false);
      markActiveFocus(button.dataset.focus);
      setView(views[button.dataset.focus] || views.all, button.dataset.focus === "all" ? null : button.dataset.focus);
    });
  });

  const zoomAt = (factor, clientX = null, clientY = null) => {
    const rect = svg.getBoundingClientRect();
    const px = clientX == null ? .5 : (clientX - rect.left) / rect.width;
    const py = clientY == null ? .5 : (clientY - rect.top) / rect.height;
    const anchorX = view.x + view.w * px;
    const anchorY = view.y + view.h * py;
    const nextW = view.w * factor;
    const nextH = view.h * factor;
    view = { x: anchorX - nextW * px, y: anchorY - nextH * py, w: nextW, h: nextH };
    updateViewBox();
  };
  document.getElementById("zoom-in")?.addEventListener("click", () => zoomAt(.82));
  document.getElementById("zoom-out")?.addEventListener("click", () => zoomAt(1.22));
  document.getElementById("fit-graph")?.addEventListener("click", () => { markActiveFocus(null); setView(densityMode === "overview" ? views.overview : views.all, null); });

  window.addEventListener("resize", () => syncViewportHeight());

  viewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomAt(event.deltaY < 0 ? .88 : 1.14, event.clientX, event.clientY);
  }, { passive: false });

  let dragState = null;
  viewport.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest?.(".cg-node,.overview-node")) return;
    dragState = { x: event.clientX, y: event.clientY, view: { ...view } };
    viewport.classList.add("is-dragging");
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!dragState) return;
    const rect = svg.getBoundingClientRect();
    view.x = dragState.view.x - (event.clientX - dragState.x) * (dragState.view.w / rect.width);
    view.y = dragState.view.y - (event.clientY - dragState.y) * (dragState.view.h / rect.height);
    updateViewBox();
  });
  const finishDrag = (event) => {
    if (!dragState) return;
    dragState = null;
    viewport.classList.remove("is-dragging");
    if (viewport.hasPointerCapture(event.pointerId)) viewport.releasePointerCapture(event.pointerId);
  };
  viewport.addEventListener("pointerup", finishDrag);
  viewport.addEventListener("pointercancel", finishDrag);

  viewport.addEventListener("keydown", (event) => {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault(); zoomAt(.84);
    } else if (event.key === "-") {
      event.preventDefault(); zoomAt(1.19);
    } else if (event.key === "0") {
      event.preventDefault(); markActiveFocus(null); setView(densityMode === "overview" ? views.overview : views.all);
    } else if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault();
      const stepX = view.w * .08;
      const stepY = view.h * .08;
      if (event.key === "ArrowLeft") view.x -= stepX;
      if (event.key === "ArrowRight") view.x += stepX;
      if (event.key === "ArrowUp") view.y -= stepY;
      if (event.key === "ArrowDown") view.y += stepY;
      updateViewBox();
    } else if (event.key === "Escape") {
      applyPathFocus("clear");
      inspector.classList.remove("is-open");
      graphStage?.classList.remove("has-inspector");
      openInspectorButton.hidden = false;
    }
  });

  const checks = [
    ["T", dims.T, dims.history + dims.future],
    ["grid", derived.grid, 16],
    ["S", derived.S, 256],
    ["tokens", derived.totalTokens, 3072],
    ["head_dim", derived.headDim, 72],
    ["QKV", derived.qkv, 3456],
    ["4H", derived.mlp, 4608],
    ["6H", derived.sixH, 6912],
    ["Final", derived.finalProjection, 32],
  ];
  const failed = checks.filter(([, actual, expected]) => actual !== expected);
  const fanoutFailures = [];
  const seenFanoutIds = new Set();
  const seenFanoutEdges = new Set();
  fanouts.forEach((fanout) => {
    if (seenFanoutIds.has(fanout.id)) fanoutFailures.push(`duplicate:${fanout.id}`);
    seenFanoutIds.add(fanout.id);
    if (!nodeById.has(fanout.from)) fanoutFailures.push(`source:${fanout.id}`);
    fanout.branches.forEach((branch) => {
      const semanticEdge = edgeById.get(branch.edgeId);
      if (!semanticEdge || semanticEdge.from !== fanout.from || !nodeById.has(semanticEdge.to)) fanoutFailures.push(`branch:${branch.edgeId}`);
      if (seenFanoutEdges.has(branch.edgeId)) fanoutFailures.push(`reused:${branch.edgeId}`);
      seenFanoutEdges.add(branch.edgeId);
    });
    const sourcePorts = portLayer.querySelectorAll(`.fanout-source[data-fanout-id="${fanout.id}"]`).length;
    if (sourcePorts !== 1) fanoutFailures.push(`source-port:${fanout.id}`);
  });
  overviewFanouts.forEach((fanout) => {
    if (seenFanoutIds.has(fanout.id)) fanoutFailures.push(`duplicate:${fanout.id}`);
    seenFanoutIds.add(fanout.id);
    if (!overviewNodeById.has(fanout.from)) fanoutFailures.push(`overview-source:${fanout.id}`);
    fanout.branches.forEach((branch) => {
      if (!overviewNodeById.has(branch.to)) fanoutFailures.push(`overview-target:${branch.id}`);
    });
  });
  const layoutFailures = [...labelLayoutIssues];
  const pointInsideNode = (point, node, padding = 12) => point[0] > node.x + padding && point[0] < node.x + node.w - padding && point[1] > node.y + padding && point[1] < node.y + node.h - padding;
  const segmentCrossesNode = (start, end, node) => {
    const distance = Math.hypot(end[0] - start[0], end[1] - start[1]);
    const steps = Math.max(2, Math.ceil(distance / 10));
    for (let index = 1; index < steps; index += 1) {
      const ratio = index / steps;
      const point = [start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio];
      if (pointInsideNode(point, node)) return true;
    }
    return false;
  };
  for (let index = 0; index < nodes.length; index += 1) {
    for (let other = index + 1; other < nodes.length; other += 1) {
      if (rectsOverlap(nodes[index], nodes[other])) layoutFailures.push(`node-overlap:${nodes[index].id}/${nodes[other].id}`);
    }
  }
  panelHeaderRects.forEach((header) => {
    nodes.forEach((node) => {
      if (rectsOverlap(header, node)) layoutFailures.push(`header-overlap:${header.id}/${node.id}`);
    });
  });
  placedLabelRects.forEach((label, index) => {
    nodes.forEach((node) => {
      if (rectsOverlap(label, node)) layoutFailures.push(`label-node:${label.id}/${node.id}`);
    });
    panelHeaderRects.forEach((header) => {
      if (rectsOverlap(label, header)) layoutFailures.push(`label-header:${label.id}/${header.id}`);
    });
    placedLabelRects.slice(index + 1).forEach((other) => {
      if (rectsOverlap(label, other)) layoutFailures.push(`label-overlap:${label.id}/${other.id}`);
    });
  });
  routedSegments.forEach((route) => {
    nodes.forEach((node) => {
      if (route.ignore.has(node.id)) return;
      for (let index = 0; index < route.points.length - 1; index += 1) {
        if (segmentCrossesNode(route.points[index], route.points[index + 1], node)) {
          layoutFailures.push(`route-node:${route.id}/${node.id}`);
          break;
        }
      }
    });
  });
  nodes.forEach((node) => {
    if (node.note && node.h >= 96 && node.h - 12 - 67 < 16) layoutFailures.push(`node-text:${node.id}`);
  });
  const status = document.getElementById("shape-check-status");
  if (failed.length === 0 && fanoutFailures.length === 0 && layoutFailures.length === 0) {
    status.textContent = `✓ ${checks.length} 项 shape · ${fanouts.length + overviewFanouts.length} 组 fan-out · 布局避障通过`;
  } else {
    const failures = [...failed.map(([name]) => name), ...fanoutFailures, ...layoutFailures];
    status.textContent = `计算图校验失败：${failures.join(", ")}`;
    status.classList.add("is-error");
  }

  setDensityMode("overview");
})();
