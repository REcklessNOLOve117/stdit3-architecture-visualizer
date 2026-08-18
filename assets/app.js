(function () {
  "use strict";

  const OPEN_SORA_SOURCE = "Open-Sora/opensora/models/stdit/stdit3.py";
  const WMPO_SOURCE = "WMPO/dependencies/opensora/opensora/models/stdit/stdit3.py";
  const DATASET_SOURCE = "WMPO/dependencies/opensora/opensora/datasets/simplevla_webdataset.py";

  const steps = [
    {
      eyebrow: "STEP 01 · INPUT CONTRACT",
      title: "先分清模型真正收到什么",
      intro: "两者都接收已经过 VAE 压缩的视频 latent，而不是直接处理 RGB。最大的语义变化是：原生 y 代表文本条件，WMPO 复用 y 参数传入连续机器人动作。",
      native: {
        subtitle: "视频生成模型的输入契约",
        nodes: [
          { label: "Video latent x", shape: "[B, 16, T, H, W]", note: "v1.3 默认 16 latent channels", kind: "shared" },
          { label: "Timestep", shape: "[B]", note: "当前噪声 / flow 时间", kind: "condition" },
          { label: "FPS", shape: "[B]", note: "视频帧率条件", kind: "condition" },
          { label: "Text feature y", shape: "[B, 1, N, 4096]", note: "T5 特征，后续映射至 1152", kind: "condition" }
        ],
        points: [
          "x 是 VAE latent；STDiT3 只负责 latent 空间中的去噪或速度预测。",
          "原生 v1.3 默认 in_channels=16，pred_sigma=true 时最终 out_channels=32。",
          "文本、fps 与 timestep 都是条件，不会先与 video token 直接拼接。"
        ],
        formula: "x ∈ ℝ[B×16×T×H×W],  y_text ∈ ℝ[B×1×N×4096]",
        source: OPEN_SORA_SOURCE + " · L203–228, L424–445",
        code: [
          "in_channels=16,",
          "hidden_size=1152, depth=28, num_heads=16,",
          "",
          "def forward(self, x, timestep, y, ..., fps=None, ...):",
          "    x = x.to(dtype)",
          "    y = y.to(dtype)"
        ].join("\n")
      },
      wmpo: {
        subtitle: "动作条件世界模型的输入契约",
        nodes: [
          { label: "Video latent x", shape: "[B, 4, 12, H, W]", note: "当前 fork 默认 4 channels", kind: "drift" },
          { label: "Timestep", shape: "[B]", note: "全局扩散时间", kind: "condition" },
          { label: "Actions y", shape: "[B, 8, 7]", note: "Ta=8 个连续机器人动作", kind: "action" },
          { label: "History slots", shape: "To=4", note: "为历史帧补 NONE action", kind: "action" }
        ],
        points: [
          "SimpleVLAWebDataset 默认取 4 帧 observation history 与 8 个未来动作，共 12 帧窗口。",
          "action_dim=7；数据归一化到约 [-1, 1] 后作为 y 传入 STDiT3。",
          "当前 WMPO fork 默认 in_channels=4，这属于底层 VAE / fork 版本差异。"
        ],
        formula: "T = To + Ta = 4 + 8 = 12,  a ∈ ℝ[B×8×7]",
        source: WMPO_SOURCE + " · L212–236, L412–423；" + DATASET_SOURCE + " · L24–35, L101–118",
        code: [
          "Ta: int = 8",
          "To: int = 4",
          "action_dim: int = 7",
          "",
          "y = batch.pop(\"action\")",
          "model_args = {\"y\": y.to(device, dtype)}"
        ].join("\n")
      },
      diffs: [
        { type: "core", tag: "核心设计", text: "WMPO 把文本条件入口 y 改成连续 action 序列，让模型学习 p(未来画面 | 历史画面, 动作)。" },
        { type: "drift", tag: "FORK 差异", text: "16→4 latent channels 来自两套 Open-Sora/VAE 版本，不应直接理解为 WMPO 的 action-conditioning 创新。" }
      ]
    },
    {
      eyebrow: "STEP 02 · TOKENIZATION",
      title: "PatchEmbed3D：把视频变成时空 token",
      intro: "patch_size=(1,2,2) 不压缩 latent 时间轴，只把每帧的 H×W 划成 2×2 patch。随后加入二维空间位置编码，得到 C=1152 的 token。",
      native: {
        subtitle: "先展平，再恢复 T×S 语义",
        nodes: [
          { label: "x", shape: "[B,16,T,H,W]", note: "latent 视频", kind: "shared" },
          { label: "PatchEmbed3D", shape: "kernel=1×2×2", note: "Conv3D / patch projection", kind: "shared" },
          { label: "Flat tokens", shape: "[B,T·S,1152]", note: "S=(H/2)·(W/2)", kind: "shared" },
          { label: "+ 2D pos", shape: "[B,T,S,1152]", note: "每帧共享空间位置语义", kind: "condition" }
        ],
        points: [
          "时间 patch 大小为 1，因此输入 latent 的 T 与 token 帧数 T 相同。",
          "原生 forward 先得到 [B,N,C]，再 rearrange 为 [B,T,S,C] 加位置编码。",
          "进入 block 前，原生实现再次展平为 [B,T·S,C]。"
        ],
        formula: "S = (H / 2) × (W / 2),  N = T × S,  C = 1152",
        source: OPEN_SORA_SOURCE + " · L277–291, L446–471, L495–511",
        code: [
          "self.x_embedder = PatchEmbed3D((1, 2, 2), 16, 1152)",
          "x = self.x_embedder(x)              # [B, T*S, C]",
          "x = rearrange(x, \"B (T S) C -> B T S C\", T=T, S=S)",
          "x = x + pos_emb",
          "x = rearrange(x, \"B T S C -> B (T S) C\")"
        ].join("\n")
      },
      wmpo: {
        subtitle: "保留显式 B×T×S×C 布局",
        nodes: [
          { label: "x", shape: "[B,4,12,H,W]", note: "latent 视频窗口", kind: "drift" },
          { label: "PatchEmbed3D", shape: "kernel=1×2×2", note: "共享 tokenization 思路", kind: "shared" },
          { label: "Flat tokens", shape: "[B,12·S,1152]", note: "先由 embedder 展平", kind: "shared" },
          { label: "+ 2D pos", shape: "[B,12,S,1152]", note: "后续保持四维布局", kind: "action" }
        ],
        points: [
          "PatchEmbed3D 与位置编码的基本目标没有改变。",
          "WMPO 不在进入 blocks 前重新展平；显式 T 维方便逐帧 action 条件广播。",
          "这使 Spatial Block 可以直接接收每帧不同的 AdaLN 参数。"
        ],
        formula: "x_tokens ∈ ℝ[B×12×S×1152]",
        source: WMPO_SOURCE + " · L284–293, L419–446, L472–482",
        code: [
          "self.x_embedder = PatchEmbed3D((1, 2, 2), 4, 1152)",
          "x = self.x_embedder(x)",
          "x = rearrange(x, \"B (T S) C -> B T S C\", T=T, S=S)",
          "x = x + pos_emb",
          "# 保留 [B, T, S, C]"
        ].join("\n")
      },
      diffs: [
        { type: "shared", tag: "共享结构", text: "两者的 patch_size、hidden size 与二维位置编码思想一致。" },
        { type: "core", tag: "核心设计", text: "WMPO 保留显式 T 轴，为下一步把逐帧 action condition 对齐到 Spatial Block 做准备。" }
      ]
    },
    {
      eyebrow: "STEP 03 · CONDITION ENCODING",
      title: "条件支路：文本/fps 被 action 序列取代",
      intro: "原生模型产生一个样本级 timestep condition，并单独编码文本；WMPO 则把 timestep 复制到每帧，再与对应 action embedding 相加。",
      native: {
        subtitle: "全局时间条件 + 独立文本条件",
        nodes: [
          { label: "timestep", shape: "[B]", note: "标量时间", kind: "condition" },
          { label: "TimestepEmbedder", shape: "[B,1152]", note: "频率编码 + MLP", kind: "condition" },
          { label: "+ FPS embed", shape: "[B,1152]", note: "全局视频元信息", kind: "condition" },
          { label: "Text CaptionEmbedder", shape: "[B,N,1152]", note: "独立送入 cross-attention", kind: "condition" }
        ],
        points: [
          "t = timestep_embedding + fps_embedding，整段视频共享同一个 t。",
          "文本 y 经过 CaptionEmbedder 后不加到 t，而是保留为 cross-attention 的 key/value。",
          "因此原生 STDiT3 同时有 AdaLN 条件通路与文本 cross-attention 通路。"
        ],
        formula: "c_global = E_t(timestep) + E_fps(fps)",
        source: OPEN_SORA_SOURCE + " · L293–305, L473–493",
        code: [
          "t = self.t_embedder(timestep, dtype=x.dtype)  # [B,C]",
          "fps = self.fps_embedder(fps.unsqueeze(1), B)",
          "t = t + fps",
          "",
          "y, y_lens = self.encode_text(y, mask)"
        ].join("\n")
      },
      wmpo: {
        subtitle: "逐帧 action condition + 全局 temporal condition",
        nodes: [
          { label: "Actions", shape: "[B,8,7]", note: "连续控制量", kind: "action" },
          { label: "ActionEncoder", shape: "7→4608→1152", note: "Linear · SiLU · Linear", kind: "action" },
          { label: "Prepend NONE", shape: "[B,12,1152]", note: "前 4 帧无对应未来 action", kind: "action" },
          { label: "repeat(t)+action", shape: "[B·12,1152]", note: "Spatial condition", kind: "action" },
          { label: "t only", shape: "[B,1152]", note: "Temporal condition", kind: "condition" }
        ],
        points: [
          "每个 7D action 独立经过 MLP：7→4C→C，其中 C=1152。",
          "可学习的 none_action 被重复 To=4 次并放在 action 序列前面。",
          "t_s 逐帧加 action；t_t 仍是样本级 timestep，因此 action 不直接调制 Temporal Block。"
        ],
        formula: "c_spatial[b,i] = E_t(t_b) + E_a(a_b,i),  c_temporal[b] = E_t(t_b)",
        source: WMPO_SOURCE + " · L37–72, L291–305, L423–456",
        code: [
          "self.fc = Sequential(",
          "    Linear(action_dim, hidden_size * 4), SiLU(),",
          "    Linear(hidden_size * 4, hidden_size))",
          "output = cat([NONE × pad_action_num, output], dim=1)",
          "",
          "t_s = repeat(t, 'b d -> (b c) d', c=T)",
          "t_s = t_s + action_embedding"
        ].join("\n")
      },
      diffs: [
        { type: "core", tag: "核心设计", text: "ActionEncoder 不直接修改 video token；它先改变 condition，再由 AdaLN 控制每帧的 Spatial Block。" },
        { type: "drift", tag: "FORK 差异", text: "WMPO 当前 forward 注释掉 fps 与文本编码；这是当前实现事实，但需与 action 注入这一核心设计分开理解。" }
      ]
    },
    {
      eyebrow: "STEP 04 · ADALN PARAMETERS",
      title: "t_block：把 condition 变成六组控制量",
      intro: "t_block 的输出不是新 token，而是每个 Block 都会使用的 shift、scale 与 gate。它们分别控制 attention 与 MLP 两个残差分支。",
      native: {
        subtitle: "一个全局 condition 调制所有帧",
        nodes: [
          { label: "c_global", shape: "[B,1152]", note: "timestep + fps", kind: "condition" },
          { label: "SiLU", shape: "[B,1152]", note: "非线性", kind: "shared" },
          { label: "Linear", shape: "1152→6912", note: "6 × hidden_size", kind: "shared" },
          { label: "6-way chunk", shape: "6 × [B,1152]", note: "两套 shift/scale/gate", kind: "condition" }
        ],
        points: [
          "norm1 与 norm2 的 affine=False，固定 gamma/beta 不在 LayerNorm 内部。",
          "每个 Block 再把 t_block 输出与自己的 scale_shift_table 相加。",
          "原生同一组 t_mlp 同时传给 Spatial 与 Temporal Block。"
        ],
        formula: "(β_attn, γ_attn, g_attn, β_mlp, γ_mlp, g_mlp) = t_block(c_global)",
        source: OPEN_SORA_SOURCE + " · L67–83, L112–120, L295–298, L473–483",
        code: [
          "self.t_block = Sequential(",
          "    SiLU(), Linear(1152, 6 * 1152))",
          "",
          "t_mlp = self.t_block(t)  # [B,6912]",
          "params = (scale_shift_table + t.reshape(B,6,-1)).chunk(6, dim=1)"
        ].join("\n")
      },
      wmpo: {
        subtitle: "Spatial 每帧一组，Temporal 每段一组",
        nodes: [
          { label: "c_spatial", shape: "[B·12,1152]", note: "含 action", kind: "action" },
          { label: "t_block", shape: "1152→6912", note: "共享同一 MLP 权重", kind: "shared" },
          { label: "t_s_mlp", shape: "[B·12,6912]", note: "reshape 为 [B,T,6,C]", kind: "action" },
          { label: "t_t_mlp", shape: "[B,6912]", note: "全局 temporal 调制", kind: "condition" }
        ],
        points: [
          "t_s_mlp 包含每一帧各自的 action 信息。",
          "Spatial Block 内部 reshape 为 [B,T,6,C]，让每帧获得不同 shift/scale/gate。",
          "Temporal Block 使用 t_t_mlp，reshape 为 [B,1,6,C] 并沿空间位置共享。"
        ],
        formula: "AdaLN(x,c) = (1 + scale(c)) · LN(x) + shift(c)",
        source: WMPO_SOURCE + " · L102–118, L141–154, L295–298, L452–462",
        code: [
          "t_s_mlp = self.t_block(t_s)  # [B*T, 6C]",
          "t_t_mlp = self.t_block(t)    # [B, 6C]",
          "",
          "if temporal:",
          "    t = t.reshape(B, 1, 6, C)",
          "else:",
          "    t = t.reshape(B, T, 6, C)"
        ].join("\n")
      },
      diffs: [
        { type: "shared", tag: "共享机制", text: "两者都使用 AdaLN-Zero 风格的 6C 调制和每层可学习 scale_shift_table。" },
        { type: "core", tag: "核心设计", text: "WMPO 的本质变化是把原本样本级的 Spatial AdaLN 参数提升为逐帧参数，并让 action 决定它们。" }
      ]
    },
    {
      eyebrow: "STEP 05 · SPATIAL BLOCK",
      title: "Spatial Block：action 真正影响画面的地方",
      intro: "Spatial Attention 只在同一帧的 S 个 patch 间建模。WMPO 用逐帧 action 生成的 AdaLN 参数调制该帧所有 patch，因此动作首先改变每帧的空间特征处理方式。",
      native: {
        subtitle: "AdaLN → 空间自注意力 → 文本 Cross-Attn → MLP",
        nodes: [
          { label: "LN + AdaLN", shape: "[B,T·S,C]", note: "全局 t_mlp", kind: "condition" },
          { label: "Spatial Self-Attn", shape: "[(B·T),S,C]", note: "同帧 patch 互相注意", kind: "shared" },
          { label: "Gate + Residual", shape: "[B,T·S,C]", note: "g_attn 控制残差", kind: "shared" },
          { label: "Text Cross-Attn", shape: "Q=x, KV=y", note: "每个 Block 都注入文本", kind: "condition" },
          { label: "AdaLN + MLP", shape: "1152→4608→1152", note: "g_mlp 后再残差", kind: "shared" }
        ],
        points: [
          "Spatial Block 把 [B,T·S,C] 临时变为 [(B·T),S,C] 做 attention。",
          "Self-Attention 后存在显式文本 cross-attention。",
          "attention 和 MLP 前各做一次无 affine LayerNorm + AdaLN。"
        ],
        formula: "x ← x + g_attn·SA(AdaLN₁(x));  x ← x + CrossAttn(x,y);  x ← x + g_mlp·MLP(AdaLN₂(x))",
        source: OPEN_SORA_SOURCE + " · L96–197",
        code: [
          "x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)",
          "x_m = rearrange(x_m, \"B (T S) C -> (B T) S C\")",
          "x = x + gate_msa * self.attn(x_m)",
          "x = x + self.cross_attn(x, y, mask)",
          "x = x + gate_mlp * self.mlp(t2i_modulate(self.norm2(x), ...))"
        ].join("\n")
      },
      wmpo: {
        subtitle: "逐帧 Action-AdaLN → 空间自注意力 → MLP",
        nodes: [
          { label: "LN + Action-AdaLN", shape: "[B,12,S,C]", note: "每帧不同参数", kind: "action" },
          { label: "Spatial Self-Attn", shape: "[(B·12),S,C]", note: "同帧 patch 互相注意", kind: "shared" },
          { label: "Gate + Residual", shape: "[B,12,S,C]", note: "逐帧 g_attn", kind: "action" },
          { label: "Text Cross-Attn", shape: "disabled", note: "代码路径被注释", kind: "removed" },
          { label: "Action-AdaLN + MLP", shape: "1152→4608→1152", note: "逐帧 g_mlp", kind: "action" }
        ],
        points: [
          "shift/scale/gate 的形状可广播为 [B,T,1,C]：同帧所有 S 个 patch 共享一个 action condition。",
          "action 不进入 Q/K/V；它先改变归一化后的特征幅度与偏移，再控制残差门。",
          "文本 cross-attention 模块仍被构造，但 forward 中调用被注释。"
        ],
        formula: "x[b,t] ← x[b,t] + g(a[b,t]) · SA((1+γ(a[b,t]))LN(x[b,t])+β(a[b,t]))",
        source: WMPO_SOURCE + " · L75–205",
        code: [
          "t = t.reshape(B, T, 6, -1)",
          "shift_msa, scale_msa, gate_msa, ... = t.chunk(6, dim=-1)",
          "x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)",
          "x_m = rearrange(x_m, \"B T S C -> (B T) S C\")",
          "# x = x + self.cross_attn(x, y, mask)",
          "x = x + gate_mlp * self.mlp(...)"
        ].join("\n")
      },
      diffs: [
        { type: "core", tag: "核心设计", text: "ActionEncoder 通过每个 Spatial Block 的 AdaLN 与两个 gate 反复影响每帧特征，而不是只在输入处加一次 embedding。" },
        { type: "core", tag: "条件变化", text: "原生依靠文本 cross-attention 提供语义控制；WMPO 当前 world-model 路径将其禁用，主要依靠历史视觉与 action 控制。" }
      ]
    },
    {
      eyebrow: "STEP 06 · TEMPORAL BLOCK",
      title: "Temporal Block：沿时间传播变化",
      intro: "Temporal Attention 固定一个空间位置，在 T 帧之间做注意力。WMPO 不把 action 直接加入 Temporal AdaLN，而是让 Spatial Block 已被 action 改写的特征沿时间传播。",
      native: {
        subtitle: "全序列时间建模",
        nodes: [
          { label: "LN + AdaLN", shape: "[B,T·S,C]", note: "全局 t_mlp", kind: "condition" },
          { label: "Rearrange", shape: "[(B·S),T,C]", note: "固定空间位置看时间", kind: "shared" },
          { label: "Temporal Attn", shape: "T ↔ T", note: "源码未设置 causal mask", kind: "shared" },
          { label: "Text Cross-Attn", shape: "Q=x, KV=y", note: "同样执行文本条件", kind: "condition" },
          { label: "AdaLN + MLP", shape: "1152→4608→1152", note: "残差输出", kind: "shared" }
        ],
        points: [
          "每个空间 patch 位置形成一条长度 T 的时间序列。",
          "原生 Attention 构造未传 is_causal，因此默认可同时观察前后帧。",
          "Temporal Block 与 Spatial Block 使用同一 STDiT3Block 类，只通过 temporal=True 改 reshape 与 RoPE。"
        ],
        formula: "[(B·S), T, C] → SelfAttention over T → [B, T·S, C]",
        source: OPEN_SORA_SOURCE + " · L37–84, L128–177, L328–349",
        code: [
          "STDiT3Block(..., temporal=True, rope=self.rope.rotate_queries_or_keys)",
          "",
          "x_m = rearrange(x_m, \"B (T S) C -> (B S) T C\")",
          "x_m = self.attn(x_m)",
          "x_m = rearrange(x_m, \"(B S) T C -> B (T S) C\")"
        ].join("\n")
      },
      wmpo: {
        subtitle: "因果时间建模，action 间接传播",
        nodes: [
          { label: "LN + AdaLN", shape: "[B,12,S,C]", note: "只用全局 timestep", kind: "condition" },
          { label: "Rearrange", shape: "[(B·S),12,C]", note: "每个 patch 的时间轨迹", kind: "shared" },
          { label: "Causal Temporal Attn", shape: "past → present", note: "is_causal=True", kind: "action" },
          { label: "Text Cross-Attn", shape: "disabled", note: "forward 中注释", kind: "removed" },
          { label: "AdaLN + MLP", shape: "1152→4608→1152", note: "全局 t 调制", kind: "shared" }
        ],
        points: [
          "temporal=True 时 Attention 被显式设置 is_causal=True。",
          "Temporal AdaLN 使用 t_t_mlp=[B,6C]，不直接包含 action embedding。",
          "action 的影响通过前一个 Spatial Block 输出进入 Temporal Attention，保持信息流因果性。"
        ],
        formula: "Temporal_i receives Spatial_i(x, action);  attention(t) only reads frames ≤ t",
        source: WMPO_SOURCE + " · L90–118, L141–171, L325–343, L452–487",
        code: [
          "self.attn = Attention(..., is_causal=temporal)",
          "",
          "if self.temporal:",
          "    t = t.reshape(B, 1, 6, -1)",
          "    x_m = rearrange(x_m, \"B T S C -> (B S) T C\")",
          "    x_m = self.attn(x_m)"
        ].join("\n")
      },
      diffs: [
        { type: "core", tag: "核心设计", text: "WMPO 的 action 是“Spatial 直接调制、Temporal 间接传播”：这让动作影响画面变化，同时保持时间注意力的统一条件。" },
        { type: "core", tag: "因果性", text: "WMPO 显式 causal temporal attention，符合从历史和动作逐步预测未来的世界模型用途。" }
      ]
    },
    {
      eyebrow: "STEP 07 · BACKBONE × 28",
      title: "不是一个 Block，而是 28 次交替更新",
      intro: "STDiT3-XL/2 有 28 个 Spatial Block 与 28 个 Temporal Block，按 Spatialᵢ→Temporalᵢ 交替执行。理解这一点才能理解 action 为什么会被反复注入。",
      native: {
        subtitle: "28 ×（Spatial + Temporal）",
        nodes: [
          { label: "Pair 01", shape: "Spatial₁ → Temporal₁", note: "t_mlp + text y", kind: "shared" },
          { label: "Pair 02", shape: "Spatial₂ → Temporal₂", note: "t_mlp + text y", kind: "shared" },
          { label: "…", shape: "repeat", note: "gradient checkpoint", kind: "condition" },
          { label: "Pair 28", shape: "Spatial₂₈ → Temporal₂₈", note: "t_mlp + text y", kind: "shared" }
        ],
        points: [
          "depth=28 表示两套 ModuleList 各有 28 个 Block，而不是总共 28 个半块。",
          "每个 Spatial/Temporal Block 都拥有独立 attention、MLP 和 scale_shift_table 参数。",
          "t_mlp 与文本 y 在不同层间复用，但各层如何响应由自身权重决定。"
        ],
        formula: "xᵢ⁺ = Spatialᵢ(xᵢ, y, t);  xᵢ₊₁ = Temporalᵢ(xᵢ⁺, y, t),  i=1…28",
        source: OPEN_SORA_SOURCE + " · L307–350, L513–547, L609–620",
        code: [
          "for spatial_block, temporal_block in zip(",
          "        self.spatial_blocks, self.temporal_blocks):",
          "    x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, ...)",
          "    x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, ...)"
        ].join("\n")
      },
      wmpo: {
        subtitle: "28 次 Action-Spatial 注入 + Causal-Temporal 传播",
        nodes: [
          { label: "Pair 01", shape: "Action-Spatial₁ → Causal-Temporal₁", note: "t_s / t_t 分流", kind: "action" },
          { label: "Pair 02", shape: "Action-Spatial₂ → Causal-Temporal₂", note: "动作再次调制", kind: "action" },
          { label: "…", shape: "repeat", note: "action condition 始终复用", kind: "condition" },
          { label: "Pair 28", shape: "Action-Spatial₂₈ → Causal-Temporal₂₈", note: "得到动作一致的 latent", kind: "action" }
        ],
        points: [
          "同一 action embedding 生成的 t_s_mlp 会送入全部 28 个 Spatial Block。",
          "每层 scale_shift_table 不同，因此每层会学习不同的 action 响应方式。",
          "Temporal Block 始终拿 t_t_mlp；动作影响通过每一对 Block 的中间特征逐层积累。"
        ],
        formula: "xᵢ⁺ = Spatialᵢ(xᵢ, t_s(action));  xᵢ₊₁ = CausalTemporalᵢ(xᵢ⁺, t_t)",
        source: WMPO_SOURCE + " · L307–344, L484–487, L530–553",
        code: [
          "for spatial_block, temporal_block in zip(",
          "        self.spatial_blocks, self.temporal_blocks):",
          "    x = auto_grad_checkpoint(spatial_block, x, t_s_mlp, ...)",
          "    x = auto_grad_checkpoint(temporal_block, x, t_t_mlp, ...)"
        ].join("\n")
      },
      diffs: [
        { type: "core", tag: "核心设计", text: "action conditioning 不是只发生一次；同一逐帧 condition 会在 28 个 Spatial Block 中反复控制 attention 与 MLP。" },
        { type: "shared", tag: "共享骨架", text: "主干深度、hidden size、heads、MLP ratio 与 Spatial/Temporal 交替结构保持一致，便于继承预训练能力。" }
      ]
    },
    {
      eyebrow: "STEP 08 · OUTPUT PROJECTION",
      title: "Final Layer：token 回到视频 latent",
      intro: "主干输出先经过最后一次 timestep-conditioned modulation，再线性投影到每个 patch 的输出通道，最后 unpatchify 恢复 [B,Cout,T,H,W]。",
      native: {
        subtitle: "全局 timestep 调制后恢复 16/32 通道 latent",
        nodes: [
          { label: "Backbone tokens", shape: "[B,T·S,1152]", note: "28 对 Block 输出", kind: "shared" },
          { label: "T2IFinalLayer", shape: "LN + AdaLN + Linear", note: "使用全局 t", kind: "condition" },
          { label: "Patch output", shape: "[B,T·S,1·2·2·Cout]", note: "Cout=32 if pred_sigma", kind: "output" },
          { label: "Unpatchify", shape: "[B,Cout,T,H,W]", note: "裁掉 padding", kind: "output" }
        ],
        points: [
          "Final Layer 使用原始 t embedding，而不是 6C 的 t_mlp。",
          "pred_sigma=true 时 out_channels=2×in_channels；调度器通常再按通道解释预测。",
          "unpatchify 把 patch 内 1×2×2 的像素位置重新排回时空网格。"
        ],
        formula: "[B,T·S,4·Cout] → rearrange → [B,Cout,T,H,W]",
        source: OPEN_SORA_SOURCE + " · L261–264, L352–355, L574–606",
        code: [
          "self.out_channels = in_channels * 2 if pred_sigma else in_channels",
          "self.final_layer = T2IFinalLayer(1152, 4, self.out_channels)",
          "",
          "x = self.final_layer(x, t, x_mask, t0, T, S)",
          "x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)"
        ].join("\n")
      },
      wmpo: {
        subtitle: "action 已写入 hidden state，最终层仍用全局 timestep",
        nodes: [
          { label: "Action-shaped tokens", shape: "[B,12,S,1152]", note: "动作影响已累积 28 层", kind: "action" },
          { label: "T2IFinalLayer", shape: "LN + AdaLN + Linear", note: "直接条件仍是全局 t", kind: "condition" },
          { label: "Patch output", shape: "[B,12,S,1·2·2·Cout]", note: "Cout=8 if pred_sigma", kind: "output" },
          { label: "Unpatchify", shape: "[B,Cout,12,H,W]", note: "恢复 latent 视频", kind: "output" }
        ],
        points: [
          "action 不直接进入 Final Layer；它已经通过 28 个 Spatial Block 改变了 x。",
          "当前 fork 默认 in_channels=4，因此 pred_sigma=true 时 Cout=8。",
          "WMPO unpatchify 接收显式 [B,T,S,C]，其 rearrange 模式与原生展平布局不同。"
        ],
        formula: "output = Unpatchify(FinalLayer(x_action-conditioned, E_t(timestep)))",
        source: WMPO_SOURCE + " · L268–270, L345–349, L495–527",
        code: [
          "x = self.final_layer(x, t, x_mask, t0, T, S)",
          "x = rearrange(",
          "    x,",
          "    \"B N_t (N_h N_w) (T_p H_p W_p C_out)",
          "       -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)\")",
          "return x.to(torch.float32)"
        ].join("\n")
      },
      diffs: [
        { type: "shared", tag: "共享结构", text: "两者都以 T2IFinalLayer + unpatchify 结束，最终输出仍是供 VAE 解码或 scheduler 使用的视频 latent。" },
        { type: "drift", tag: "FORK 差异", text: "输出通道 32 vs 8 与输入 latent 通道 16 vs 4 对应，是底层版本差异；核心 action 效果已经在 hidden state 中形成。" }
      ]
    }
  ];

  const state = {
    mode: "compare",
    step: 0,
    timer: null
  };

  const architectureView = document.getElementById("architecture-view");
  const differenceContent = document.getElementById("difference-content");
  const stepEyebrow = document.getElementById("step-eyebrow");
  const stepTitle = document.getElementById("step-title");
  const stepIntro = document.getElementById("step-intro");
  const progressFill = document.getElementById("progress-fill");
  const progressTrack = document.querySelector(".progress-track");
  const prevButton = document.getElementById("prev-step");
  const nextButton = document.getElementById("next-step");
  const autoButton = document.getElementById("auto-play");
  const modeButtons = Array.from(document.querySelectorAll(".mode-button"));
  const stepButtons = Array.from(document.querySelectorAll(".step-button"));

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function buildFlow(nodes) {
    const flow = element("div", "flow");
    nodes.forEach(function (node, index) {
      const box = element("div", "flow-node " + node.kind);
      box.appendChild(element("span", "node-label", node.label));
      box.appendChild(element("span", "node-shape", node.shape));
      box.appendChild(element("span", "node-note", node.note));
      flow.appendChild(box);

      if (index < nodes.length - 1) {
        flow.appendChild(element("span", "flow-arrow", "→"));
      }
    });
    return flow;
  }

  function buildPanel(kind, content) {
    const isNative = kind === "native";
    const article = element("article", "model-panel " + kind);
    article.setAttribute("aria-label", isNative ? "原生 Open-Sora 当前步骤" : "WMPO 当前步骤");

    const header = element("header", "model-header");
    const headerCopy = element("div");
    headerCopy.appendChild(element("h3", "", isNative ? "原生 Open-Sora v1.3" : "WMPO World Model"));
    headerCopy.appendChild(element("p", "", content.subtitle));
    header.appendChild(headerCopy);
    header.appendChild(element("span", "model-badge", isNative ? "STDiT3-XL/2" : "ACTION-CONDITIONED"));
    article.appendChild(header);

    const flowWrap = element("div", "flow-wrap");
    flowWrap.appendChild(buildFlow(content.nodes));
    article.appendChild(flowWrap);

    const detail = element("div", "panel-detail");
    const explanation = element("div", "explanation");
    const list = element("ul", "point-list");
    content.points.forEach(function (point) {
      list.appendChild(element("li", "", point));
    });
    explanation.appendChild(list);
    explanation.appendChild(element("div", "formula", content.formula));

    const source = element("div", "source-block");
    source.appendChild(element("p", "source-label", "SOURCE TRACE"));
    source.appendChild(element("span", "source-path", content.source));
    const pre = element("pre");
    pre.appendChild(element("code", "", content.code));
    source.appendChild(pre);

    detail.appendChild(explanation);
    detail.appendChild(source);
    article.appendChild(detail);
    return article;
  }

  function renderDifferences(step) {
    differenceContent.replaceChildren();
    step.diffs.forEach(function (diff) {
      const item = element("div", "diff-item " + diff.type);
      item.appendChild(element("span", "diff-tag", diff.tag));
      item.appendChild(element("p", "", diff.text));
      differenceContent.appendChild(item);
    });
  }

  function render() {
    const step = steps[state.step];
    stepEyebrow.textContent = step.eyebrow;
    stepTitle.textContent = step.title;
    stepIntro.textContent = step.intro;

    architectureView.replaceChildren();
    architectureView.className = "architecture-view " + (state.mode === "compare" ? "compare" : "single");
    if (state.mode === "native" || state.mode === "compare") {
      architectureView.appendChild(buildPanel("native", step.native));
    }
    if (state.mode === "wmpo" || state.mode === "compare") {
      architectureView.appendChild(buildPanel("wmpo", step.wmpo));
    }
    renderDifferences(step);

    modeButtons.forEach(function (button) {
      const active = button.dataset.mode === state.mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    stepButtons.forEach(function (button, index) {
      const active = index === state.step;
      button.classList.toggle("is-active", active);
      if (active) {
        button.setAttribute("aria-current", "step");
      } else {
        button.removeAttribute("aria-current");
      }
    });

    const progress = ((state.step + 1) / steps.length) * 100;
    progressFill.style.width = progress + "%";
    progressTrack.setAttribute("aria-valuenow", String(state.step + 1));
    prevButton.disabled = state.step === 0;
    nextButton.disabled = state.step === steps.length - 1;
  }

  function stopAutoPlay() {
    if (state.timer !== null) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
    autoButton.setAttribute("aria-pressed", "false");
    autoButton.textContent = "▶ 自动播放";
  }

  function setStep(index, fromAutoPlay) {
    state.step = Math.max(0, Math.min(steps.length - 1, index));
    render();
    if (!fromAutoPlay) {
      stopAutoPlay();
    }
  }

  function toggleAutoPlay() {
    if (state.timer !== null) {
      stopAutoPlay();
      return;
    }

    if (state.step === steps.length - 1) {
      state.step = 0;
      render();
    }
    autoButton.setAttribute("aria-pressed", "true");
    autoButton.textContent = "❚❚ 暂停播放";
    state.timer = window.setInterval(function () {
      if (state.step >= steps.length - 1) {
        stopAutoPlay();
        return;
      }
      setStep(state.step + 1, true);
      if (state.step >= steps.length - 1) {
        stopAutoPlay();
      }
    }, 4500);
  }

  modeButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      state.mode = button.dataset.mode;
      render();
    });
  });

  stepButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      setStep(Number(button.dataset.step), false);
    });
  });

  prevButton.addEventListener("click", function () {
    setStep(state.step - 1, false);
  });

  nextButton.addEventListener("click", function () {
    setStep(state.step + 1, false);
  });

  autoButton.addEventListener("click", toggleAutoPlay);

  document.addEventListener("keydown", function (event) {
    const tagName = event.target && event.target.tagName;
    if (tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") {
      return;
    }
    if (event.key === "ArrowLeft" && state.step > 0) {
      event.preventDefault();
      setStep(state.step - 1, false);
    }
    if (event.key === "ArrowRight" && state.step < steps.length - 1) {
      event.preventDefault();
      setStep(state.step + 1, false);
    }
  });

  render();
}());
