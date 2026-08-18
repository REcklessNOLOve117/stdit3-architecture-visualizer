(() => {
  "use strict";

  const WMPO_SOURCE = "WMPO/dependencies/opensora/opensora/models/stdit/stdit3.py · L141–205";
  const NATIVE_SOURCE = "Open-Sora/opensora/models/stdit/stdit3.py · L112–185";

  const details = {
    spatial_input: {
      kind: "WMPO · SPATIAL INPUT", title: "输入 hidden x",
      description: "每一帧包含 S 个空间 patch token。Spatial Block 会把 B 和 T 合并，让每一帧独立执行空间注意力。",
      shape: "[B,T,S,1152]", formula: "x → SpatialBlockℓ(x, t_spatial)", source: WMPO_SOURCE
    },
    spatial_norm1: {
      kind: "WMPO · PRE-NORM", title: "LayerNorm 1",
      description: "先对每个 token 的 1152 个通道归一化。affine=False 表示 LayerNorm 自己不保存固定 scale/shift，动态参数交给 Action-AdaLN。",
      shape: "[B,T,S,1152] → [B,T,S,1152]", formula: "n₁ = LN₁(x)", source: WMPO_SOURCE
    },
    spatial_adaln1: {
      kind: "WMPO · ACTION ADALN", title: "Action-AdaLN 1：Attention 前",
      description: "这不是一个独立 Action Layer。逐帧 action 产生的 shift_msa 与 scale_msa 直接作用在 Norm 1 的输出上，然后才送进 Spatial Self-Attention。",
      shape: "video [B,T,S,1152]；参数 [B,T,1,1152]，沿 S 广播",
      formula: "h_attn=(1+scale_msa)·LN₁(x)+shift_msa", source: WMPO_SOURCE
    },
    spatial_reshape_in: {
      kind: "WMPO · SHAPE TRANSFORM", title: "进入 Spatial Attention",
      description: "合并 batch 与 frame 维；因此每一帧的 S 个 patch 互相注意，不会在这里跨帧。",
      shape: "[B,T,S,C] → [B·T,S,C]", formula: "h = rearrange(h, 'b t s c → (b t) s c')", source: WMPO_SOURCE
    },
    spatial_attention: {
      kind: "WMPO · SPATIAL ATTENTION", title: "Spatial Multi-Head Self-Attention",
      description: "16 个 attention heads 在单帧内部建立 patch-to-patch 关系。Action 已通过前面的 AdaLN 改变了送入 Q/K/V 投影的 hidden。",
      shape: "[B·T,S,1152] → [B·T,S,1152]", formula: "A = MHA_spatial(h_attn)", source: WMPO_SOURCE
    },
    spatial_reshape_out: {
      kind: "WMPO · SHAPE TRANSFORM", title: "恢复视频结构",
      description: "把合并的 B·T 拆回 batch 与 frame，供逐帧 gate、residual 和之后的 Temporal Block 使用。",
      shape: "[B·T,S,C] → [B,T,S,C]", formula: "A = rearrange(A, '(b t) s c → b t s c')", source: WMPO_SOURCE
    },
    spatial_gate_msa: {
      kind: "WMPO · ACTION GATE", title: "gate_msa：控制 Attention 更新量",
      description: "第三组 action-conditioned 参数不是送入 Attention，而是逐元素缩放 Attention 的输出，再与原输入做 residual add。",
      shape: "A [B,T,S,C]；gate [B,T,1,C]", formula: "x₁ = x + gate_msa ⊙ A", source: WMPO_SOURCE
    },
    spatial_cross_off: {
      kind: "WMPO · DISABLED", title: "Text Cross-Attention 关闭",
      description: "当前 WMPO world-model forward 中，这一调用被注释。图中保留空位，是为了准确对照原生 Open-Sora 的 Block 顺序。",
      shape: "不执行；x₁ 原样向上", formula: "x₁ → x₁", source: WMPO_SOURCE
    },
    spatial_norm2: {
      kind: "WMPO · PRE-NORM", title: "LayerNorm 2",
      description: "对 Attention residual 后的 x₁ 再做一次归一化，为 MLP 子层准备输入。",
      shape: "[B,T,S,1152] → [B,T,S,1152]", formula: "n₂ = LN₂(x₁)", source: WMPO_SOURCE
    },
    spatial_adaln2: {
      kind: "WMPO · ACTION ADALN", title: "Action-AdaLN 2：MLP 前",
      description: "同一个逐帧 action condition 的第四、第五组参数，直接调制 Norm 2 的输出。它是 action 进入 Spatial Block 内部的第二处。",
      shape: "video [B,T,S,1152]；参数 [B,T,1,1152]，沿 S 广播",
      formula: "h_mlp=(1+scale_mlp)·LN₂(x₁)+shift_mlp", source: WMPO_SOURCE
    },
    spatial_mlp: {
      kind: "WMPO · FEED-FORWARD", title: "MLP / Feed-Forward Network",
      description: "对每个 token 独立进行通道混合，先扩展到 4C=4608，再投影回 C=1152。",
      shape: "1152 → 4608 → 1152", formula: "M = W₂·GELU(W₁·h_mlp)", source: WMPO_SOURCE
    },
    spatial_gate_mlp: {
      kind: "WMPO · ACTION GATE", title: "gate_mlp：控制 MLP 更新量",
      description: "第六组 action-conditioned 参数缩放 MLP 输出。随后和 x₁ 残差相加，得到 Spatial Block 的最终输出 x₂。",
      shape: "M [B,T,S,C]；gate [B,T,1,C]", formula: "x₂ = x₁ + gate_mlp ⊙ M", source: WMPO_SOURCE
    },
    spatial_condition: {
      kind: "WMPO · CONDITION SOURCE", title: "逐帧 Action + Timestep 条件",
      description: "未来帧使用各自的 7D action；4 个历史帧补 NONE action。t_block 先得到六组基础参数，再与第 ℓ 层自己的 scale_shift_table 相加。",
      shape: "actions [B,8,7] → padded [B,12,7] → params [B,12,6,1152]",
      formula: "pᵢ,ℓ = t_block(Eₜ(t)+Eₐ(aᵢ)) + Rℓ", source: WMPO_SOURCE
    },
    spatial_params_msa: {
      kind: "WMPO · PARAMETER ROUTING", title: "Attention 前的两组参数",
      description: "shift_msa 与 scale_msa 只连接到第一个 AdaLN。它们不连接到一个独立的 Action 层，也不直接加在 video token 上。",
      shape: "2 × [B,T,1,1152]", formula: "(shift_msa, scale_msa) → AdaLN₁", source: WMPO_SOURCE
    },
    spatial_params_mlp: {
      kind: "WMPO · PARAMETER ROUTING", title: "MLP 前的两组参数",
      description: "shift_mlp 与 scale_mlp 只连接到第二个 AdaLN，使相同 video hidden 在不同 action 下得到不同的 MLP 输入。",
      shape: "2 × [B,T,1,1152]", formula: "(shift_mlp, scale_mlp) → AdaLN₂", source: WMPO_SOURCE
    },

    temporal_input: {
      kind: "WMPO · TEMPORAL INPUT", title: "输入 x₂",
      description: "这是前一个 Spatial Block 已经融合 action 的输出。Temporal Block 没有再直接接收 action embedding，但 action 信息已经存在于 hidden x₂ 中。",
      shape: "[B,T,S,1152]", formula: "x₂(spatial) → TemporalBlockℓ", source: WMPO_SOURCE
    },
    temporal_norm1: {
      kind: "WMPO · PRE-NORM", title: "Temporal LayerNorm 1",
      description: "对 Spatial Block 输出做 Pre-Norm，后面的动态 scale/shift 只由全局 timestep condition 生成。",
      shape: "[B,T,S,1152] → [B,T,S,1152]", formula: "n₁ᵀ = LN₁ᵀ(x₂)", source: WMPO_SOURCE
    },
    temporal_adaln1: {
      kind: "WMPO · TIMESTEP ADALN", title: "Timestep-AdaLN 1",
      description: "Temporal Block 没有直接加 action embedding；这里使用全局 diffusion timestep 参数调制 Attention 输入。",
      shape: "hidden [B,T,S,C]；参数 [B,1,1,C]，沿 T、S 广播",
      formula: "hᵀ=(1+scale_msa(t))·LN(x₂)+shift_msa(t)", source: WMPO_SOURCE
    },
    temporal_reshape_in: {
      kind: "WMPO · SHAPE TRANSFORM", title: "进入 Temporal Attention",
      description: "合并 batch 与空间 patch 维，同一空间位置在 T 帧之间执行时间注意力。",
      shape: "[B,T,S,C] → [B·S,T,C]", formula: "h = rearrange(h, 'b t s c → (b s) t c')", source: WMPO_SOURCE
    },
    temporal_attention: {
      kind: "WMPO · CAUSAL ATTENTION", title: "Causal Temporal Self-Attention",
      description: "每个时间 token 只能查看自己与过去帧，不能查看未来帧。action 不直接注入这里，而是经 Spatial 输出间接影响它。",
      shape: "[B·S,T,1152] → [B·S,T,1152]", formula: "Aᵀ = MHA_temporal(hᵀ, is_causal=True)", source: WMPO_SOURCE
    },
    temporal_reshape_out: {
      kind: "WMPO · SHAPE TRANSFORM", title: "恢复 [B,T,S,C]",
      description: "将每个空间位置的时间序列重新组合为视频 token 网格。",
      shape: "[B·S,T,C] → [B,T,S,C]", formula: "Aᵀ = rearrange(Aᵀ, '(b s) t c → b t s c')", source: WMPO_SOURCE
    },
    temporal_gate_msa: {
      kind: "WMPO · TIMESTEP GATE", title: "Temporal gate_msa",
      description: "用 timestep gate 控制 causal attention 更新量，然后与 Temporal Block 输入做 residual add。",
      shape: "[B,T,S,C]", formula: "z₁ = x₂ + gate_msa(t) ⊙ Aᵀ", source: WMPO_SOURCE
    },
    temporal_norm2: {
      kind: "WMPO · PRE-NORM", title: "Temporal LayerNorm 2",
      description: "对 causal attention residual 的结果做第二次 Pre-Norm。",
      shape: "[B,T,S,C] → [B,T,S,C]", formula: "n₂ᵀ = LN₂ᵀ(z₁)", source: WMPO_SOURCE
    },
    temporal_adaln2: {
      kind: "WMPO · TIMESTEP ADALN", title: "Timestep-AdaLN 2",
      description: "第二组 timestep scale/shift 调制 MLP 前的归一化结果，不直接使用 action。",
      shape: "[B,T,S,C]", formula: "h_mlpᵀ=(1+scale_mlp(t))·LN(z₁)+shift_mlp(t)", source: WMPO_SOURCE
    },
    temporal_mlp: {
      kind: "WMPO · FEED-FORWARD", title: "Temporal Block MLP",
      description: "和 Spatial Block 一样，对每个 token 做 4× 通道扩展的前馈网络。",
      shape: "1152 → 4608 → 1152", formula: "Mᵀ = MLP(h_mlpᵀ)", source: WMPO_SOURCE
    },
    temporal_gate_mlp: {
      kind: "WMPO · TIMESTEP GATE", title: "Temporal gate_mlp",
      description: "缩放 Temporal MLP 的更新量并执行第二次 residual add，输出给下一组 Spatial Block。",
      shape: "[B,T,S,C]", formula: "z₂ = z₁ + gate_mlp(t) ⊙ Mᵀ", source: WMPO_SOURCE
    },
    temporal_condition: {
      kind: "WMPO · GLOBAL CONDITION", title: "Temporal 只接收全局 timestep condition",
      description: "Temporal Block 的六组 AdaLN 参数由 diffusion timestep 生成，并与当前层的 scale_shift_table 相加。Action 的直接注入仅发生在 Spatial Block。",
      shape: "[B,1,6,1152]，广播到所有 T 与 S", formula: "pᵀℓ = t_block(Eₜ(t)) + Rᵀℓ", source: WMPO_SOURCE
    },

    native_input: {
      kind: "OPEN-SORA · INPUT", title: "原生 Spatial Block 输入",
      description: "原生 STDiT3 的 Spatial Block 同样执行帧内 attention，但没有 WMPO 的逐帧 action condition。",
      shape: "[B,T·S,1152]（Block 内按 frame 重排）", formula: "x → SpatialBlockℓ", source: NATIVE_SOURCE
    },
    native_norm1: {
      kind: "OPEN-SORA · PRE-NORM", title: "LayerNorm 1",
      description: "无 affine 的归一化，动态 scale/shift 由 timestep 与 fps 条件提供。",
      shape: "[B,T,S,C] → [B,T,S,C]", formula: "n₁ = LN₁(x)", source: NATIVE_SOURCE
    },
    native_adaln1: {
      kind: "OPEN-SORA · ADALN", title: "AdaLN 1：timestep + fps",
      description: "原生模型的第一处 AdaLN 不含机器人 action；它使用全局生成条件调制 Spatial Attention 输入。",
      shape: "hidden [B,T,S,C]；condition [B,1,6,C]", formula: "h=(1+scale_msa(t,fps))·LN₁(x)+shift_msa(t,fps)", source: NATIVE_SOURCE
    },
    native_attention: {
      kind: "OPEN-SORA · SPATIAL ATTENTION", title: "Spatial Multi-Head Self-Attention",
      description: "在每帧内部让空间 patch 互相注意，规格同样是 C=1152、16 heads。",
      shape: "[B·T,S,1152] → [B·T,S,1152]", formula: "A = MHA_spatial(h)", source: NATIVE_SOURCE
    },
    native_gate_msa: {
      kind: "OPEN-SORA · GATE", title: "原生 gate_msa",
      description: "由 timestep+fps condition 控制 self-attention 更新量，然后执行 residual add。",
      shape: "[B,T,S,C]", formula: "x₁ = x + gate_msa ⊙ A", source: NATIVE_SOURCE
    },
    native_cross: {
      kind: "OPEN-SORA · TEXT CONDITION", title: "Text Cross-Attention：原生开启",
      description: "video hidden 作为 Query，caption embedding 作为 Key/Value。这个子层在 WMPO 当前 world-model forward 中被注释掉。",
      shape: "Q [B,T·S,C]；KV [B,L_text,C]", formula: "x₂ = x₁ + CrossAttn(Q=x₁, KV=y)", source: NATIVE_SOURCE
    },
    native_norm2: {
      kind: "OPEN-SORA · PRE-NORM", title: "LayerNorm 2",
      description: "对 text cross-attention 后的结果做 MLP 前归一化。",
      shape: "[B,T,S,C] → [B,T,S,C]", formula: "n₂ = LN₂(x₂)", source: NATIVE_SOURCE
    },
    native_adaln2: {
      kind: "OPEN-SORA · ADALN", title: "AdaLN 2：MLP 前",
      description: "使用 timestep+fps 生成的 shift_mlp 与 scale_mlp 调制第二次归一化结果。",
      shape: "[B,T,S,C]", formula: "h_mlp=(1+scale_mlp)·LN₂(x₂)+shift_mlp", source: NATIVE_SOURCE
    },
    native_mlp: {
      kind: "OPEN-SORA · FEED-FORWARD", title: "原生 MLP",
      description: "逐 token 的通道前馈网络，扩展倍率为 4。",
      shape: "1152 → 4608 → 1152", formula: "M = MLP(h_mlp)", source: NATIVE_SOURCE
    },
    native_gate_mlp: {
      kind: "OPEN-SORA · GATE", title: "原生 gate_mlp",
      description: "控制 MLP 更新量，并通过第二条 residual skip 得到 Block 输出。",
      shape: "[B,T,S,C]", formula: "x₃ = x₂ + gate_mlp ⊙ M", source: NATIVE_SOURCE
    },
    native_condition: {
      kind: "OPEN-SORA · GLOBAL CONDITION", title: "Timestep + FPS 条件",
      description: "原生 STDiT3 使用 timestep 和 fps embedding 生成六组 AdaLN 参数；没有 ActionEncoder 和 NONE action padding。",
      shape: "[B,1,6,1152]", formula: "pℓ = t_block(Eₜ(t)+E_fps(fps)) + Rℓ", source: NATIVE_SOURCE
    }
  };

  const viewMeta = {
    spatial: { kind: "WMPO SPATIAL", title: "Action-conditioned Spatial Transformer Block", defaultKey: "spatial_adaln1" },
    temporal: { kind: "WMPO TEMPORAL", title: "Causal Temporal Transformer Block", defaultKey: "temporal_attention" },
    native: { kind: "OPEN-SORA NATIVE", title: "Native Spatial Transformer Block", defaultKey: "native_cross" }
  };

  const tabs = [...document.querySelectorAll(".view-tab")];
  const panels = [...document.querySelectorAll(".diagram-view")];
  const interactiveNodes = [...document.querySelectorAll("[data-key]")];
  const viewKind = document.getElementById("view-kind");
  const viewTitle = document.getElementById("view-title");
  const moduleKind = document.getElementById("module-kind");
  const moduleTitle = document.getElementById("module-title");
  const moduleDescription = document.getElementById("module-description");
  const moduleShape = document.getElementById("module-shape");
  const moduleFormula = document.getElementById("module-formula");
  const moduleSource = document.getElementById("module-source");
  const flowButtons = [...document.querySelectorAll(".flow-button")];
  const diagramStage = document.getElementById("diagram-stage");
  const backwardOverlay = document.getElementById("backward-overlay");

  function showDetail(key) {
    const item = details[key];
    if (!item) return;
    interactiveNodes.forEach((node) => node.classList.toggle("is-selected", node.dataset.key === key && !node.closest("[hidden]")));
    moduleKind.textContent = item.kind;
    moduleTitle.textContent = item.title;
    moduleDescription.textContent = item.description;
    moduleShape.textContent = item.shape;
    moduleFormula.textContent = item.formula;
    moduleSource.textContent = item.source;
  }

  function switchView(view) {
    const meta = viewMeta[view];
    if (!meta) return;
    tabs.forEach((tab) => {
      const active = tab.dataset.view === view;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-pressed", String(active));
    });
    panels.forEach((panel) => {
      const active = panel.dataset.viewPanel === view;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });
    viewKind.textContent = meta.kind;
    viewTitle.textContent = meta.title;
    showDetail(meta.defaultKey);
  }

  tabs.forEach((tab) => tab.addEventListener("click", () => switchView(tab.dataset.view)));

  function setFlowMode(mode) {
    const backward = mode === "backward";
    diagramStage.classList.toggle("backward-mode", backward);
    backwardOverlay.hidden = !backward;
    flowButtons.forEach((button) => {
      const active = button.dataset.flow === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (backward) {
      moduleKind.textContent = "BACKWARD · GRADIENT";
      moduleTitle.textContent = "Loss → gate / AdaLN → t_block → ActionEncoder";
      moduleDescription.textContent = "Attention 与 MLP 两条梯度支路都回到六组调制参数，再经逐帧 condition 返回 ActionEncoder；Spatial Block 参数也同时获得梯度。";
      moduleShape.textContent = "∂L/∂x₂ → ∂L/∂pᵢ → ∂L/∂θ_action";
      moduleFormula.textContent = "∂L/∂cᵢ = (∂L/∂pᵢ) · (∂t_block/∂cᵢ)";
      moduleSource.textContent = "PyTorch autograd · 对应 WMPO forward L141–205, L448–487";
    } else {
      const activeView = tabs.find((tab) => tab.classList.contains("is-active"))?.dataset.view || "spatial";
      showDetail(viewMeta[activeView].defaultKey);
    }
  }

  flowButtons.forEach((button) => button.addEventListener("click", () => setFlowMode(button.dataset.flow)));

  interactiveNodes.forEach((node) => {
    node.addEventListener("click", () => showDetail(node.dataset.key));
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showDetail(node.dataset.key);
      }
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    const current = tabs.findIndex((tab) => tab.classList.contains("is-active"));
    const delta = event.key === "ArrowRight" ? 1 : -1;
    const next = (current + delta + tabs.length) % tabs.length;
    switchView(tabs[next].dataset.view);
    tabs[next].focus();
  });

  switchView("spatial");
  setFlowMode("forward");
})();
