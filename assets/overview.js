(function () {
  "use strict";

  const CANVAS_WIDTH = 2700;
  const CANVAS_HEIGHT = 1700;
  const OPEN_SORA = "Open-Sora/opensora/models/stdit/stdit3.py";
  const WMPO = "WMPO/dependencies/opensora/opensora/models/stdit/stdit3.py";

  const nodeLayer = document.getElementById("node-layer");
  const connectorLayer = document.getElementById("connector-layer");
  const canvas = document.getElementById("diagram-canvas");
  const viewport = document.getElementById("diagram-viewport");
  const zoomValue = document.getElementById("zoom-value");
  const minimapWindow = document.getElementById("minimap-window");
  const viewButtons = Array.from(document.querySelectorAll(".view-button"));
  const chapterButtons = Array.from(document.querySelectorAll(".chapter-button"));

  const inspector = {
    kind: document.getElementById("inspector-kind"),
    title: document.getElementById("inspector-title"),
    description: document.getElementById("inspector-description"),
    input: document.getElementById("inspector-input"),
    output: document.getElementById("inspector-output"),
    formula: document.getElementById("inspector-formula"),
    meaning: document.getElementById("inspector-meaning"),
    source: document.getElementById("inspector-source"),
    classification: document.getElementById("inspector-class"),
    count: document.getElementById("inspector-count")
  };

  const state = {
    scale: 0.5,
    panX: 0,
    panY: 0,
    dragging: false,
    pointerId: null,
    startX: 0,
    startY: 0,
    startPanX: 0,
    startPanY: 0,
    view: "compare",
    selectedId: null
  };

  const nodes = [];
  const nodeMap = new Map();
  const connections = [];

  function addNode(config) {
    nodes.push(config);
    nodeMap.set(config.id, config);
    return config;
  }

  function commonMeta(overrides) {
    return Object.assign({
      description: "",
      input: "—",
      output: "—",
      formula: "—",
      meaning: "",
      source: "—",
      classification: "共享骨架",
      className: "shared",
      count: ""
    }, overrides);
  }

  function addCoreNodes() {
    addNode({
      id: "n_input", lane: "native", x: 110, y: 190, w: 155, h: 92,
      type: "INPUT", label: "Video latent x", shape: "[B,16,T,H,W]", category: "shared",
      meta: commonMeta({
        description: "原生 Open-Sora v1.3 的 STDiT3 接收经过 VAE 压缩的 16 通道视频 latent。",
        input: "VAE latent [B,16,T,H,W]", output: "同形状，转为模型 dtype",
        formula: "x = x.to(dtype)", meaning: "STDiT3 不直接处理 RGB；空间和时间尺寸都是 latent 尺度。",
        source: OPEN_SORA + " · L203–228, L424–445"
      })
    });
    addNode({
      id: "n_patch", lane: "native", x: 310, y: 190, w: 155, h: 92,
      type: "TOKENIZE", label: "PatchEmbed3D", shape: "(1,2,2) → C=1152", category: "shared",
      meta: commonMeta({
        description: "用 1×2×2 的 3D patch 投影把 latent 视频变成 token。时间 patch 为 1，不降低 T。",
        input: "[B,16,T,H,W]", output: "[B,T·S,1152]",
        formula: "S=(H/2)·(W/2), N=T·S", meaning: "每个 token 对应一帧中的 2×2 latent patch。",
        source: OPEN_SORA + " · L277–291, L495–503"
      })
    });
    addNode({
      id: "n_pos", lane: "native", x: 510, y: 190, w: 155, h: 92,
      type: "POSITION", label: "+ 2D Pos Embed", shape: "[B,T,S,1152]", category: "condition",
      meta: commonMeta({
        description: "PositionEmbedding2D 按 H×W 生成空间位置编码，并广播到每一帧。",
        input: "[B,T,S,1152]", output: "[B,T,S,1152]",
        formula: "x_tokens ← x_tokens + pos(H,W,scale)", meaning: "Spatial Attention 需要知道 patch 在画面中的位置。",
        source: OPEN_SORA + " · L277–285, L446–471, L502–511"
      })
    });

    addNode({
      id: "n_time", lane: "native", x: 110, y: 330, w: 140, h: 70,
      type: "CONDITION", label: "Timestep", shape: "[B] → [B,1152]", category: "condition",
      meta: commonMeta({
        description: "扩散或 Rectified Flow 的时间标量通过 TimestepEmbedder 变成 1152 维条件。",
        input: "timestep [B]", output: "E_t [B,1152]",
        formula: "E_t = TimestepEmbedder(t)", meaning: "告诉网络当前输入处在怎样的噪声强度。",
        source: OPEN_SORA + " · L293–298, L473–477"
      })
    });
    addNode({
      id: "n_fps", lane: "native", x: 285, y: 330, w: 140, h: 70,
      type: "CONDITION", label: "FPS Embed", shape: "[B] → [B,1152]", category: "condition",
      meta: commonMeta({
        description: "原生模型用 SizeEmbedder 编码 fps，并与 timestep embedding 相加。",
        input: "fps [B]", output: "E_fps [B,1152]",
        formula: "c_global = E_t + E_fps", meaning: "同一动作在不同帧率下对应不同的视觉速度。",
        source: OPEN_SORA + " · L293–298, L473–483"
      })
    });
    addNode({
      id: "n_tblock", lane: "native", x: 470, y: 330, w: 170, h: 70,
      type: "ADALN PARAMS", label: "t_block", shape: "1152 → 6×1152", category: "condition",
      meta: commonMeta({
        description: "SiLU + Linear 把全局条件投影成 attention/MLP 两组 shift、scale、gate。",
        input: "[B,1152]", output: "[B,6912]",
        formula: "(βa,γa,ga,βm,γm,gm)=t_block(c)", meaning: "这些值不是 token，而是控制每个 Transformer Block 的动态参数。",
        source: OPEN_SORA + " · L295–298, L473–483"
      })
    });
    addNode({
      id: "n_text", lane: "native", x: 110, y: 420, w: 140, h: 64,
      type: "CONDITION", label: "Text y", shape: "[B,1,N,4096]", category: "condition",
      meta: commonMeta({
        description: "T5 文本特征是原生视频生成模型的语义条件。",
        input: "[B,1,N,4096]", output: "送入 CaptionEmbedder",
        formula: "y_text = T5(prompt)", meaning: "文本说明模型应该生成什么内容。",
        source: OPEN_SORA + " · L299–305, L409–422"
      })
    });
    addNode({
      id: "n_caption", lane: "native", x: 285, y: 420, w: 170, h: 64,
      type: "CONDITION", label: "CaptionEmbedder", shape: "[B,N,1152]", category: "condition",
      meta: commonMeta({
        description: "将 4096 维文本特征映射到 STDiT3 hidden size，并整理有效 token。",
        input: "[B,1,N,4096]", output: "[B,N,1152]",
        formula: "y = CaptionEmbedder(y)", meaning: "每一个 Spatial/Temporal Block 都可通过 Cross-Attention 读取文本。",
        source: OPEN_SORA + " · L299–305, L485–493"
      })
    });

    addNode({
      id: "n_final", lane: "native", x: 2280, y: 190, w: 155, h: 92,
      type: "OUTPUT HEAD", label: "T2IFinalLayer", shape: "1152 → 4·Cout", category: "output",
      meta: commonMeta({
        description: "最终 LayerNorm + timestep AdaLN + Linear，把每个 token 投影为一个 1×2×2 输出 patch。",
        input: "[B,T·S,1152]", output: "[B,T·S,4·Cout]",
        formula: "patch = Linear(AdaLN(LN(x),t))", meaning: "主干特征在这里转成 scheduler 所需的 latent 预测。",
        source: OPEN_SORA + " · L352–355, L574–606"
      })
    });
    addNode({
      id: "n_output", lane: "native", x: 2480, y: 190, w: 155, h: 92,
      type: "UNPATCHIFY", label: "Output latent", shape: "[B,32,T,H,W]", category: "output",
      meta: commonMeta({
        description: "unpatchify 恢复时空网格。默认 in_channels=16 且 pred_sigma=true，因此 Cout=32。",
        input: "[B,T·S,4·Cout]", output: "[B,Cout,T,H,W]",
        formula: "Cout = 2·Cin = 32", meaning: "输出仍在 latent 空间，之后由调度器解释并最终交给 VAE 解码。",
        source: OPEN_SORA + " · L261–264, L582–606"
      })
    });

    addNode({
      id: "w_input", lane: "wmpo", x: 110, y: 650, w: 155, h: 92,
      type: "INPUT", label: "Video latent x", shape: "[B,4,12,H,W]", category: "drift",
      meta: commonMeta({
        description: "WMPO 当前 fork 接收 4 通道、12 帧 latent；12=4 帧历史 + 8 帧动作预测窗口。",
        input: "[B,4,12,H,W]", output: "同形状，转为模型 dtype",
        formula: "T=To+Ta=4+8=12", meaning: "4 通道属于 fork/VAE 版本差异，不是 action conditioning 的核心创新。",
        source: WMPO + " · L212–236, L412–423", classification: "fork / 版本差异", className: "drift"
      })
    });
    addNode({
      id: "w_patch", lane: "wmpo", x: 310, y: 650, w: 155, h: 92,
      type: "TOKENIZE", label: "PatchEmbed3D", shape: "(1,2,2) → C=1152", category: "shared",
      meta: commonMeta({
        description: "WMPO 保留原生 PatchEmbed3D 设计，但在进入 blocks 时维持显式 [B,T,S,C] 布局。",
        input: "[B,4,12,H,W]", output: "[B,12,S,1152]",
        formula: "x = rearrange(x,\"B (T S) C → B T S C\")", meaning: "显式 T 轴让逐帧 action AdaLN 容易广播。",
        source: WMPO + " · L284–293, L472–482"
      })
    });
    addNode({
      id: "w_pos", lane: "wmpo", x: 510, y: 650, w: 155, h: 92,
      type: "POSITION", label: "+ 2D Pos Embed", shape: "[B,12,S,1152]", category: "shared",
      meta: commonMeta({
        description: "与原生相同，为每帧 patch 添加二维空间位置编码。",
        input: "[B,12,S,1152]", output: "[B,12,S,1152]",
        formula: "x_tokens ← x_tokens + pos(H,W,scale)", meaning: "共享原生空间表征能力。",
        source: WMPO + " · L284–293, L419–446"
      })
    });

    addNode({
      id: "w_action", lane: "wmpo", x: 110, y: 790, w: 140, h: 70,
      type: "ACTION", label: "Actions", shape: "[B,8,7]", category: "action",
      meta: commonMeta({
        description: "8 个未来机器人动作，每个动作通常包含位姿增量与夹爪状态等 7 个连续值。",
        input: "[B,8,7]", output: "送入 ActionEncoder",
        formula: "a_i∈ℝ⁷, i=1…8", meaning: "这是 WMPO 世界模型区别于文本视频模型的控制信号。",
        source: WMPO + " · L37–72, L412–423", classification: "WMPO 核心设计", className: "action"
      })
    });
    addNode({
      id: "w_action_encoder", lane: "wmpo", x: 285, y: 790, w: 155, h: 70,
      type: "ACTION", label: "ActionEncoder", shape: "7→4608→1152", category: "action",
      meta: commonMeta({
        description: "每个连续 action 独立经过 Linear、SiLU、Linear，映射到 hidden size。",
        input: "[B,8,7]", output: "[B,8,1152]",
        formula: "E_a(a)=W₂·SiLU(W₁a)", meaning: "把低维机器人控制量翻译成 Transformer 可用的条件向量。",
        source: WMPO + " · L37–62", classification: "WMPO 核心设计", className: "action"
      })
    });
    addNode({
      id: "w_none", lane: "wmpo", x: 475, y: 790, w: 155, h: 70,
      type: "ACTION", label: "Prepend NONE", shape: "[B,12,1152]", category: "action",
      meta: commonMeta({
        description: "可学习 none_action 重复 4 次并放在 8 个 action embedding 前，对齐 12 帧视频。",
        input: "[B,8,1152]", output: "[B,12,1152]",
        formula: "[NONE×4, E_a(a₁)…E_a(a₈)]", meaning: "历史 observation 帧没有对应的未来动作，因此使用 NONE 占位。",
        source: WMPO + " · L51–71", classification: "WMPO 核心设计", className: "action"
      })
    });
    addNode({
      id: "w_spatial_cond", lane: "wmpo", x: 665, y: 790, w: 170, h: 70,
      type: "ACTION-ADALN", label: "t_s + action", shape: "[B·12,1152]", category: "action",
      meta: commonMeta({
        description: "timestep embedding 被复制到每帧，再与该帧 action embedding 相加。",
        input: "E_t [B,1152] + E_a [B,12,1152]", output: "c_spatial [B·12,1152]",
        formula: "c_s[b,i]=E_t(t_b)+E_a(a_b,i)", meaning: "ActionEncoder 不直接加到 video token，而是改变 Spatial AdaLN 条件。",
        source: WMPO + " · L448–456", classification: "WMPO 核心设计", className: "action"
      })
    });
    addNode({
      id: "w_ts_block", lane: "wmpo", x: 665, y: 885, w: 210, h: 64,
      type: "ACTION-ADALN PARAMS", label: "逐帧参数 · 广播到 28 层", shape: "[B·12,6×1152]", category: "action",
      meta: commonMeta({
        description: "逐帧 spatial condition 被投影为 6C AdaLN 参数，并复用于全部 28 个 Spatial Block。",
        input: "[B·12,1152]", output: "[B·12,6·1152]",
        formula: "t_s_mlp=t_block(c_s)", meaning: "这是送入 Spatial Block 内部两处 AdaLN 的参数，不是 token，也不是独立的 Action Layer。",
        source: WMPO + " · L295–298, L452–456", classification: "WMPO 核心设计", className: "action", count: "→ 28 Spatial Blocks"
      })
    });
    addNode({
      id: "w_time", lane: "wmpo", x: 110, y: 885, w: 140, h: 64,
      type: "CONDITION", label: "Timestep", shape: "[B]→[B,1152]", category: "condition",
      meta: commonMeta({
        description: "全局 timestep 同时是 spatial 条件的基底，也是 Temporal Block 的唯一直接 AdaLN 条件。",
        input: "[B]", output: "[B,1152]",
        formula: "t=E_t(timestep)", meaning: "Temporal Block 不直接接收 action embedding。",
        source: WMPO + " · L448–462"
      })
    });
    addNode({
      id: "w_tt_block", lane: "wmpo", x: 285, y: 885, w: 155, h: 64,
      type: "ADALN PARAMS", label: "t_t_block", shape: "[B,6912]", category: "condition",
      meta: commonMeta({
        description: "全局 timestep 经过同一个 t_block，生成 Temporal Block 使用的 6C 参数。",
        input: "[B,1152]", output: "[B,6912]",
        formula: "t_t_mlp=t_block(t)", meaning: "action 通过前一个 Spatial Block 的 hidden state 间接进入 Temporal Block。",
        source: WMPO + " · L452–487", count: "→ 28 Temporal Blocks"
      })
    });
    addNode({
      id: "w_text_off", lane: "wmpo", x: 475, y: 885, w: 155, h: 64,
      type: "DISABLED", label: "Text Cross-Attn", shape: "forward 中注释", category: "removed",
      meta: commonMeta({
        description: "cross_attn 模块仍被构造，但当前 WMPO STDiT3Block.forward 中调用被注释。",
        input: "原计划 Q=x, KV=text", output: "当前无文本残差",
        formula: "# x = x + self.cross_attn(...)", meaning: "当前 world-model 路径主要依靠历史视觉与 action，而非文本控制。",
        source: WMPO + " · L112–118, L183–185", classification: "WMPO 核心设计", className: "action"
      })
    });

    addNode({
      id: "w_final", lane: "wmpo", x: 2280, y: 650, w: 155, h: 92,
      type: "OUTPUT HEAD", label: "T2IFinalLayer", shape: "1152 → 4·Cout", category: "output",
      meta: commonMeta({
        description: "动作影响已写入 hidden state；Final Layer 本身仍由全局 timestep 调制。",
        input: "[B,12,S,1152]", output: "[B,12,S,4·Cout]",
        formula: "patch=FinalLayer(x_action,t)", meaning: "action 无需再次直接进入输出头。",
        source: WMPO + " · L345–349, L495–527"
      })
    });
    addNode({
      id: "w_output", lane: "wmpo", x: 2480, y: 650, w: 155, h: 92,
      type: "UNPATCHIFY", label: "Output latent", shape: "[B,8,12,H,W]", category: "output",
      meta: commonMeta({
        description: "显式 [B,T,S,C] token 被 rearrange 回视频 latent。默认 Cin=4 且 pred_sigma=true，Cout=8。",
        input: "[B,12,S,4·Cout]", output: "[B,Cout,12,H,W]",
        formula: "Cout=2·Cin=8", meaning: "之后由 scheduler 使用，并最终通过共享 VAE 解码成视频。",
        source: WMPO + " · L268–270, L503–527", classification: "fork / 版本差异", className: "drift"
      })
    });
  }

  function blockMeta(lane, kind, index) {
    const isNative = lane === "native";
    const isSpatial = kind === "spatial";
    const layerName = (isSpatial ? "Spatial Block " : "Temporal Block ") + String(index).padStart(2, "0");
    if (isNative && isSpatial) {
      return commonMeta({
        description: "第 " + index + " 个原生 Spatial Block：全局 AdaLN、帧内 Self-Attention、文本 Cross-Attention、MLP。",
        input: "[B,T·S,1152]", output: "[B,T·S,1152]",
        formula: "x←x+g·SA(AdaLN(x)); x←x+CrossAttn(x,y); x←x+g·MLP(AdaLN(x))",
        meaning: "同一帧的 S 个 patch 相互建模；每层拥有独立 attention、MLP 与 scale_shift_table。",
        source: OPEN_SORA + " · L96–197", count: layerName
      });
    }
    if (isNative) {
      return commonMeta({
        description: "第 " + index + " 个原生 Temporal Block：固定空间位置，对 T 帧进行非 causal Self-Attention。",
        input: "[B,T·S,1152]", output: "[B,T·S,1152]",
        formula: "[B,T·S,C]→[(B·S),T,C]→Attention→[B,T·S,C]",
        meaning: "每个空间位置沿完整时间序列交换信息；源码未显式设置 causal mask。",
        source: OPEN_SORA + " · L128–177, L328–349", count: layerName
      });
    }
    if (isSpatial) {
      return commonMeta({
        description: "第 " + index + " 个 WMPO Spatial Block：使用逐帧 action 生成的 AdaLN 参数控制 attention 与 MLP。",
        input: "[B,12,S,1152]", output: "[B,12,S,1152]",
        formula: "x[b,t]←x[b,t]+g(a[b,t])·SA(AdaLN(x[b,t],a[b,t]))",
        meaning: "action condition 在 28 个 Spatial Block 中反复注入，而不是只加在输入一次。",
        source: WMPO + " · L130–205", classification: "WMPO 核心设计", className: "action", count: layerName
      });
    }
    return commonMeta({
      description: "第 " + index + " 个 WMPO Temporal Block：沿 T 维进行 causal Self-Attention，直接 AdaLN 条件仅为 timestep。",
      input: "[B,12,S,1152]", output: "[B,12,S,1152]",
      formula: "Temporal_i(CausalAttention(Spatial_i(x,action)))",
      meaning: "action 先改变空间特征，再通过 causal 时间注意力向未来传播。",
      source: WMPO + " · L90–118, L141–171", classification: "WMPO 核心设计", className: "action", count: layerName
    });
  }

  function addBackbone(lane, y, prefix) {
    const startX = 930;
    const stepX = 180;
    const displayed = [1, 2, 3, "gap", 26, 27, 28];
    let previousId = null;
    displayed.forEach(function (index, position) {
      const x = startX + position * stepX;
      if (index === "gap") {
        const gapId = prefix + "_gap";
        addNode({
          id: gapId, lane: lane, x: x, y: y + 23, w: 110, h: 84,
          type: "×22 PAIRS", label: "⋯", shape: "Blocks 4–25",
          category: lane === "wmpo" ? "action" : "shared", mini: true, omission: true,
          meta: commonMeta({
            description: "为了保持总览清晰，这里省略显示第 4–25 对 Block；运行时它们仍逐对执行。",
            input: lane === "wmpo" ? "[B,12,S,1152]" : "[B,T·S,1152]",
            output: lane === "wmpo" ? "[B,12,S,1152]" : "[B,T·S,1152]",
            formula: "[Spatial_i → Temporal_i], i=4…25",
            meaning: "省略的是画法，不是网络计算。STDiT3-XL/2 仍有完整 28 个 Spatial 和 28 个 Temporal Block。",
            source: lane === "wmpo" ? WMPO + " · L307–344, L484–487" : OPEN_SORA + " · L307–350, L513–547",
            classification: lane === "wmpo" ? "WMPO 主干" : "共享骨架",
            className: lane === "wmpo" ? "action" : "shared",
            count: "22 pairs omitted visually"
          })
        });
        if (previousId) connections.push({ from: previousId, to: gapId, className: lane });
        previousId = gapId;
        return;
      }
      const spatialId = prefix + "_s_" + index;
      const temporalId = prefix + "_t_" + index;
      addNode({
        id: spatialId, lane: lane, x: x, y: y, w: 110, h: 54,
        type: "SPATIAL", label: "#" + index, shape: "Self-Attn + MLP", category: lane === "wmpo" ? "action" : "shared",
        mini: true, meta: blockMeta(lane, "spatial", index)
      });
      addNode({
        id: temporalId, lane: lane, x: x, y: y + 76, w: 110, h: 54,
        type: "TEMPORAL", label: "#" + index, shape: lane === "wmpo" ? "Causal Self-Attn" : "Self-Attn + MLP",
        category: lane === "wmpo" ? "action" : "shared",
        mini: true, meta: blockMeta(lane, "temporal", index)
      });
      if (previousId) connections.push({ from: previousId, to: spatialId, className: lane, stepped: true });
      connections.push({ from: spatialId, to: temporalId, className: lane, vertical: true });
      previousId = temporalId;
    });
  }

  function addExplodedNodes() {
    const panels = [
      { lane: "native", x: 70, y: 1100, w: 620, h: 430, title: "原生 Spatial Block", subtitle: "全局 AdaLN + 文本 Cross-Attention" },
      { lane: "wmpo", x: 720, y: 1100, w: 1220, h: 430, title: "WMPO Spatial Block 内部的 Action-AdaLN", subtitle: "六个参数直接注入 Self-Attention 与 MLP 两条残差支路；不存在独立 Action Layer" },
      { lane: "both", x: 1970, y: 1100, w: 660, h: 430, title: "Temporal Block 对比", subtitle: "输入是 Spatial 输出 x₂；Action-AdaLN 参数不直接进入" }
    ];
    panels.forEach(function (panel) {
      const div = document.createElement("div");
      div.className = "exploded-panel";
      div.dataset.lane = panel.lane;
      div.style.left = panel.x + "px";
      div.style.top = panel.y + "px";
      div.style.width = panel.w + "px";
      div.style.height = panel.h + "px";
      const title = document.createElement("h3");
      title.textContent = panel.title;
      const subtitle = document.createElement("p");
      subtitle.textContent = panel.subtitle;
      div.appendChild(title);
      div.appendChild(subtitle);
      nodeLayer.appendChild(div);
    });

    [
      { x: 745, text: "① Self-Attention 前调制" },
      { x: 1360, text: "② MLP 前调制" }
    ].forEach(function (caption) {
      const div = document.createElement("div");
      div.className = "branch-caption";
      div.dataset.lane = "wmpo";
      div.style.left = caption.x + "px";
      div.style.top = "1355px";
      div.textContent = caption.text;
      nodeLayer.appendChild(div);
    });

    const nativeDetail = [
      ["nd_adaln1", 90, "AdaLN₁", "[B,T·S,C]", "condition"],
      ["nd_attn", 210, "Spatial SA", "[(B·T),S,C]", "shared"],
      ["nd_gate", 330, "Gate + Residual", "[B,T·S,C]", "shared"],
      ["nd_cross", 450, "Text Cross-Attn", "Q=x, KV=y", "condition"],
      ["nd_mlp", 570, "AdaLN₂ + MLP", "1152→4608→1152", "shared"]
    ];
    nativeDetail.forEach(function (item) {
      addNode({
        id: item[0], lane: "native", x: item[1], y: 1250, w: 100, h: 90,
        type: "SPATIAL", label: item[2], shape: item[3], category: item[4],
        meta: item[0] === "nd_cross" ? commonMeta({
          description: "原生每个 Block 都执行文本 Cross-Attention。",
          input: "Q=x; KV=y_text", output: "[B,T·S,C]",
          formula: "x←x+CrossAttn(x,y,mask)", meaning: "文本条件持续注入生成语义。",
          source: OPEN_SORA + " · L157–177"
        }) : blockMeta("native", "spatial", 1)
      });
    });

    const actionConditionDetail = [
      ["wa_action", 745, 105, "Action aᵢ", "[B,8,7]", "action", nodeMap.get("w_action").meta],
      ["wa_encode", 865, 120, "ActionEncoder", "7→4608→1152", "action", nodeMap.get("w_action_encoder").meta],
      ["wa_none", 1000, 120, "NONE ×4", "[B,12,1152]", "action", nodeMap.get("w_none").meta],
      ["wa_repeat", 1135, 120, "repeat(t)", "[B,12,1152]", "condition", nodeMap.get("w_time").meta],
      ["wa_sum", 1270, 100, "逐帧相加", "t + Eₐ(aᵢ)", "action", nodeMap.get("w_spatial_cond").meta],
      ["wa_tblock", 1385, 120, "t_block", "1152→6912", "action", nodeMap.get("w_ts_block").meta],
      ["wa_reshape", 1520, 150, "reshape", "[B,T,6,C]", "action", commonMeta({
        description: "把 [B·T,6C] 恢复成 [B,T,6,C]，让每一帧拥有自己的一组六类 AdaLN 控制量。",
        input: "[B·T,6C]", output: "[B,T,6,C]",
        formula: "t_s_mlp.reshape(B,T,6,C)", meaning: "T 维不再被折叠，因此 action 与视频帧一一对应。",
        source: WMPO + " · L141–149", classification: "WMPO 核心设计", className: "action"
      })],
      ["wa_chunk", 1685, 225, "chunk(6)", "shift / scale / gate ×2", "action", commonMeta({
        description: "6C 被切成 attention 与 MLP 各自的 shift、scale、gate。",
        input: "[B,T,6,C]", output: "6 × [B,T,1,C]",
        formula: "(βa,γa,ga,βm,γm,gm)=chunk(t,6)", meaning: "shift/scale 改变归一化特征，gate 控制两个残差分支的强度。",
        source: WMPO + " · L141–154", classification: "WMPO 核心设计", className: "action"
      })]
    ];
    actionConditionDetail.forEach(function (item) {
      addNode({
        id: item[0], lane: "wmpo", x: item[1], y: 1180, w: item[2], h: 82,
        type: "CONDITION PARAMS", label: item[3], shape: item[4], category: item[5], meta: item[6]
      });
    });

    const attnModMeta = commonMeta({
      description: "Self-Attention 前的 Action-AdaLN。shift_msa 与 scale_msa 直接调制 LN₁(x)，并不产生新的 Action Layer。",
      input: "LN₁(x) [B,T,S,C] + shift_msa/scale_msa [B,T,1,C]", output: "h_attn [B,T,S,C]",
      formula: "h_attn=(1+scale_msa)·LN₁(x)+shift_msa", meaning: "每帧 action 参数沿该帧所有 S 个空间 patch 广播。",
      source: WMPO + " · L141–181", classification: "WMPO 核心设计", className: "action"
    });
    const mlpModMeta = commonMeta({
      description: "MLP 前的第二处 Action-AdaLN。它使用独立的 shift_mlp 与 scale_mlp 调制 LN₂(x₁)。",
      input: "LN₂(x₁) [B,T,S,C] + shift_mlp/scale_mlp [B,T,1,C]", output: "h_mlp [B,T,S,C]",
      formula: "h_mlp=(1+scale_mlp)·LN₂(x₁)+shift_mlp", meaning: "Attention 与 MLP 两条支路使用不同的调制参数。",
      source: WMPO + " · L186–205", classification: "WMPO 核心设计", className: "action"
    });
    const gateMsaMeta = commonMeta({
      description: "Spatial Self-Attention 的输出乘以 gate_msa 后才加回残差。",
      input: "SA(h_attn) + gate_msa", output: "x₁ [B,T,S,C]",
      formula: "x₁=x+gate_msa·SpatialSA(h_attn)", meaning: "gate_msa 控制 Attention 残差支路强度。",
      source: WMPO + " · L163–181", classification: "WMPO 核心设计", className: "action"
    });
    const gateMlpMeta = commonMeta({
      description: "MLP 输出乘以 gate_mlp 后加回 x₁，得到 Spatial Block 的最终输出 x₂。",
      input: "MLP(h_mlp) + gate_mlp", output: "x₂ [B,T,S,C]",
      formula: "x₂=x₁+gate_mlp·MLP(h_mlp)", meaning: "只有更新后的 x₂ 继续进入 Temporal Block；六组 Action-AdaLN 参数不会作为 token 进入 Temporal Block。",
      source: WMPO + " · L186–205", classification: "WMPO 核心设计", className: "action"
    });

    const actionFeatureDetail = [
      ["wa_shift_msa", 920, 1290, 190, 60, "ATTN PARAMS", "shift_msa + scale_msa", "→ AdaLN₁", "action", attnModMeta],
      ["wa_gate_msa", 1180, 1290, 140, 60, "ATTN PARAM", "gate_msa", "→ Attention residual", "action", gateMsaMeta],
      ["wa_cross_off", 1350, 1290, 140, 60, "NO OP", "Cross-Attn OFF", "源码中注释", "removed", nodeMap.get("w_text_off").meta],
      ["wa_shift_mlp", 1510, 1290, 190, 60, "MLP PARAMS", "shift_mlp + scale_mlp", "→ AdaLN₂", "action", mlpModMeta],
      ["wa_gate_mlp", 1780, 1290, 130, 60, "MLP PARAM", "gate_mlp", "→ MLP residual", "action", gateMlpMeta],
      ["wa_x", 745, 1380, 70, 90, "FEATURE", "输入 x", "[B,T,S,C]", "shared", blockMeta("wmpo", "spatial", 1)],
      ["wa_ln1", 830, 1380, 80, 90, "PRE-NORM", "LN₁(x)", "affine=False", "shared", blockMeta("wmpo", "spatial", 1)],
      ["wa_mod1", 925, 1380, 185, 90, "ACTION-ADALN₁", "Self-Attention 前调制", "(1+scale_msa)·LN₁+shift_msa", "action", attnModMeta],
      ["wa_attn", 1125, 1380, 100, 90, "SPATIAL", "Self-Attention", "帧内 S×S", "shared", blockMeta("wmpo", "spatial", 1)],
      ["wa_gate1", 1240, 1380, 105, 90, "RESIDUAL", "×gate_msa + x", "得到 x₁", "action", gateMsaMeta],
      ["wa_x1", 1360, 1380, 60, 90, "STATE", "x₁", "[B,T,S,C]", "shared", blockMeta("wmpo", "spatial", 1)],
      ["wa_ln2", 1435, 1380, 75, 90, "PRE-NORM", "LN₂(x₁)", "affine=False", "shared", blockMeta("wmpo", "spatial", 1)],
      ["wa_mod2", 1525, 1380, 175, 90, "ACTION-ADALN₂", "MLP 前调制", "(1+scale_mlp)·LN₂+shift_mlp", "action", mlpModMeta],
      ["wa_mlp2", 1715, 1380, 85, 90, "CHANNEL", "MLP", "1152→4608→1152", "shared", blockMeta("wmpo", "spatial", 1)],
      ["wa_gate2", 1805, 1380, 110, 90, "×GATE_MLP + RESIDUAL", "得到 Spatial 输出 x₂", "直接进入 Temporal Block", "action", gateMlpMeta]
    ];
    actionFeatureDetail.forEach(function (item) {
      addNode({
        id: item[0], lane: "wmpo", x: item[1], y: item[2], w: item[3], h: item[4],
        type: item[5], label: item[6], shape: item[7], category: item[8], meta: item[9]
      });
    });

    const temporalDetail = [
      ["td_reshape", 1995, "接收 x₂ 并 Rearrange", "[(B·S),T,C]", "shared"],
      ["td_native", 2150, "原生 Temporal", "T ↔ T", "shared"],
      ["td_wmpo", 2310, "WMPO Causal", "past → present", "action"],
      ["td_mlp", 2470, "MLP + Residual", "[B,T,S,C]", "shared"]
    ];
    temporalDetail.forEach(function (item) {
      addNode({
        id: item[0], lane: item[0] === "td_native" ? "native" : (item[0] === "td_wmpo" ? "wmpo" : "both"),
        x: item[1], y: 1270, w: 140, h: 90,
        type: "TEMPORAL", label: item[2], shape: item[3], category: item[4],
        meta: item[0] === "td_wmpo" ? blockMeta("wmpo", "temporal", 1) : blockMeta("native", "temporal", 1)
      });
    });

    [
      { id: "diff_action", x: 100, label: "ActionEncoder", shape: "7→4608→1152", category: "action", meta: nodeMap.get("w_action_encoder").meta },
      { id: "diff_frame", x: 455, label: "逐帧 Spatial AdaLN", shape: "[B,T,6,C]", category: "action", meta: nodeMap.get("w_spatial_cond").meta },
      { id: "diff_causal", x: 810, label: "Causal Temporal", shape: "is_causal=True", category: "action", meta: blockMeta("wmpo", "temporal", 1) },
      { id: "diff_text", x: 1165, label: "Text Cross-Attn OFF", shape: "核心路径变化", category: "removed", meta: nodeMap.get("w_text_off").meta },
      { id: "diff_fps", x: 1520, label: "FPS OFF", shape: "forward 注释", category: "drift", meta: commonMeta({
        description: "WMPO 当前 forward 注释掉 fps_embedder 的使用。",
        input: "原生 fps [B]", output: "当前无 fps 条件",
        formula: "# fps = self.fps_embedder(...)", meaning: "属于当前 fork 实现差异，应与 action 注入区分。",
        source: WMPO + " · L448–456", classification: "fork / 版本差异", className: "drift"
      }) },
      { id: "diff_channels", x: 1875, label: "Latent 16→4", shape: "VAE/fork 差异", category: "drift", meta: nodeMap.get("w_input").meta },
      { id: "diff_mask", x: 2230, label: "mask t₀: 0→15", shape: "条件帧策略差异", category: "drift", meta: commonMeta({
        description: "原生 x_mask 分支使用 timestep=0；WMPO 当前实现使用常数 15。",
        input: "x_mask + timestep", output: "t0_mlp",
        formula: "Open-Sora: t₀=0; WMPO: t₀=15", meaning: "这是条件帧/噪声处理的实现差异，不是 ActionEncoder 的必要组成。",
        source: OPEN_SORA + " · L479–483；" + WMPO + " · L457–462", classification: "fork / 版本差异", className: "drift"
      }) }
    ].forEach(function (item) {
      addNode({
        id: item.id, lane: "wmpo", x: item.x, y: 1558, w: 295, h: 72,
        type: item.category === "drift" ? "FORK DIFFERENCE" : "WMPO CHANGE",
        label: item.label, shape: item.shape, category: item.category, meta: item.meta
      });
    });
  }

  function addBaseDecorations() {
    [
      ["lane-background native", "", ""],
      ["lane-background wmpo", "", ""],
      ["detail-zone", "", ""]
    ].forEach(function (item) {
      const div = document.createElement("div");
      div.className = item[0];
      nodeLayer.appendChild(div);
    });

    const nativeTitle = document.createElement("div");
    nativeTitle.className = "lane-title native";
    nativeTitle.innerHTML = "<span>NATIVE VIDEO GENERATOR</span><strong>Open-Sora v1.3</strong>";
    nodeLayer.appendChild(nativeTitle);

    const wmpoTitle = document.createElement("div");
    wmpoTitle.className = "lane-title wmpo";
    wmpoTitle.innerHTML = "<span>ACTION-CONDITIONED WORLD MODEL</span><strong>WMPO</strong>";
    nodeLayer.appendChild(wmpoTitle);

    const detailTitle = document.createElement("div");
    detailTitle.className = "detail-zone-title";
    detailTitle.innerHTML = "<p>EXPLODED BLOCK VIEW</p><h2>Block 内部与关键改动</h2>";
    nodeLayer.appendChild(detailTitle);

    [
      { lane: "native", x: 900, y: 178, w: 1320, label: "28 × [Spatial → Temporal] · 展示 1–3 / 26–28，中间 22 对省略" },
      { lane: "wmpo", x: 900, y: 638, w: 1320, label: "28 × [Action-Spatial → Causal-Temporal] · 展示 1–3 / 26–28" }
    ].forEach(function (config) {
      const bracket = document.createElement("div");
      bracket.className = "backbone-bracket";
      bracket.dataset.lane = config.lane;
      bracket.style.left = config.x + "px";
      bracket.style.top = config.y + "px";
      bracket.style.width = config.w + "px";
      const label = document.createElement("span");
      label.className = "bracket-label";
      label.textContent = config.label;
      bracket.appendChild(label);
      nodeLayer.appendChild(bracket);
    });
  }

  function connectCore() {
    [
      ["n_input", "n_patch", "native"], ["n_patch", "n_pos", "native"], ["n_pos", "n_s_1", "native"],
      ["n_t_28", "n_final", "native"], ["n_final", "n_output", "native"],
      ["n_time", "n_tblock", "condition"], ["n_fps", "n_tblock", "condition"],
      ["n_tblock", "n_s_1", "condition"], ["n_text", "n_caption", "condition"], ["n_caption", "n_s_1", "condition"],
      ["w_input", "w_patch", "wmpo"], ["w_patch", "w_pos", "wmpo"], ["w_pos", "w_s_1", "wmpo"],
      ["w_t_28", "w_final", "wmpo"], ["w_final", "w_output", "wmpo"],
      ["w_action", "w_action_encoder", "action"], ["w_action_encoder", "w_none", "action"],
      ["w_none", "w_spatial_cond", "action"], ["w_spatial_cond", "w_ts_block", "action", "vertical"],
      ["w_time", "w_tt_block", "condition"],
      ["w_tt_block", "w_t_1", "condition"],
      ["nd_adaln1", "nd_attn", "native"], ["nd_attn", "nd_gate", "native"],
      ["nd_gate", "nd_cross", "native"], ["nd_cross", "nd_mlp", "native"],
      ["wa_action", "wa_encode", "action"], ["wa_encode", "wa_none", "action"],
      ["wa_none", "wa_sum", "action"], ["wa_repeat", "wa_sum", "condition"],
      ["wa_sum", "wa_tblock", "action"], ["wa_tblock", "wa_reshape", "action"],
      ["wa_reshape", "wa_chunk", "action"],
      ["wa_x", "wa_ln1", "wmpo"], ["wa_ln1", "wa_mod1", "action"],
      ["wa_mod1", "wa_attn", "action"], ["wa_attn", "wa_gate1", "action"],
      ["wa_gate1", "wa_x1", "wmpo"], ["wa_x1", "wa_ln2", "wmpo"],
      ["wa_ln2", "wa_mod2", "action"], ["wa_mod2", "wa_mlp2", "action"],
      ["wa_mlp2", "wa_gate2", "action"], ["wa_gate2", "td_reshape", "action"],
      ["wa_shift_msa", "wa_mod1", "action", "vertical"], ["wa_gate_msa", "wa_gate1", "action", "vertical"],
      ["wa_shift_mlp", "wa_mod2", "action", "vertical"], ["wa_gate_mlp", "wa_gate2", "action", "vertical"],
      ["td_reshape", "td_native", "native"],
      ["td_reshape", "td_wmpo", "wmpo"], ["td_native", "td_mlp", "native"], ["td_wmpo", "td_mlp", "wmpo"]
    ].forEach(function (item) {
      connections.push({ from: item[0], to: item[1], className: item[2], vertical: item[3] === "vertical" });
    });
  }

  function createNodeElement(node) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "arch-node " + node.category +
      (node.mini ? " mini-block" : "") +
      (node.omission ? " omission-block" : "");
    button.dataset.nodeId = node.id;
    button.dataset.lane = node.lane;
    button.style.left = node.x + "px";
    button.style.top = node.y + "px";
    button.style.width = node.w + "px";
    button.style.height = node.h + "px";
    button.setAttribute("aria-label", node.label + "，" + node.shape);

    const type = document.createElement("span");
    type.className = "node-type";
    type.textContent = node.type;
    const name = document.createElement("span");
    name.className = "node-name";
    name.textContent = node.label;
    const shape = document.createElement("span");
    shape.className = "node-shape";
    shape.textContent = node.shape;
    button.appendChild(type);
    button.appendChild(name);
    button.appendChild(shape);
    button.addEventListener("click", function () {
      selectNode(node.id);
    });
    return button;
  }

  function marker(id, color) {
    return "<marker id=\"" + id + "\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerWidth=\"6\" markerHeight=\"6\" orient=\"auto-start-reverse\"><path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"" + color + "\"></path></marker>";
  }

  function renderConnectors() {
    connectorLayer.innerHTML = "<defs>" +
      marker("arrow-neutral", "#385568") +
      marker("arrow-native", "#69a8ff") +
      marker("arrow-wmpo", "#5ee0a5") +
      marker("arrow-condition", "#ffb45f") +
      "</defs>";

    connections.forEach(function (connection) {
      const from = nodeMap.get(connection.from);
      const to = nodeMap.get(connection.to);
      if (!from || !to) return;

      let startX = from.x + from.w;
      let startY = from.y + from.h / 2;
      let endX = to.x;
      let endY = to.y + to.h / 2;
      let pathData;

      if (connection.vertical) {
        startX = from.x + from.w / 2;
        startY = from.y + from.h;
        endX = to.x + to.w / 2;
        endY = to.y;
        pathData = "M " + startX + " " + startY + " L " + endX + " " + endY;
      } else if (connection.stepped || Math.abs(endY - startY) > 80) {
        const midX = startX + Math.max(18, (endX - startX) / 2);
        pathData = "M " + startX + " " + startY + " L " + midX + " " + startY +
          " L " + midX + " " + endY + " L " + endX + " " + endY;
      } else {
        pathData = "M " + startX + " " + startY + " L " + endX + " " + endY;
      }

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", pathData);
      path.setAttribute("class", "connector-path " + connection.className);
      connectorLayer.appendChild(path);
    });

    renderActionAdaLNBus();
    renderSpatialParameterRouting();

    [
      { x: 1475, y: 326, text: "t_mlp + text → every block" },
      { x: 1050, y: 614, text: "6 组调制参数 → 每个 Spatial Block 内部的 Self-Attention / MLP 两处", className: "action-bus-label" },
      { x: 1450, y: 856, text: "t_t(global) → Temporal ×28" },
      { x: 1690, y: 1505, text: "Spatial 输出 x₂  ─────────→  Temporal Block", className: "spatial-output-label" },
      { x: 112, y: 1640, text: "核心 action 设计" },
      { x: 1520, y: 1640, text: "fork / 版本差异" }
    ].forEach(function (label) {
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", label.x);
      text.setAttribute("y", label.y);
      text.setAttribute("class", "connector-label" + (label.className ? " " + label.className : ""));
      text.textContent = label.text;
      connectorLayer.appendChild(text);
    });
  }

  function renderActionAdaLNBus() {
    const source = nodeMap.get("w_ts_block");
    const targetIds = ["w_s_1", "w_s_2", "w_s_3", "w_gap", "w_s_26", "w_s_27", "w_s_28"];
    const targets = targetIds.map(function (id) { return nodeMap.get(id); }).filter(Boolean);
    if (!source || !targets.length) return;

    const busY = 625;
    const sourceX = source.x + source.w;
    const sourceY = source.y + source.h / 2;
    const elbowX = sourceX + 25;
    const firstX = targets[0].x + targets[0].w / 2;
    const lastX = targets[targets.length - 1].x + targets[targets.length - 1].w / 2;
    const busStartX = Math.min(firstX, elbowX);
    const busEndX = Math.max(lastX, elbowX);

    const feed = document.createElementNS("http://www.w3.org/2000/svg", "path");
    feed.setAttribute("d", "M " + sourceX + " " + sourceY + " L " + elbowX + " " + sourceY +
      " L " + elbowX + " " + busY);
    feed.setAttribute("class", "connector-path action action-bus-feed");
    connectorLayer.appendChild(feed);

    const trunk = document.createElementNS("http://www.w3.org/2000/svg", "path");
    trunk.setAttribute("d", "M " + busStartX + " " + busY + " L " + busEndX + " " + busY);
    trunk.setAttribute("class", "connector-path action action-bus");
    connectorLayer.appendChild(trunk);

    const feedJunction = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    feedJunction.setAttribute("cx", elbowX);
    feedJunction.setAttribute("cy", busY);
    feedJunction.setAttribute("r", "5");
    feedJunction.setAttribute("class", "action-bus-junction");
    connectorLayer.appendChild(feedJunction);

    targets.forEach(function (target) {
      const targetX = target.x + target.w / 2;
      const injection = document.createElementNS("http://www.w3.org/2000/svg", "path");
      injection.setAttribute("d", "M " + targetX + " " + busY + " L " + targetX + " " + target.y);
      injection.setAttribute("class", "connector-path action action-injection");
      connectorLayer.appendChild(injection);

      const junction = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      junction.setAttribute("cx", targetX);
      junction.setAttribute("cy", busY);
      junction.setAttribute("r", "5");
      junction.setAttribute("class", "action-bus-junction");
      connectorLayer.appendChild(junction);
    });
  }

  function renderSpatialParameterRouting() {
    const source = nodeMap.get("wa_chunk");
    const targetIds = ["wa_shift_msa", "wa_gate_msa", "wa_shift_mlp", "wa_gate_mlp"];
    const targets = targetIds.map(function (id) { return nodeMap.get(id); }).filter(Boolean);
    if (!source || targets.length !== targetIds.length) return;

    const busY = 1274;
    const sourceX = source.x + source.w / 2;
    const sourceY = source.y + source.h;
    const centers = targets.map(function (target) { return target.x + target.w / 2; });
    const busStartX = Math.min.apply(null, centers);
    const busEndX = Math.max.apply(null, centers.concat([sourceX]));

    const feed = document.createElementNS("http://www.w3.org/2000/svg", "path");
    feed.setAttribute("d", "M " + sourceX + " " + sourceY + " L " + sourceX + " " + busY);
    feed.setAttribute("class", "connector-path action spatial-param-feed");
    connectorLayer.appendChild(feed);

    const bus = document.createElementNS("http://www.w3.org/2000/svg", "path");
    bus.setAttribute("d", "M " + busStartX + " " + busY + " L " + busEndX + " " + busY);
    bus.setAttribute("class", "connector-path action spatial-param-bus");
    connectorLayer.appendChild(bus);

    targets.forEach(function (target, index) {
      const injection = document.createElementNS("http://www.w3.org/2000/svg", "path");
      injection.setAttribute("d", "M " + centers[index] + " " + busY + " L " + centers[index] + " " + target.y);
      injection.setAttribute("class", "connector-path action spatial-param-injection");
      connectorLayer.appendChild(injection);
    });
  }

  function renderNodes() {
    nodes.forEach(function (node) {
      nodeLayer.appendChild(createNodeElement(node));
    });
  }

  function renderInspector(meta, title, kind) {
    inspector.kind.textContent = kind || "MODULE";
    inspector.title.textContent = title;
    inspector.description.textContent = meta.description;
    inspector.input.textContent = meta.input;
    inspector.output.textContent = meta.output;
    inspector.formula.textContent = meta.formula;
    inspector.meaning.textContent = meta.meaning;
    inspector.source.textContent = meta.source;
    inspector.classification.textContent = meta.classification;
    inspector.classification.className = "classification " + meta.className;
    inspector.count.textContent = meta.count;
    inspector.count.hidden = !meta.count;
  }

  function selectNode(id) {
    const node = nodeMap.get(id);
    if (!node) return;
    state.selectedId = id;
    document.querySelectorAll(".arch-node.is-selected").forEach(function (element) {
      element.classList.remove("is-selected");
    });
    const selected = document.querySelector('[data-node-id="' + id + '"]');
    if (selected) selected.classList.add("is-selected");
    const title = node.omission
      ? "Blocks 4–25（省略显示）"
      : (node.mini
        ? (node.type === "SPATIAL" ? "Spatial Block " : "Temporal Block ") + node.label
        : node.label);
    renderInspector(node.meta, title, node.type);
  }

  function renderOverviewInspector() {
    renderInspector(commonMeta({
      description: "两条对齐的完整 forward：上方是原生 Open-Sora v1.3，下方是 WMPO action-conditioned 世界模型。图中展示第 1–3 与第 26–28 对 Spatial/Temporal Block，第 4–25 对折叠省略；实际运行仍是完整 28 对。",
      input: "原生: latent + timestep + fps + text；WMPO: latent + timestep + actions",
      output: "video latent prediction",
      formula: "PatchEmbed → [Spatial → Temporal] ×28 → FinalLayer → Unpatchify",
      meaning: "先比较共享骨架，再沿绿色 action 路径观察 WMPO 如何改变每个 Spatial Block。",
      source: OPEN_SORA + "；" + WMPO,
      count: "28 Spatial + 28 Temporal / model"
    }), "完整模型地图", "OVERVIEW");
  }

  function clampScale(value) {
    return Math.max(0.22, Math.min(1.4, value));
  }

  function clampPan() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    const scaledWidth = CANVAS_WIDTH * state.scale;
    const scaledHeight = CANVAS_HEIGHT * state.scale;
    const margin = 120;

    if (scaledWidth <= width) {
      state.panX = (width - scaledWidth) / 2;
    } else {
      state.panX = Math.min(margin, Math.max(width - scaledWidth - margin, state.panX));
    }
    if (scaledHeight <= height) {
      state.panY = (height - scaledHeight) / 2;
    } else {
      state.panY = Math.min(margin, Math.max(height - scaledHeight - margin, state.panY));
    }
  }

  function applyTransform() {
    clampPan();
    canvas.style.transform = "translate(" + state.panX + "px," + state.panY + "px) scale(" + state.scale + ")";
    zoomValue.textContent = Math.round(state.scale * 100) + "%";
    updateMinimap();
  }

  function fitView() {
    const scaleX = (viewport.clientWidth - 36) / CANVAS_WIDTH;
    const scaleY = (viewport.clientHeight - 36) / CANVAS_HEIGHT;
    state.scale = clampScale(Math.min(scaleX, scaleY));
    state.panX = (viewport.clientWidth - CANVAS_WIDTH * state.scale) / 2;
    state.panY = (viewport.clientHeight - CANVAS_HEIGHT * state.scale) / 2;
    applyTransform();
  }

  function setZoom(nextScale, clientX, clientY) {
    const oldScale = state.scale;
    const next = clampScale(nextScale);
    const rect = viewport.getBoundingClientRect();
    const anchorX = clientX === undefined ? rect.left + rect.width / 2 : clientX;
    const anchorY = clientY === undefined ? rect.top + rect.height / 2 : clientY;
    const localX = anchorX - rect.left;
    const localY = anchorY - rect.top;
    const canvasX = (localX - state.panX) / oldScale;
    const canvasY = (localY - state.panY) / oldScale;

    state.scale = next;
    state.panX = localX - canvasX * next;
    state.panY = localY - canvasY * next;
    applyTransform();
  }

  function focusRegion(region) {
    if (region.id === "full") {
      fitView();
      renderOverviewInspector();
      return;
    }
    state.scale = clampScale(region.scale);
    state.panX = viewport.clientWidth / 2 - region.cx * state.scale;
    state.panY = viewport.clientHeight / 2 - region.cy * state.scale;
    applyTransform();
    if (region.nodeId) selectNode(region.nodeId);
  }

  function updateMinimap() {
    const miniWidth = 162;
    const miniHeight = 88;
    const visibleX = Math.max(0, -state.panX / state.scale);
    const visibleY = Math.max(0, -state.panY / state.scale);
    const visibleW = Math.min(CANVAS_WIDTH, viewport.clientWidth / state.scale);
    const visibleH = Math.min(CANVAS_HEIGHT, viewport.clientHeight / state.scale);
    minimapWindow.style.left = (visibleX / CANVAS_WIDTH) * miniWidth + "px";
    minimapWindow.style.top = (visibleY / CANVAS_HEIGHT) * miniHeight + "px";
    minimapWindow.style.width = Math.max(8, (visibleW / CANVAS_WIDTH) * miniWidth) + "px";
    minimapWindow.style.height = Math.max(8, (visibleH / CANVAS_HEIGHT) * miniHeight) + "px";
  }

  function setView(view) {
    state.view = view;
    canvas.classList.remove("mode-native", "mode-wmpo", "mode-compare");
    canvas.classList.add("mode-" + view);
    viewButtons.forEach(function (button) {
      const active = button.dataset.view === view;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function initInteractions() {
    viewButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        setView(button.dataset.view);
      });
    });

    document.getElementById("shape-toggle").addEventListener("change", function (event) {
      canvas.classList.toggle("hide-shapes", !event.target.checked);
    });
    document.getElementById("zoom-in").addEventListener("click", function () {
      setZoom(state.scale + 0.1);
    });
    document.getElementById("zoom-out").addEventListener("click", function () {
      setZoom(state.scale - 0.1);
    });
    document.getElementById("fit-view").addEventListener("click", fitView);
    document.getElementById("reset-view").addEventListener("click", function () {
      state.scale = 1;
      state.panX = 40;
      state.panY = 40;
      applyTransform();
    });

    const regions = {
      full: { id: "full" },
      inputs: { id: "inputs", cx: 260, cy: 500, scale: 0.78, nodeId: "w_input" },
      tokens: { id: "tokens", cx: 500, cy: 500, scale: 0.82, nodeId: "w_patch" },
      conditions: { id: "conditions", cx: 520, cy: 760, scale: 0.72, nodeId: "w_action_encoder" },
      spatial: { id: "spatial", cx: 1410, cy: 1320, scale: 0.52, nodeId: "wa_mod1" },
      temporal: { id: "temporal", cx: 2170, cy: 1300, scale: 0.72, nodeId: "td_wmpo" },
      backbone: { id: "backbone", cx: 1390, cy: 540, scale: 0.5, nodeId: "w_gap" },
      output: { id: "output", cx: 2425, cy: 500, scale: 0.78, nodeId: "w_final" },
      differences: { id: "differences", cx: 1350, cy: 1570, scale: 0.58, nodeId: "diff_action" }
    };

    chapterButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        chapterButtons.forEach(function (item) {
          item.classList.toggle("is-active", item === button);
        });
        focusRegion(regions[button.dataset.focus]);
      });
    });

    viewport.addEventListener("pointerdown", function (event) {
      if (event.target.closest(".arch-node")) return;
      state.dragging = true;
      state.pointerId = event.pointerId;
      state.startX = event.clientX;
      state.startY = event.clientY;
      state.startPanX = state.panX;
      state.startPanY = state.panY;
      viewport.classList.add("is-dragging");
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener("pointermove", function (event) {
      if (!state.dragging || event.pointerId !== state.pointerId) return;
      state.panX = state.startPanX + event.clientX - state.startX;
      state.panY = state.startPanY + event.clientY - state.startY;
      applyTransform();
    });
    function endDrag(event) {
      if (event.pointerId !== state.pointerId) return;
      state.dragging = false;
      state.pointerId = null;
      viewport.classList.remove("is-dragging");
    }
    viewport.addEventListener("pointerup", endDrag);
    viewport.addEventListener("pointercancel", endDrag);
    viewport.addEventListener("wheel", function (event) {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.1 : 0.9;
      setZoom(state.scale * factor, event.clientX, event.clientY);
    }, { passive: false });

    viewport.addEventListener("keydown", function (event) {
      if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        setZoom(state.scale + 0.1);
      } else if (event.key === "-") {
        event.preventDefault();
        setZoom(state.scale - 0.1);
      } else if (event.key === "0") {
        event.preventDefault();
        fitView();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        state.panX += 60;
        applyTransform();
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        state.panX -= 60;
        applyTransform();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        state.panY += 60;
        applyTransform();
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        state.panY -= 60;
        applyTransform();
      }
    });

    window.addEventListener("resize", function () {
      applyTransform();
    });
  }

  function init() {
    addBaseDecorations();
    addCoreNodes();
    addBackbone("native", 210, "n");
    addBackbone("wmpo", 670, "w");
    addExplodedNodes();
    connectCore();
    renderConnectors();
    renderNodes();
    renderOverviewInspector();
    setView("compare");
    initInteractions();
    window.requestAnimationFrame(fitView);
  }

  init();
}());
