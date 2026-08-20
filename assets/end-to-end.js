(() => {
  "use strict";
  const CW = 2850, CH = 1800;
  const viewport = document.getElementById("architecture-viewport");
  const canvas = document.getElementById("architecture-canvas");
  const zoomValue = document.getElementById("zoom-value");
  const state = { scale: .5, x: 0, y: 0, drag: false, pointer: null, startX: 0, startY: 0, originX: 0, originY: 0, density: "learn" };
  const semanticLevelEl=document.getElementById("semantic-level");
  const densityButtons=[...document.querySelectorAll(".density-button")];
  const clearPathButton=document.getElementById("clear-path");
  const dimensionTrace=document.getElementById("dimension-trace");
  const dimensionMessage=document.getElementById("dimension-message");

  const details = {
    rgb:["OUTSIDE STDiT3","RGB 视频","数据层面的原始输入。VAE 编码和加噪发生在 STDiT3.forward 之前。","[B,3,12,H₀,W₀]","模型外部：dataset / VAE"],
    vae:["OUTSIDE STDiT3","VAE Encoder + 加噪","VAE 把 RGB 压缩到 4 通道 latent，生成过程再依据 timestep 构造噪声 latent xₜ。","RGB → latent xₜ","模型外部：VAE / sampler"],
    latent:["MODEL INPUT","噪声 latent xₜ","这是 WMPO STDiT3.forward 真正接收的视频张量；12 帧示例对应 4 个历史位置和 8 个未来位置。","[B,4,12,Hx,Wx]","WMPO stdit3.py · L412–421"],
    patch:["VIDEO TOKENIZATION","PatchEmbed3D","Conv3D 使用 kernel=stride=(1,2,2)，保持帧数不变，每个 2×2 latent patch 投影为一个 1152 维 token。","[B,4,12,Hx,Wx] → [B,12,S,1152]","layers/blocks.py · L79–129"],
    position:["POSITION","2D sin/cos 位置编码","为空间 H×W patch 生成位置编码，同一套编码沿 12 个时间位置广播。","[1,S,1152] + [B,12,S,1152]","layers/blocks.py · L944–993"],
    pairs:["BACKBONE","28 × Spatial → Temporal","位置编码后的 video hidden x₀ 进入 Pair 1；每对先执行 action-conditioned Spatial Block，再把输出直接交给 timestep-conditioned causal Temporal Block，随后传入下一 Pair。28 对参数彼此独立。","x₀ → Pair₁ → … → Pair₂₈；全程 [B,12,S,1152]","WMPO stdit3.py · L484–487"],
    final:["OUTPUT HEAD","T2IFinalLayer","全局 timestep 先经过 SiLU + Linear 生成 final shift/scale，再用它调制无 affine 的 LayerNorm 输出，最后预测每个 latent patch。","Eₜ(t) → SiLU → Linear(1152,2304) → shift / scale","layers/blocks.py · L709–721"],
    unpatchify:["OUTPUT SHAPE","unpatchify","把每个 token 的 1×2×2 patch prediction 重新排列为 5D latent 网格。","[B,12,S,4·Cout] → [B,Cout,12,Hx,Wx]","WMPO stdit3.py · L503–520"],
    decoder:["OUTSIDE STDiT3","Sampler + VAE Decoder","模型输出仍是 latent 预测；采样器更新 xₜ，最终由 VAE Decoder 得到可见视频。","latent prediction → RGB","模型外部：sampling pipeline"],
    actions:["ACTION INPUT","8 个未来 7D actions","每个未来时间位置对应一个机器人动作向量，典型维度包含平移、旋转和 gripper。","[B,8,7]","simplevla_webdataset.py · Ta=8, action_dim=7"],
    actionEncoder:["ACTION CONDITION","ActionEncoder","每个 7D action 独立经过 Linear、SiLU、Linear 投影到 hidden size。这里的 SiLU 属于 action 编码器；编码结果不直接加到 video token。","Linear(7,4608) → SiLU → Linear(4608,1152)","WMPO stdit3.py · L37–72"],
    none:["ACTION ALIGNMENT","prepend NONE ×4","在 8 个未来 action embedding 前补 4 个可学习 NONE，使条件与 12 帧位置一一对齐。","[B,8,1152] → [B,12,1152]","WMPO stdit3.py · L54–65"],
    spatialCondition:["SPATIAL PARAMS","逐帧 Action + timestep 参数","每帧条件 cₛ=Eₜ(t)+Eₐ(aᵢ) 先过 SiLU，再由 Linear(1152,6912) 生成六组参数。绿色虚线只表示 shift / scale / gate 的路由，不是 Spatial 输出或 x₂ 广播。","cₛ → SiLU → Linear(1152,6·1152) → chunk(6)","WMPO stdit3.py · L295–298、L448–457"],
    spatialAdaLN1:["ACTION ADALN","Attention 前的 Action-AdaLN","shift_msa 与 scale_msa 逐通道调制 LN₁(x)，因此 action 在 Q/K/V 生成之前改变 video hidden。","[B,T,S,C]；参数 [B,T,1,C] 沿 S 广播","WMPO stdit3.py · L147–167"],
    spatialAdaLN2:["ACTION ADALN","MLP 前的 Action-AdaLN","shift_mlp 与 scale_mlp 调制 LN₂(x₁)，这是 action 在 Spatial Block 内的第二个直接注入点。","[B,T,S,C]；参数 [B,T,1,C] 沿 S 广播","WMPO stdit3.py · L186–204"],
    gateMsa:["ACTION GATE","gate_msa：Attention 更新写回","gate_msa 不进入 Attention；它在 Attention 输出之后逐通道控制更新量写回 residual x 的强度。","x₁=x+gate_msa⊙Attention(AdaLN₁(x))","WMPO stdit3.py · L156–181"],
    gateMlp:["ACTION GATE","gate_mlp：MLP 更新写回","gate_mlp 控制 MLP 更新量写回 x₁，最终产生已经包含 action 影响的 Spatial 输出 x₂。","x₂=x₁+gate_mlp⊙MLP(AdaLN₂(x₁))","WMPO stdit3.py · L186–205"],
    crossOff:["GHOST BRANCH","Text Cross-Attention：不执行","当前 WMPO world-model forward 中该调用被注释，因此它只作为源码差异旁路显示，不属于 x₁→LayerNorm 2 的主计算流。","x₁ 直接进入 LayerNorm 2","WMPO stdit3.py · L181–186"],
    timestep:["DIFFUSION CONDITION","timestep","标量 diffusion / flow 时间决定当前噪声强度。","[B]","WMPO stdit3.py · L448–462"],
    tEmbed:["DIFFUSION CONDITION","TimestepEmbedder","先构造 256 维 sin/cos frequency embedding，再经过 Linear、SiLU、Linear 得到 1152 维全局条件。","[B] → sin/cos₍₂₅₆₎ → Linear → SiLU → Linear → [B,1152]","layers/blocks.py · L792–825"],
    temporalCondition:["TEMPORAL PARAMS","Temporal t_block","全局 Eₜ(t) 经过同一个 t_block：SiLU + Linear(1152,6912)，再切成六组 Temporal AdaLN/gate 参数；这里不再加入 action embedding。","Eₜ(t) → SiLU → Linear(1152,6·1152) → chunk(6)","WMPO stdit3.py · L295–298、L456–488"],
    finalCondition:["FINAL PARAMS","Final shift / scale","Final Layer 使用自己的 SiLU + Linear(1152,2304) 生成 shift/scale。它不直接读取 action；action 影响已经保存在 28 对 Block 输出的 hidden 中。","Eₜ(t) → SiLU → Linear(1152,2·1152) → shift, scale","layers/blocks.py · L709–721"],
    spatialInput:["VIDEO HIDDEN INPUT","Video hidden xℓ 进入 Spatial Block","这是 Block 真正处理的视频 token，而不是 action embedding。Pair 1 接收 PatchEmbed3D + 位置编码得到的 x₀；后续 Pair 接收上一 Pair 的 Temporal 输出。xℓ 同时走主分支和两条 residual 支路。","[B,T,S,1152] → LayerNorm 1 / residual","WMPO stdit3.py · L448–488"],
    spatialAttn:["SPATIAL ATTENTION","Spatial Self-Attention","每帧的 S 个 patch 互相注意。Action 已通过前面的 Action-AdaLN 改变 Q/K/V 的输入。","[B·T,S,1152] → [B·T,S,1152]","WMPO stdit3.py · L156–181"],
    spatialX2:["SPATIAL OUTPUT","x₂：已经包含 action 影响","第二条 residual 把 action-conditioned MLP 更新量写回 hidden。Action 参数本身不会作为新 token 传给 Temporal；它们造成的数值变化保留在 x₂ 中。","x₂ = x₁ + gate_mlp(a,t) · MLP(AdaLN₂(x₁,a,t))","WMPO stdit3.py · L186–206"],
    temporalHandoff:["HIDDEN STATE HANDOFF","Spatial → Temporal 直接传递 x₂","两个 Block 之间没有独立 Action Layer，也没有先 reshape。Spatial Block 返回的 [B,T,S,C] x₂ 原样成为 Temporal Block 的 x。","[B,T,S,1152] → [B,T,S,1152]","WMPO stdit3.py · L485–488"],
    temporalX2:["TEMPORAL INPUT","Temporal 收到的 x₂ 仍含 action 影响","Temporal Block 的 AdaLN 参数只由全局 timestep 生成，但它处理的 hidden 已被前一个 Spatial Block 的 action-conditioned Attention 和 MLP 改写。","x_temporal,in = x₂","WMPO stdit3.py · L142–167"],
    temporalAttn:["TEMPORAL ATTENTION","Causal Temporal Attention","同一空间位置的 T 帧执行注意力，Q/K 使用 RoPE，并加入下三角 causal mask。","[B·S,T,1152] → [B·S,T,1152]","WMPO stdit3.py · L163–181；blocks.py L187–218"],
    mlp:["FEED-FORWARD","MLP","逐 token 进行通道混合：先扩展到 4608，approx GELU，再投影回 1152。AdaLN 与 gate 位于 MLP 类外部。","1152 → 4608 → 1152","WMPO stdit3.py · L113–116、L186–204"]
  };
  const inspector = { kind:document.getElementById("detail-kind"), title:document.getElementById("detail-title"), description:document.getElementById("detail-description"), shape:document.getElementById("detail-shape"), source:document.getElementById("detail-source") };
  const inspectorPanel=document.getElementById("canvas-inspector");
  const inspectorReopen=document.getElementById("open-inspector");
  const tensorTrace=document.getElementById("tensor-trace");
  const tensorFields={name:document.getElementById("tensor-name"),produced:document.getElementById("tensor-produced"),consumed:document.getElementById("tensor-consumed"),action:document.getElementById("tensor-action")};
  const tensorDetails={
    spatialInput:{name:"xℓ · [B,T,S,1152]",produced:"Pair 1: PatchEmbed + PE；Pair ℓ>1: Temporal Block ℓ−1",consumed:"Spatial LN₁ + residual",action:"Pair 1 输入未直接拼接 action；后续 xℓ 携带前面 Pair 的 action 影响"},
    spatialX2:{name:"x₂ · [B,T,S,1152]",produced:"Spatial MLP residual",consumed:"Temporal Block ℓ",action:"包含：YES · 直接 condition：NO"},
    temporalHandoff:{name:"x₂ · shape 不变",produced:"Spatial Block return",consumed:"Temporal Block input",action:"数值携带 action；参数不广播"},
    temporalX2:{name:"x₂ · [B,T,S,1152]",produced:"Spatial Block ℓ",consumed:"Temporal LN₁",action:"包含：YES · Temporal 不直接读 action"}
  };

  const connectorGroup = document.getElementById("dynamic-connectors");
  const NS = "http://www.w3.org/2000/svg";
  const connections = [
    {from:"#node-rgb",to:"#node-vae",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#node-vae",to:"#node-latent",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#node-latent",to:"#node-patch",kind:"video",fromSide:"right",toSide:"left"},
    {from:"#node-patch",to:"#node-position",kind:"video",fromSide:"bottom",toSide:"top"},
    {from:"#node-position",to:"#node-pairs",kind:"video",fromSide:"right",toSide:"left"},
    {from:"#node-pairs",to:"#spatial-input-x",kind:"video",fromSide:"right",toSide:"left",label:"video hidden xℓ",labelAt:.56,labelDy:-2},
    {from:"#node-pairs",to:"#pair-1",kind:"video",fromSide:"bottom",toSide:"top"},
    {from:"#pair-1",to:"#pair-2",kind:"video",fromSide:"bottom",toSide:"top"},
    {from:"#pair-27",to:"#pair-28",kind:"video",fromSide:"bottom",toSide:"top"},
    {from:"#pair-28",to:"#node-final",kind:"video",fromSide:"bottom",toSide:"top"},
    {from:"#node-final",to:"#node-unpatchify",kind:"main",fromSide:"bottom",toSide:"top"},
    {from:"#node-unpatchify",to:"#node-decoder",kind:"main",fromSide:"bottom",toSide:"top"},
    {from:"#spatial-output-x2",to:"#spatial-temporal-handoff",kind:"handoff",fromSide:"right",toSide:"left"},
    {from:"#spatial-temporal-handoff",to:"#temporal-input-x2",kind:"handoff",fromSide:"right",toSide:"left",route:"handoffRail",railGap:18},

    {from:"#cond-spatial-params",to:"#sp-adaln1",kind:"action",fromSide:"right",toSide:"left",route:"bus",busRef:"#block-panel",busRefSide:"left",busOffset:-18},
    {from:"#cond-spatial-params",to:"#sp-gate1",kind:"action",fromSide:"right",toSide:"left",route:"bus",busRef:"#block-panel",busRefSide:"left",busOffset:-18},
    {from:"#cond-spatial-params",to:"#sp-adaln2",kind:"action",fromSide:"right",toSide:"left",route:"bus",busRef:"#block-panel",busRefSide:"left",busOffset:-18},
    {from:"#cond-spatial-params",to:"#sp-gate2",kind:"action",fromSide:"right",toSide:"left",route:"bus",busRef:"#block-panel",busRefSide:"left",busOffset:-18},
    {from:"#cond-temporal-params",to:"#tm-adaln1",kind:"time",fromSide:"right",toSide:"right",route:"bus",busRef:"#block-panel",busRefSide:"right",busOffset:18},
    {from:"#cond-temporal-params",to:"#tm-gate1",kind:"time",fromSide:"right",toSide:"right",route:"bus",busRef:"#block-panel",busRefSide:"right",busOffset:18},
    {from:"#cond-temporal-params",to:"#tm-adaln2",kind:"time",fromSide:"right",toSide:"right",route:"bus",busRef:"#block-panel",busRefSide:"right",busOffset:18},
    {from:"#cond-temporal-params",to:"#tm-gate2",kind:"time",fromSide:"right",toSide:"right",route:"bus",busRef:"#block-panel",busRefSide:"right",busOffset:18},
    {from:"#cond-final-params",to:"#final-expanded-adaln",kind:"time",fromSide:"right",toSide:"top",route:"bus",busRef:"#final-panel",busRefSide:"left",busOffset:-18,label:"Final shift / scale(t)"}
  ];

  function anchor(selector, side){
    const element=document.querySelector(selector); if(!element)return null;
    const r=element.getBoundingClientRect(), c=canvas.getBoundingClientRect(), s=state.scale || 1;
    const left=(r.left-c.left)/s, top=(r.top-c.top)/s, width=r.width/s, height=r.height/s;
    if(side==="left")return{x:left,y:top+height/2};
    if(side==="right")return{x:left+width,y:top+height/2};
    if(side==="top")return{x:left+width/2,y:top};
    return{x:left+width/2,y:top+height};
  }
  function svgElement(tag,attrs){const el=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,String(v)));return el;}
  function pathFor(a,b,cfg){
    if(cfg.route==="bus"){const ref=cfg.busRef?anchor(cfg.busRef,cfg.busRefSide||"left"):null;const bx=ref?ref.x+(cfg.busOffset||0):cfg.busX;return `M${a.x},${a.y} H${bx} V${b.y} H${b.x}`;}
    if(cfg.route==="handoffRail"){const rx=Math.min(b.x-10,a.x+(cfg.railGap||18));return `M${a.x},${a.y} H${rx} V${b.y} H${b.x}`;}
    if((cfg.fromSide==="bottom"||cfg.fromSide==="top")&&(cfg.toSide==="top"||cfg.toSide==="bottom")){const my=(a.y+b.y)/2;return `M${a.x},${a.y} C${a.x},${my} ${b.x},${my} ${b.x},${b.y}`;}
    const mx=(a.x+b.x)/2;return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`;
  }
  function drawConnections(){
    connectorGroup.replaceChildren();
    const renderedPorts=new Set();
    connections.forEach(cfg=>{
      const a=anchor(cfg.from,cfg.fromSide||"right"),b=anchor(cfg.to,cfg.toSide||"left");if(!a||!b)return;
      const kindClass=cfg.kind==="main"?"main-wire":cfg.kind==="zoom"?"zoom-wire":cfg.kind==="action"?"action-wire":cfg.kind==="video"?"video-wire":cfg.kind==="handoff"?"handoff-wire":"time-wire";
      const path=svgElement("path",{d:pathFor(a,b,cfg),class:kindClass,"data-from":cfg.from,"data-to":cfg.to});
      const startRadius=cfg.kind==="handoff"?3:5,endRadius=cfg.kind==="handoff"?4:6;
      connectorGroup.append(path);
      const startKey=`${cfg.from}:${cfg.fromSide||"right"}:${cfg.kind}`;
      const endKey=`${cfg.to}:${cfg.toSide||"left"}:${cfg.kind}`;
      if(!renderedPorts.has(startKey)){connectorGroup.append(svgElement("circle",{cx:a.x,cy:a.y,r:startRadius,class:`connector-port ${cfg.kind} start-port`}));renderedPorts.add(startKey);}
      if(!renderedPorts.has(endKey)){connectorGroup.append(svgElement("circle",{cx:b.x,cy:b.y,r:endRadius,class:`connector-port ${cfg.kind} end-port`}));renderedPorts.add(endKey);}
      if(cfg.label){
        const point=path.getPointAtLength(path.getTotalLength()*(cfg.labelAt||.5));
        const text=svgElement("text",{x:point.x+(cfg.labelDx||0),y:point.y-7+(cfg.labelDy||0),class:`connector-label ${cfg.kind}`});
        text.textContent=cfg.label;connectorGroup.append(text);
      }
    });
  }

  function render(){ canvas.style.transform=`translate(${state.x}px,${state.y}px) scale(${state.scale})`; zoomValue.textContent=Math.round(state.scale*100)+"%"; applySemanticZoom(); }
  function applySemanticZoom(){const level=state.scale<.45?"overview":state.scale<.75?"structure":"detail";canvas.classList.remove("semantic-overview","semantic-structure","semantic-detail");canvas.classList.add("semantic-"+level);semanticLevelEl.textContent={overview:"总览层级",structure:"结构层级",detail:"计算层级"}[level];}
  function setDensity(density){state.density=density;canvas.classList.remove("density-learn","density-structure","density-source");canvas.classList.add("density-"+density);densityButtons.forEach(button=>{const active=button.dataset.density===density;button.classList.toggle("is-active",active);button.setAttribute("aria-pressed",String(active));});requestAnimationFrame(drawConnections);}
  function clamp(){ const w=viewport.clientWidth,h=viewport.clientHeight,m=70,sw=CW*state.scale,sh=CH*state.scale; state.x=sw<w?(w-sw)/2:Math.min(m,Math.max(w-sw-m,state.x)); state.y=sh<h?(h-sh)/2:Math.min(m,Math.max(h-sh-m,state.y)); }
  function fit(){ const sx=(viewport.clientWidth-30)/CW,sy=(viewport.clientHeight-30)/CH; state.scale=Math.max(.1,Math.min(.9,Math.min(sx,sy))); state.x=(viewport.clientWidth-CW*state.scale)/2; state.y=(viewport.clientHeight-CH*state.scale)/2; render(); }
  function zoom(next,cx=viewport.clientWidth/2,cy=viewport.clientHeight/2){ const old=state.scale; next=Math.max(.1,Math.min(1.35,next)); const px=(cx-state.x)/old,py=(cy-state.y)/old; state.scale=next; state.x=cx-px*next;state.y=cy-py*next;clamp();render(); }
  const regions={input:{x:20,y:100,w:820,h:900},video:{x:260,y:120,w:1320,h:1370},condition:{x:20,y:950,w:820,h:850},block:{x:820,y:110,w:790,h:1370},handoff:{x:820,y:110,w:790,h:1370},attention:{x:1570,y:110,w:720,h:1250},output:{x:830,y:1440,w:1460,h:340}};
  function setFocusMode(name){
    canvas.classList.toggle("is-handoff-focus",name==="handoff");
    canvas.classList.toggle("is-routing-focus",name==="condition");
    canvas.classList.toggle("is-video-focus",name==="video");
    document.querySelectorAll("[data-focus]").forEach(button=>{
      const active=button.dataset.focus===name;
      button.classList.toggle("is-active",active);
      button.setAttribute("aria-pressed",String(active));
    });
  }
  function focus(name){ const r=regions[name]; if(!r)return; setFocusMode(name); const s=Math.min((viewport.clientWidth-80)/r.w,(viewport.clientHeight-80)/r.h,.95); const minimum=name==="handoff"?.48:["condition","block","attention"].includes(name)?.76:.48; state.scale=Math.max(minimum,s); state.x=viewport.clientWidth/2-(r.x+r.w/2)*state.scale; state.y=viewport.clientHeight/2-(r.y+r.h/2)*state.scale; clamp();render(); }
  function setInspectorOpen(open){inspectorPanel.classList.toggle("is-collapsed",!open);inspectorReopen.classList.toggle("is-visible",!open);}
  const pathGroups={
    spatialInput:["node-latent","node-patch","node-position","node-pairs","spatial-input-x","sp-norm1","sp-adaln1","sp-attn","sp-gate1","sp-add1","sp-norm2","sp-adaln2","sp-mlp","sp-gate2","sp-add2","spatial-output-x2","spatial-temporal-handoff","temporal-input-x2"],
    spatialAdaLN1:["cond-actions","cond-action-encoder","cond-none","cond-spatial-params","spatial-input-x","sp-norm1","sp-adaln1","sp-attn"],
    gateMsa:["cond-actions","cond-action-encoder","cond-none","cond-spatial-params","sp-gate1","sp-add1"],
    spatialAdaLN2:["cond-actions","cond-action-encoder","cond-none","cond-spatial-params","sp-norm2","sp-adaln2","sp-mlp"],
    gateMlp:["cond-actions","cond-action-encoder","cond-none","cond-spatial-params","sp-gate2","sp-add2","spatial-output-x2"],
    spatialX2:["sp-mlp","sp-gate2","sp-add2","spatial-output-x2","spatial-temporal-handoff","temporal-input-x2"],
    temporalHandoff:["spatial-output-x2","spatial-temporal-handoff","temporal-input-x2"],
    temporalX2:["spatial-output-x2","spatial-temporal-handoff","temporal-input-x2","tm-norm1"],
    spatialAttn:["sp-adaln1","sp-attn","attn-expanded-input"],
    temporalAttn:["temporal-input-x2","tm-norm1","tm-adaln1","tm-attn-reshape","tm-attn","tm-attn-restore"]
  };
  function showDetail(key,origin){ const d=details[key]; if(!d)return; document.querySelectorAll(".detail-node").forEach(n=>n.classList.toggle("is-selected",n===origin)); inspector.kind.textContent=d[0];inspector.title.textContent=d[1];inspector.description.textContent=d[2];inspector.shape.textContent=d[3];inspector.source.textContent=d[4];showTensorTrace(key);highlightPath(key,origin);updateDimensionTrace(key);setInspectorOpen(true); }
  function showTensorTrace(key){const item=tensorDetails[key];tensorTrace.hidden=!item;if(!item)return;tensorFields.name.textContent=item.name;tensorFields.produced.textContent=item.produced;tensorFields.consumed.textContent=item.consumed;tensorFields.action.textContent=item.action;}
  function highlightPath(key,origin){clearPath(false);const ids=new Set(pathGroups[key]||[origin?.id].filter(Boolean));if(!ids.size)return;canvas.classList.add("is-path-focus");ids.forEach(id=>document.getElementById(id)?.classList.add("path-active"));document.querySelectorAll("#dynamic-connectors path").forEach(path=>{const from=(path.dataset.from||"").replace("#","");const to=(path.dataset.to||"").replace("#","");path.classList.toggle("path-active",ids.has(from)&&ids.has(to));});clearPathButton.hidden=false;requestAnimationFrame(()=>framePath(ids));}
  function framePath(ids){const elements=[...ids].map(id=>document.getElementById(id)).filter(Boolean);if(elements.length<2)return;const cr=canvas.getBoundingClientRect(),s=state.scale||1;const boxes=elements.map(el=>{const r=el.getBoundingClientRect();return{l:(r.left-cr.left)/s,t:(r.top-cr.top)/s,r:(r.right-cr.left)/s,b:(r.bottom-cr.top)/s};});const left=Math.min(...boxes.map(b=>b.l)),top=Math.min(...boxes.map(b=>b.t)),right=Math.max(...boxes.map(b=>b.r)),bottom=Math.max(...boxes.map(b=>b.b)),pad=110;const next=Math.min(.92,(viewport.clientWidth-100)/(right-left+pad*2),(viewport.clientHeight-100)/(bottom-top+pad*2));state.scale=Math.max(.48,next);state.x=viewport.clientWidth/2-(left+right)/2*state.scale;state.y=viewport.clientHeight/2-(top+bottom)/2*state.scale;clamp();render();}
  function clearPath(hideButton=true){canvas.classList.remove("is-path-focus");document.querySelectorAll(".path-active").forEach(el=>el.classList.remove("path-active"));if(hideButton)clearPathButton.hidden=true;}
  function updateDimensionTrace(key){const temporal=/temporal/i.test(key);const spatial=/spatial|gateMsa|gateMlp|mlp/i.test(key)&&!temporal;dimensionTrace.classList.toggle("spatial",spatial);dimensionTrace.classList.toggle("temporal",temporal);dimensionMessage.textContent=temporal?"Temporal：B×S → B·S，Attention axis=T。":spatial?"Spatial：B×T → B·T，Attention axis=S。":"视频身份保持 [B,T,S,C]。";}
  document.querySelectorAll(".detail-node").forEach(n=>n.addEventListener("click",()=>showDetail(n.dataset.detail,n)));
  densityButtons.forEach(button=>button.addEventListener("click",()=>setDensity(button.dataset.density)));
  clearPathButton.addEventListener("click",()=>clearPath(true));
  document.getElementById("close-inspector").addEventListener("click",()=>setInspectorOpen(false));
  inspectorReopen.addEventListener("click",()=>setInspectorOpen(true));
  document.getElementById("zoom-in").addEventListener("click",()=>zoom(state.scale+.1));
  document.getElementById("zoom-out").addEventListener("click",()=>zoom(state.scale-.1));
  document.getElementById("fit-canvas").addEventListener("click",()=>{setFocusMode(null);clearPath(true);fit();});
  document.getElementById("actual-size").addEventListener("click",()=>zoom(1));
  document.querySelectorAll("[data-focus]").forEach(b=>b.addEventListener("click",()=>focus(b.dataset.focus)));
  viewport.addEventListener("pointerdown",e=>{if(e.target.closest("button,a,.canvas-inspector"))return;state.drag=true;state.pointer=e.pointerId;state.startX=e.clientX;state.startY=e.clientY;state.originX=state.x;state.originY=state.y;viewport.classList.add("is-dragging");viewport.setPointerCapture(e.pointerId)});
  viewport.addEventListener("pointermove",e=>{if(!state.drag||e.pointerId!==state.pointer)return;state.x=state.originX+e.clientX-state.startX;state.y=state.originY+e.clientY-state.startY;clamp();render()});
  function endDrag(e){if(!state.drag||e.pointerId!==state.pointer)return;state.drag=false;state.pointer=null;viewport.classList.remove("is-dragging")}
  viewport.addEventListener("pointerup",endDrag);viewport.addEventListener("pointercancel",endDrag);
  viewport.addEventListener("wheel",e=>{e.preventDefault();const r=viewport.getBoundingClientRect();zoom(state.scale*(e.deltaY<0?1.1:.9),e.clientX-r.left,e.clientY-r.top)},{passive:false});
  viewport.addEventListener("keydown",e=>{if(e.key==="0")fit();else if(e.key==="+"||e.key==="=")zoom(state.scale+.1);else if(e.key==="-")zoom(state.scale-.1);else if(e.key==="ArrowLeft")state.x+=60;else if(e.key==="ArrowRight")state.x-=60;else if(e.key==="ArrowUp")state.y+=60;else if(e.key==="ArrowDown")state.y-=60;else return;clamp();render()});
  window.addEventListener("resize",()=>{fit();drawConnections();});
  window.addEventListener("load",()=>requestAnimationFrame(()=>requestAnimationFrame(drawConnections)));

  const modes={spatial:{input:"[B·T,S,1152]",q:"[B·T,16,S,72]",k:"[B·T,16,S,72]",v:"[B·T,16,S,72]",score:"[B·T,16,S,S]",weighted:"[B·T,16,S,72]",output:"[B·T,S,1152]",rope:"Spatial：无 RoPE",mask:"OFF",causal:false},temporal:{input:"[B·S,T,1152]",q:"[B·S,16,T,72]",k:"[B·S,16,T,72]",v:"[B·S,16,T,72]",score:"[B·S,16,T,T]",weighted:"[B·S,16,T,72]",output:"[B·S,T,1152]",rope:"Temporal：Q/K 使用 RoPE",mask:"下三角 ON",causal:true}};
  function setMode(key){const m=modes[key];document.getElementById("attn-input").textContent=m.input;document.getElementById("q-shape").textContent=m.q;document.getElementById("k-shape").textContent=m.k;document.getElementById("v-shape").textContent=m.v;document.getElementById("score-shape").textContent=m.score;document.getElementById("weighted-shape").textContent=m.weighted;document.getElementById("attn-output").textContent=m.output;document.getElementById("rope-state").textContent=m.rope;document.querySelector("#causal-mask code").textContent=m.mask;document.getElementById("causal-mask").classList.toggle("is-off",!m.causal);document.querySelectorAll("[data-attention]").forEach(b=>{const on=b.dataset.attention===key;b.classList.toggle("is-active",on);b.setAttribute("aria-pressed",String(on))})}
  document.querySelectorAll("[data-attention]").forEach(b=>b.addEventListener("click",()=>setMode(b.dataset.attention)));
  if("ResizeObserver" in window){const connectorResizeObserver=new ResizeObserver(()=>requestAnimationFrame(drawConnections));const selectors=new Set(connections.flatMap(item=>[item.from,item.to,item.busRef].filter(Boolean)));selectors.forEach(selector=>{const element=document.querySelector(selector);if(element)connectorResizeObserver.observe(element);});}
  setFocusMode(null);setDensity("learn");showDetail("latent");setInspectorOpen(false);setMode("spatial");requestAnimationFrame(()=>{fit();drawConnections();});
  if(document.fonts?.ready)document.fonts.ready.then(drawConnections);
})();
