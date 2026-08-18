(() => {
  "use strict";
  const CW = 2850, CH = 1700;
  const viewport = document.getElementById("architecture-viewport");
  const canvas = document.getElementById("architecture-canvas");
  const zoomValue = document.getElementById("zoom-value");
  const state = { scale: .5, x: 0, y: 0, drag: false, pointer: null, startX: 0, startY: 0, originX: 0, originY: 0 };

  const details = {
    rgb:["OUTSIDE STDiT3","RGB 视频","数据层面的原始输入。VAE 编码和加噪发生在 STDiT3.forward 之前。","[B,3,12,H₀,W₀]","模型外部：dataset / VAE"],
    vae:["OUTSIDE STDiT3","VAE Encoder + 加噪","VAE 把 RGB 压缩到 4 通道 latent，生成过程再依据 timestep 构造噪声 latent xₜ。","RGB → latent xₜ","模型外部：VAE / sampler"],
    latent:["MODEL INPUT","噪声 latent xₜ","这是 WMPO STDiT3.forward 真正接收的视频张量；12 帧示例对应 4 个历史位置和 8 个未来位置。","[B,4,12,Hx,Wx]","WMPO stdit3.py · L412–421"],
    patch:["VIDEO TOKENIZATION","PatchEmbed3D","Conv3D 使用 kernel=stride=(1,2,2)，保持帧数不变，每个 2×2 latent patch 投影为一个 1152 维 token。","[B,4,12,Hx,Wx] → [B,12,S,1152]","layers/blocks.py · L79–129"],
    position:["POSITION","2D sin/cos 位置编码","为空间 H×W patch 生成位置编码，同一套编码沿 12 个时间位置广播。","[1,S,1152] + [B,12,S,1152]","layers/blocks.py · L944–993"],
    pairs:["BACKBONE","28 × Spatial → Temporal","每对先执行 action-conditioned Spatial Block，再执行 timestep-conditioned causal Temporal Block；28 对参数彼此独立。","全程 [B,12,S,1152]","WMPO stdit3.py · L484–487"],
    final:["OUTPUT HEAD","T2IFinalLayer","全局 timestep 先经过 SiLU + Linear 生成 final shift/scale，再用它调制无 affine 的 LayerNorm 输出，最后预测每个 latent patch。","Eₜ(t) → SiLU → Linear(1152,2304) → shift / scale","layers/blocks.py · L709–721"],
    unpatchify:["OUTPUT SHAPE","unpatchify","把每个 token 的 1×2×2 patch prediction 重新排列为 5D latent 网格。","[B,12,S,4·Cout] → [B,Cout,12,Hx,Wx]","WMPO stdit3.py · L503–520"],
    decoder:["OUTSIDE STDiT3","Sampler + VAE Decoder","模型输出仍是 latent 预测；采样器更新 xₜ，最终由 VAE Decoder 得到可见视频。","latent prediction → RGB","模型外部：sampling pipeline"],
    actions:["ACTION INPUT","8 个未来 7D actions","每个未来时间位置对应一个机器人动作向量，典型维度包含平移、旋转和 gripper。","[B,8,7]","simplevla_webdataset.py · Ta=8, action_dim=7"],
    actionEncoder:["ACTION CONDITION","ActionEncoder","每个 7D action 独立经过 Linear、SiLU、Linear 投影到 hidden size。这里的 SiLU 属于 action 编码器；编码结果不直接加到 video token。","Linear(7,4608) → SiLU → Linear(4608,1152)","WMPO stdit3.py · L37–72"],
    none:["ACTION ALIGNMENT","prepend NONE ×4","在 8 个未来 action embedding 前补 4 个可学习 NONE，使条件与 12 帧位置一一对齐。","[B,8,1152] → [B,12,1152]","WMPO stdit3.py · L54–65"],
    spatialCondition:["SPATIAL PARAMS","逐帧 Action + timestep 参数","每帧条件 cₛ=Eₜ(t)+Eₐ(aᵢ) 先过 SiLU，再由 Linear(1152,6912) 生成六组参数。绿色虚线只表示 shift / scale / gate 的路由，不是 Spatial 输出或 x₂ 广播。","cₛ → SiLU → Linear(1152,6·1152) → chunk(6)","WMPO stdit3.py · L295–298、L448–457"],
    timestep:["DIFFUSION CONDITION","timestep","标量 diffusion / flow 时间决定当前噪声强度。","[B]","WMPO stdit3.py · L448–462"],
    tEmbed:["DIFFUSION CONDITION","TimestepEmbedder","先构造 256 维 sin/cos frequency embedding，再经过 Linear、SiLU、Linear 得到 1152 维全局条件。","[B] → sin/cos₍₂₅₆₎ → Linear → SiLU → Linear → [B,1152]","layers/blocks.py · L792–825"],
    temporalCondition:["TEMPORAL PARAMS","Temporal t_block","全局 Eₜ(t) 经过同一个 t_block：SiLU + Linear(1152,6912)，再切成六组 Temporal AdaLN/gate 参数；这里不再加入 action embedding。","Eₜ(t) → SiLU → Linear(1152,6·1152) → chunk(6)","WMPO stdit3.py · L295–298、L456–488"],
    finalCondition:["FINAL PARAMS","Final shift / scale","Final Layer 使用自己的 SiLU + Linear(1152,2304) 生成 shift/scale。它不直接读取 action；action 影响已经保存在 28 对 Block 输出的 hidden 中。","Eₜ(t) → SiLU → Linear(1152,2·1152) → shift, scale","layers/blocks.py · L709–721"],
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

  const connectorGroup = document.getElementById("dynamic-connectors");
  const NS = "http://www.w3.org/2000/svg";
  const connections = [
    {from:"#node-rgb",to:"#node-vae",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#node-vae",to:"#node-latent",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#node-latent",to:"#node-patch",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#node-patch",to:"#node-position",kind:"main",fromSide:"bottom",toSide:"top"},
    {from:"#node-position",to:"#node-pairs",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#node-pairs",to:"#pair-1",kind:"main",fromSide:"bottom",toSide:"top"},
    {from:"#pair-1",to:"#pair-2",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#pair-2",to:"#pair-27",kind:"main",fromSide:"bottom",toSide:"top",label:"… 中间 24 对 …"},
    {from:"#pair-27",to:"#pair-28",kind:"main",fromSide:"right",toSide:"left"},
    {from:"#pair-28",to:"#node-final",kind:"main",fromSide:"bottom",toSide:"top"},
    {from:"#node-final",to:"#node-unpatchify",kind:"main",fromSide:"bottom",toSide:"top"},
    {from:"#node-unpatchify",to:"#node-decoder",kind:"main",fromSide:"bottom",toSide:"top"},
    {from:"#sp-add2",to:"#spatial-output-x2",kind:"handoff",fromSide:"bottom",toSide:"top"},
    {from:"#spatial-output-x2",to:"#spatial-temporal-handoff",kind:"handoff",fromSide:"right",toSide:"left"},
    {from:"#spatial-temporal-handoff",to:"#temporal-input-x2",kind:"handoff",fromSide:"right",toSide:"left",route:"handoffRail",railGap:18},
    {from:"#temporal-input-x2",to:"#tm-norm1",kind:"handoff",fromSide:"bottom",toSide:"top"},

    {from:"#node-pairs",to:"#block-title",kind:"zoom",fromSide:"right",toSide:"left",label:"展开 Pair ℓ"},
    {from:"#sp-attn",to:"#attn-expanded-input",kind:"zoom",fromSide:"right",toSide:"left",label:"展开 Spatial Q/K/V"},
    {from:"#tm-attn",to:"#attn-expanded-input",kind:"zoom",fromSide:"right",toSide:"left",label:"Temporal 使用同一 Attention 类"},
    {from:"#sp-mlp",to:"#mlp-expanded-input",kind:"zoom",fromSide:"right",toSide:"left",label:"展开 Spatial MLP"},
    {from:"#tm-mlp",to:"#mlp-expanded-input",kind:"zoom",fromSide:"right",toSide:"left",label:"Temporal 复用 MLP 结构"},
    {from:"#pair-28",to:"#final-expanded-input",kind:"zoom",fromSide:"right",toSide:"left",label:"Pair 28 输出"},

    {from:"#cond-spatial-params",to:"#sp-adaln1",kind:"action",fromSide:"right",toSide:"left",route:"bus",busX:850},
    {from:"#cond-spatial-params",to:"#sp-gate1",kind:"action",fromSide:"right",toSide:"left",route:"bus",busX:860},
    {from:"#cond-spatial-params",to:"#sp-adaln2",kind:"action",fromSide:"right",toSide:"left",route:"bus",busX:870},
    {from:"#cond-spatial-params",to:"#sp-gate2",kind:"action",fromSide:"right",toSide:"left",route:"bus",busX:880},
    {from:"#cond-temporal-params",to:"#tm-adaln1",kind:"time",fromSide:"right",toSide:"right",route:"bus",busX:1534},
    {from:"#cond-temporal-params",to:"#tm-gate1",kind:"time",fromSide:"right",toSide:"right",route:"bus",busX:1544},
    {from:"#cond-temporal-params",to:"#tm-adaln2",kind:"time",fromSide:"right",toSide:"right",route:"bus",busX:1554},
    {from:"#cond-temporal-params",to:"#tm-gate2",kind:"time",fromSide:"right",toSide:"right",route:"bus",busX:1564},
    {from:"#cond-final-params",to:"#final-expanded-adaln",kind:"time",fromSide:"right",toSide:"top",route:"bus",busX:840,label:"Final shift / scale(t)"}
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
    if(cfg.route==="bus"){const bx=cfg.busX;return `M${a.x},${a.y} H${bx} V${b.y} H${b.x}`;}
    if(cfg.route==="handoffRail"){const rx=Math.min(b.x-10,a.x+(cfg.railGap||18));return `M${a.x},${a.y} H${rx} V${b.y} H${b.x}`;}
    if((cfg.fromSide==="bottom"||cfg.fromSide==="top")&&(cfg.toSide==="top"||cfg.toSide==="bottom")){const my=(a.y+b.y)/2;return `M${a.x},${a.y} C${a.x},${my} ${b.x},${my} ${b.x},${b.y}`;}
    const mx=(a.x+b.x)/2;return `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`;
  }
  function drawConnections(){
    connectorGroup.replaceChildren();
    connections.forEach(cfg=>{
      const a=anchor(cfg.from,cfg.fromSide||"right"),b=anchor(cfg.to,cfg.toSide||"left");if(!a||!b)return;
      const kindClass=cfg.kind==="main"?"main-wire":cfg.kind==="zoom"?"zoom-wire":cfg.kind==="action"?"action-wire":cfg.kind==="handoff"?"handoff-wire":"time-wire";
      const path=svgElement("path",{d:pathFor(a,b,cfg),class:kindClass,"data-from":cfg.from,"data-to":cfg.to});
      const start=svgElement("circle",{cx:a.x,cy:a.y,r:5,class:`connector-port ${cfg.kind} start-port`});
      const end=svgElement("circle",{cx:b.x,cy:b.y,r:6,class:`connector-port ${cfg.kind} end-port`});
      connectorGroup.append(start,end,path);
      if(cfg.label){
        connectorGroup.append(path);
        const point=path.getPointAtLength(path.getTotalLength()*(cfg.labelAt||.5));
        const text=svgElement("text",{x:point.x+(cfg.labelDx||0),y:point.y-7+(cfg.labelDy||0),class:`connector-label ${cfg.kind}`});
        text.textContent=cfg.label;connectorGroup.append(text);
      }
    });
  }

  function render(){ canvas.style.transform=`translate(${state.x}px,${state.y}px) scale(${state.scale})`; zoomValue.textContent=Math.round(state.scale*100)+"%"; }
  function clamp(){ const w=viewport.clientWidth,h=viewport.clientHeight,m=70,sw=CW*state.scale,sh=CH*state.scale; state.x=sw<w?(w-sw)/2:Math.min(m,Math.max(w-sw-m,state.x)); state.y=sh<h?(h-sh)/2:Math.min(m,Math.max(h-sh-m,state.y)); }
  function fit(){ const sx=(viewport.clientWidth-30)/CW,sy=(viewport.clientHeight-30)/CH; state.scale=Math.max(.1,Math.min(.9,Math.min(sx,sy))); state.x=(viewport.clientWidth-CW*state.scale)/2; state.y=(viewport.clientHeight-CH*state.scale)/2; render(); }
  function zoom(next,cx=viewport.clientWidth/2,cy=viewport.clientHeight/2){ const old=state.scale; next=Math.max(.1,Math.min(1.35,next)); const px=(cx-state.x)/old,py=(cy-state.y)/old; state.scale=next; state.x=cx-px*next;state.y=cy-py*next;clamp();render(); }
  const regions={input:{x:20,y:100,w:820,h:900},condition:{x:20,y:950,w:820,h:750},block:{x:820,y:110,w:790,h:1270},handoff:{x:840,y:230,w:710,h:810},attention:{x:1570,y:110,w:720,h:1250},output:{x:830,y:1340,w:1460,h:340}};
  function setFocusMode(name){
    canvas.classList.toggle("is-handoff-focus",name==="handoff");
    canvas.classList.toggle("is-routing-focus",name==="condition");
    document.querySelectorAll("[data-focus]").forEach(button=>{
      const active=button.dataset.focus===name;
      button.classList.toggle("is-active",active);
      button.setAttribute("aria-pressed",String(active));
    });
  }
  function focus(name){ const r=regions[name]; if(!r)return; setFocusMode(name); const s=Math.min((viewport.clientWidth-80)/r.w,(viewport.clientHeight-80)/r.h,.95); state.scale=Math.max(.3,s); state.x=viewport.clientWidth/2-(r.x+r.w/2)*state.scale; state.y=viewport.clientHeight/2-(r.y+r.h/2)*state.scale; clamp();render(); }
  function setInspectorOpen(open){inspectorPanel.classList.toggle("is-collapsed",!open);inspectorReopen.classList.toggle("is-visible",!open);}
  function showDetail(key){ const d=details[key]; if(!d)return; document.querySelectorAll(".detail-node").forEach(n=>n.classList.toggle("is-selected",n.dataset.detail===key)); inspector.kind.textContent=d[0];inspector.title.textContent=d[1];inspector.description.textContent=d[2];inspector.shape.textContent=d[3];inspector.source.textContent=d[4];setInspectorOpen(true); }
  document.querySelectorAll(".detail-node").forEach(n=>n.addEventListener("click",()=>showDetail(n.dataset.detail)));
  document.getElementById("close-inspector").addEventListener("click",()=>setInspectorOpen(false));
  inspectorReopen.addEventListener("click",()=>setInspectorOpen(true));
  document.getElementById("zoom-in").addEventListener("click",()=>zoom(state.scale+.1));
  document.getElementById("zoom-out").addEventListener("click",()=>zoom(state.scale-.1));
  document.getElementById("fit-canvas").addEventListener("click",()=>{setFocusMode(null);fit();});
  document.getElementById("actual-size").addEventListener("click",()=>zoom(1));
  document.querySelectorAll("[data-focus]").forEach(b=>b.addEventListener("click",()=>focus(b.dataset.focus)));
  viewport.addEventListener("pointerdown",e=>{if(e.target.closest("button,a,.canvas-inspector"))return;state.drag=true;state.pointer=e.pointerId;state.startX=e.clientX;state.startY=e.clientY;state.originX=state.x;state.originY=state.y;viewport.classList.add("is-dragging");viewport.setPointerCapture(e.pointerId)});
  viewport.addEventListener("pointermove",e=>{if(!state.drag||e.pointerId!==state.pointer)return;state.x=state.originX+e.clientX-state.startX;state.y=state.originY+e.clientY-state.startY;clamp();render()});
  function endDrag(e){if(!state.drag||e.pointerId!==state.pointer)return;state.drag=false;state.pointer=null;viewport.classList.remove("is-dragging")}
  viewport.addEventListener("pointerup",endDrag);viewport.addEventListener("pointercancel",endDrag);
  viewport.addEventListener("wheel",e=>{e.preventDefault();const r=viewport.getBoundingClientRect();zoom(state.scale*(e.deltaY<0?1.1:.9),e.clientX-r.left,e.clientY-r.top)},{passive:false});
  viewport.addEventListener("keydown",e=>{if(e.key==="0")fit();else if(e.key==="+"||e.key==="=")zoom(state.scale+.1);else if(e.key==="-")zoom(state.scale-.1);else if(e.key==="ArrowLeft")state.x+=60;else if(e.key==="ArrowRight")state.x-=60;else if(e.key==="ArrowUp")state.y+=60;else if(e.key==="ArrowDown")state.y-=60;else return;clamp();render()});
  window.addEventListener("resize",()=>{fit();drawConnections();});

  const modes={spatial:{input:"[B·T,S,1152]",q:"[B·T,16,S,72]",k:"[B·T,16,S,72]",v:"[B·T,16,S,72]",score:"[B·T,16,S,S]",weighted:"[B·T,16,S,72]",output:"[B·T,S,1152]",rope:"Spatial：无 RoPE",mask:"OFF",causal:false},temporal:{input:"[B·S,T,1152]",q:"[B·S,16,T,72]",k:"[B·S,16,T,72]",v:"[B·S,16,T,72]",score:"[B·S,16,T,T]",weighted:"[B·S,16,T,72]",output:"[B·S,T,1152]",rope:"Temporal：Q/K 使用 RoPE",mask:"下三角 ON",causal:true}};
  function setMode(key){const m=modes[key];document.getElementById("attn-input").textContent=m.input;document.getElementById("q-shape").textContent=m.q;document.getElementById("k-shape").textContent=m.k;document.getElementById("v-shape").textContent=m.v;document.getElementById("score-shape").textContent=m.score;document.getElementById("weighted-shape").textContent=m.weighted;document.getElementById("attn-output").textContent=m.output;document.getElementById("rope-state").textContent=m.rope;document.querySelector("#causal-mask code").textContent=m.mask;document.getElementById("causal-mask").classList.toggle("is-off",!m.causal);document.querySelectorAll("[data-attention]").forEach(b=>{const on=b.dataset.attention===key;b.classList.toggle("is-active",on);b.setAttribute("aria-pressed",String(on))})}
  document.querySelectorAll("[data-attention]").forEach(b=>b.addEventListener("click",()=>setMode(b.dataset.attention)));
  setFocusMode(null);showDetail("latent");setInspectorOpen(false);setMode("spatial");requestAnimationFrame(()=>{fit();drawConnections();});
  if(document.fonts?.ready)document.fonts.ready.then(drawConnections);
})();
