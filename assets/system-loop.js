(() => {
  "use strict";
  const buttons = [...document.querySelectorAll(".phase-button")];
  const parts = [...document.querySelectorAll("#loop-diagram .diagram-part")];
  const fields = {
    kind: document.getElementById("phase-kind"),
    title: document.getElementById("phase-title"),
    description: document.getElementById("phase-description"),
    data: document.getElementById("phase-data"),
    update: document.getElementById("phase-update"),
    frozen: document.getElementById("phase-frozen"),
    formula: document.getElementById("phase-formula")
  };

  const phases = {
    all: {
      kind: "COMPLETE SYSTEM",
      title: "三个模型各司其职，形成可训练的 imagined environment",
      description: "Policy 与 World Model 交替生成完整 imagined trajectory；Reward Model 给出任务结果；GRPO 使用组内相对奖励更新 Policy。",
      data: "frames → actions → future frames → trajectory → reward",
      update: "Policy θ；行为对齐阶段单独更新 World Model φ",
      frozen: "在 Policy GRPO 阶段：World Model φ 与 Reward Model ψ",
      formula: "aₜ ~ πθ(a|s)　·　ŝₜ₊₁ ~ pφ(s'|s,a)　·　r = Rψ(τ)"
    },
    act: {
      kind: "STEP 1 · POLICY ACTION",
      title: "VLA Policy 根据视觉状态和指令采样 action chunk",
      description: "Policy 读取最近 m 帧与语言指令，一次预测未来 K=8 个动作。动作不是 World Model 自己生成的。",
      data: "Iᵢ₋ₘ:ᵢ + instruction g → aᵢ:ᵢ₊K",
      update: "rollout 阶段不立即更新参数",
      frozen: "Policy θ、World φ、Reward ψ 均用于前向采样",
      formula: "aᵢ:ᵢ₊K ~ πθ(a | Iᵢ₋ₘ:ᵢ, g)"
    },
    imagine: {
      kind: "STEP 2 · WORLD IMAGINATION",
      title: "World Model 把 action 变成可见的未来后果",
      description: "STDiT3 接收历史帧和逐帧 action，生成下一段 K 帧。最新 imagined frames 回到 Policy，继续下一轮 action–frame 交替。",
      data: "history frames + action chunk → K imagined frames",
      update: "Policy optimization 时不更新 World Model",
      frozen: "World Model φ 作为 learned environment",
      formula: "Îᵢ:ᵢ₊K ~ pφ(Iᵢ:ᵢ₊K | Iᵢ₋c:ᵢ, aᵢ:ᵢ₊K)"
    },
    score: {
      kind: "STEP 3 · TRAJECTORY REWARD",
      title: "Reward Model 评价完整轨迹，而不是单个动作",
      description: "当 Policy–World 交替生成完整 trial 后，VideoMAE Reward Model 对视频 clip 滑窗评估，输出任务成功概率和二值结果。",
      data: "imagined trajectory τ → success probability → y∈{0,1}",
      update: "本阶段只做 Reward Model 前向评估",
      frozen: "Reward Model ψ 在 Policy GRPO 时冻结",
      formula: "y = Rψ(τ),　y ∈ {0,1}"
    },
    update: {
      kind: "STEP 4 · GRPO POLICY UPDATE",
      title: "同一初始状态的多条轨迹形成组内相对奖励",
      description: "WMPO 从同一 s₀ 采样一组 trajectories，计算归一化 advantage 和 clipped policy objective，只更新 VLA Policy。",
      data: "group rewards {R₁…Rᴳ} + policy log-probabilities",
      update: "VLA Policy θ",
      frozen: "World Model φ、Reward Model ψ",
      formula: "Âᵢ = (Rᵢ − mean(R)) / std(R)　→　update θ"
    },
    align: {
      kind: "STEP 5 · POLICY BEHAVIOR ALIGNMENT",
      title: "用当前 Policy 的真实行为重新对齐 World Model",
      description: "专家数据偏成功，不能覆盖 Policy 的碰撞和失败。WMPO 收集真实 Policy rollout，单独微调 World Model，使 imagined failures 更可信。",
      data: "real policy rollouts → video/action training pairs",
      update: "World Model φ",
      frozen: "这不是 GRPO 的 reward gradient，也不更新 Reward Model",
      formula: "φ ← fine-tune(real trajectories from πθ)"
    }
  };

  function setPhase(key) {
    const phase = phases[key];
    if (!phase) return;
    buttons.forEach(button => {
      const active = button.dataset.phase === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    parts.forEach(part => {
      const tags = (part.dataset.part || "").split(/\s+/);
      part.classList.toggle("is-muted", key !== "all" && !tags.includes(key));
    });
    Object.keys(fields).forEach(name => { fields[name].textContent = phase[name]; });
  }

  buttons.forEach(button => button.addEventListener("click", () => setPhase(button.dataset.phase)));
  setPhase("all");
})();
