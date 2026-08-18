(function () {
  "use strict";

  const buttons = Array.from(document.querySelectorAll(".step-button"));
  const panels = Array.from(document.querySelectorAll(".lesson-step"));
  const prevButton = document.getElementById("prev-step");
  const nextButton = document.getElementById("next-step");
  const counter = document.getElementById("step-counter");
  const title = document.getElementById("step-title");
  const progress = document.querySelector(".progress");
  const progressBar = document.getElementById("progress-bar");
  const realToggle = document.getElementById("real-toggle");
  const stage = document.getElementById("lesson-stage");

  const titles = [
    "ActionEncoder：把物理动作翻译成 condition",
    "Action embedding 与 timestep embedding 相加",
    "6H = 6 组 × 每组 H 维：把 12 个数切成 6 个二维向量",
    "Video hidden 先经过 LayerNorm",
    "Self-Attention 前的第一处 Action-AdaLN",
    "Action 改变 Q/K/V，gate 控制残差写回",
    "MLP 前的第二处 Action-AdaLN",
    "Spatial 输出 x₂ 再进入 Temporal Block",
    "同一 condition 复用于 28 层，每层参数仍不同"
  ];

  let current = 0;

  function clampStep(index) {
    return Math.max(0, Math.min(panels.length - 1, index));
  }

  function showStep(index, focusStage) {
    current = clampStep(index);
    buttons.forEach(function (button, buttonIndex) {
      const active = buttonIndex === current;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-current", active ? "step" : "false");
    });
    panels.forEach(function (panel, panelIndex) {
      const active = panelIndex === current;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });

    const number = current + 1;
    counter.textContent = number + " / " + panels.length;
    title.textContent = titles[current];
    progress.setAttribute("aria-valuenow", String(number));
    progressBar.style.width = ((number / panels.length) * 100) + "%";
    prevButton.disabled = current === 0;
    nextButton.disabled = current === panels.length - 1;

    if (focusStage) {
      stage.scrollIntoView({ behavior: "smooth", block: "start" });
      stage.focus({ preventScroll: true });
    }
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      showStep(Number(button.dataset.step), true);
    });
  });

  prevButton.addEventListener("click", function () {
    showStep(current - 1, true);
  });

  nextButton.addEventListener("click", function () {
    showStep(current + 1, true);
  });

  realToggle.addEventListener("change", function () {
    document.body.classList.toggle("show-real", realToggle.checked);
  });

  document.addEventListener("keydown", function (event) {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      showStep(current - 1, false);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      showStep(current + 1, false);
    }
  });

  panels.forEach(function (panel) {
    panel.hidden = true;
  });
  showStep(0, false);
}());
