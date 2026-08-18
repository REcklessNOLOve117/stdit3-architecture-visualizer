(function () {
  "use strict";

  const key = "stdit3-display-preferences";
  const defaults = { comfort: false, font: "100", grid: true, motion: true };
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(key) || "{}"); } catch (_error) { saved = {}; }
  const prefs = Object.assign({}, defaults, saved);

  const root = document.documentElement;
  const body = document.body;
  const shell = document.createElement("details");
  shell.className = "display-settings";
  shell.innerHTML = [
    "<summary>阅读设置</summary>",
    "<div class=\"display-settings-panel\">",
    "<strong>Comfort Dark</strong>",
    "<label><span>柔和背景</span><input type=\"checkbox\" data-pref=\"comfort\"></label>",
    "<label><span>字号</span><select data-pref=\"font\"><option value=\"90\">90%</option><option value=\"100\">100%</option><option value=\"115\">115%</option></select></label>",
    "<label><span>网格</span><input type=\"checkbox\" data-pref=\"grid\"></label>",
    "<label><span>动画</span><input type=\"checkbox\" data-pref=\"motion\"></label>",
    "</div>"
  ].join("");
  body.appendChild(shell);

  const controls = {
    comfort: shell.querySelector('[data-pref="comfort"]'),
    font: shell.querySelector('[data-pref="font"]'),
    grid: shell.querySelector('[data-pref="grid"]'),
    motion: shell.querySelector('[data-pref="motion"]')
  };

  function apply() {
    body.classList.toggle("comfort-theme", prefs.comfort);
    body.classList.toggle("grid-off", !prefs.grid);
    body.classList.toggle("motion-off", !prefs.motion);
    root.dataset.fontScale = prefs.font;
    controls.comfort.checked = prefs.comfort;
    controls.font.value = prefs.font;
    controls.grid.checked = prefs.grid;
    controls.motion.checked = prefs.motion;
    try { localStorage.setItem(key, JSON.stringify(prefs)); } catch (_error) { /* local file privacy mode */ }
  }

  controls.comfort.addEventListener("change", function () { prefs.comfort = controls.comfort.checked; apply(); });
  controls.font.addEventListener("change", function () { prefs.font = controls.font.value; apply(); });
  controls.grid.addEventListener("change", function () { prefs.grid = controls.grid.checked; apply(); });
  controls.motion.addEventListener("change", function () { prefs.motion = controls.motion.checked; apply(); });
  apply();
}());
