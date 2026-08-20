(() => {
  const svg = document.getElementById('alignment-timeline');
  const videoHzSelect = document.getElementById('video-hz');
  const actionHzSelect = document.getElementById('action-hz');
  const actionButtons = [...document.querySelectorAll('.action-button')];
  const presetButtons = [...document.querySelectorAll('.preset-button')];
  let selectedAction = 0;

  const sub = (base, offset) => {
    if (offset === 0) return base;
    return `${base}${offset > 0 ? '+' : ''}${offset}`;
  };

  const ms = value => `${value.toFixed(1)} ms`;

  function render() {
    const videoHz = Number(videoHzSelect.value);
    const actionHz = Number(actionHzSelect.value);
    const framePeriod = 1000 / videoHz;
    const controlPeriod = 1000 / actionHz;
    const horizonMs = 8 * framePeriod;
    const rawCount = Math.ceil((horizonMs / controlPeriod) - 1e-9);
    const ratio = actionHz / videoHz;

    document.getElementById('video-period').textContent = ms(framePeriod);
    document.getElementById('action-period').textContent = ms(controlPeriod);
    document.getElementById('horizon-duration').textContent = ms(horizonMs);
    document.getElementById('raw-action-count').textContent = `约 ${rawCount} 个`;

    const verdict = document.getElementById('frequency-verdict');
    if (Math.abs(ratio - 1) < 1e-6) {
      verdict.innerHTML = `当前示例为 <b>1:1</b>：每个 ${framePeriod.toFixed(1)} ms 的相邻视频帧区间恰好对应一个控制周期，可直接形成 8 个 action 条件。`;
    } else if (ratio > 1) {
      verdict.innerHTML = `控制器比相机快 <b>${ratio.toFixed(2)}×</b>：每个视频帧区间约含 ${ratio.toFixed(2)} 个原始控制周期，必须按时间戳聚合或组合成一个 frame-aligned action。`;
    } else {
      verdict.innerHTML = `控制器比相机慢：一个控制指令平均覆盖 <b>${(1 / ratio).toFixed(2)}</b> 个视频帧区间，应按实际保持规则扩展，不能假设每帧都有一个新指令。`;
    }

    drawTimeline(videoHz, actionHz);
    updateRelation();
  }

  function drawTimeline(videoHz, actionHz) {
    const startX = 190;
    const stepX = 105;
    const xForOffset = offset => startX + (offset + 3) * stepX;
    const xAnchor = xForOffset(0);
    const framePeriod = 1000 / videoHz;
    const horizonMs = framePeriod * 8;
    const offsets = Array.from({ length: 12 }, (_, i) => i - 3);

    let html = `
      <title id="svg-title">WMPO 历史帧、动作、未来帧和 VAE latent 时间对齐图</title>
      <desc id="svg-desc">四个历史帧位于锚点 t 之前；八个动作分别作用于相邻帧之间，并生成八个未来帧；逐帧 VAE 保留相同时间位置；STDiT 前四个 action 条件为 NONE，后八个条件对应八个未来 latent。</desc>
      <defs>
        <marker id="axis-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10Z" fill="var(--muted)"/></marker>
        <marker id="action-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="var(--green)"/></marker>
        <marker id="map-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="var(--green)"/></marker>
      </defs>
      <text x="25" y="47" class="lane-label">统一时间</text>
      <text x="25" y="69" class="lane-note">τ · 单位 ms</text>
      <line x1="${startX - 56}" y1="70" x2="${xForOffset(8) + 70}" y2="70" class="axis"/>
      <text x="25" y="145" class="lane-label">RGB frame</text>
      <text x="25" y="166" class="lane-note">相机采样点</text>
      <text x="25" y="251" class="lane-label">Action</text>
      <text x="25" y="272" class="lane-note">作用于帧间区间</text>
      <text x="25" y="355" class="lane-label">VAE latent</text>
      <text x="25" y="376" class="lane-note">2D VAE · T 不压缩</text>
      <text x="25" y="472" class="lane-label">STDiT condition</text>
      <text x="25" y="493" class="lane-note">12 个时间位置</text>
      <text x="25" y="604" class="lane-label">原始控制流</text>
      <text x="25" y="625" class="lane-note">按 fₐ 采样 / 保持</text>
      <rect x="${startX - 50}" y="91" width="${xAnchor - startX + 100}" height="434" class="history-zone"/>
      <rect x="${xAnchor + stepX / 2}" y="91" width="${xForOffset(8) - xAnchor - stepX / 2 + 50}" height="434" class="future-zone"/>
      <text x="${(startX + xAnchor) / 2}" y="105" class="zone-label" text-anchor="middle">HISTORY · 4 FRAMES</text>
      <text x="${(xAnchor + stepX / 2 + xForOffset(8)) / 2}" y="105" class="zone-label" text-anchor="middle">FUTURE TARGET · 8 FRAMES</text>
      <line x1="${xAnchor + stepX / 2}" y1="91" x2="${xAnchor + stepX / 2}" y2="525" stroke="var(--green)" stroke-dasharray="4 7" opacity=".65"/>
      <text x="${xAnchor + stepX / 2 + 8}" y="120" class="time-label">预测边界</text>`;

    offsets.forEach(offset => {
      const x = xForOffset(offset);
      const time = offset * framePeriod;
      const frameClass = offset < 0 ? 'history-frame' : offset === 0 ? 'anchor-frame' : 'future-frame';
      const frameKind = offset < 0 ? 'history' : offset === 0 ? 'anchor / latest' : 'future target';
      const latentName = `z${sub('t', offset)}`;
      const tokenPos = offset + 3;
      const condition = offset <= 0 ? 'NONE' : `a${sub('t', offset - 1)}`;
      html += `
        <line x1="${x}" y1="76" x2="${x}" y2="525" class="guide${offset === 0 ? ' anchor' : ''}"/>
        <text x="${x}" y="53" class="time-label" text-anchor="middle">${time === 0 ? '0' : `${time > 0 ? '+' : ''}${time.toFixed(1)}`} ms</text>
        <rect x="${x - 38}" y="119" width="76" height="62" rx="8" class="${frameClass}"/>
        <text x="${x}" y="145" class="frame-main">I${sub('t', offset)}</text>
        <text x="${x}" y="166" class="frame-kind">${frameKind}</text>
        <circle cx="${x}" cy="355" r="22" class="vae-node"/>
        <text x="${x}" y="359" class="vae-label">${latentName}</text>
        <line x1="${x}" y1="181" x2="${x}" y2="330" stroke="var(--blue)" stroke-width="1.3" opacity=".55" marker-end="url(#axis-arrow)"/>
        <rect x="${x - 39}" y="440" width="78" height="68" rx="7" class="token ${offset <= 0 ? 'history-token' : ''}${offset > 0 && offset - 1 === selectedAction ? ' selected' : ''}" data-token-action="${offset - 1}"/>
        <text x="${x}" y="461" class="token-pos">position ${tokenPos}</text>
        <text x="${x}" y="486" class="token-action ${offset <= 0 ? 'none-text' : ''}">${condition}</text>
        <text x="${x}" y="500" class="token-pos">for ${latentName}</text>`;
    });

    for (let k = 0; k < 8; k += 1) {
      const fromX = xForOffset(k);
      const toX = xForOffset(k + 1);
      const midX = (fromX + toX) / 2;
      const targetX = toX;
      const selected = k === selectedAction ? ' selected' : '';
      html += `
        <g class="action-transition${selected}" data-action-index="${k}">
          <line x1="${fromX + 11}" y1="236" x2="${toX - 12}" y2="236" class="transition"/>
          <line x1="${fromX + 2}" y1="236" x2="${toX - 2}" y2="236" class="transition-hit"/>
          <text x="${midX}" y="220" class="action-label">a${sub('t', k)}</text>
        </g>
        <path d="M${midX} 251 C${midX} 285 ${targetX + 45} 285 ${targetX + 45} 322 L${targetX + 45} 385 C${targetX + 45} 416 ${targetX} 410 ${targetX} 433" class="mapping${selected}" data-map-index="${k}"/>
        ${selected ? `<text x="${midX}" y="282" class="selection-note">controls transition</text>` : ''}`;
    }

    const controlPeriodMs = 1000 / actionHz;
    const rawSampleCount = Math.ceil((horizonMs / controlPeriodMs) - 1e-9);
    for (let j = 0; j < rawSampleCount; j += 1) {
      const sampleMs = j * controlPeriodMs;
      const nextMs = Math.min((j + 1) * controlPeriodMs, horizonMs);
      const x1 = xAnchor + (sampleMs / framePeriod) * stepX;
      const x2 = xAnchor + (nextMs / framePeriod) * stepX;
      html += `
        <line x1="${x1}" y1="602" x2="${x2}" y2="602" class="raw-hold"/>
        <circle cx="${x1}" cy="602" r="6" class="raw-sample"/>
        <text x="${x1}" y="629" class="raw-label">u${j}</text>`;
    }
    html += `
      <line x1="${xAnchor}" y1="560" x2="${xForOffset(8)}" y2="560" class="axis"/>
      <text x="${(xAnchor + xForOffset(8)) / 2}" y="679" class="lane-note" text-anchor="middle">8 个 future frame intervals = ${(horizonMs).toFixed(1)} ms · 原始控制经过时间戳对齐后 → 8 个有效 action 条件</text>`;

    svg.innerHTML = html;
    svg.querySelectorAll('.action-transition').forEach(group => {
      group.addEventListener('click', () => selectAction(Number(group.dataset.actionIndex)));
    });
  }

  function updateRelation() {
    const k = selectedAction;
    const position = k + 4;
    document.getElementById('selected-relation').innerHTML = `
      <div><span>执行区间</span><strong>a<sub>${sub('t', k)}</sub> 作用于 [τ<sub>${sub('t', k)}</sub>, τ<sub>${sub('t', k + 1)}</sub>)</strong></div>
      <div class="relation-arrow">→</div>
      <div><span>状态转移</span><strong>I<sub>${sub('t', k)}</sub> ─a<sub>${sub('t', k)}</sub>→ I<sub>${sub('t', k + 1)}</sub></strong></div>
      <div class="relation-arrow">→</div>
      <div><span>条件落点</span><strong>STDiT position ${position} · z<sub>${sub('t', k + 1)}</sub></strong></div>`;
  }

  function selectAction(index) {
    selectedAction = index;
    actionButtons.forEach(button => {
      const active = Number(button.dataset.actionIndex) === index;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    render();
  }

  actionButtons.forEach(button => button.addEventListener('click', () => selectAction(Number(button.dataset.actionIndex))));
  presetButtons.forEach(button => button.addEventListener('click', () => {
    videoHzSelect.value = button.dataset.videoHz;
    actionHzSelect.value = button.dataset.actionHz;
    presetButtons.forEach(item => {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    render();
  }));
  [videoHzSelect, actionHzSelect].forEach(select => select.addEventListener('change', () => {
    presetButtons.forEach(button => {
      const active = button.dataset.videoHz === videoHzSelect.value && button.dataset.actionHz === actionHzSelect.value;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    render();
  }));

  render();
})();
