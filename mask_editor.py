"""Green, high-clarity browser mask editor for the Gradio 3 UI."""

EDITOR_HTML = r"""
<div id="green-mask-editor" class="mask-editor">
  <div class="mask-toolbar">
    <label class="file-button"><span class="button-icon">＋</span> 打开目标原图
      <input id="green-mask-file" type="file" accept="image/*" />
    </label>
    <span class="toolbar-divider"></span>
    <span class="tool-label">绘制工具</span>
    <label class="mask-mode-option">
      <input type="radio" name="green-mask-mode" value="rectangle" checked />
      <span class="mode-label">▭ 矩形框 <kbd>R</kbd></span>
    </label>
    <label class="mask-mode-option">
      <input type="radio" name="green-mask-mode" value="brush" />
      <span class="mode-label">● 画笔 <kbd>B</kbd></span>
    </label>
    <label class="brush-control">画笔 <input id="green-brush-size" type="number" min="1" max="1000" step="10" value="80" /> px</label>
    <button id="green-mask-undo" type="button">↶ 撤销</button>
    <button id="green-mask-clear" class="danger-lite" type="button">清空</button>
  </div>
  <div class="mask-help-row">
    <span class="legend-chip add"><i></i>绿色区域：将被 Reference 替换</span>
    <span class="legend-chip erase"><i></i>右键：擦除选区</span>
    <span>框不必贴合物体轮廓，只需明确目标位置和大致大小</span>
  </div>
  <div class="mask-canvas-wrap">
    <canvas id="green-mask-canvas"></canvas>
    <div id="green-mask-empty">
      <div class="empty-icon">▧</div>
      <strong>尚未打开目标图</strong>
      <span>点击左上角“打开目标原图”开始绘制</span>
    </div>
  </div>
  <div class="mask-status-row">
    <span id="green-mask-status" class="status-ready-dot">等待目标图</span>
    <span>左键添加｜右键擦除｜Ctrl+Z 撤销｜滚轮调整画笔</span>
  </div>
</div>
"""


EDITOR_CSS = r"""
:root { --app-green: #16803c; --app-green-2: #20a653; --app-green-soft: #e8f5ec; --app-red: #c83d3d; }
#green-source-data, #green-mask-data { display: none !important; }
.gradio-container { max-width: 1460px !important; margin: 0 auto !important; padding-bottom: 50px !important; }
.app-hero { border: 1px solid #cce7d5; border-radius: 12px; padding: 20px 24px; margin-bottom: 18px; background: linear-gradient(135deg, #f4fbf6 0%, #ffffff 58%, #eff8f2 100%); }
.app-hero h1 { margin: 0 0 8px !important; color: #105f30; font-size: 27px !important; }
.app-hero p { margin: 0 !important; color: #42614c; }
.input-map { display: flex; flex-wrap: wrap; align-items: center; gap: 7px; margin-top: 13px; }
.input-map span { display: inline-flex; align-items: center; min-height: 27px; padding: 3px 9px; border-radius: 999px; background: #e8f5ec; color: #126b34; font-weight: 600; font-size: 12px; }
.input-map b { color: #7a8e80; font-size: 13px; }
.workflow-section { margin-top: 22px; padding-top: 4px; }
.step-title { display: flex; align-items: center; gap: 10px; margin: 0 0 12px; }
.step-badge { display: inline-flex; width: 31px; height: 31px; align-items: center; justify-content: center; border-radius: 50%; background: var(--app-green); color: white; font-weight: 800; }
.step-title strong { font-size: 20px; color: var(--body-text-color); }
.step-title small { color: var(--body-text-color-subdued); }
.reference-panel, .generate-panel { border: 1px solid var(--border-color-primary); border-radius: 9px; padding: 13px; background: var(--background-fill-secondary); }
.status-panel { min-height: 48px; padding: 9px 12px; border-left: 4px solid var(--app-green); border-radius: 5px; background: var(--background-fill-secondary); }
.status-panel p { margin: 0 !important; }
.hint-panel { padding: 8px 11px; border-radius: 5px; background: var(--app-green-soft); color: #1b6437; font-size: 13px; }
.sam-mode label { border-radius: 5px !important; }
.reference-ready img { background: #fff !important; }
.generate-action { min-height: 50px !important; font-size: 17px !important; font-weight: 700 !important; }
.generate-action.primary { background: var(--app-green) !important; border-color: var(--app-green) !important; }
.result-gallery { border-top: 3px solid var(--app-green); padding-top: 10px; }
.generation-progress { padding: 11px 13px; border: 1px solid #b8d9c3; border-radius: 7px; background: var(--background-fill-secondary); }
.generation-progress-head { display: flex; justify-content: space-between; gap: 14px; margin-bottom: 8px; font-size: 14px; }
.generation-progress-head span { overflow-wrap: anywhere; }
.generation-progress-head strong { color: #126b34; white-space: nowrap; }
.generation-progress-track { height: 13px; overflow: hidden; border-radius: 5px; background: var(--border-color-primary); }
.generation-progress-fill { height: 100%; min-width: 0; border-radius: 5px; background: linear-gradient(90deg, #16803c, #29b85c); transition: width .3s ease; }
.generation-progress:not(.completed):not(.failed) .generation-progress-fill { background: linear-gradient(90deg, #126b34 0%, #35c76a 45%, #126b34 100%); background-size: 220% 100%; animation: green-progress-flow 1.4s linear infinite; }
@keyframes green-progress-flow { from { background-position: 110% 0; } to { background-position: -110% 0; } }
.generation-progress.completed .generation-progress-fill { background: #16803c; }
.generation-progress.failed { border-color: #e2baba; }
.generation-progress.failed .generation-progress-fill { background: #c83d3d; }
.generation-progress.failed .generation-progress-head strong { color: #a82f2f; }
.task-metrics { margin-top: 9px; padding: 12px; border: 1px solid #c7dfce; border-radius: 8px; background: var(--background-fill-secondary); }
.task-metrics-title { margin-bottom: 10px; color: #126b34; font-size: 15px; font-weight: 800; }
.task-metrics-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; }
.metric-card { display: flex; min-width: 0; flex-direction: column; gap: 3px; padding: 9px 10px; border-radius: 6px; background: var(--app-green-soft); }
.metric-card span { color: #4d6a57; font-size: 11px; }
.metric-card strong { color: #105f30; font-size: 16px; overflow-wrap: anywhere; }
.metric-card small { color: #63766a; font-size: 10px; overflow-wrap: anywhere; }
.metrics-breakdown { margin-top: 9px; padding-top: 8px; border-top: 1px dashed #b9d1c0; color: var(--body-text-color-subdued); font-size: 12px; }
.task-metrics-empty { color: var(--body-text-color-subdued); font-size: 13px; }
.poll-trigger { display: none !important; }
.mask-editor { border: 1px solid #b8d9c3; border-radius: 9px; overflow: hidden; box-shadow: 0 2px 10px rgba(15, 80, 40, .06); }
.mask-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 9px; padding: 10px 12px; background: var(--background-fill-secondary); }
.mask-toolbar label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; }
.toolbar-divider { width: 1px; height: 26px; background: var(--border-color-primary); }
.tool-label { color: var(--body-text-color-subdued); font-size: 13px; }
.mask-mode-option input { position: absolute; width: 1px; height: 1px; opacity: 0; }
.mask-mode-option { gap: 0 !important; }
.mode-label { display: inline-flex; align-items: center; gap: 6px; min-height: 35px; padding: 5px 11px; border: 1px solid var(--border-color-primary); border-radius: 6px; color: var(--body-text-color); background: var(--button-secondary-background-fill); }
.mode-label kbd { padding: 0 4px; border: 1px solid #bbc8be; border-radius: 3px; background: rgba(255,255,255,.7); font-size: 10px; }
.mask-mode-option input:checked + .mode-label { color: #126b34; border-color: var(--app-green); background: var(--app-green-soft); font-weight: 700; box-shadow: inset 0 0 0 1px var(--app-green); }
.mask-mode-option input:focus-visible + .mode-label { outline: 2px solid var(--app-green); outline-offset: 2px; }
.mask-toolbar button, .file-button { border: 1px solid var(--border-color-primary); border-radius: 6px; padding: 7px 11px; background: var(--button-secondary-background-fill); cursor: pointer; }
.file-button { color: white !important; border-color: var(--app-green) !important; background: var(--app-green) !important; font-weight: 700; }
.file-button:hover { background: #126b34 !important; }
.file-button input { display: none; }
.button-icon { font-size: 18px; line-height: 12px; }
.danger-lite { color: var(--app-red) !important; }
.brush-control { font-size: 13px; color: var(--body-text-color-subdued); }
#green-brush-size { width: 70px; padding: 5px; border: 1px solid var(--border-color-primary); border-radius: 4px; }
.mask-help-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 7px 12px; border-top: 1px solid var(--border-color-primary); border-bottom: 1px solid var(--border-color-primary); background: #f5faf6; color: #52685a; font-size: 12px; }
.legend-chip { display: inline-flex; align-items: center; gap: 5px; font-weight: 600; }
.legend-chip i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.legend-chip.add i { background: #20c866; }
.legend-chip.erase i { background: #ed4a4a; }
.mask-canvas-wrap { position: relative; height: min(65vh, 700px); min-height: 430px; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #1d2420; }
#green-mask-canvas { display: none; max-width: 100%; max-height: 100%; width: auto; height: auto; cursor: crosshair; touch-action: none; user-select: none; }
#green-mask-empty { display: flex; flex-direction: column; align-items: center; gap: 6px; color: #d8e3db; }
#green-mask-empty span { color: #95a89b; font-size: 13px; }
.empty-icon { font-size: 42px; color: #58aa73; }
.mask-status-row { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; padding: 9px 12px; font-size: 13px; background: var(--background-fill-secondary); }
#green-mask-status { font-weight: 700; color: #126b34; }
.status-ready-dot::before { content: ""; display: inline-block; width: 8px; height: 8px; margin-right: 7px; border-radius: 50%; background: var(--app-green-2); }
@media (max-width: 900px) {
  .gradio-container { padding-left: 8px !important; padding-right: 8px !important; }
  .mask-canvas-wrap { min-height: 350px; height: 56vh; }
  .toolbar-divider { display: none; }
  .reference-row, .generate-row { flex-direction: column !important; }
  .reference-row > *, .generate-row > * { width: 100% !important; min-width: 0 !important; flex: 1 1 auto !important; }
  .task-metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


MASK_EDITOR_JS = r"""
() => {
  if (!window.__fluxGreenPollTimer) {
    window.__fluxGreenPollTimer = window.setInterval(() => {
      const root = document.getElementById("green-generation-poll");
      const button = root && (root.matches("button") ? root : root.querySelector("button"));
      if (button && !button.disabled) button.click();
    }, 1000);
  }
  const root = document.getElementById("green-mask-editor");
  if (!root || root.dataset.ready === "1") return [];
  root.dataset.ready = "1";
  const fileInput = document.getElementById("green-mask-file");
  const canvas = document.getElementById("green-mask-canvas");
  const emptyTip = document.getElementById("green-mask-empty");
  const status = document.getElementById("green-mask-status");
  const brushInput = document.getElementById("green-brush-size");
  const ctx = canvas.getContext("2d");
  const sourceCanvas = document.createElement("canvas");
  const sourceCtx = sourceCanvas.getContext("2d");
  const maskCanvas = document.createElement("canvas");
  const maskCtx = maskCanvas.getContext("2d", {willReadFrequently: true});
  const overlayCanvas = document.createElement("canvas");
  const overlayCtx = overlayCanvas.getContext("2d");
  let loaded = false, sourceData = "", drawing = false, activeButton = 0;
  let start = null, last = null, undoStack = [], renderFrame = 0, renderPoint = null;
  let originalWidth = 1, originalHeight = 1, workingScale = 1;
  const MAX_WORK_SIDE = 1600;
  const modeInputs = Array.from(root.querySelectorAll('input[name="green-mask-mode"]'));
  let activeMode = (modeInputs.find(input => input.checked) || {}).value || "rectangle";

  function mode() {
    return activeMode;
  }
  function setMode(nextMode, announce = true) {
    activeMode = nextMode === "brush" ? "brush" : "rectangle";
    modeInputs.forEach(input => { input.checked = input.value === activeMode; });
    root.dataset.activeMode = activeMode;
    if (announce) {
      status.textContent = activeMode === "brush"
        ? `画笔模式已启用｜大小 ${brushInput.value}px｜左键添加，右键擦除`
        : "矩形框模式已启用｜左键添加，右键擦除";
    }
  }
  modeInputs.forEach(input => {
    input.addEventListener("change", () => { if (input.checked) setMode(input.value); });
    const label = input.closest("label");
    if (label) label.addEventListener("click", () => setMode(input.value));
  });
  setMode(activeMode, false);
  function setGradioValue(rootId, value) {
    const el = document.querySelector(`#${rootId} textarea, #${rootId} input`);
    if (!el) return;
    const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value").set.call(el, value);
    el.dispatchEvent(new Event("input", {bubbles: true}));
    el.dispatchEvent(new Event("change", {bubbles: true}));
  }
  function syncSource() {
    if (!loaded) return;
    setGradioValue("green-source-data", sourceData);
  }
  function syncMask() {
    if (!loaded) return;
    setGradioValue("green-mask-data", maskCanvas.toDataURL("image/png"));
  }
  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvas.width - 1, (event.clientX - rect.left) * canvas.width / rect.width)),
      y: Math.max(0, Math.min(canvas.height - 1, (event.clientY - rect.top) * canvas.height / rect.height))
    };
  }
  function pushUndo() {
    // Binary masks compress extremely well as PNG. Keeping compressed snapshots
    // avoids hundreds of MB of raw RGBA ImageData on 4K uploads.
    undoStack.push(maskCanvas.toDataURL("image/png"));
    if (undoStack.length > 12) undoStack.shift();
  }
  function updateStats(prefix = "目标区域已更新") {
    const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
    let selected = 0, minX = maskCanvas.width, minY = maskCanvas.height, maxX = -1, maxY = -1;
    for (let y = 0; y < maskCanvas.height; y++) {
      for (let x = 0; x < maskCanvas.width; x++) {
        if (data[(y * maskCanvas.width + x) * 4 + 3] > 0) {
          selected++; minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
        }
      }
    }
    if (!selected) { status.textContent = `${prefix}｜当前选区为空`; return; }
    const pct = selected * 100 / (maskCanvas.width * maskCanvas.height);
    const fullW = Math.max(1, Math.round((maxX-minX+1) / workingScale));
    const fullH = Math.max(1, Math.round((maxY-minY+1) / workingScale));
    status.textContent = `${prefix}｜绿色选区 ${pct.toFixed(1)}%｜原图范围约 ${fullW}×${fullH}px`;
  }
  function render(previewPoint = null) {
    if (!loaded) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(sourceCanvas, 0, 0);
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.globalCompositeOperation = "source-over";
    overlayCtx.fillStyle = "rgb(26,205,96)";
    overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.globalCompositeOperation = "destination-in";
    overlayCtx.drawImage(maskCanvas, 0, 0);
    ctx.save(); ctx.globalAlpha = 0.40; ctx.drawImage(overlayCanvas, 0, 0); ctx.restore();
    if (drawing && mode() === "rectangle" && start && previewPoint) {
      ctx.save(); ctx.strokeStyle = activeButton === 2 ? "#ed4a4a" : "#1fd46a";
      ctx.lineWidth = Math.max(3, Math.min(canvas.width, canvas.height) / 250);
      ctx.setLineDash([12, 7]); ctx.shadowColor = "rgba(0,0,0,.55)"; ctx.shadowBlur = 3;
      ctx.strokeRect(start.x, start.y, previewPoint.x - start.x, previewPoint.y - start.y); ctx.restore();
    }
  }
  function scheduleRender(previewPoint = null) {
    renderPoint = previewPoint;
    if (renderFrame) return;
    renderFrame = window.requestAnimationFrame(() => {
      renderFrame = 0;
      render(renderPoint);
    });
  }
  function paintBrush(a, b, erase) {
    const sourceWidth = Math.max(1, Math.min(1000, Number(brushInput.value) || 80));
    const width = Math.max(1, sourceWidth * workingScale);
    maskCtx.save(); maskCtx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    maskCtx.strokeStyle = "white"; maskCtx.fillStyle = "white"; maskCtx.lineCap = "round"; maskCtx.lineJoin = "round"; maskCtx.lineWidth = width;
    maskCtx.beginPath(); maskCtx.moveTo(a.x, a.y); maskCtx.lineTo(b.x, b.y); maskCtx.stroke();
    maskCtx.beginPath(); maskCtx.arc(b.x, b.y, width / 2, 0, Math.PI * 2); maskCtx.fill(); maskCtx.restore();
  }
  function finish(event) {
    if (!drawing || !loaded) return;
    const p = point(event);
    if (mode() === "rectangle" && start) {
      const left = Math.min(start.x, p.x), top = Math.min(start.y, p.y);
      const width = Math.max(1, Math.abs(p.x - start.x)), height = Math.max(1, Math.abs(p.y - start.y));
      maskCtx.save(); maskCtx.globalCompositeOperation = activeButton === 2 ? "destination-out" : "source-over";
      maskCtx.fillStyle = "white"; maskCtx.fillRect(left, top, width, height); maskCtx.restore();
    }
    drawing = false; start = null; last = null; render(); syncMask();
    updateStats(activeButton === 2 ? "已擦除区域" : "已添加区域");
  }
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0]; if (!file) return;
    const reader = new FileReader(); reader.onload = () => {
      const img = new Image(); img.onload = () => {
        originalWidth = img.naturalWidth; originalHeight = img.naturalHeight;
        workingScale = Math.min(1, MAX_WORK_SIDE / Math.max(originalWidth, originalHeight));
        const workWidth = Math.max(1, Math.round(originalWidth * workingScale));
        const workHeight = Math.max(1, Math.round(originalHeight * workingScale));
        canvas.width = sourceCanvas.width = maskCanvas.width = overlayCanvas.width = workWidth;
        canvas.height = sourceCanvas.height = maskCanvas.height = overlayCanvas.height = workHeight;
        sourceCtx.clearRect(0, 0, workWidth, workHeight);
        sourceCtx.drawImage(img, 0, 0, workWidth, workHeight);
        maskCtx.clearRect(0, 0, workWidth, workHeight);
        // Preserve the uploaded JPEG/WebP bytes instead of expanding every
        // source image to a much larger PNG data URL.
        sourceData = reader.result; loaded = true; undoStack = [];
        canvas.style.display = "block"; emptyTip.style.display = "none";
        render(); syncSource(); syncMask();
        const workNote = workingScale < 1 ? `｜画布加速 ${workWidth}×${workHeight}` : "";
        status.textContent = `目标图已就绪｜${originalWidth}×${originalHeight}${workNote}｜请绘制绿色区域`;
      }; img.src = reader.result;
    }; reader.readAsDataURL(file);
  });
  canvas.addEventListener("pointerdown", event => {
    if (!loaded || (event.button !== 0 && event.button !== 2)) return;
    event.preventDefault();
    try { if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId); } catch (_) {}
    pushUndo(); drawing = true;
    activeButton = event.button; start = last = point(event);
    if (mode() === "brush") { paintBrush(start, start, activeButton === 2); scheduleRender(); }
  });
  canvas.addEventListener("pointermove", event => {
    if (!drawing) return; event.preventDefault(); const p = point(event);
    if (mode() === "brush") { paintBrush(last, p, activeButton === 2); last = p; scheduleRender(); } else scheduleRender(p);
  });
  canvas.addEventListener("pointerup", finish); canvas.addEventListener("pointercancel", finish);
  canvas.addEventListener("contextmenu", event => event.preventDefault());
  canvas.addEventListener("wheel", event => {
    if (!loaded) return; event.preventDefault();
    brushInput.value = Math.max(1, Math.min(1000, Number(brushInput.value) + (event.deltaY < 0 ? 10 : -10)));
    status.textContent = `画笔大小已调整为 ${brushInput.value}px`;
  }, {passive: false});
  document.getElementById("green-mask-undo").addEventListener("click", () => {
    if (!loaded || !undoStack.length) { status.textContent = "没有可撤销的操作"; return; }
    const snapshot = new Image();
    snapshot.onload = () => {
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      maskCtx.drawImage(snapshot, 0, 0);
      render(); syncMask(); updateStats("已撤销上一步");
    };
    snapshot.src = undoStack.pop();
  });
  document.getElementById("green-mask-clear").addEventListener("click", () => {
    if (!loaded) return; pushUndo(); maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    render(); syncMask(); updateStats("已清空（可撤销）");
  });
  document.addEventListener("keydown", event => {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (event.key.toLowerCase() === "r") setMode("rectangle");
    if (event.key.toLowerCase() === "b") setMode("brush");
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); document.getElementById("green-mask-undo").click(); }
  });
  return [];
}
"""
