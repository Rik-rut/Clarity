/* MaskCanvas: first-frame target painter for MatAnyone2.
 * Layered canvases over the extracted first frame; strokes are composited
 * onto an offscreen mask canvas (white=target). Undo/redo are stroke-based.
 * "Detect" mode forwards clicks to a host callback (server-side auto
 * detection) and merges the returned white-on-transparent PNG into strokes.
 */
(function () {
  "use strict";

  const BRUSH_RADII = { small: 8, medium: 20, large: 44 };

  function create(container, imageSrc, options) {
    const opts = options || {};
    container.innerHTML =
      '<div class="ma-stage">' +
      '<img class="ma-frame" alt="first frame">' +
      '<canvas class="ma-overlay"></canvas>' +
      "</div>";
    const img = container.querySelector(".ma-frame");
    const overlay = container.querySelector(".ma-overlay");
    const offscreen = document.createElement("canvas");
    const octx = offscreen.getContext("2d");
    const vctx = overlay.getContext("2d");
    const undoStack = [];
    const redoStack = [];
    // Start in detect mode: the UI highlights Auto-Detect on load
    // (index.html ma-autodetect .active) and clicks should auto-detect.
    let mode = "detect";
    let radius = BRUSH_RADII.medium;
    let drawing = false;
    let lastPt = null;
    let previewOn = false;
    let ready = false;

    img.onload = function () {
      [offscreen, overlay].forEach(function (c) {
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
      });
      ready = true;
      repaint();
    };
    img.src = imageSrc;

    function canvasPoint(event) {
      const rect = overlay.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * (overlay.width / rect.width),
        y: (event.clientY - rect.top) * (overlay.height / rect.height),
      };
    }

    function pushUndoState() {
      undoStack.push(octx.getImageData(0, 0, offscreen.width, offscreen.height));
      if (undoStack.length > 40) undoStack.shift();
      redoStack.length = 0;
    }

    function strokeTo(pt) {
      octx.strokeStyle = "#ffffff";
      octx.lineCap = "round";
      octx.lineJoin = "round";
      octx.lineWidth = radius * 2;
      octx.beginPath();
      octx.moveTo(lastPt.x, lastPt.y);
      octx.lineTo(pt.x, pt.y);
      octx.stroke();
      lastPt = pt;
    }

    function onPointerDown(event) {
      if (!ready) return;
      if (mode === "detect") {
        event.preventDefault();
        if (typeof opts.onDetect === "function") opts.onDetect(canvasPoint(event));
        return;
      }
      if (mode === "remove") eraseStroke(event);
      else beginStroke(event);
    }

    function beginStroke(event) {
      event.preventDefault();
      drawing = true;
      lastPt = canvasPoint(event);
      pushUndoState();
      strokeTo(lastPt);
      repaint();
    }

    function moveStroke(event) {
      if (!drawing) return;
      event.preventDefault();
      const pt = canvasPoint(event);
      if (mode === "remove") {
        eraseTo(pt);
      } else {
        strokeTo(pt);
      }
      repaint();
    }

    function endStroke() {
      drawing = false;
      lastPt = null;
      repaint();
    }

    function compositeMask() {
      // white strokes -> magenta translucent preview; erase-mode handled by
      // destination-out during stroke, so mask pixels are always additive.
      return octx.getImageData(0, 0, offscreen.width, offscreen.height);
    }

    function repaint() {
      vctx.clearRect(0, 0, overlay.width, overlay.height);
      if (!previewOn) return;
      const data = compositeMask();
      const out = vctx.createImageData(overlay.width, overlay.height);
      for (let i = 0; i < data.data.length; i += 4) {
        const a = data.data[i]; // red channel carries the grayscale mask
        out.data[i] = 255;
        out.data[i + 1] = 0;
        out.data[i + 2] = 255;
        out.data[i + 3] = Math.round(a * 0.55);
      }
      vctx.putImageData(out, 0, 0);
    }

    overlay.addEventListener("pointerdown", onPointerDown);
    overlay.addEventListener("pointermove", moveStroke);
    window.addEventListener("pointerup", endStroke);

    function eraseStroke(event) {
      event.preventDefault();
      drawing = true;
      lastPt = canvasPoint(event);
      pushUndoState();
      eraseTo(lastPt);
      repaint();
    }

    function eraseTo(pt) {
      octx.save();
      octx.globalCompositeOperation = "destination-out";
      octx.strokeStyle = "rgba(0,0,0,1)";
      octx.lineCap = "round";
      octx.lineWidth = radius * 2;
      octx.beginPath();
      octx.moveTo(lastPt.x, lastPt.y);
      octx.lineTo(pt.x, pt.y);
      octx.stroke();
      octx.restore();
      lastPt = pt;
    }

    // pointer events carry the button/modifiers; remove-mode toggles here:
    container.addEventListener("contextmenu", function (e) { e.preventDefault(); });

    function applyMode(nextMode) {
      mode = nextMode;
    }

    function flattenGrayscale() {
      const data = octx.getImageData(0, 0, offscreen.width, offscreen.height).data;
      const canvas = document.createElement("canvas");
      canvas.width = offscreen.width;
      canvas.height = offscreen.height;
      const ctx = canvas.getContext("2d");
      const image = ctx.createImageData(canvas.width, canvas.height);
      for (let i = 0; i < data.length; i += 4) {
        const value = data[i]; // strokes painted pure white
        image.data[i] = value;
        image.data[i + 1] = value;
        image.data[i + 2] = value;
        image.data[i + 3] = 255;
      }
      ctx.putImageData(image, 0, 0);
      return canvas;
    }

    return {
      setMode: applyMode,
      setBrushSize: function (name) { radius = BRUSH_RADII[name] || BRUSH_RADII.medium; },
      togglePreview: function (on) { previewOn = !!on; repaint(); },
      isReady: function () { return ready; },
      setFromAlphaImage: function (image) {
        // Merge a white-on-transparent mask PNG (any resolution) into the
        // stroke canvas as one undoable step.
        if (!ready || !image) return false;
        pushUndoState();
        const tmp = document.createElement("canvas");
        tmp.width = offscreen.width;
        tmp.height = offscreen.height;
        tmp.getContext("2d").drawImage(image, 0, 0, offscreen.width, offscreen.height);
        octx.drawImage(tmp, 0, 0);
        repaint();
        return true;
      },
      undo: function () {
        if (!undoStack.length) return;
        redoStack.push(octx.getImageData(0, 0, offscreen.width, offscreen.height));
        octx.putImageData(undoStack.pop(), 0, 0);
        repaint();
      },
      redo: function () {
        if (!redoStack.length) return;
        undoStack.push(octx.getImageData(0, 0, offscreen.width, offscreen.height));
        octx.putImageData(redoStack.pop(), 0, 0);
        repaint();
      },
      clear: function () {
        pushUndoState();
        octx.clearRect(0, 0, offscreen.width, offscreen.height);
        repaint();
      },
      isEmpty: function () {
        const data = octx.getImageData(0, 0, offscreen.width, offscreen.height).data;
        for (let i = 0; i < data.length; i += 4) if (data[i] > 0) return false;
        return true;
      },
      getMaskPngB64: function () {
        if (this.isEmpty()) return null;
        return flattenGrayscale().toDataURL("image/png");
      },
      resetForImage: function (nextSrc) {
        ready = false;
        undoStack.length = 0;
        redoStack.length = 0;
        octx.clearRect(0, 0, offscreen.width, offscreen.height);
        img.src = nextSrc;
      },
    };
  }

  window.MaskCanvas = { create: create };
})();
