/**
 * Clarity Studio — First-Run Setup Wizard
 *
 * Drives the desktop provisioning server (video_upscaler/desktop/server.py)
 * over loopback HTTP. There is deliberately no desktop IPC in this page: the
 * wizard is a page of the product, so the exact same file renders in the Tauri
 * webview and in a plain browser, and "works in the browser" means "works on
 * the desktop".
 */

(function () {
  'use strict';

  const POLL_MS = 500;

  // Provisioning order and the endpoint that starts each step.
  const STEP_ENDPOINTS = {
    gpu: '/api/setup/detect-gpu',
    runtime: '/api/setup/runtime',
    models: '/api/setup/models',
    verify: '/api/setup/verify',
    complete: '/api/setup/complete'
  };

  // Stage display metadata and stepper mapping. `runtime` covers both the venv
  // and the dependency install; the server reports which phase it is in.
  const STAGE_CONFIG = {
    init: {
      stepIndex: 0,
      title: 'Initializing Environment...',
      taskBadge: 'Preparing Workspace'
    },
    gpu: {
      stepIndex: 1,
      title: 'Detecting GPU Hardware...',
      taskBadge: 'Detecting'
    },
    python: {
      stepIndex: 1,
      title: 'Downloading & Installing Python 3.11 Runtime...',
      taskBadge: 'Step 1 of 4'
    },
    runtime: {
      stepIndex: 2,
      title: 'Creating Isolated Virtual Environment...',
      taskBadge: 'Step 2 of 4'
    },
    venv: {
      stepIndex: 2,
      title: 'Creating Isolated Virtual Environment...',
      taskBadge: 'Step 2 of 4'
    },
    dependencies: {
      stepIndex: 3,
      title: 'Installing PyTorch & AI Engine Libraries...',
      taskBadge: 'Step 3 of 4'
    },
    models: {
      stepIndex: 4,
      title: 'Fetching Neural Weights (Real-CUGAN & AMT-S)...',
      taskBadge: 'Step 4 of 4'
    },
    verify: {
      stepIndex: 5,
      title: 'Verifying the Provisioned Environment...',
      taskBadge: 'Verifying'
    },
    complete: {
      stepIndex: 5,
      title: 'AI Engine Setup Completed!',
      taskBadge: 'Setup Ready'
    }
  };

  // State
  const state = {
    percent: 0,
    stage: 'init',
    speed: '',
    logs: [],
    seenLogLines: 0,
    startTime: Date.now(),
    isComplete: false,
    hasError: false,
    busy: false,
    nextStep: '',
    stepStatuses: {},
    pollTimer: null,
    unavailableSince: 0
  };

  // DOM Elements cache
  let dom = {};

  function initDOM() {
    dom = {
      stepper: document.getElementById('setup-stepper'),
      stepPython: document.getElementById('step-python'),
      stepVenv: document.getElementById('step-venv'),
      stepDependencies: document.getElementById('step-dependencies'),
      stepModels: document.getElementById('step-models'),
      stageTitle: document.getElementById('setup-stage-title'),
      statusMessage: document.getElementById('setup-status-message'),
      percentText: document.getElementById('setup-percent-text'),
      progressBar: document.getElementById('setup-progress-bar'),
      speedBadge: document.getElementById('setup-speed-badge'),
      speedText: document.getElementById('setup-speed-text'),
      etaBadge: document.getElementById('setup-eta-badge'),
      etaText: document.getElementById('setup-eta-text'),
      activeTaskBadge: document.getElementById('setup-active-task-badge'),
      errorBanner: document.getElementById('setup-error-banner'),
      errorMessage: document.getElementById('setup-error-message'),
      btnRetry: document.getElementById('btn-retry-setup'),
      successBanner: document.getElementById('setup-success-banner'),
      successSubtitle: document.getElementById('setup-success-subtitle'),
      logsAccordion: document.getElementById('logs-accordion'),
      btnToggleLogs: document.getElementById('btn-toggle-logs'),
      logsToggleLabel: document.getElementById('logs-toggle-label'),
      logsCountBadge: document.getElementById('logs-count-badge'),
      logsDrawer: document.getElementById('logs-drawer'),
      logsTerminal: document.getElementById('logs-terminal'),
      btnClearLogs: document.getElementById('btn-clear-logs'),
      btnCopyLogs: document.getElementById('btn-copy-logs')
    };
  }

  /**
   * JSON fetch helper. Returns null when the wizard server is unreachable, so
   * the caller can keep the last known UI state instead of flickering.
   */
  async function request(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error((payload && payload.error) || `Setup server returned ${response.status}`);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  function getJSON(path) {
    return request(path, { method: 'GET', headers: { Accept: 'application/json' } });
  }

  function postJSON(path, body) {
    return request(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {})
    });
  }

  /**
   * Updates stepper step item UI classes.
   */
  function updateStepper(stepIndex) {
    const steps = [
      { el: dom.stepPython, id: 1 },
      { el: dom.stepVenv, id: 2 },
      { el: dom.stepDependencies, id: 3 },
      { el: dom.stepModels, id: 4 }
    ];

    steps.forEach(({ el, id }) => {
      if (!el) return;
      el.classList.remove('active', 'completed');
      const numEl = el.querySelector('.step-num');

      if (id < stepIndex) {
        el.classList.add('completed');
        if (numEl) numEl.innerHTML = '&#10003;';
      } else if (id === stepIndex) {
        el.classList.add('active');
        if (numEl) numEl.textContent = id;
      } else {
        if (numEl) numEl.textContent = id;
      }
    });
  }

  /**
   * Append status or log lines to technical console.
   */
  function appendLog(message, type = 'info') {
    if (!message) return;
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const line = `[${timeStr}] ${message}`;
    state.logs.push({ time: timeStr, text: line, type });

    if (dom.logsTerminal) {
      const lineSpan = document.createElement('div');
      lineSpan.className = type === 'error'
        ? 'log-line-err'
        : type === 'uv'
          ? 'log-line-uv'
          : 'log-line-info';
      lineSpan.textContent = line;
      dom.logsTerminal.appendChild(lineSpan);
      dom.logsTerminal.scrollTop = dom.logsTerminal.scrollHeight;
    }

    if (dom.logsCountBadge) {
      dom.logsCountBadge.textContent = `${state.logs.length} lines`;
    }
  }

  /**
   * Sync the log console with a server snapshot (only unseen lines are added).
   */
  function syncLogs(snapshot) {
    const lines = Array.isArray(snapshot.lines) ? snapshot.lines : [];
    const total = typeof snapshot.total === 'number' ? snapshot.total : lines.length;
    const unseen = Math.max(0, Math.min(lines.length, total - state.seenLogLines));
    const fresh = lines.slice(lines.length - unseen);
    state.seenLogLines = total;

    const step = snapshot.phase || snapshot.step || snapshot.stage || state.stage;
    fresh.forEach((text) => {
      const isUv = /%|Downloading|Installing|Prepared|Resolved|Building/.test(text);
      appendLog(text, isUv ? 'uv' : step === 'verify' ? 'info' : 'info');
    });
  }

  /**
   * Format ETA estimate string based on progress percentage and elapsed time.
   */
  function calculateETA(percent) {
    if (percent <= 2 || percent >= 100) return null;
    const elapsedSec = (Date.now() - state.startTime) / 1000;
    if (elapsedSec < 3) return null;

    const estimatedTotalSec = (elapsedSec / percent) * 100;
    const remainingSec = Math.max(0, Math.round(estimatedTotalSec - elapsedSec));

    if (remainingSec < 60) {
      return `~${remainingSec}s`;
    }
    const mins = Math.floor(remainingSec / 60);
    const secs = remainingSec % 60;
    return `~${mins}m ${secs}s`;
  }

  /**
   * Render a progress snapshot from the setup server.
   *
   * @param {Object} event
   * @param {string} [event.step]  - 'gpu' | 'runtime' | 'models' | 'verify' | 'complete'
   * @param {string} [event.phase] - sub-phase of 'runtime' ('venv' | 'dependencies')
   * @param {number} [event.percent]
   * @param {string} [event.speed]
   * @param {string} [event.message]
   */
  function handleProgress(event) {
    if (!event) return;

    syncLogs(event);

    const stageKey = String(event.phase || event.step || event.stage || state.stage || 'init');
    const config = STAGE_CONFIG[stageKey] || {
      stepIndex: 1,
      title: event.message || 'Processing...',
      taskBadge: 'In Progress'
    };

    state.stage = stageKey;
    const percent = Math.min(100, Math.max(0, Number(event.percent) || 0));
    state.percent = Math.max(state.percent, percent);
    state.speed = event.speed || '';

    if (dom.percentText) {
      dom.percentText.textContent = `${state.percent.toFixed(1)}%`;
    }

    if (dom.progressBar) {
      dom.progressBar.style.width = `${state.percent}%`;
      const track = dom.progressBar.parentElement;
      if (track) track.setAttribute('aria-valuenow', Math.round(state.percent));
    }

    if (dom.stageTitle) dom.stageTitle.textContent = config.title;
    if (dom.statusMessage && event.message) dom.statusMessage.textContent = event.message;
    if (dom.activeTaskBadge) dom.activeTaskBadge.textContent = config.taskBadge;

    updateStepper(config.stepIndex);

    if (dom.speedBadge && dom.speedText) {
      if (state.speed && state.speed.trim() !== '') {
        dom.speedText.textContent = state.speed;
        dom.speedBadge.classList.remove('hidden');
      } else {
        dom.speedBadge.classList.add('hidden');
      }
    }

    if (dom.etaBadge && dom.etaText) {
      const eta = calculateETA(state.percent);
      if (eta) {
        dom.etaText.textContent = `ETA: ${eta}`;
        dom.etaBadge.classList.remove('hidden');
      } else {
        dom.etaBadge.classList.add('hidden');
      }
    }
  }

  /**
   * Handle setup completion.
   */
  function handleComplete(payload = {}) {
    if (state.isComplete) return;
    state.isComplete = true;
    state.hasError = false;
    stopPolling();

    handleProgress({
      step: 'complete',
      percent: 100.0,
      speed: '',
      message: 'Setup finalized successfully. The studio opens automatically.'
    });

    updateStepper(5);

    if (dom.successBanner) dom.successBanner.classList.remove('hidden');
    if (dom.errorBanner) dom.errorBanner.classList.add('hidden');
    if (dom.successSubtitle) {
      dom.successSubtitle.textContent = 'Environment ready — launching Clarity Studio…';
    }

    appendLog('✓ First-time setup complete!', 'info');
  }

  /**
   * Handle a setup failure with the retry affordance.
   */
  function handleError(payload) {
    state.hasError = true;
    const errorMsg = typeof payload === 'string'
      ? payload
      : (payload && (payload.error || payload.message)) || 'Unknown installation failure';

    if (dom.errorMessage) dom.errorMessage.textContent = errorMsg;
    if (dom.errorBanner) dom.errorBanner.classList.remove('hidden');
    if (dom.successBanner) dom.successBanner.classList.add('hidden');
    if (dom.stageTitle) dom.stageTitle.textContent = 'Setup Paused on Error';
    if (dom.statusMessage) {
      dom.statusMessage.textContent = 'Click "Retry Setup" to resume or check technical logs below.';
    }
    if (dom.speedBadge) dom.speedBadge.classList.add('hidden');
    if (dom.etaBadge) dom.etaBadge.classList.add('hidden');

    appendLog(`[ERROR] ${errorMsg}`, 'error');
    expandLogs();
  }

  /**
   * Re-run the failed (or current) step and let the polling loop resume.
   */
  async function retrySetup() {
    const step = failedStep() || state.nextStep || 'gpu';
    state.hasError = false;
    state.percent = 0;
    if (dom.errorBanner) dom.errorBanner.classList.add('hidden');
    appendLog(`Retrying setup from step '${step}'...`, 'info');

    try {
      await postJSON('/api/setup/retry', { step });
      state.seenLogLines = 0;
      if (dom.logsTerminal) dom.logsTerminal.innerHTML = '';
      await refresh();
    } catch (err) {
      handleError(err && err.message ? err.message : String(err));
    }
  }

  function failedStep() {
    const statuses = state.stepStatuses || {};
    const order = Object.keys(STEP_ENDPOINTS);
    for (const step of order) {
      if (statuses[step] === 'failed') return step;
    }
    return '';
  }

  /**
   * Expand log accordion.
   */
  function expandLogs() {
    if (dom.logsAccordion && dom.logsDrawer) {
      dom.logsAccordion.classList.add('open');
      dom.logsDrawer.classList.remove('hidden');
      if (dom.btnToggleLogs) dom.btnToggleLogs.setAttribute('aria-expanded', 'true');
      if (dom.logsToggleLabel) dom.logsToggleLabel.textContent = 'Hide Log Details';
      if (dom.logsTerminal) dom.logsTerminal.scrollTop = dom.logsTerminal.scrollHeight;
    }
  }

  /**
   * Toggle log accordion.
   */
  function toggleLogs() {
    if (!dom.logsAccordion || !dom.logsDrawer) return;
    const isOpen = dom.logsAccordion.classList.toggle('open');
    if (isOpen) {
      dom.logsDrawer.classList.remove('hidden');
      if (dom.btnToggleLogs) dom.btnToggleLogs.setAttribute('aria-expanded', 'true');
      if (dom.logsToggleLabel) dom.logsToggleLabel.textContent = 'Hide Log Details';
      if (dom.logsTerminal) dom.logsTerminal.scrollTop = dom.logsTerminal.scrollHeight;
    } else {
      dom.logsDrawer.classList.add('hidden');
      if (dom.btnToggleLogs) dom.btnToggleLogs.setAttribute('aria-expanded', 'false');
      if (dom.logsToggleLabel) dom.logsToggleLabel.textContent = 'Show Log Details';
    }
  }

  /**
   * Reset UI to initial zero state.
   */
  function resetUI() {
    stopPolling();
    state.percent = 0;
    state.stage = 'init';
    state.speed = '';
    state.seenLogLines = 0;
    state.isComplete = false;
    state.hasError = false;
    state.startTime = Date.now();

    if (dom.progressBar) dom.progressBar.style.width = '0%';
    if (dom.percentText) dom.percentText.textContent = '0.0%';
    if (dom.stageTitle) dom.stageTitle.textContent = 'Initializing Setup...';
    if (dom.statusMessage) dom.statusMessage.textContent = 'Preparing environment directories...';
    if (dom.activeTaskBadge) dom.activeTaskBadge.textContent = 'Step 1 of 4';
    if (dom.speedBadge) dom.speedBadge.classList.add('hidden');
    if (dom.etaBadge) dom.etaBadge.classList.add('hidden');
    if (dom.errorBanner) dom.errorBanner.classList.add('hidden');
    if (dom.successBanner) dom.successBanner.classList.add('hidden');
    updateStepper(0);
  }

  /**
   * Start one provisioning step if (and only if) it is still pending. A failed
   * step is never auto-retried: that is the Retry button's job.
   */
  async function advance(status) {
    if (!status || state.isComplete || state.hasError) return;
    const step = status.next_step;
    if (!step || !STEP_ENDPOINTS[step]) return;
    if ((status.state && status.state.steps && status.state.steps[step]) !== 'pending') return;
    if (status.running) return;

    try {
      await postJSON(STEP_ENDPOINTS[step], {});
    } catch (err) {
      // 409 just means the server is busy with this step; keep polling.
      if (err.status === 409) return;
      handleError(err && err.message ? err.message : String(err));
    }
  }

  /**
   * Pull one status + progress pair and reconcile the UI with it.
   */
  async function refresh() {
    if (state.busy || state.isComplete) return;
    state.busy = true;

    try {
      const status = await getJSON('/api/setup/status');
      state.unavailableSince = 0;
      state.nextStep = status.next_step;
      state.stepStatuses = (status.state && status.state.steps) || {};

      if (status.complete) {
        handleComplete(status);
        return;
      }

      const progress = await getJSON('/api/setup/progress');
      handleProgress(progress);

      if (progress.error && !progress.running) {
        handleError(progress.error);
        return;
      }

      if (!progress.running) {
        await advance(status);
      }
    } catch (err) {
      onUnavailable(err);
    } finally {
      state.busy = false;
    }
  }

  /**
   * The wizard server went away. Before provisioning it is the shell's job;
   * after completion it is expected (the shell moved on to the studio).
   */
  function onUnavailable(err) {
    if (!state.unavailableSince) {
      state.unavailableSince = Date.now();
      appendLog(`Setup server unavailable: ${err.message || err}`, 'error');
      return;
    }
    if (Date.now() - state.unavailableSince > 60000) {
      handleError('Lost contact with the setup server. Relaunch Clarity to continue.');
      stopPolling();
    }
  }

  function startPolling() {
    stopPolling();
    state.pollTimer = setInterval(refresh, POLL_MS);
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  /**
   * Bind event listeners to DOM controls.
   */
  function bindEvents() {
    if (dom.btnToggleLogs) {
      dom.btnToggleLogs.addEventListener('click', toggleLogs);
    }
    if (dom.btnRetry) {
      dom.btnRetry.addEventListener('click', retrySetup);
    }
    if (dom.btnClearLogs) {
      dom.btnClearLogs.addEventListener('click', () => {
        state.logs = [];
        if (dom.logsTerminal) dom.logsTerminal.innerHTML = '';
        if (dom.logsCountBadge) dom.logsCountBadge.textContent = '0 lines';
      });
    }
    if (dom.btnCopyLogs) {
      dom.btnCopyLogs.addEventListener('click', () => {
        const text = state.logs.map((l) => l.text).join('\n');
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(() => {
            const orig = dom.btnCopyLogs.textContent;
            dom.btnCopyLogs.textContent = 'Copied!';
            setTimeout(() => { dom.btnCopyLogs.textContent = orig; }, 1500);
          }).catch(() => {});
        }
      });
    }
  }

  /**
   * Main initialization
   */
  async function init() {
    initDOM();
    bindEvents();
    resetUI();

    appendLog('Connected to the Clarity setup server. Provisioning the runtime...', 'info');
    await refresh();
    startPolling();
  }

  // Export testing hooks for automated unit test suites and diagnostics
  window.__ClaritySetup = {
    state,
    handleProgress,
    handleComplete,
    handleError,
    retrySetup,
    resetUI,
    appendLog,
    refresh,
    STAGE_CONFIG,
    STEP_ENDPOINTS
  };

  // Launch on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
