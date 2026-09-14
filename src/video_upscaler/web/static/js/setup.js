/**
 * Clarity Studio — Setup Wizard & Desktop IPC Bridge
 * Coordinates first-run environment provisioning with Tauri backend.
 */

(function () {
  'use strict';

  // State
  const state = {
    isTauri: typeof window !== 'undefined' && Boolean(window.__TAURI__),
    percent: 0,
    stage: 'init',
    speed: '',
    logs: [],
    startTime: Date.now(),
    isComplete: false,
    hasError: false,
    mockTimer: null
  };

  // Stage display metadata and stepper mapping
  const STAGE_CONFIG = {
    init: {
      stepIndex: 0,
      title: 'Initializing Environment...',
      taskBadge: 'Preparing Workspace'
    },
    python: {
      stepIndex: 1,
      title: 'Downloading & Installing Python 3.11 Runtime...',
      taskBadge: 'Step 1 of 4'
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
    complete: {
      stepIndex: 5,
      title: 'AI Engine Setup Completed!',
      taskBadge: 'Setup Ready'
    }
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
      btnCopyLogs: document.getElementById('btn-copy-logs'),
      browserTestBar: document.getElementById('browser-test-bar'),
      btnTestSimulate: document.getElementById('btn-test-simulate'),
      btnTestError: document.getElementById('btn-test-error'),
      btnTestReset: document.getElementById('btn-test-reset')
    };
  }

  /**
   * Tauri IPC invoke abstraction supporting both Tauri v2 (`core.invoke`) and v1 (`invoke`).
   */
  async function invokeTauri(cmd, args = {}) {
    if (!state.isTauri) {
      throw new Error(`Tauri environment not detected: cannot invoke "${cmd}"`);
    }

    if (window.__TAURI__.core && typeof window.__TAURI__.core.invoke === 'function') {
      return await window.__TAURI__.core.invoke(cmd, args);
    }
    if (typeof window.__TAURI__.invoke === 'function') {
      return await window.__TAURI__.invoke(cmd, args);
    }
    throw new Error('Tauri invoke API unavailable');
  }

  /**
   * Tauri event listener abstraction.
   */
  async function listenTauri(eventName, callback) {
    if (!state.isTauri) return null;

    if (window.__TAURI__.event && typeof window.__TAURI__.event.listen === 'function') {
      return await window.__TAURI__.event.listen(eventName, callback);
    }
    return null;
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
    state.logs.push({ time: timeStr, text: message, type });

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
   * Process a progress event from Tauri backend or test harness.
   *
   * @param {Object} event
   * @param {string} event.stage - Stage identifier ('init', 'python', 'venv', 'dependencies', 'models', 'complete')
   * @param {number} event.percent - Progress percentage (0.0 - 100.0)
   * @param {string} [event.speed] - Transfer rate (e.g. '24.5MB/s')
   * @param {string} [event.message] - Status log message
   */
  function handleProgress(event) {
    if (!event) return;
    state.hasError = false;
    if (dom.errorBanner) dom.errorBanner.classList.add('hidden');

    const stageKey = (event.stage || state.stage || 'init').toLowerCase();
    const config = STAGE_CONFIG[stageKey] || {
      stepIndex: 1,
      title: event.stage || 'Processing...',
      taskBadge: 'In Progress'
    };

    state.stage = stageKey;
    const percent = Math.min(100, Math.max(0, Number(event.percent) || state.percent || 0));
    state.percent = percent;
    state.speed = event.speed || '';

    // Update Percentage Text
    if (dom.percentText) {
      dom.percentText.textContent = `${percent.toFixed(1)}%`;
    }

    // Update Smooth Progress Bar
    if (dom.progressBar) {
      dom.progressBar.style.width = `${percent}%`;
      const track = dom.progressBar.parentElement;
      if (track) track.setAttribute('aria-valuenow', Math.round(percent));
    }

    // Update Stage Titles
    if (dom.stageTitle) {
      dom.stageTitle.textContent = config.title;
    }
    if (dom.statusMessage && event.message) {
      dom.statusMessage.textContent = event.message;
    }
    if (dom.activeTaskBadge) {
      dom.activeTaskBadge.textContent = config.taskBadge;
    }

    // Update Stepper
    updateStepper(config.stepIndex);

    // Update Speed Display
    if (dom.speedBadge && dom.speedText) {
      if (state.speed && state.speed.trim() !== '') {
        dom.speedText.textContent = state.speed;
        dom.speedBadge.classList.remove('hidden');
      } else {
        dom.speedBadge.classList.add('hidden');
      }
    }

    // Update ETA Display
    if (dom.etaBadge && dom.etaText) {
      const eta = calculateETA(percent);
      if (eta) {
        dom.etaText.textContent = `ETA: ${eta}`;
        dom.etaBadge.classList.remove('hidden');
      } else {
        dom.etaBadge.classList.add('hidden');
      }
    }

    // Append to Logs
    if (event.message) {
      const isUv = state.speed || event.message.includes('%') || event.message.includes('Downloading');
      appendLog(event.message, isUv ? 'uv' : 'info');
    }
  }

  /**
   * Handle setup completion event.
   */
  function handleComplete(payload = {}) {
    state.isComplete = true;
    state.hasError = false;

    handleProgress({
      stage: 'complete',
      percent: 100.0,
      speed: '',
      message: 'Setup finalized successfully. Launching Studio...'
    });

    updateStepper(5);

    if (dom.successBanner) {
      dom.successBanner.classList.remove('hidden');
    }
    if (dom.errorBanner) {
      dom.errorBanner.classList.add('hidden');
    }

    appendLog('✓ First-time setup complete! Redirecting to Clarity Web Studio...', 'info');

    // Smooth transition / redirect
    const redirectUrl = payload.url || (payload.port ? `http://127.0.0.1:${payload.port}/` : '/');
    setTimeout(() => {
      if (state.isTauri) {
        // In Tauri standalone, invoke launch or navigate
        try {
          invokeTauri('launch_main_app', { url: redirectUrl }).catch(() => {
            window.location.href = redirectUrl;
          });
        } catch (_) {
          window.location.href = redirectUrl;
        }
      } else {
        // Browser fallback
        if (dom.successSubtitle) {
          dom.successSubtitle.textContent = `Ready! Redirect target: ${redirectUrl}`;
        }
      }
    }, 1200);
  }

  /**
   * Handle setup error event with retry affordance.
   */
  function handleError(payload) {
    state.hasError = true;
    const errorMsg = typeof payload === 'string'
      ? payload
      : (payload && (payload.error || payload.message)) || 'Unknown installation failure';

    if (dom.errorMessage) {
      dom.errorMessage.textContent = errorMsg;
    }
    if (dom.errorBanner) {
      dom.errorBanner.classList.remove('hidden');
    }
    if (dom.successBanner) {
      dom.successBanner.classList.add('hidden');
    }

    if (dom.stageTitle) {
      dom.stageTitle.textContent = 'Setup Paused on Error';
    }
    if (dom.statusMessage) {
      dom.statusMessage.textContent = 'Click "Retry Setup" to resume or check technical logs below.';
    }

    appendLog(`[ERROR] ${errorMsg}`, 'error');

    // Automatically expand the logs accordion on error so technical diagnostics are visible
    expandLogs();
  }

  /**
   * Re-trigger setup after an error.
   */
  async function retrySetup() {
    state.hasError = false;
    if (dom.errorBanner) dom.errorBanner.classList.add('hidden');
    appendLog('Retrying setup...', 'info');

    if (state.isTauri) {
      try {
        // Try retry_setup command, fall back to start_setup
        try {
          await invokeTauri('retry_setup');
        } catch (_) {
          await invokeTauri('start_setup');
        }
      } catch (err) {
        handleError(err && err.message ? err.message : String(err));
      }
    } else {
      runMockSimulation();
    }
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
    if (state.mockTimer) {
      clearInterval(state.mockTimer);
      state.mockTimer = null;
    }
    state.percent = 0;
    state.stage = 'init';
    state.speed = '';
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
   * Browser mock simulation for testing outside Tauri shell.
   */
  function runMockSimulation(simulateError = false) {
    resetUI();
    appendLog('Starting browser preview mock simulation...', 'info');

    const steps = [
      { stage: 'init', percent: 5, speed: '', msg: 'Created %LOCALAPPDATA%\\Clarity runtime directories' },
      { stage: 'python', percent: 12, speed: '18.5 MB/s', msg: 'Downloading python-3.11.9-windows-x86_64.tar.gz' },
      { stage: 'python', percent: 22, speed: '24.2 MB/s', msg: 'Unpacking Python standalone runtime' },
      { stage: 'venv', percent: 28, speed: '', msg: 'Creating virtual environment in %LOCALAPPDATA%\\Clarity\\env' },
      { stage: 'venv', percent: 34, speed: '', msg: 'Configured isolated site-packages and launcher' },
      { stage: 'dependencies', percent: 45, speed: '32.1 MB/s', msg: 'Fetching torch-2.3.0+cu126-cp311-win_amd64.whl' },
      { stage: 'dependencies', percent: 65, speed: '28.4 MB/s', msg: 'Installing torchvision and onnxruntime-gpu' }
    ];

    if (simulateError) {
      steps.push({
        isError: true,
        error: 'Network timeout downloading torch wheels from download.pytorch.org: Connection reset by peer'
      });
    } else {
      steps.push(
        { stage: 'dependencies', percent: 78, speed: '15.2 MB/s', msg: 'Installing Clarity Studio core package' },
        { stage: 'models', percent: 85, speed: '12.8 MB/s', msg: 'Downloading Real-CUGAN models (2x, 3x, 4x weights)' },
        { stage: 'models', percent: 95, speed: '16.0 MB/s', msg: 'Downloading AMT-S interpolation checkpoint' },
        { stage: 'complete', percent: 100, speed: '', msg: 'Writing .setup_complete marker' }
      );
    }

    let idx = 0;
    state.mockTimer = setInterval(() => {
      if (idx >= steps.length) {
        clearInterval(state.mockTimer);
        state.mockTimer = null;
        return;
      }

      const item = steps[idx];
      idx++;

      if (item.isError) {
        handleError(item.error);
        clearInterval(state.mockTimer);
        state.mockTimer = null;
      } else if (item.stage === 'complete') {
        handleComplete({ port: 7860 });
        clearInterval(state.mockTimer);
        state.mockTimer = null;
      } else {
        handleProgress({
          stage: item.stage,
          percent: item.percent,
          speed: item.speed,
          message: item.msg
        });
      }
    }, 450);
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
        const text = state.logs.map(l => `[${l.time}] ${l.text}`).join('\n');
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(() => {
            const orig = dom.btnCopyLogs.textContent;
            dom.btnCopyLogs.textContent = 'Copied!';
            setTimeout(() => { dom.btnCopyLogs.textContent = orig; }, 1500);
          }).catch(() => {});
        }
      });
    }

    // Browser Preview Controls
    if (dom.btnTestSimulate) {
      dom.btnTestSimulate.addEventListener('click', () => runMockSimulation(false));
    }
    if (dom.btnTestError) {
      dom.btnTestError.addEventListener('click', () => runMockSimulation(true));
    }
    if (dom.btnTestReset) {
      dom.btnTestReset.addEventListener('click', resetUI);
    }
  }

  /**
   * Main Initialization
   */
  async function init() {
    initDOM();
    bindEvents();

    if (state.isTauri) {
      appendLog('Tauri desktop shell connected. Initializing setup listeners...', 'info');

      try {
        // Listen to setup progress stream from Rust backend
        await listenTauri('setup-progress', (event) => {
          handleProgress(event.payload);
        });

        // Listen to setup complete event
        await listenTauri('setup-complete', (event) => {
          handleComplete(event.payload);
        });

        // Listen to setup error event
        await listenTauri('setup-error', (event) => {
          handleError(event.payload);
        });

        // Check if setup is already complete or start it
        try {
          const status = await invokeTauri('get_setup_status');
          if (status === 'completed' || (status && status.complete)) {
            handleComplete(status);
            return;
          }
        } catch (_) {
          // Command not yet registered or error, proceed to start_setup
        }

        appendLog('Invoking start_setup command...', 'info');
        await invokeTauri('start_setup').catch((err) => {
          console.warn('start_setup invocation returned:', err);
        });

      } catch (err) {
        console.error('Failed to configure Tauri setup stream:', err);
        handleError(err && err.message ? err.message : String(err));
      }
    } else {
      // Standalone browser preview mode
      if (dom.browserTestBar) {
        dom.browserTestBar.classList.remove('hidden');
      }
      appendLog('Running in standalone browser mode. Test controls active.', 'info');

      if (typeof URLSearchParams !== 'undefined' && window.location && window.location.search) {
        const urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('mock')) {
          runMockSimulation(urlParams.get('mock') === 'error');
        }
      }
    }
  }

  // Export testing hooks for automated unit test suites and diagnostics
  window.__ClaritySetup = {
    state,
    handleProgress,
    handleComplete,
    handleError,
    retrySetup,
    resetUI,
    runMockSimulation,
    appendLog
  };

  // Launch on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
