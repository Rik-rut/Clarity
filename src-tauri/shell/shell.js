/*
 * Clarity desktop boot shell.
 *
 * This page is the only thing the Tauri asset origin ever serves. It renders
 * instantly, needs no backend and no /api calls, and gets out of the way as
 * soon as the Python server is up — the studio UI is only ever loaded from the
 * backend origin. That separation is what keeps "broken backend" from looking
 * like "broken app": failures surface here as text plus a retry, never as a
 * dead studio.
 */

(function () {
  'use strict';

  const dom = {
    status: document.getElementById('status'),
    dot: document.getElementById('dot'),
    error: document.getElementById('error'),
    errorMessage: document.getElementById('error-message'),
    errorDetails: document.getElementById('error-details'),
    btnRetry: document.getElementById('btn-retry'),
    btnDetails: document.getElementById('btn-details'),
    hint: document.getElementById('hint'),
    progress: document.getElementById('progress'),
    progressBar: document.getElementById('progress-bar'),
    progressPercent: document.getElementById('progress-percent'),
    progressPhase: document.getElementById('progress-phase'),
    progressLog: document.getElementById('progress-log')
  };

  const tauri = typeof window !== 'undefined' ? window.__TAURI__ : undefined;
  const hasIpc = Boolean(tauri && tauri.core && typeof tauri.core.invoke === 'function');
  let detailsHidden = true;

  function setStatus(message) {
    if (!message) return;
    dom.status.textContent = message;
    dom.hint.hidden = true;
  }

  function showError(message, details) {
    dom.progress.hidden = true;
    dom.errorMessage.textContent = message || 'Unknown error.';
    dom.dot.classList.remove('busy');
    dom.error.classList.add('show');
    dom.status.textContent = 'Startup stopped.';

    if (details) {
      dom.errorDetails.textContent = details;
      dom.errorDetails.hidden = detailsHidden;
      dom.btnDetails.hidden = false;
    } else {
      dom.errorDetails.hidden = true;
      dom.btnDetails.hidden = true;
    }
  }

  async function retry() {
    dom.btnRetry.disabled = true;
    setStatus('Retrying…');
    showError('Retrying…', null);
    dom.error.classList.add('show');

    try {
      if (hasIpc) {
        await tauri.core.invoke('retry_boot');
      } else {
        window.location.reload();
      }
    } catch (err) {
      showError(String((err && err.message) || err), null);
    } finally {
      dom.btnRetry.disabled = false;
    }
  }

  function toggleDetails() {
    detailsHidden = !detailsHidden;
    dom.errorDetails.hidden = detailsHidden;
    dom.btnDetails.textContent = detailsHidden ? 'Show details' : 'Hide details';
  }

  const LOG_LIMIT = 12;
  const logLines = [];

  /**
   * Provisioning progress from Rust, which is parsing the provisioner's
   * `PROGRESS <percent>|<phase>|<message>` lines. The shell renders it; it never
   * decides what to install.
   */
  function showProgress(percent, phase, message) {
    const value = Number(percent);
    dom.progress.hidden = false;
    dom.progressBar.style.width = `${Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : 0}%`;
    dom.progressPercent.textContent = `${Number.isFinite(value) ? value.toFixed(1) : '0'}%`;
    dom.progressPhase.textContent = phase || '';
    if (message) {
      logLines.push(message);
      while (logLines.length > LOG_LIMIT) logLines.shift();
      dom.progressLog.hidden = false;
      dom.progressLog.textContent = logLines.join('\n');
      dom.progressLog.scrollTop = dom.progressLog.scrollHeight;
    }
  }

  dom.btnRetry.addEventListener('click', retry);
  dom.btnDetails.addEventListener('click', toggleDetails);

  async function listen(event, handler) {
    if (!tauri || !tauri.event || typeof tauri.event.listen !== 'function') return;
    try {
      await tauri.event.listen(event, (payload) => handler(payload && payload.payload));
    } catch (err) {
      // Missing permission must not take the boot screen down; URL state and
      // polling still report progress.
      console.debug('boot shell listen failed:', event, err);
    }
  }

  /**
   * Fallback channel: the shell can also be *navigated* to with the failure in
   * the query string, which works even without any IPC permissions.
   */
  function applyUrlState() {
    const params = new URLSearchParams(window.location.search);
    const error = params.get('error');
    if (error) {
      showError(error, params.get('details'));
      return true;
    }
    if (params.get('status')) setStatus(params.get('status'));
    return false;
  }

  async function init() {
    if (applyUrlState()) return;

    await listen('boot-status', (payload) => setStatus(payload && payload.message));
    await listen('boot-progress', (payload) =>
      showProgress(payload && payload.percent, payload && payload.phase, payload && payload.message)
    );
    await listen('boot-error', (payload) =>
      showError(payload && payload.message, payload && payload.details)
    );

    if (hasIpc) {
      try {
        const config = await tauri.core.invoke('get_app_config');
        if (config && config.log_dir) {
          dom.errorDetails.textContent = 'Logs: ' + config.log_dir;
        }
      } catch (err) {
        console.debug('boot shell config unavailable:', err);
      }
    }
  }

  window.__ClarityShell = { setStatus, showError, showProgress, retry, hasIpc };

  init();
})();
