/**
 * Automated Unit Tests for setup.js & Desktop IPC Bridge
 * Runs in Node.js with a mock browser DOM and mock Tauri environment.
 */

const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { test, describe, beforeEach } = require('node:test');

// Simple DOM Mock Helper
function createMockElement(id = '', tag = 'div') {
  const listeners = {};
  const children = [];
  const classes = new Set();
  const attributes = {};

  const el = {
    id,
    tagName: tag.toUpperCase(),
    textContent: '',
    innerHTML: '',
    style: {},
    className: '',
    children,
    parentElement: null,
    scrollTop: 0,
    scrollHeight: 100,

    classList: {
      add: (...cls) => {
        cls.forEach(c => classes.add(c));
        el.className = Array.from(classes).join(' ');
      },
      remove: (...cls) => {
        cls.forEach(c => classes.delete(c));
        el.className = Array.from(classes).join(' ');
      },
      toggle: (c) => {
        if (classes.has(c)) {
          classes.delete(c);
          el.className = Array.from(classes).join(' ');
          return false;
        } else {
          classes.add(c);
          el.className = Array.from(classes).join(' ');
          return true;
        }
      },
      contains: (c) => classes.has(c)
    },

    setAttribute: (k, v) => { attributes[k] = String(v); },
    getAttribute: (k) => attributes[k] || null,

    addEventListener: (event, handler) => {
      listeners[event] = listeners[event] || [];
      listeners[event].push(handler);
    },

    dispatchEvent: (event) => {
      const handlers = listeners[event.type] || [];
      handlers.forEach(h => h(event));
    },

    click: () => {
      const handlers = listeners['click'] || [];
      handlers.forEach(h => h({ type: 'click', target: el }));
    },

    appendChild: (child) => {
      child.parentElement = el;
      children.push(child);
      return child;
    },

    querySelector: (selector) => {
      if (selector === '.step-num') {
        let found = children.find(c => c.classList.contains('step-num'));
        if (!found) {
          found = createMockElement('', 'div');
          found.classList.add('step-num');
          el.appendChild(found);
        }
        return found;
      }
      return null;
    }
  };

  return el;
}

function setupMockEnvironment(customTauri = null) {
  const elements = {};
  const elementIds = [
    'setup-stepper', 'step-python', 'step-venv', 'step-dependencies', 'step-models',
    'setup-stage-title', 'setup-status-message', 'setup-percent-text', 'setup-progress-bar',
    'setup-speed-badge', 'setup-speed-text', 'setup-eta-badge', 'setup-eta-text',
    'setup-active-task-badge', 'setup-error-banner', 'setup-error-message', 'btn-retry-setup',
    'setup-success-banner', 'setup-success-subtitle', 'logs-accordion', 'btn-toggle-logs',
    'logs-toggle-label', 'logs-count-badge', 'logs-drawer', 'logs-terminal', 'btn-clear-logs',
    'btn-copy-logs', 'browser-test-bar', 'btn-test-simulate', 'btn-test-error', 'btn-test-reset',
    'setup-storage-row', 'setup-storage-path', 'setup-storage-note', 'setup-storage-error',
    'btn-change-storage'
  ];

  elementIds.forEach(id => {
    elements[id] = createMockElement(id);
  });

  // Apply default classes matching setup.html
  elements['logs-drawer'].classList.add('hidden');
  elements['setup-error-banner'].classList.add('hidden');
  elements['setup-success-banner'].classList.add('hidden');
  elements['browser-test-bar'].classList.add('hidden');
  elements['setup-speed-badge'].classList.add('hidden');
  elements['setup-eta-badge'].classList.add('hidden');

  // Track parent element for progress bar
  const track = createMockElement('progress-track');
  track.appendChild(elements['setup-progress-bar']);

  const domListeners = {};
  const mockDocument = {
    readyState: 'complete',
    getElementById: (id) => elements[id] || null,
    createElement: (tag) => createMockElement('', tag),
    addEventListener: (ev, cb) => {
      domListeners[ev] = domListeners[ev] || [];
      domListeners[ev].push(cb);
    }
  };

  const mockWindow = {
    document: mockDocument,
    location: { href: 'http://localhost/setup.html', search: '' },
    __TAURI__: customTauri,
    Date: Date,
    addEventListener: (ev, cb) => {
      domListeners[ev] = domListeners[ev] || [];
      domListeners[ev].push(cb);
    }
  };

  return { elements, mockDocument, mockWindow, domListeners };
}

/**
 * Test doubles for the HTTP wizard protocol.
 *
 * The wizard page talks to video_upscaler/desktop/server.py over loopback HTTP,
 * so the only environment it needs is a fetch stub. The Tauri spy exists purely
 * to prove the page no longer depends on desktop IPC.
 */
function setupJsPathForTests() {
  return path.resolve(__dirname, '../src/video_upscaler/web/static/js/setup.js');
}

function makeFetchStub(calls, routes) {
  return async function fetchStub(path, options) {
    const opts = options || {};
    const record = {
      path,
      method: opts.method || 'GET',
      body: opts.body ? JSON.parse(opts.body) : null
    };
    calls.push(record);

    const table = typeof routes === 'function' ? routes(record) : routes[path];
    if (table === undefined) {
      return { ok: false, status: 404, json: async () => ({ error: 'unexpected call ' + path }) };
    }
    return { ok: true, status: 200, json: async () => table };
  };
}

function makeTauriSpy(ipc, results) {
  return {
    event: {
      listen: (name) => {
        ipc.listened.push(name);
        return Promise.resolve(() => {});
      }
    },
    core: {
      invoke: (cmd, args) => {
        ipc.invoked.push({ cmd, args });
        const table = results || {};
        return Promise.resolve(table[cmd] === undefined ? null : table[cmd]);
      }
    }
  };
}

async function runSetupJs(env, fetchImpl, settleMs) {
  const vm = require('node:vm');
  const sandbox = {
    window: env.mockWindow,
    document: env.mockDocument,
    console,
    fetch: fetchImpl,
    navigator: { clipboard: { writeText: () => Promise.resolve() } },
    // The wizard polls, so its timers must not hold the test process open.
    setTimeout: (fn, ms) => { const t = setTimeout(fn, ms); if (t && t.unref) t.unref(); return t; },
    clearTimeout,
    setInterval: (fn, ms) => { const t = setInterval(fn, ms); if (t && t.unref) t.unref(); return t; },
    clearInterval,
    URLSearchParams,
    Date
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(setupJsPathForTests(), 'utf8'), sandbox);
  await new Promise((resolve) => setTimeout(resolve, settleMs === undefined ? 60 : settleMs));
  return sandbox;
}

describe('Setup Wizard (setup.js) Unit Tests', () => {
  const setupJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/setup.js');
  const setupJsContent = fs.readFileSync(setupJsPath, 'utf8');

  test('wizard talks HTTP to the provisioning server and never uses desktop IPC', async () => {
    const calls = [];
    const ipc = { invoked: [], listened: [] };

    const env = setupMockEnvironment(makeTauriSpy(ipc));
    await runSetupJs(env, makeFetchStub(calls, {
      '/api/setup/status': {
        complete: false,
        running: false,
        next_step: 'gpu',
        state: { steps: { gpu: 'pending' } }
      },
      '/api/setup/progress': { running: false, step: 'gpu', percent: 0, lines: [], total: 0 }
    }));

    const paths = calls.map((call) => call.path);
    assert.ok(paths.includes('/api/setup/status'), 'wizard must poll /api/setup/status');
    assert.ok(paths.includes('/api/setup/progress'), 'wizard must poll /api/setup/progress');
    assert.deepStrictEqual(
      calls.filter((call) => call.method === 'POST').map((call) => call.path),
      ['/api/setup/detect-gpu'],
      'the pending first step must be started over HTTP'
    );

    // This page used to be driven by Tauri events on a foreign origin, where the
    // event bridge is not connected: the bar sat at 0% forever. IPC is gone.
    assert.deepStrictEqual(ipc.listened, [], 'wizard must not register Tauri event listeners');
    assert.deepStrictEqual(
      ipc.invoked.map((entry) => entry.cmd),
      [],
      'wizard must not invoke Tauri commands'
    );
  });

  test('progress snapshot updates percent, speed, message and stepper', async () => {
    const env = setupMockEnvironment(null);
    await runSetupJs(env, makeFetchStub([], {
      '/api/setup/status': {
        complete: false,
        running: true,
        next_step: 'runtime',
        state: { steps: { gpu: 'done', runtime: 'running' } }
      },
      '/api/setup/progress': {
        running: true,
        step: 'runtime',
        phase: 'dependencies',
        percent: 65.4,
        speed: '24.5 MB/s',
        message: 'Downloading torch wheel',
        lines: [],
        total: 0
      }
    }));

    assert.strictEqual(env.elements['setup-percent-text'].textContent, '65.4%');
    assert.strictEqual(env.elements['setup-progress-bar'].style.width, '65.4%');
    assert.strictEqual(env.elements['setup-speed-text'].textContent, '24.5 MB/s');
    assert.strictEqual(env.elements['setup-speed-badge'].classList.contains('hidden'), false);
    assert.strictEqual(env.elements['setup-status-message'].textContent, 'Downloading torch wheel');
    assert.strictEqual(
      env.elements['setup-stage-title'].textContent,
      'Installing PyTorch & AI Engine Libraries...'
    );

    // dependencies is stepper item 3, so 1-2 are done and models has not started.
    assert.strictEqual(env.elements['step-python'].classList.contains('completed'), true);
    assert.strictEqual(env.elements['step-venv'].classList.contains('completed'), true);
    assert.strictEqual(env.elements['step-dependencies'].classList.contains('active'), true);
    assert.strictEqual(env.elements['step-models'].classList.contains('active'), false);
  });

  test('progress percentage never regresses', async () => {
    const env = setupMockEnvironment(null);
    await runSetupJs(env, makeFetchStub([], {
      '/api/setup/status': {
        complete: false,
        running: true,
        next_step: 'models',
        state: { steps: { gpu: 'done' } }
      },
      '/api/setup/progress': { running: true, step: 'models', percent: 65.4, lines: [], total: 0 }
    }));

    env.mockWindow.__ClaritySetup.handleProgress({
      step: 'models',
      percent: 20,
      message: 'Resuming a partially downloaded file'
    });

    assert.strictEqual(env.elements['setup-percent-text'].textContent, '65.4%');
    assert.strictEqual(env.mockWindow.__ClaritySetup.state.percent, 65.4);
  });

  test('a failed step shows the banner and Retry re-runs it over HTTP', async () => {
    const calls = [];
    const env = setupMockEnvironment(null);
    await runSetupJs(env, makeFetchStub(calls, {
      '/api/setup/status': {
        complete: false,
        running: false,
        next_step: 'runtime',
        state: { steps: { gpu: 'done', runtime: 'failed' } }
      },
      '/api/setup/progress': {
        running: false,
        step: 'runtime',
        percent: 40,
        error: 'Failed to extract uv package archive',
        lines: [],
        total: 0
      }
    }));

    assert.strictEqual(env.elements['setup-error-banner'].classList.contains('hidden'), false);
    assert.strictEqual(
      env.elements['setup-error-message'].textContent,
      'Failed to extract uv package archive'
    );
    // The logs must open themselves: that is the only clue the user can act on.
    assert.strictEqual(env.elements['logs-drawer'].classList.contains('hidden'), false);
    assert.strictEqual(env.elements['logs-accordion'].classList.contains('open'), true);

    const before = calls.length;
    env.elements['btn-retry-setup'].click();

    // Cleared synchronously, before the request is even sent.
    assert.strictEqual(env.elements['setup-error-banner'].classList.contains('hidden'), true);

    await new Promise((resolve) => setTimeout(resolve, 60));
    const retry = calls.slice(before).find((call) => call.path === '/api/setup/retry');
    assert.ok(retry, 'Retry must POST /api/setup/retry');
    // It must resume the step that failed, not restart the whole wizard.
    assert.strictEqual(retry.body.step, 'runtime');
  });

  test('completion fills the bar and shows the success banner', async () => {
    const env = setupMockEnvironment(null);
    await runSetupJs(env, makeFetchStub([], {
      '/api/setup/status': {
        complete: true,
        running: false,
        next_step: 'complete',
        state: { steps: { gpu: 'done', runtime: 'done', models: 'done', verify: 'done' } }
      },
      '/api/setup/progress': { running: false, percent: 0, lines: [], total: 0 }
    }));

    assert.strictEqual(env.elements['setup-percent-text'].textContent, '100.0%');
    assert.strictEqual(env.elements['setup-progress-bar'].style.width, '100%');
    assert.strictEqual(env.elements['setup-success-banner'].classList.contains('hidden'), false);
    assert.strictEqual(env.elements['setup-error-banner'].classList.contains('hidden'), true);
    assert.strictEqual(env.elements['step-models'].classList.contains('completed'), true);
  });

  test('Standalone mode initializes safely and supports log toggling', async () => {
    // No Tauri and no reachable wizard server: the page must still boot.
    const env = setupMockEnvironment(null);
    await runSetupJs(env, undefined, 30);

    // Test toggle logs
    assert.strictEqual(env.elements['logs-drawer'].classList.contains('hidden'), true);
    env.elements['btn-toggle-logs'].click();
    assert.strictEqual(env.elements['logs-drawer'].classList.contains('hidden'), false);
    env.elements['btn-toggle-logs'].click();
    assert.strictEqual(env.elements['logs-drawer'].classList.contains('hidden'), true);
  });
});

describe('Desktop IPC & Notification Bridge (app.js) Unit Tests', () => {
  const appJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/app.js');
  const appJsContent = fs.readFileSync(appJsPath, 'utf8');

  /**
   * Extract top-level helpers out of the app.js IIFE so they can be exercised in
   * isolation. Everything in that file is indented two spaces inside the closure,
   * and each function ends at the first line that is exactly '  }'.
   */
  function bridgeSource(headers) {
    let source = '';
    for (const header of headers) {
      const start = appJsContent.indexOf(header);
      assert.notStrictEqual(start, -1, 'app.js must define ' + header);
      const end = appJsContent.indexOf('\n  }', start);
      assert.notStrictEqual(end, -1, 'unterminated ' + header);
      source += appJsContent.slice(start, end + 4) + '\n\n';
    }
    return source;
  }

  const BRIDGE_HEADERS = [
    'function tauriInvoke()',
    'function sendWebNotification(title, body)',
    'async function initDesktopBridge()',
    'function sendDesktopNotification(title, body)'
  ];

  test('app.js source contains notification bridge functions', () => {
    for (const header of BRIDGE_HEADERS) {
      assert.ok(appJsContent.includes(header), 'app.js must define ' + header);
    }
    assert.ok(appJsContent.includes("'plugin:notification|notify'"));
    assert.ok(appJsContent.includes("'plugin:notification|request_permission'"));
    assert.ok(appJsContent.includes('Clarity — Render Complete'));
  });

  test('initDesktopBridge requests permission through IPC when running in Tauri', async () => {
    const invoked = [];
    const bridge = bridgeSource(BRIDGE_HEADERS);
    const initDesktopBridge = new Function(
      'window',
      'Notification',
      bridge + '\nreturn initDesktopBridge;'
    )(
      {
        __TAURI__: {
          core: {
            invoke: (cmd, args) => {
              invoked.push({ cmd, args });
              return Promise.resolve('granted');
            }
          }
        }
      },
      undefined
    );

    await initDesktopBridge();

    assert.deepStrictEqual(
      invoked.map((call) => call.cmd),
      ['plugin:notification|request_permission'],
      'the desktop build must ask the notification plugin, not the browser'
    );
  });

  test('initDesktopBridge asks the browser for permission without Tauri', async () => {
    let requested = false;
    const bridge = bridgeSource(BRIDGE_HEADERS);
    const initDesktopBridge = new Function(
      'window',
      'Notification',
      bridge + '\nreturn initDesktopBridge;'
    )(
      {},
      {
        permission: 'default',
        requestPermission: () => {
          requested = true;
          return Promise.resolve('granted');
        }
      }
    );

    await initDesktopBridge();
    assert.strictEqual(requested, true);
  });

  test('sendDesktopNotification prefers the notification plugin IPC command', async () => {
    const invokedCommands = [];
    const webNotifications = [];

    function NotificationMock(title, options) {
      webNotifications.push({ title, body: options.body });
    }
    NotificationMock.permission = 'granted';

    const bridge = bridgeSource(BRIDGE_HEADERS);
    const sendDesktopNotification = new Function(
      'window',
      'Notification',
      bridge + '\nreturn sendDesktopNotification;'
    )(
      {
        __TAURI__: {
          core: {
            invoke: (cmd, args) => {
              invokedCommands.push({ cmd, args });
              return Promise.resolve();
            }
          }
        }
      },
      NotificationMock
    );

    sendDesktopNotification('Clarity — Render Complete', 'Render finished');
    await new Promise((resolve) => setTimeout(resolve, 5));

    assert.strictEqual(invokedCommands.length, 1);
    assert.strictEqual(invokedCommands[0].cmd, 'plugin:notification|notify');
    assert.strictEqual(invokedCommands[0].args.options.title, 'Clarity — Render Complete');
    assert.strictEqual(invokedCommands[0].args.options.body, 'Render finished');
    assert.strictEqual(webNotifications.length, 0, 'native toast must win over the browser API');
  });

  test('sendDesktopNotification falls back to the web Notification API without usable IPC', () => {
    const shown = [];

    function NotificationMock(title, options) {
      shown.push({ title, body: options.body });
    }
    NotificationMock.permission = 'granted';

    const bridge = bridgeSource(BRIDGE_HEADERS);
    const sendDesktopNotification = new Function(
      'window',
      'Notification',
      bridge + '\nreturn sendDesktopNotification;'
    )({ __TAURI__: {} }, NotificationMock);

    sendDesktopNotification('Clarity — Render Complete', 'Video finished');

    assert.strictEqual(shown.length, 1);
    assert.strictEqual(shown[0].title, 'Clarity — Render Complete');
    assert.strictEqual(shown[0].body, 'Video finished');
  });

  test('a rejected native toast still produces a browser notification', async () => {
    const shown = [];

    function NotificationMock(title, options) {
      shown.push({ title, body: options.body });
    }
    NotificationMock.permission = 'granted';

    const bridge = bridgeSource(BRIDGE_HEADERS);
    const sendDesktopNotification = new Function(
      'window',
      'Notification',
      bridge + '\nreturn sendDesktopNotification;'
    )(
      {
        __TAURI__: {
          core: {
            invoke: () => Promise.reject(new Error('ACL denied for this origin'))
          }
        }
      },
      NotificationMock
    );

    sendDesktopNotification('Clarity — Render Complete', 'Fallback body');
    await new Promise((resolve) => setTimeout(resolve, 5));

    assert.strictEqual(shown.length, 1, 'a denied plugin permission must not swallow the notice');
    assert.strictEqual(shown[0].body, 'Fallback body');
  });
});
