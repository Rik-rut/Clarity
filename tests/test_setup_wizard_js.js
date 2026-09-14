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
    'btn-copy-logs', 'browser-test-bar', 'btn-test-simulate', 'btn-test-error', 'btn-test-reset'
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

describe('Setup Wizard (setup.js) Unit Tests', () => {
  const setupJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/setup.js');
  const setupJsContent = fs.readFileSync(setupJsPath, 'utf8');

  test('Tauri environment: listeners registered and start_setup invoked', async () => {
    const tauriEvents = {};
    const tauriInvocations = [];

    const mockTauri = {
      event: {
        listen: (event, handler) => {
          tauriEvents[event] = handler;
          return Promise.resolve(() => {});
        }
      },
      core: {
        invoke: (cmd, args) => {
          tauriInvocations.push({ cmd, args });
          if (cmd === 'get_setup_status') {
            return Promise.resolve({ complete: false });
          }
          if (cmd === 'start_setup') {
            return Promise.resolve({ started: true });
          }
          return Promise.resolve(null);
        }
      }
    };

    const env = setupMockEnvironment(mockTauri);

    // Execute setup.js in simulated sandbox
    const vm = require('node:vm');
    const sandbox = {
      window: env.mockWindow,
      document: env.mockDocument,
      console: console,
      setTimeout: setTimeout,
      clearTimeout: clearTimeout,
      setInterval: setInterval,
      clearInterval: clearInterval,
      URLSearchParams: URLSearchParams,
      Date: Date
    };
    vm.createContext(sandbox);
    vm.runInContext(setupJsContent, sandbox);

    // Allow async init to settle
    await new Promise(r => setTimeout(r, 50));

    // Verify listeners registered
    assert.strictEqual(typeof tauriEvents['setup-progress'], 'function');
    assert.strictEqual(typeof tauriEvents['setup-complete'], 'function');
    assert.strictEqual(typeof tauriEvents['setup-error'], 'function');

    // Verify start_setup was invoked
    const startInvoked = tauriInvocations.some(inv => inv.cmd === 'start_setup');
    assert.strictEqual(startInvoked, true, 'Expected start_setup command to be invoked');
  });

  test('Progress event updates UI elements correctly', async () => {
    const tauriEvents = {};
    const mockTauri = {
      event: {
        listen: (event, handler) => {
          tauriEvents[event] = handler;
          return Promise.resolve(() => {});
        }
      },
      core: {
        invoke: () => Promise.resolve(null)
      }
    };

    const env = setupMockEnvironment(mockTauri);
    const vm = require('node:vm');
    const sandbox = {
      window: env.mockWindow,
      document: env.mockDocument,
      console: console,
      setTimeout: setTimeout,
      clearTimeout: clearTimeout,
      setInterval: setInterval,
      clearInterval: clearInterval,
      URLSearchParams: URLSearchParams,
      Date: Date
    };
    vm.createContext(sandbox);
    vm.runInContext(setupJsContent, sandbox);
    await new Promise(r => setTimeout(r, 30));

    // Simulate setup-progress event
    const progressHandler = tauriEvents['setup-progress'];
    assert.strictEqual(typeof progressHandler, 'function');

    progressHandler({
      payload: {
        stage: 'dependencies',
        percent: 65.4,
        speed: '24.5 MB/s',
        message: 'Downloading torch-2.3.0 wheel...'
      }
    });

    assert.strictEqual(env.elements['setup-percent-text'].textContent, '65.4%');
    assert.strictEqual(env.elements['setup-progress-bar'].style.width, '65.4%');
    assert.strictEqual(env.elements['setup-speed-text'].textContent, '24.5 MB/s');
    assert.strictEqual(env.elements['setup-speed-badge'].classList.contains('hidden'), false);
    assert.strictEqual(env.elements['setup-status-message'].textContent, 'Downloading torch-2.3.0 wheel...');

    // Stepper checks: Step 1 (python) and 2 (venv) completed, Step 3 (dependencies) active
    assert.strictEqual(env.elements['step-python'].classList.contains('completed'), true);
    assert.strictEqual(env.elements['step-venv'].classList.contains('completed'), true);
    assert.strictEqual(env.elements['step-dependencies'].classList.contains('active'), true);
  });

  test('Error event triggers error banner, logs message and expands logs drawer', async () => {
    const tauriEvents = {};
    const tauriInvocations = [];
    const mockTauri = {
      event: {
        listen: (event, handler) => {
          tauriEvents[event] = handler;
          return Promise.resolve(() => {});
        }
      },
      core: {
        invoke: (cmd, args) => {
          tauriInvocations.push({ cmd, args });
          return Promise.resolve(null);
        }
      }
    };

    const env = setupMockEnvironment(mockTauri);
    const vm = require('node:vm');
    const sandbox = {
      window: env.mockWindow,
      document: env.mockDocument,
      console: console,
      setTimeout: setTimeout,
      clearTimeout: clearTimeout,
      setInterval: setInterval,
      clearInterval: clearInterval,
      URLSearchParams: URLSearchParams,
      Date: Date
    };
    vm.createContext(sandbox);
    vm.runInContext(setupJsContent, sandbox);
    await new Promise(r => setTimeout(r, 30));

    // Simulate error event
    const errorHandler = tauriEvents['setup-error'];
    errorHandler({ payload: { error: 'Failed to extract uv package archive' } });

    assert.strictEqual(env.elements['setup-error-banner'].classList.contains('hidden'), false);
    assert.strictEqual(env.elements['setup-error-message'].textContent, 'Failed to extract uv package archive');
    // Logs drawer should be auto-expanded
    assert.strictEqual(env.elements['logs-drawer'].classList.contains('hidden'), false);
    assert.strictEqual(env.elements['logs-accordion'].classList.contains('open'), true);

    // Click retry button
    env.elements['btn-retry-setup'].click();
    await new Promise(r => setTimeout(r, 30));

    assert.strictEqual(env.elements['setup-error-banner'].classList.contains('hidden'), true);
    const retryInvoked = tauriInvocations.some(inv => inv.cmd === 'retry_setup' || inv.cmd === 'start_setup');
    assert.strictEqual(retryInvoked, true, 'Retry should invoke retry_setup or start_setup');
  });

  test('Complete event shows success banner and sets 100%', async () => {
    const tauriEvents = {};
    const mockTauri = {
      event: {
        listen: (event, handler) => {
          tauriEvents[event] = handler;
          return Promise.resolve(() => {});
        }
      },
      core: {
        invoke: () => Promise.resolve(null)
      }
    };

    const env = setupMockEnvironment(mockTauri);
    const vm = require('node:vm');
    const sandbox = {
      window: env.mockWindow,
      document: env.mockDocument,
      console: console,
      setTimeout: setTimeout,
      clearTimeout: clearTimeout,
      setInterval: setInterval,
      clearInterval: clearInterval,
      URLSearchParams: URLSearchParams,
      Date: Date
    };
    vm.createContext(sandbox);
    vm.runInContext(setupJsContent, sandbox);
    await new Promise(r => setTimeout(r, 30));

    const completeHandler = tauriEvents['setup-complete'];
    completeHandler({ payload: { port: 7860 } });

    assert.strictEqual(env.elements['setup-percent-text'].textContent, '100.0%');
    assert.strictEqual(env.elements['setup-progress-bar'].style.width, '100%');
    assert.strictEqual(env.elements['setup-success-banner'].classList.contains('hidden'), false);
    assert.strictEqual(env.elements['setup-error-banner'].classList.contains('hidden'), true);
  });

  test('Standalone mode initializes safely and supports log toggling', async () => {
    // No Tauri
    const env = setupMockEnvironment(null);
    const vm = require('node:vm');
    const sandbox = {
      window: env.mockWindow,
      document: env.mockDocument,
      console: console,
      setTimeout: setTimeout,
      clearTimeout: clearTimeout,
      setInterval: setInterval,
      clearInterval: clearInterval,
      URLSearchParams: URLSearchParams,
      Date: Date
    };
    vm.createContext(sandbox);
    vm.runInContext(setupJsContent, sandbox);
    await new Promise(r => setTimeout(r, 30));

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

  test('app.js source contains notification bridge functions', () => {
    assert.strictEqual(appJsContent.includes('function initDesktopBridge()'), true);
    assert.strictEqual(appJsContent.includes('function sendDesktopNotification(title, body)'), true);
    assert.strictEqual(appJsContent.includes('window.__TAURI__.notification.sendNotification'), true);
    assert.strictEqual(appJsContent.includes('Clarity — Render Complete'), true);
  });

  test('initDesktopBridge checks and requests Tauri notification permission', async () => {
    let permissionRequested = false;
    let permissionChecked = false;

    const mockTauri = {
      notification: {
        isPermissionGranted: () => {
          permissionChecked = true;
          return Promise.resolve(false);
        },
        requestPermission: () => {
          permissionRequested = true;
          return Promise.resolve('granted');
        },
        sendNotification: () => {}
      }
    };

    // Extract initDesktopBridge function and test its logic
    const initMatch = appJsContent.match(/async function initDesktopBridge\(\) \{[\s\S]*?\n  \}/);
    assert.ok(initMatch, 'Should find initDesktopBridge in app.js');

    const fn = new Function('window', `return (${initMatch[0]});`)(({ __TAURI__: mockTauri }));
    await fn();

    assert.strictEqual(permissionChecked, true, 'Should check if permission is granted');
    assert.strictEqual(permissionRequested, true, 'Should request permission if not granted');
  });

  test('sendDesktopNotification invokes Tauri sendNotification', () => {
    const sentNotifications = [];

    const mockTauri = {
      notification: {
        sendNotification: (options) => {
          sentNotifications.push(options);
        }
      }
    };

    const sendMatch = appJsContent.match(/function sendDesktopNotification\(title, body\) \{[\s\S]*?\n  \}/);
    assert.ok(sendMatch, 'Should find sendDesktopNotification in app.js');

    const fn = new Function('window', `return (${sendMatch[0]});`)(({ __TAURI__: mockTauri }));
    fn('Clarity — Render Complete', 'Video "test.mp4" has finished processing!');

    assert.strictEqual(sentNotifications.length, 1);
    assert.strictEqual(sentNotifications[0].title, 'Clarity — Render Complete');
    assert.strictEqual(sentNotifications[0].body, 'Video "test.mp4" has finished processing!');
  });

  test('sendDesktopNotification falls back to window.__TAURI__.core.invoke if notification object is missing', async () => {
    const invokedCommands = [];

    const mockTauri = {
      core: {
        invoke: (cmd, args) => {
          invokedCommands.push({ cmd, args });
          return Promise.resolve();
        }
      }
    };

    const sendMatch = appJsContent.match(/function sendDesktopNotification\(title, body\) \{[\s\S]*?\n  \}/);
    const fn = new Function('window', `return (${sendMatch[0]});`)(({ __TAURI__: mockTauri }));
    fn('Clarity — Render Complete', 'Render finished');

    assert.strictEqual(invokedCommands.length, 1);
    assert.strictEqual(invokedCommands[0].cmd, 'plugin:notification|notify');
    assert.strictEqual(invokedCommands[0].args.options.title, 'Clarity — Render Complete');
  });
});

