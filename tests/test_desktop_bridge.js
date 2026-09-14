/**
 * Automated Unit Tests for the Desktop IPC Bridge (app.js).
 * Runs in Node.js with a mock Tauri environment.
 *
 * (The first-run wizard and its setup.js suite were removed: the installer
 * and the boot shell own setup now.)
 */

const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { test, describe } = require('node:test');

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

describe('Render queue lifecycle (Phase C)', () => {
  const appJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/app.js');
  const src = fs.readFileSync(appJsPath, 'utf8');

  // Same extraction strategy as the bridge suite: app.js internals live two
  // spaces deep, each function ending at the first line that is exactly '  }'.
  function extractFn(header) {
    const start = src.indexOf(header);
    assert.notStrictEqual(start, -1, 'app.js must define ' + header);
    const end = src.indexOf('\n  }', start);
    assert.notStrictEqual(end, -1, 'unterminated ' + header);
    return src.slice(start, end + 4);
  }

  test('done-state hides Cancel (resetRenderUI)', () => {
    const fnSrc = extractFn('function resetRenderUI()');
    function makeEl() {
      const added = [];
      const removed = [];
      return { added, removed, classList: { add: (c) => added.push(c), remove: (c) => removed.push(c) } };
    }
    const btnRender = makeEl();
    const btnCancel = makeEl();
    const renderProgressCard = makeEl();
    const resetRenderUI = new Function('elems', fnSrc + '\nreturn resetRenderUI;')({
      btnRender,
      btnCancel,
      renderProgressCard
    });

    resetRenderUI();

    assert.ok(btnCancel.added.includes('hidden'), 'Cancel must be hidden after reset');
    assert.ok(!btnCancel.removed.includes('hidden'), 'reset must not re-show Cancel');
  });

  test('cancel click hits /api/jobs/${id}/cancel', () => {
    assert.ok(
      src.includes('/api/jobs/${state.activeJob.job_id}/cancel'),
      'cancel must POST to /api/jobs/${id}/cancel'
    );
    assert.ok(!src.includes('/api/jobs/cancel/'), 'old /api/jobs/cancel/${id} route must be gone');
  });

  test('job_cancelled resets UI and clears the active job', () => {
    assert.ok(src.includes('job_cancelled'), 'ws handler must handle job_cancelled');
    const idx = src.indexOf('job_cancelled');
    const region = src.slice(idx, idx + 800);
    assert.ok(region.includes('state.activeJob'), 'cancelled branch must clear state.activeJob');
    assert.ok(region.includes('resetRenderUI'), 'cancelled branch must reset the render UI');
  });

  test('completed clears the queue selection (failed keeps it)', () => {
    const idx = src.indexOf("data.type === 'job_completed'");
    assert.notStrictEqual(idx, -1, 'ws handler must handle job_completed');
    const region = src.slice(idx, idx + 3000);
    assert.ok(region.includes('state.selectedVideos = []'), 'completed must clear selectedVideos');
    assert.ok(region.includes('updateQueueUI'), 'completed must re-render the queue UI');
  });

  test('delete unloads the preview BEFORE issuing DELETE', () => {
    const fnSrc = extractFn('async function handleDeleteVideo(');
    const fetchIdx = fnSrc.indexOf('/api/videos/delete');
    assert.notStrictEqual(fetchIdx, -1, 'handleDeleteVideo must call the delete endpoint');
    const before = fnSrc.slice(0, fetchIdx);
    assert.ok(before.includes('clearPreview'), 'preview must be unloaded before DELETE');
  });
});
