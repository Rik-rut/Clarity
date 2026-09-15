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
    assert.ok(before.includes('unloadPlayerIfShowing'), 'matching previews must be unloaded before DELETE');
    assert.ok(!before.includes('clearPreview'), 'delete must not blanket-clear unrelated previews/mask');
  });
});

describe('Scoped unload on deletes (F3)', () => {
  const appJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/app.js');
  const src = fs.readFileSync(appJsPath, 'utf8');

  // Same extraction strategy as the suites above: app.js internals live two
  // spaces deep, each function ending at the first line that is exactly '  }'.
  function extractFn(header) {
    const start = src.indexOf(header);
    assert.notStrictEqual(start, -1, 'app.js must define ' + header);
    const end = src.indexOf('\n  }', start);
    assert.notStrictEqual(end, -1, 'unterminated ' + header);
    return src.slice(start, end + 4);
  }

  function loadScopedHelpers() {
    const fnSrc =
      extractFn('function playerShowsVideo(') + '\n\n' + extractFn('function unloadPlayerIfShowing(');
    return new Function(fnSrc + '\nreturn { playerShowsVideo, unloadPlayerIfShowing };')();
  }

  function makePlayer(rawSrc) {
    const calls = [];
    return {
      calls,
      src: rawSrc,
      currentSrc: '',
      getAttribute(name) {
        return name === 'src' ? this.src : null;
      },
      removeAttribute(name) {
        calls.push('remove:' + name);
        if (name === 'src') this.src = '';
      },
      pause() {
        calls.push('pause');
      },
      load() {
        calls.push('load');
      }
    };
  }

  function makePlaceholder() {
    const removed = [];
    return { removed, classList: { remove: (c) => removed.push(c) } };
  }

  test('delete path no longer blanket-clears previews or mask state', () => {
    const fnSrc = extractFn('async function handleDeleteVideo(');
    assert.ok(!fnSrc.includes('clearPreview'), 'delete must not blanket clearPreview (wipes mask + unrelated previews)');
    assert.ok(!fnSrc.includes('__maMask'), 'delete must leave mask state alone');
    assert.ok(!fnSrc.includes('maMaskStage'), 'delete must leave the mask stage alone');
  });

  test('deleting file A while the player shows file B leaves B playing', () => {
    const { unloadPlayerIfShowing } = loadScopedHelpers();
    const playerB = makePlayer('/api/stream/video?path=' + encodeURIComponent('/vids/B.mp4'));
    const placeholder = makePlaceholder();

    const unloaded = unloadPlayerIfShowing(playerB, placeholder, { name: 'A.mp4', path: '/vids/A.mp4' });

    assert.strictEqual(unloaded, false, 'unrelated player must not be unloaded');
    assert.deepStrictEqual(playerB.calls, [], 'unrelated player must keep playing (no pause/load)');
    assert.strictEqual(
      playerB.src,
      '/api/stream/video?path=' + encodeURIComponent('/vids/B.mp4'),
      'unrelated player src must be untouched'
    );
    assert.deepStrictEqual(placeholder.removed, [], 'no placeholder churn for unrelated deletes');
  });

  test('deleting the file the player shows unloads that player', () => {
    const { unloadPlayerIfShowing } = loadScopedHelpers();
    const playerA = makePlayer('/api/stream/video?path=' + encodeURIComponent('/vids/A.mp4'));
    const placeholder = makePlaceholder();

    const unloaded = unloadPlayerIfShowing(playerA, placeholder, { name: 'A.mp4', path: '/vids/A.mp4' });

    assert.strictEqual(unloaded, true, 'matching player must be unloaded');
    assert.deepStrictEqual(playerA.calls, ['pause', 'remove:src', 'load'], 'matching player must pause + drop src + load');
    assert.strictEqual(playerA.src, '', 'matching player src must be released');
  });

  test('input-delete path unloads the input player showing the target', () => {
    const fnSrc = extractFn('async function handleDeleteVideo(');
    assert.ok(fnSrc.includes('videoLeft'), 'input-delete path must consider the input player (videoLeft)');
    assert.ok(fnSrc.includes('videoMaInput'), 'input-delete path must consider the MA input player (videoMaInput)');

    const { unloadPlayerIfShowing } = loadScopedHelpers();
    const videoLeft = makePlayer('/api/stream/video?path=' + encodeURIComponent('C:\\Clarity\\input\\clip.mp4'));
    const placeholderLeft = makePlaceholder();

    const unloaded = unloadPlayerIfShowing(videoLeft, placeholderLeft, {
      name: 'clip.mp4',
      path: 'C:\\Clarity\\input\\clip.mp4'
    });

    assert.strictEqual(unloaded, true, 'input player showing the deleted file must be unloaded');
    assert.deepStrictEqual(videoLeft.calls, ['pause', 'remove:src', 'load']);
  });

  test('scoped unload runs BEFORE fetch(DELETE)', () => {
    const fnSrc = extractFn('async function handleDeleteVideo(');
    const fetchIdx = fnSrc.indexOf('/api/videos/delete');
    assert.notStrictEqual(fetchIdx, -1, 'handleDeleteVideo must call the delete endpoint');
    const unloadIdx = fnSrc.indexOf('unloadPlayerIfShowing');
    assert.notStrictEqual(unloadIdx, -1, 'handleDeleteVideo must use the scoped unload helper');
    assert.ok(unloadIdx < fetchIdx, 'scoped unload must run before fetch(DELETE) so the stream handle is released');
  });
});
