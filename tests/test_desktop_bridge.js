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
    'function tauriInvoke(cmd, args)',
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
    const fnSrc = extractFn('function handleJobCancelled(');
    assert.ok(fnSrc.includes('state.activeJob = null'), 'cancelled handler must clear state.activeJob');
    assert.ok(fnSrc.includes('resetRenderUI'), 'cancelled handler must reset the render UI');
    assert.ok(src.includes("data.type === 'job_cancelled'"), 'ws handler must still route job_cancelled');
    const idx = src.indexOf("data.type === 'job_cancelled'");
    const region = src.slice(idx, idx + 300);
    assert.ok(region.includes('handleJobCancelled'), 'cancelled branch must use the shared handler');
  });

  test('completed keeps the queue selection', () => {
    const fnSrc = extractFn('function handleJobCompleted(');
    assert.strictEqual(fnSrc.includes('state.selectedVideos = []'), false, 'completed must keep selectedVideos');
    assert.ok(src.includes('handleJobCompleted(data.job)'), 'ws branch must delegate to the shared handler');
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

describe('Status poll reconciler (F4)', () => {
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

  function makeElem() {
    const added = [];
    const removed = [];
    const el = {
      added,
      removed,
      classList: { add: (c) => added.push(c), remove: (c) => removed.push(c) },
      textContent: '',
      src: '',
      loadCalls: 0,
      load() { el.loadCalls++; }
    };
    return el;
  }

  function completedSnapshot() {
    return {
      job_id: 'j1',
      status: 'completed',
      percent: 100,
      stage: 'Processing complete!',
      current_file_index: 1,
      total_files: 1,
      current_file_name: 'a.mp4',
      elapsed_formatted: '0:03',
      eta_formatted: '--',
      output_files: [],
      error_message: null
    };
  }

  // Build the real poller + shared terminal handlers with stubbed siblings.
  function loadPollerHarness(overrides) {
    overrides = overrides || {};
    const intervalMatch = src.match(/STATUS_POLL_INTERVAL_MS\s*=\s*(\d+)/);
    assert.ok(intervalMatch, 'app.js must define STATUS_POLL_INTERVAL_MS');
    const graceMatch = src.match(/STATUS_POLL_GRACE_MS\s*=\s*(\d+)/);
    assert.ok(graceMatch, 'app.js must define STATUS_POLL_GRACE_MS');
    assert.strictEqual(Number(intervalMatch[1]), 2500, 'poll interval must stay conservative (2.5s)');
    assert.strictEqual(Number(graceMatch[1]), 5000, 'grace window must be 5s (no WS heartbeat constant exists)');

    const fetchCalls = [];
    const fetchImpl = overrides.fetch || (() => {
      fetchCalls.push('/api/jobs/status');
      return Promise.resolve({ json: () => Promise.resolve({ active: true, job: { job_id: 'srv' } }) });
    });
    const fetchMock = (...args) => fetchImpl(...args);
    const intervals = [];
    const cleared = [];
    let timerSeq = 0;
    const setIntervalMock = (cb, ms) => {
      timerSeq++;
      intervals.push({ id: timerSeq, cb, ms });
      return timerSeq;
    };
    const clearIntervalMock = (id) => { cleared.push(id); };
    let nowMs = 1000000;
    const dateMock = { now: () => nowMs };

    const prelude = [
      'const STATUS_POLL_INTERVAL_MS = ' + intervalMatch[1] + ';',
      'const STATUS_POLL_GRACE_MS = ' + graceMatch[1] + ';',
      'let statusPollTimer = null;',
      'let lastWsEventAt = 0;',
      'const calls = { toasts: [], notifications: [], queueUpdates: 0, maUpdates: 0, outputLoads: 0, maResults: 0 };',
      'function showToast(msg, kind) { calls.toasts.push({ msg: msg, kind: kind }); }',
      'function updateMaRenderButton() { calls.maUpdates++; }',
      'function updateQueueUI() { calls.queueUpdates++; }',
      'function sendDesktopNotification(title, body) { calls.notifications.push({ title: title, body: body }); }',
      'function loadOutputVideos(silent) { calls.outputLoads++; }',
      'function populateMaResultWindows(files) { calls.maResults++; }',
      'function fetchActivity() { calls.activityFetches = (calls.activityFetches || 0) + 1; }'
    ].join('\n');
    const fns = [
      extractFn('function resetRenderUI()'),
      extractFn('function handleJobCompleted('),
      extractFn('function handleJobCancelled('),
      extractFn('function startStatusPoller('),
      extractFn('function stopStatusPoller('),
      extractFn('async function pollJobStatus(')
    ].join('\n\n');
    const factory = new Function(
      'state',
      'elems',
      'fetch',
      'setInterval',
      'clearInterval',
      'Date',
      prelude + '\n' + fns +
      '\nreturn { startStatusPoller, stopStatusPoller, pollJobStatus,' +
      ' handleJobCompleted, handleJobCancelled, calls,' +
      ' getTimer: () => statusPollTimer,' +
      ' setLastWsEvent: (v) => { lastWsEventAt = v; } };'
    );
    const state = {
      activeJob: null,
      selectedVideos: [{ name: 'a.mp4' }],
      activeTab: 'upscale',
      isOutputDropdownOpen: false
    };
    const elems = {
      btnRender: makeElem(),
      btnCancel: makeElem(),
      renderProgressCard: makeElem(),
      renderCompletedBanner: makeElem(),
      renderTotalTime: makeElem(),
      videoRight: makeElem(),
      placeholderRight: makeElem()
    };
    const api = factory(state, elems, fetchMock, setIntervalMock, clearIntervalMock, dateMock);
    api.now = () => nowMs;
    api.advance = (ms) => { nowMs += ms; };
    return { api, state, elems, fetchCalls, intervals, cleared, advance: api.advance };
  }

  test('poller starts on render start and stops on WS completion (no stray intervals)', () => {
    const h = loadPollerHarness();
    assert.strictEqual(h.api.getTimer(), null, 'no timer before a render starts');
    h.api.startStatusPoller();
    const first = h.api.getTimer();
    assert.notStrictEqual(first, null, 'render start must arm the poller');
    assert.strictEqual(h.intervals.length, 1);
    assert.strictEqual(h.intervals[0].ms, 2500);
    h.api.startStatusPoller();
    assert.deepStrictEqual(h.cleared, [first], 'restart must clear the old interval (single poller)');
    assert.strictEqual(h.intervals.length, 2);

    h.state.activeJob = completedSnapshot();
    h.api.handleJobCompleted(h.state.activeJob);
    assert.strictEqual(h.state.activeJob, null, 'WS completion must clear the active job');
    assert.strictEqual(h.api.getTimer(), null, 'WS completion must stop the poller');
    assert.ok(h.cleared.length >= 2, 'completion must clear the interval');
    assert.ok(h.elems.btnCancel.added.includes('hidden'), 'Cancel must hide after completion');
  });

  test('missed broadcast reconciles through the shared terminal handler exactly once', async () => {
    let statusPayload = { active: false, job: null };
    const h = loadPollerHarness({
      fetch: () => {
        hRef.fetchCalls.push('/api/jobs/status');
        return Promise.resolve({ json: () => Promise.resolve(statusPayload) });
      }
    });
    const hRef = h;
    h.state.activeJob = completedSnapshot();
    h.api.startStatusPoller();
    h.api.setLastWsEvent(h.api.now() - 10000);

    await h.api.pollJobStatus();

    assert.strictEqual(h.state.activeJob, null, 'reconciler must clear the stuck active job');
    assert.strictEqual(h.api.getTimer(), null, 'poller must stop after reconcile');
    assert.strictEqual(h.api.calls.toasts.length, 1, 'shared terminal path must run once');
    assert.strictEqual(h.api.calls.toasts[0].kind, 'success');
    assert.deepStrictEqual(h.state.selectedVideos, [{ name: 'a.mp4' }], 'reconcile must keep the queue like WS completion');
    assert.ok(h.elems.renderCompletedBanner.removed.includes('hidden'), 'completion banner must show');

    const toastCount = h.api.calls.toasts.length;
    h.api.handleJobCompleted(completedSnapshot());
    assert.strictEqual(h.api.calls.toasts.length, toastCount, 'late WS completion must not double-complete');
  });

  test('fresh WS traffic suppresses the reconcile (grace window)', async () => {
    const h = loadPollerHarness({
      fetch: () => {
        hRef.fetchCalls.push('/api/jobs/status');
        return Promise.resolve({ json: () => Promise.resolve({ active: false, job: null }) });
      }
    });
    const hRef = h;
    h.state.activeJob = completedSnapshot();
    h.api.startStatusPoller();
    await h.api.pollJobStatus();
    assert.notStrictEqual(h.state.activeJob, null, 'recent WS event must suppress the reconcile');
    assert.strictEqual(h.api.calls.toasts.length, 0, 'no terminal UI while inside the grace window');
  });

  test('no status fetch when idle (no activeJob)', async () => {
    const h = loadPollerHarness();
    h.state.activeJob = null;
    h.api.startStatusPoller();
    h.api.setLastWsEvent(h.api.now() - 60000);
    await h.api.pollJobStatus();
    assert.strictEqual(h.fetchCalls.length, 0, 'idle poller must not hit the network');
    assert.strictEqual(h.api.getTimer(), null, 'idle poller must stand down');
  });

  test('status reconciler is wired to the render lifecycle', () => {
    assert.ok(src.includes('/api/jobs/status'), 'must poll GET /api/jobs/status');
    const startSrc = extractFn('function startStatusPoller(');
    assert.ok(startSrc.includes('stopStatusPoller'), 'restart must clear the old interval');
    assert.ok(startSrc.includes('STATUS_POLL_INTERVAL_MS'), 'start must use the shared interval constant');
    const pollSrc = extractFn('async function pollJobStatus(');
    assert.ok(pollSrc.includes('/api/jobs/status'), 'poller must fetch the status endpoint');
    assert.ok(pollSrc.includes('STATUS_POLL_GRACE_MS'), 'poller must honor the grace window');
    assert.ok(pollSrc.includes('handleJobCompleted'), 'poller must reuse the shared terminal path');
  });
});


describe('1x scale UI', () => {
  test('index.html contains 1x scale button and explanatory hint', () => {
    const htmlPath = path.resolve(__dirname, '../src/video_upscaler/web/static/index.html');
    const html = fs.readFileSync(htmlPath, 'utf8');
    assert.ok(html.includes('data-scale="1"'), 'index.html must have 1x scale button');
    assert.ok(html.includes('1x keeps the original resolution'), 'index.html must contain 1x hint');
  });

  test('selecting scale sets profile to Nx_Balanced', () => {
    const appJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/app.js');
    const appJs = fs.readFileSync(appJsPath, 'utf8');
    assert.ok(appJs.includes("state.upscaleConfig.profile = `${scale}x_Balanced`"));
  });
});

describe('Recent Activity Panel UI', () => {
  const appJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/app.js');
  const appJs = fs.readFileSync(appJsPath, 'utf8');
  const htmlPath = path.resolve(__dirname, '../src/video_upscaler/web/static/index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');
  const cssPath = path.resolve(__dirname, '../src/video_upscaler/web/static/css/app.css');
  const css = fs.readFileSync(cssPath, 'utf8');

  test('index.html contains activity button, badge, and dropdown container', () => {
    assert.ok(html.includes('id="btn-activity"'), 'index.html must have btn-activity');
    assert.ok(html.includes('id="activity-badge"'), 'index.html must have activity-badge');
    assert.ok(html.includes('id="activity-panel"'), 'index.html must have activity-panel');
    assert.ok(html.includes('id="activity-list"'), 'index.html must have activity-list');
    assert.ok(html.includes('id="btn-activity-clear"'), 'index.html must have btn-activity-clear');
    const bellPos = html.indexOf('id="btn-activity"');
    const badgePos = html.indexOf('id="system-info-badge"');
    assert.ok(bellPos < badgePos, 'activity button must sit left of system-info-badge');
  });

  test('app.css contains activity dropdown and item styling', () => {
    assert.ok(css.includes('.btn-header-activity'), 'must style btn-header-activity');
    assert.ok(css.includes('.activity-badge'), 'must style activity-badge');
    assert.ok(css.includes('.activity-panel'), 'must style activity-panel');
    assert.ok(css.includes('.activity-item'), 'must style activity-item');
    assert.ok(css.includes('.activity-copy-btn'), 'must style activity-copy-btn');
  });

  test('app.js defines activity panel lifecycle functions', () => {
    assert.ok(appJs.includes('function fetchActivity('));
    assert.ok(appJs.includes('function renderActivityList('));
    assert.ok(appJs.includes('function markActivitySeen('));
    assert.ok(appJs.includes('function clearActivityHistory('));
    assert.ok(appJs.includes('function copyActivityPath('));
    assert.ok(appJs.includes('function openActivityPanel('));
    assert.ok(appJs.includes('function closeActivityPanel('));
  });

  test('copyActivityPath prefers navigator.clipboard and falls back gracefully', async () => {
    let copiedText = null;
    let toastMsg = null;
    const mockNavigator = {
      clipboard: {
        writeText: async (t) => {
          copiedText = t;
        }
      }
    };
    const showToast = (msg) => { toastMsg = msg; };

    const start = appJs.indexOf('async function copyActivityPath(');
    const end = appJs.indexOf('\n  }', start);
    const fnSrc = appJs.slice(start, end + 4);

    const copyFn = new Function('navigator', 'showToast', `
      ${fnSrc}
      return copyActivityPath;
    `)(mockNavigator, showToast);

    await copyFn('/output/test_video_1x.mp4');
    assert.strictEqual(copiedText, '/output/test_video_1x.mp4');
    assert.strictEqual(toastMsg, 'Path copied to clipboard!');
  });
});

describe('Desktop Auto Update UI & Bridge', () => {
  const appJsPath = path.resolve(__dirname, '../src/video_upscaler/web/static/js/app.js');
  const appJs = fs.readFileSync(appJsPath, 'utf8');
  const tauriConfPath = path.resolve(__dirname, '../src-tauri/tauri.conf.json');
  const tauriConf = JSON.parse(fs.readFileSync(tauriConfPath, 'utf8'));
  const capPath = path.resolve(__dirname, '../src-tauri/capabilities/local-backend.json');
  const cap = JSON.parse(fs.readFileSync(capPath, 'utf8'));
  const hooksPath = path.resolve(__dirname, '../src-tauri/hooks.nsh');
  const hooks = fs.readFileSync(hooksPath, 'utf8');

  test('tauri.conf.json configures updater endpoint, pubkey, and artifacts', () => {
    assert.strictEqual(tauriConf.bundle.createUpdaterArtifacts, true);
    assert.ok(tauriConf.plugins && tauriConf.plugins.updater, 'updater plugin must be configured');
    assert.ok(tauriConf.plugins.updater.pubkey, 'must have pubkey');
    assert.ok(tauriConf.plugins.updater.endpoints.some((e) => e.includes('Rik-rut/Clarity/releases/latest/download/latest.json')));
    assert.strictEqual(tauriConf.plugins.updater.windows.installMode, 'passive');
  });

  test('capabilities and hooks configure updater permissions and update mode', () => {
    assert.ok(cap.permissions.includes('updater:default'), 'local-backend must include updater:default');
    assert.ok(hooks.includes('$UpdateMode = 1'), 'hooks.nsh must branch on $UpdateMode');
    assert.ok(hooks.includes('clarity_trt_upd_no'), 'hooks.nsh must skip TensorRT dialog on update');
  });

  test('app.js defines update functions and check update button', () => {
    assert.ok(appJs.includes('async function checkForUpdates('));
    assert.ok(appJs.includes('async function installUpdate('));
    assert.ok(appJs.includes('function simulateUpdate('));
    assert.ok(appJs.includes('window.__simulateUpdate = simulateUpdate;'));
    assert.ok(appJs.includes("btnCheckUpdates: document.getElementById('btn-check-updates')"));
  });

  // Same extraction strategy as the suites above: app.js internals live two
  // spaces deep, each function ending at the first line that is exactly '  }'.
  // Tests exercise the REAL tauriInvoke/bridge from app.js; only the
  // environment (Tauri IPC, fetch, DOM) is faked.
  function extractFn(header) {
    const start = appJs.indexOf(header);
    assert.notStrictEqual(start, -1, 'app.js must define ' + header);
    const end = appJs.indexOf('\n  }', start);
    assert.notStrictEqual(end, -1, 'unterminated ' + header);
    return appJs.slice(start, end + 4);
  }

  function makeElement(tag) {
    const classes = [];
    const el = {
      tag,
      children: [],
      listeners: {},
      textContent: '',
      title: '',
      type: '',
      classList: {
        add: (c) => { if (!classes.includes(c)) classes.push(c); },
        remove: (c) => { const i = classes.indexOf(c); if (i >= 0) classes.splice(i, 1); },
        contains: (c) => classes.includes(c),
      },
      get className() { return classes.join(' '); },
      set className(v) {
        classes.length = 0;
        classes.push(...String(v).split(/\s+/).filter(Boolean));
      },
      get innerHTML() { return ''; },
      set innerHTML(v) { if (v === '') el.children.length = 0; },
      appendChild(child) { el.children.push(child); return child; },
      addEventListener(type, handler) { el.listeners[type] = handler; },
      contains() { return false; },
    };
    return el;
  }

  const DEFAULT_METADATA = {
    rid: 4271,
    version: '0.2.0',
    currentVersion: '0.1.2',
    date: '2026-11-08T00:00:00Z',
    body: 'Faster setup',
  };

  function loadBridge(options) {
    options = options || {};
    const state = { pendingUpdate: null, activeJob: null };
    const elems = {
      activityBadge: makeElement('span'),
      activityList: makeElement('div'),
      activityPanel: makeElement('div'),
    };
    elems.activityPanel.classList.add('hidden');

    const ipc = [];
    const toasts = [];
    const fetchCalls = [];
    const channels = [];
    const metadata = options.metadata === undefined ? DEFAULT_METADATA : options.metadata;

    function TauriChannel() {
      this.id = 'channel-' + channels.length;
      channels.push(this);
    }

    const win = options.noTauri
      ? {}
      : {
          __TAURI__: {
            core: {
              invoke: (cmd, args) => {
                ipc.push({ cmd, args });
                if (options.checkError) return Promise.reject(new Error('updater endpoint unreachable'));
                if (cmd === 'plugin:updater|check') return Promise.resolve(metadata);
                return Promise.resolve();
              },
              Channel: TauriChannel,
            },
          },
        };

    const fetchMock = async (url) => {
      fetchCalls.push(url);
      if (url === '/api/jobs/status') {
        return { ok: true, json: async () => ({ active: options.activeJob === true }) };
      }
      if (url === '/api/activity/seen') return { ok: true, json: async () => ({ ok: true }) };
      return { ok: true, json: async () => ({ unread: options.unread || 0, entries: options.entries || [] }) };
    };

    const bridgeSource = [
      extractFn('function tauriInvoke(cmd, args)'),
      extractFn('function sendWebNotification(title, body)'),
      extractFn('function sendDesktopNotification(title, body)'),
      extractFn('function formatRelativeTime('),
      extractFn('function renderActivityList('),
      extractFn('async function fetchActivity('),
      extractFn('async function markActivitySeen('),
      extractFn('async function checkForUpdates('),
      extractFn('function simulateUpdate('),
      extractFn('async function installUpdate('),
    ].join('\n\n');

    const api = new Function(
      'window',
      'Notification',
      'state',
      'elems',
      'fetch',
      'document',
      'showToast',
      bridgeSource + '\nreturn { tauriInvoke, sendDesktopNotification, checkForUpdates, installUpdate, simulateUpdate, fetchActivity, renderActivityList };'
    )(win, options.Notification, state, elems, fetchMock, { createElement: (tag) => makeElement(tag) },
      (msg, kind) => toasts.push({ msg, kind }));

    // app.js fires fetchActivity() without awaiting it; drain those promises.
    const settle = () => new Promise((resolve) => setTimeout(resolve, 5));

    return { api, state, elems, ipc, toasts, fetchCalls, channels, settle, TauriChannel };
  }

  test('checkForUpdates runs the updater command through the real IPC bridge', async () => {
    const h = loadBridge();
    await h.api.checkForUpdates(false);

    assert.ok(h.ipc.some((c) => c.cmd === 'plugin:updater|check'), 'must call plugin:updater|check');
    assert.strictEqual(h.state.pendingUpdate.version, '0.2.0', 'metadata must land in state.pendingUpdate');
    assert.strictEqual(h.state.pendingUpdate.rid, 4271);

    const notice = h.ipc.find((c) => c.cmd === 'plugin:notification|notify');
    assert.ok(notice, 'an available update must dispatch a desktop notification');
    assert.ok(notice.args.options.title.includes('Update Available'));
    assert.ok(notice.args.options.body.includes('0.2.0'));
    assert.strictEqual(h.toasts.length, 0, 'automatic checks must stay silent');

    await h.settle();
    assert.strictEqual(h.elems.activityBadge.textContent, '1', 'pending update alone counts as one unread');
  });

  test('checkForUpdates is silent off-desktop and tells manual checks', async () => {
    const h = loadBridge({ noTauri: true });
    await h.api.checkForUpdates(false);
    assert.strictEqual(h.toasts.length, 0, 'auto check outside desktop must be completely silent');
    await h.api.checkForUpdates(true);
    assert.ok(h.toasts.some((t) => t.msg.includes('only available in the desktop app')));
    assert.strictEqual(h.ipc.length, 0);
  });

  test('no newer release clears state and informs a manual check', async () => {
    const h = loadBridge({ metadata: null });
    await h.api.checkForUpdates(false);
    assert.strictEqual(h.state.pendingUpdate, null);
    assert.strictEqual(h.toasts.length, 0);
    await h.api.checkForUpdates(true);
    assert.ok(h.toasts.some((t) => t.msg.includes('up to date')));
    assert.ok(h.ipc.some((c) => c.cmd === 'plugin:updater|check'));
  });

  test('updater errors stay quiet on auto checks and surface on manual', async () => {
    const h = loadBridge({ checkError: true });
    await h.api.checkForUpdates(false);
    assert.strictEqual(h.toasts.length, 0, 'a failed automatic check must not interrupt the user');
    await h.api.checkForUpdates(true);
    const err = h.toasts.find((t) => t.kind === 'error');
    assert.ok(err && err.msg.includes('Failed to check for updates'), 'manual check must report the failure');
  });

  test('installUpdate passes the resource id and event channel', async () => {
    const h = loadBridge();
    await h.api.checkForUpdates(false);
    await h.api.installUpdate();

    const call = h.ipc.find((c) => c.cmd === 'plugin:updater|download_and_install');
    assert.ok(call, 'must call plugin:updater|download_and_install');
    assert.strictEqual(call.args.rid, 4271, 'Tauri v2 requires the numeric resource id');
    assert.ok(call.args.onEvent instanceof h.TauriChannel, 'must pass a core Channel for download events');
    assert.strictEqual(h.channels.length, 1, 'exactly one event channel per install');
  });

  test('installUpdate refuses while a render job is running', async () => {
    const h = loadBridge({ activeJob: true });
    await h.api.checkForUpdates(false);
    assert.strictEqual(h.state.pendingUpdate.version, '0.2.0');

    await h.api.installUpdate();
    assert.strictEqual(h.ipc.some((c) => c.cmd === 'plugin:updater|download_and_install'), false,
      'must not install while a job is active');
    assert.ok(h.toasts.some((t) => t.msg.includes('Please wait for the current render job')));
  });

  test('installUpdate refuses without a pending resource id', async () => {
    const h = loadBridge();
    await h.api.installUpdate();
    assert.strictEqual(h.ipc.some((c) => c.cmd === 'plugin:updater|download_and_install'), false);
    assert.ok(h.toasts.some((t) => t.msg.includes('No pending update found to install.')));
  });

  test('a simulated update drives notification, badge and activity banner offline', async () => {
    const h = loadBridge({ unread: 2 });
    const simulated = h.api.simulateUpdate('0.2.0', 'Faster setup');

    assert.strictEqual(simulated.version, '0.2.0');
    assert.strictEqual(typeof simulated.rid, 'number', 'rid must be the updater numeric resource id');
    assert.strictEqual(h.state.pendingUpdate.body, 'Faster setup');

    const notice = h.ipc.find((c) => c.cmd === 'plugin:notification|notify');
    assert.ok(notice, 'simulation must dispatch the desktop notification');
    assert.ok(notice.args.options.title.includes('Update Available'));
    assert.ok(h.toasts.some((t) => t.msg === 'Update available: v0.2.0'));

    await h.api.fetchActivity();
    assert.strictEqual(h.elems.activityBadge.textContent, '3', 'badge = unread entries + pending update');
    assert.strictEqual(h.elems.activityBadge.classList.contains('hidden'), false);

    const banner = h.elems.activityList.children.find((el) => el.className.includes('activity-update-banner'));
    assert.ok(banner, 'the update card must render in Recent Activity');
    assert.strictEqual(banner.children[0].children[0].textContent, '✨ Update Available: v0.2.0');
    assert.ok(banner.children.some((el) => el.textContent === 'Faster setup'), 'release notes must show');

    const actions = banner.children.find((el) => el.className.includes('activity-update-actions'));
    assert.ok(actions.children.some((el) => el.textContent === 'Install & Restart'));
    assert.ok(actions.children.some((el) => el.textContent === 'Later'));

    assert.ok(h.fetchCalls.every((u) => u === '/api/activity'),
      'simulation must only touch the stubbed local activity endpoint');
    assert.ok(h.fetchCalls.length >= 1);

    const installBtn = actions.children.find((el) => el.textContent === 'Install & Restart');
    await installBtn.listeners.click({ stopPropagation: () => {} });
    await h.settle();
    const call = h.ipc.find((c) => c.cmd === 'plugin:updater|download_and_install');
    assert.ok(call, 'the banner button must start the install');
    assert.strictEqual(call.args.rid, simulated.rid);

    const laterBtn = actions.children.find((el) => el.textContent === 'Later');
    await laterBtn.listeners.click({ stopPropagation: () => {} });
    await h.api.fetchActivity();
    assert.strictEqual(h.state.pendingUpdate, null, 'Later clears the pending update');
    assert.strictEqual(h.elems.activityBadge.textContent, '2', 'dismissed update drops out of the badge');
    assert.strictEqual(h.elems.activityList.children.find((el) => el.className.includes('activity-update-banner')), undefined);
  });
});
