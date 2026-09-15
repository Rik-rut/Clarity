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
    const fnSrc = extractFn('function handleJobCancelled(');
    assert.ok(fnSrc.includes('state.activeJob = null'), 'cancelled handler must clear state.activeJob');
    assert.ok(fnSrc.includes('resetRenderUI'), 'cancelled handler must reset the render UI');
    assert.ok(src.includes("data.type === 'job_cancelled'"), 'ws handler must still route job_cancelled');
    const idx = src.indexOf("data.type === 'job_cancelled'");
    const region = src.slice(idx, idx + 300);
    assert.ok(region.includes('handleJobCancelled'), 'cancelled branch must use the shared handler');
  });

  test('completed clears the queue selection (failed keeps it)', () => {
    const fnSrc = extractFn('function handleJobCompleted(');
    assert.ok(fnSrc.includes('state.selectedVideos = []'), 'completed must clear selectedVideos');
    assert.ok(fnSrc.includes('updateQueueUI'), 'completed must re-render the queue UI');
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
      'function populateMaResultWindows(files) { calls.maResults++; }'
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
    assert.deepStrictEqual(h.state.selectedVideos, [], 'reconcile must clear the queue like WS completion');
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
