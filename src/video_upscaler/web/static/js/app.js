/* Clarity Video AI — Web Studio Engine */

document.addEventListener('DOMContentLoaded', () => {
  const state = {
    activeTab: 'upscale',
    videos: [],
    selectedVideos: [],
    anchorVideoIndex: -1,
    selectedVideo: null,
    outputVideos: [],
    selectedOutputVideo: null,
    isOutputDropdownOpen: false,
    viewMode: 'split', // 'split' | 'single'
    lastActiveViewport: 'left', // 'left' | 'right'
    lastActiveViewportMa: 'input', // 'input' | 'mask' | 'greenscreen' | 'matte'
    upscaleConfig: { scale: 2, profile: '2x_Balanced' },
    slowmoConfig: { factor: 2, model_key: 'AMT-S' },
    dedupConfig: { factor: 2, model: 'gmfss', npass: 0 },
    activeJob: null,
    systemInfo: null,
    isUploading: false,
    videoToRename: null,
    renameTargetFolder: null,
    zoomState: {
      left: { scale: 1.0, panX: 0, panY: 0 },
      right: { scale: 1.0, panX: 0, panY: 0 },
      maInput: { scale: 1.0, panX: 0, panY: 0 },
      maMask: { scale: 1.0, panX: 0, panY: 0 },
      maGreenscreen: { scale: 1.0, panX: 0, panY: 0 },
      maMatte: { scale: 1.0, panX: 0, panY: 0 },
      isDragging: false,
      dragStart: { x: 0, y: 0 },
      targetViewport: null
    }
  };

  // DOM Elements
  const elems = {
    tabUpscale: document.getElementById('tab-upscale'),
    tabSlowmo: document.getElementById('tab-slowmo'),
    tabDedup: document.getElementById('tab-dedup'),
    tabMatanyone: document.getElementById('tab-matanyone'),
    panelUpscale: document.getElementById('panel-upscale'),
    panelSlowmo: document.getElementById('panel-slowmo'),
    panelDedup: document.getElementById('panel-dedup'),
    panelMatanyone: document.getElementById('panel-matanyone'),
    btnRefreshVideos: document.getElementById('btn-refresh-videos'),
    dropzone: document.getElementById('dropzone'),
    dropzoneLabel: document.getElementById('dropzone-label'),
    dropzoneSub: document.getElementById('dropzone-sub'),
    uploadProgressBarContainer: document.getElementById('upload-progress-bar-container'),
    uploadProgressBarFill: document.getElementById('upload-progress-bar-fill'),
    uploadProgressText: document.getElementById('upload-progress-text'),
    fileInput: document.getElementById('file-input'),
    videoList: document.getElementById('video-list'),
    btnClearVideos: document.getElementById('btn-clear-videos'),
    profileCards: document.getElementById('profile-cards'),
    amtModelCards: document.getElementById('amt-model-cards'),
    dedupModelCards: document.getElementById('dedup-model-cards'),
    outputCard: document.getElementById('output-card'),
    btnBrowseOutputDir: document.getElementById('btn-browse-output-dir'),
    btnToggleOutputDropdown: document.getElementById('btn-toggle-output-dropdown'),
    outputVideoListContainer: document.getElementById('output-video-list-container'),
    outputVideoList: document.getElementById('output-video-list'),
    btnRefreshOutputVideos: document.getElementById('btn-refresh-output-videos'),
    btnClearOutputVideos: document.getElementById('btn-clear-output-videos'),
    outputDirLabel: document.getElementById('output-dir-label'),
    outputDirInput: document.getElementById('output-dir-input'),
    btnToggleQueue: document.getElementById('btn-toggle-queue'),
    queueCountBadge: document.getElementById('queue-count-badge'),
    queueDrawerOverlay: document.getElementById('queue-drawer-overlay'),
    queueItemList: document.getElementById('queue-item-list'),
    queueStatusText: document.getElementById('queue-status-text'),
    btnQueueRender: document.getElementById('btn-queue-render'),
    btnCloseQueue: document.getElementById('btn-close-queue'),
    btnRender: document.getElementById('btn-render'),
    btnCancel: document.getElementById('btn-cancel'),
    btnViewSplit: document.getElementById('btn-view-split'),
    btnViewSingle: document.getElementById('btn-view-single'),
    playerStage: document.getElementById('player-stage'),
    playerStageMa2: document.getElementById('player-stage-ma2'),
    viewportLeft: document.getElementById('viewport-left'),
    viewportRight: document.getElementById('viewport-right'),
    wrapperLeft: document.getElementById('wrapper-left'),
    wrapperRight: document.getElementById('wrapper-right'),
    videoLeft: document.getElementById('video-left'),
    videoRight: document.getElementById('video-right'),
    placeholderLeft: document.getElementById('placeholder-left'),
    placeholderRight: document.getElementById('placeholder-right'),
    viewportMaInput: document.getElementById('viewport-ma-input'),
    viewportMaMask: document.getElementById('viewport-ma-mask'),
    viewportMaGreenscreen: document.getElementById('viewport-ma-greenscreen'),
    viewportMaMatte: document.getElementById('viewport-ma-matte'),
    wrapperMaInput: document.getElementById('wrapper-ma-input'),
    wrapperMaGreenscreen: document.getElementById('wrapper-ma-greenscreen'),
    wrapperMaMatte: document.getElementById('wrapper-ma-matte'),
    maMaskStage: document.getElementById('ma-mask-stage'),
    videoMaInput: document.getElementById('video-ma-input'),
    placeholderMaInput: document.getElementById('placeholder-ma-input'),
    maInputMetaBadge: document.getElementById('ma-input-meta-badge'),
    placeholderMaMask: document.getElementById('placeholder-ma-mask'),
    videoMaGreenscreen: document.getElementById('video-ma-greenscreen'),
    placeholderMaGreenscreen: document.getElementById('placeholder-ma-greenscreen'),
    maGsMetaBadge: document.getElementById('ma-gs-meta-badge'),
    videoMaMatte: document.getElementById('video-ma-matte'),
    placeholderMaMatte: document.getElementById('placeholder-ma-matte'),
    maMatteMetaBadge: document.getElementById('ma-matte-meta-badge'),
    btnStageBrowse: document.getElementById('btn-stage-browse'),
    beforeMetaBadge: document.getElementById('before-meta-badge'),
    afterMetaBadge: document.getElementById('after-meta-badge'),
    zoomBadge: document.getElementById('zoom-badge'),
    btnZoomOut: document.getElementById('btn-zoom-out'),
    btnZoomIn: document.getElementById('btn-zoom-in'),
    btnResetView: document.getElementById('btn-reset-view'),
    btnResetApp: document.getElementById('btn-reset-app'),
    transportBar: document.getElementById('transport-bar'),
    transportUnified: document.getElementById('transport-unified'),
    transportDual: document.getElementById('transport-dual'),
    btnPlayPause: document.getElementById('btn-play-pause'),
    playIcon: document.getElementById('play-icon'),
    btnStepBack: document.getElementById('btn-step-back'),
    btnStepFwd: document.getElementById('btn-step-fwd'),
    timecodeCurrent: document.getElementById('timecode-current'),
    timecodeTotal: document.getElementById('timecode-total'),
    timelineScrubber: document.getElementById('timeline-scrubber'),
    btnLoop: document.getElementById('btn-loop'),
    btnMute: document.getElementById('btn-mute'),
    muteIcon: document.getElementById('mute-icon'),
    btnPlayLeft: document.getElementById('btn-play-left'),
    scrubberLeft: document.getElementById('scrubber-left'),
    timecodeLeft: document.getElementById('timecode-left'),
    btnPlayRight: document.getElementById('btn-play-right'),
    scrubberRight: document.getElementById('scrubber-right'),
    timecodeRight: document.getElementById('timecode-right'),
    renderProgressCard: document.getElementById('render-progress-card'),
    progressStage: document.getElementById('progress-stage'),
    progressPercent: document.getElementById('progress-percent'),
    progressBarFill: document.getElementById('progress-bar-fill'),
    progressFileInfo: document.getElementById('progress-file-info'),
    progressElapsed: document.getElementById('progress-elapsed'),
    progressEta: document.getElementById('progress-eta'),
    renderCompletedBanner: document.getElementById('render-completed-banner'),
    renderTotalTime: document.getElementById('render-total-time'),
    btnCloseBanner: document.getElementById('btn-close-banner'),
    backendDeviceText: document.getElementById('backend-device-text'),
    dragOverlay: document.getElementById('drag-overlay'),
    toastContainer: document.getElementById('toast-container'),
    renameModalOverlay: document.getElementById('rename-modal-overlay'),
    renameInput: document.getElementById('rename-input'),
    btnCloseRename: document.getElementById('btn-close-rename'),
    btnCancelRename: document.getElementById('btn-cancel-rename'),
    btnConfirmRename: document.getElementById('btn-confirm-rename')
  };

  // Lightweight Toast System
  function showToast(message, type = 'info') {
    if (!elems.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    elems.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, 4000);
  }

  // 1. Fetch System Info & Scanned Videos
  async function loadSystemInfo() {
    try {
      const resp = await fetch('/api/system/info');
      if (resp.ok) {
        state.systemInfo = await resp.json();
        if (elems.backendDeviceText && state.systemInfo.backend_label) {
          elems.backendDeviceText.textContent = state.systemInfo.backend_label;
        }
        renderProfiles();
        renderAmtModels();
        renderDedupModels();
      }
    } catch (e) {
      console.warn('System info load warning:', e);
    }
  }

  async function loadVideos() {
    try {
      const resp = await fetch('/api/videos/scanned');
      if (resp.ok) {
        const data = await resp.json();
        state.videos = data.videos || [];
        
        // Re-validate selection
        if (state.selectedVideos.length > 0) {
          state.selectedVideos = state.videos.filter(v => state.selectedVideos.some(sv => sv.name === v.name));
        }
        if (state.selectedVideos.length === 0 && state.videos.length > 0) {
          state.selectedVideos = [state.videos[0]];
          state.anchorVideoIndex = 0;
        }
        if (state.selectedVideos.length > 0) {
          state.selectedVideo = state.selectedVideos[0];
          previewVideo(state.selectedVideo);
        } else {
          state.selectedVideo = null;
          clearPreview();
        }

        renderVideoList();
        updateRenderButtonLabel();
        updateQueueUI();
        updateMaRenderButton();
      }
    } catch (e) {
      console.warn('Video scan load warning:', e);
    }
  }

  function closeAllContextMenus() {
    document.querySelectorAll('.video-context-menu').forEach(m => m.classList.add('hidden'));
  }

  // 2. Render Input Video List with Multi-Select & Card-Anchored 3-Dots Menu
  function renderVideoList() {
    if (!elems.videoList) return;
    if (state.videos.length === 0) {
      elems.videoList.innerHTML = '<div style="font-size: 0.75rem; color: var(--text-muted); padding: 0.75rem 0; text-align: center;">No videos in input/ folder.<br/>Drag & drop or browse above.</div>';
      return;
    }

    elems.videoList.innerHTML = state.videos.map((vid, idx) => {
      const isSelected = state.selectedVideos.some(v => v.name === vid.name);
      const metaInfo = (vid.width > 0 && vid.height > 0)
        ? `${vid.width}x${vid.height} • ${vid.fps}fps • ${vid.size_formatted}`
        : `${vid.size_formatted}`;

      return `
        <div class="video-card-wrapper" data-index="${idx}">
          <div data-name="${vid.name}" class="config-card flex items-center justify-between ${isSelected ? 'active' : ''}" style="padding: 0.45rem 0.6rem; user-select: none;">
            <div class="flex items-center gap-2" style="min-width: 0; pointer-events: none;">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="${isSelected ? '#6EA4BF' : '#94a3b8'}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m10 15 5-3-5-3v6Z"/></svg>
              <div class="truncate">
                <div class="text-xs font-semibold truncate" style="color: ${isSelected ? '#6EA4BF' : '#f1f5f9'}; font-size: 0.75rem;">${vid.name}</div>
                <div class="text-xs" style="color: var(--text-muted); font-size: 0.68rem; margin-top: 1px;">
                  ${metaInfo}
                </div>
              </div>
            </div>
            <button class="btn-dots" type="button" data-dots-index="${idx}" title="Options">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="5" r="1.5"/><circle cx="12" cy="19" r="1.5"/></svg>
            </button>
          </div>
          <div id="context-menu-${idx}" class="video-context-menu hidden">
            <button class="context-menu-item" type="button" data-action="rename" data-index="${idx}">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>
              <span>Rename</span>
            </button>
            <button class="context-menu-item danger" type="button" data-action="delete" data-index="${idx}">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/></svg>
              <span>Delete</span>
            </button>
          </div>
        </div>
      `;
    }).join('');

    elems.videoList.querySelectorAll('.video-card-wrapper').forEach(wrapper => {
      const idx = parseInt(wrapper.getAttribute('data-index'), 10);
      const card = wrapper.querySelector('.config-card');
      const dotsBtn = wrapper.querySelector('.btn-dots');
      const menu = wrapper.querySelector('.video-context-menu');

      card.addEventListener('click', (ev) => {
        if (ev.target.closest('.btn-dots') || ev.target.closest('.video-context-menu')) return;
        closeAllContextMenus();
        handleVideoItemClick(idx, ev);
      });

      dotsBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const wasHidden = menu.classList.contains('hidden');
        closeAllContextMenus();
        if (wasHidden) {
          menu.classList.remove('hidden');
        }
      });

      menu.querySelectorAll('.context-menu-item').forEach(itemBtn => {
        itemBtn.addEventListener('click', (ev) => {
          ev.stopPropagation();
          closeAllContextMenus();
          const action = itemBtn.getAttribute('data-action');
          const targetVid = state.videos[idx];
          if (action === 'rename') {
            openRenameModal(targetVid, null);
          } else if (action === 'delete') {
            handleDeleteVideo(targetVid, null);
          }
        });
      });
    });
  }

  function handleVideoItemClick(index, event) {
    if (index < 0 || index >= state.videos.length) return;
    const clickedVid = state.videos[index];

    if (event.shiftKey && state.anchorVideoIndex >= 0) {
      // Range selection
      const start = Math.min(state.anchorVideoIndex, index);
      const end = Math.max(state.anchorVideoIndex, index);
      state.selectedVideos = state.videos.slice(start, end + 1);
    } else if (event.ctrlKey || event.metaKey) {
      // Toggle selection
      const existsIdx = state.selectedVideos.findIndex(v => v.name === clickedVid.name);
      if (existsIdx >= 0) {
        state.selectedVideos.splice(existsIdx, 1);
      } else {
        state.selectedVideos.push(clickedVid);
      }
      state.anchorVideoIndex = index;
    } else {
      // Single selection
      state.selectedVideos = [clickedVid];
      state.anchorVideoIndex = index;
    }

    if (state.selectedVideos.length > 0) {
      state.selectedVideo = state.selectedVideos[0];
      previewVideo(state.selectedVideo);
    } else {
      state.selectedVideo = null;
      clearPreview();
    }

    renderVideoList();
    updateRenderButtonLabel();
    updateQueueUI();
    updateMaRenderButton();
  }

  // 3. Render Output Video List with 3-Dots Menu & AFTER Viewport Loading
  async function loadOutputVideos(autoPreviewFirst = false) {
    try {
      const folder = (elems.outputDirInput && elems.outputDirInput.value) || 'output';
      const resp = await fetch('/api/videos/scanned?folder=' + encodeURIComponent(folder));
      if (resp.ok) {
        const data = await resp.json();
        state.outputVideos = data.videos || [];
        renderOutputVideoList();

        if (autoPreviewFirst && state.outputVideos.length > 0) {
          previewOutputVideo(state.outputVideos[0]);
        }
      }
    } catch (e) {
      console.warn('Output videos scan warning:', e);
    }
  }

  function renderOutputVideoList() {
    if (!elems.outputVideoList) return;
    if (state.outputVideos.length === 0) {
      elems.outputVideoList.innerHTML = '<div style="font-size: 0.725rem; color: var(--text-muted); padding: 0.75rem 0; text-align: center;">No rendered videos in output folder.</div>';
      return;
    }

    elems.outputVideoList.innerHTML = state.outputVideos.map((vid, idx) => {
      const isSelected = state.selectedOutputVideo && state.selectedOutputVideo.name === vid.name;
      const metaInfo = (vid.width > 0 && vid.height > 0)
        ? `${vid.width}x${vid.height} • ${vid.fps}fps • ${vid.size_formatted}`
        : `${vid.size_formatted}`;

      return `
        <div class="output-video-card-wrapper video-card-wrapper" data-output-index="${idx}">
          <div data-name="${vid.name}" class="config-card flex items-center justify-between ${isSelected ? 'active' : ''}" style="padding: 0.45rem 0.6rem; user-select: none;">
            <div class="flex items-center gap-2" style="min-width: 0; pointer-events: none;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="${isSelected ? '#6EA4BF' : '#10b981'}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;"><polygon points="23 7 16 12 23 17 23 7"/><rect width="15" height="14" x="1" y="5" rx="2" ry="2"/></svg>
              <div class="truncate">
                <div class="text-xs font-semibold truncate" style="color: ${isSelected ? '#6EA4BF' : '#f1f5f9'}; font-size: 0.75rem;">${vid.name}</div>
                <div class="text-xs" style="color: var(--text-muted); font-size: 0.68rem; margin-top: 1px;">
                  ${metaInfo}
                </div>
              </div>
            </div>
            <button class="btn-dots btn-output-dots" type="button" data-output-dots="${idx}" title="Options">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1.5"/><circle cx="12" cy="5" r="1.5"/><circle cx="12" cy="19" r="1.5"/></svg>
            </button>
          </div>
          <div id="output-context-menu-${idx}" class="video-context-menu hidden">
            <button class="context-menu-item" type="button" data-action="rename-output" data-index="${idx}">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>
              <span>Rename</span>
            </button>
            <button class="context-menu-item danger" type="button" data-action="delete-output" data-index="${idx}">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/></svg>
              <span>Delete</span>
            </button>
          </div>
        </div>
      `;
    }).join('');

    elems.outputVideoList.querySelectorAll('.output-video-card-wrapper').forEach(wrapper => {
      const idx = parseInt(wrapper.getAttribute('data-output-index'), 10);
      const card = wrapper.querySelector('.config-card');
      const dotsBtn = wrapper.querySelector('.btn-output-dots');
      const menu = wrapper.querySelector('.video-context-menu');

      card.addEventListener('click', (ev) => {
        if (ev.target.closest('.btn-dots') || ev.target.closest('.video-context-menu')) return;
        closeAllContextMenus();
        const vid = state.outputVideos[idx];
        state.selectedOutputVideo = vid;
        previewOutputVideo(vid);
        renderOutputVideoList();
      });

      dotsBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        const wasHidden = menu.classList.contains('hidden');
        closeAllContextMenus();
        if (wasHidden) {
          menu.classList.remove('hidden');
        }
      });

      menu.querySelectorAll('.context-menu-item').forEach(itemBtn => {
        itemBtn.addEventListener('click', (ev) => {
          ev.stopPropagation();
          closeAllContextMenus();
          const action = itemBtn.getAttribute('data-action');
          const targetVid = state.outputVideos[idx];
          const folder = (elems.outputDirInput && elems.outputDirInput.value) || 'output';
          if (action === 'rename-output') {
            openRenameModal(targetVid, folder);
          } else if (action === 'delete-output') {
            handleDeleteVideo(targetVid, folder);
          }
        });
      });
    });
  }

  document.addEventListener('click', (ev) => {
    if (!ev.target.closest('.video-card-wrapper')) {
      closeAllContextMenus();
    }
  });

  function previewOutputVideo(vid) {
    if (!vid || !elems.videoRight) return;
    elems.videoRight.src = `/api/stream/video?path=${encodeURIComponent(vid.path)}`;
    elems.videoRight.load();
    if (elems.placeholderRight) elems.placeholderRight.classList.add('hidden');
    if (elems.afterMetaBadge) {
      if (vid.width > 0 && vid.height > 0) {
        elems.afterMetaBadge.textContent = `${vid.width}x${vid.height} • ${vid.fps}fps`;
      } else {
        elems.afterMetaBadge.textContent = vid.name;
      }
    }
  }

  // 4. Output Destination Accordion & Folder Chooser
  if (elems.btnBrowseOutputDir) {
    elems.btnBrowseOutputDir.addEventListener('click', async () => {
      try {
        const initial = (elems.outputDirInput && elems.outputDirInput.value) || 'output';
        const resp = await fetch('/api/directories/browse', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ initial_dir: initial })
        });
        const data = await resp.json();
        if (data.success && data.path) {
          if (elems.outputDirInput) elems.outputDirInput.value = data.path;
          if (elems.outputDirLabel) elems.outputDirLabel.textContent = data.path;
          showToast(`Output set: ${data.path}`, 'success');
          if (state.isOutputDropdownOpen) {
            await loadOutputVideos(true);
          }
        }
      } catch (e) {
        console.warn('Browse directory error:', e);
      }
    });
  }

  if (elems.btnToggleOutputDropdown) {
    elems.btnToggleOutputDropdown.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      state.isOutputDropdownOpen = !state.isOutputDropdownOpen;
      if (elems.outputVideoListContainer) {
        elems.outputVideoListContainer.classList.toggle('hidden', !state.isOutputDropdownOpen);
      }
      elems.btnToggleOutputDropdown.classList.toggle('open', state.isOutputDropdownOpen);

      if (state.isOutputDropdownOpen) {
        await loadOutputVideos(true);
      }
    });
  }

  if (elems.btnRefreshOutputVideos) {
    elems.btnRefreshOutputVideos.addEventListener('click', () => loadOutputVideos(false));
  }

  if (elems.btnClearOutputVideos) {
    elems.btnClearOutputVideos.addEventListener('click', async () => {
      if (state.outputVideos.length === 0) {
        showToast('No output videos to clear', 'info');
        return;
      }
      const folder = (elems.outputDirInput && elems.outputDirInput.value) || 'output';
      if (!confirm(`Are you sure you want to delete all ${state.outputVideos.length} videos from the output folder (${folder})?`)) return;
      try {
        const resp = await fetch('/api/videos/clear', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder: folder })
        });
        const data = await resp.json();
        if (resp.ok && data.success) {
          showToast(`Cleared ${data.deleted_count} output video(s)`, 'success');
          state.selectedOutputVideo = null;
          if (elems.videoRight) {
            elems.videoRight.pause();
            elems.videoRight.removeAttribute('src');
            elems.videoRight.load();
          }
          if (elems.placeholderRight) elems.placeholderRight.classList.remove('hidden');
          if (elems.afterMetaBadge) elems.afterMetaBadge.textContent = 'Enhanced';
          await loadOutputVideos(false);
        } else {
          showToast(`Clear failed: ${data.detail || 'Error'}`, 'error');
        }
      } catch (e) {
        showToast(`Clear error: ${e}`, 'error');
      }
    });
  }

  // 5. Rename Video Modal (Supports Input & Output Folders)
  function openRenameModal(vid, folder) {
    state.videoToRename = vid;
    state.renameTargetFolder = folder;
    if (elems.renameInput) {
      elems.renameInput.value = vid.name;
    }
    if (elems.renameModalOverlay) {
      elems.renameModalOverlay.classList.remove('hidden');
      setTimeout(() => elems.renameInput && elems.renameInput.focus(), 50);
    }
  }

  function closeRenameModal() {
    state.videoToRename = null;
    state.renameTargetFolder = null;
    if (elems.renameModalOverlay) elems.renameModalOverlay.classList.add('hidden');
  }

  if (elems.btnCloseRename) elems.btnCloseRename.addEventListener('click', closeRenameModal);
  if (elems.btnCancelRename) elems.btnCancelRename.addEventListener('click', closeRenameModal);

  async function submitRename() {
    if (!state.videoToRename || !elems.renameInput) return;
    const newName = elems.renameInput.value.trim();
    if (!newName) {
      showToast('Filename cannot be empty', 'error');
      return;
    }
    try {
      const resp = await fetch('/api/videos/rename', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          old_name: state.videoToRename.name,
          new_name: newName,
          folder: state.renameTargetFolder
        })
      });
      const data = await resp.json();
      if (resp.ok && data.success) {
        showToast(`Renamed to: ${data.video.name}`, 'success');
        closeRenameModal();
        if (state.renameTargetFolder) {
          await loadOutputVideos(false);
        } else {
          await loadVideos();
        }
      } else {
        showToast(`Rename failed: ${data.detail || 'Error'}`, 'error');
      }
    } catch (e) {
      showToast(`Rename error: ${e}`, 'error');
    }
  }

  if (elems.btnConfirmRename) elems.btnConfirmRename.addEventListener('click', submitRename);
  if (elems.renameInput) {
    elems.renameInput.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') submitRename();
      if (ev.key === 'Escape') closeRenameModal();
    });
  }

  // 6. Delete Video (Supports Input & Output Folders)
  async function handleDeleteVideo(vid, folder) {
    const loc = folder ? 'output folder' : 'input folder';
    if (!confirm(`Delete "${vid.name}" from ${loc}?`)) return;
    try {
      const resp = await fetch('/api/videos/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_name: vid.name,
          folder: folder
        })
      });
      const data = await resp.json();
      if (resp.ok && data.success) {
        showToast(`Deleted ${vid.name}`, 'info');
        if (folder) {
          if (state.selectedOutputVideo && state.selectedOutputVideo.name === vid.name) {
            state.selectedOutputVideo = null;
            if (elems.videoRight) {
              elems.videoRight.pause();
              elems.videoRight.removeAttribute('src');
              elems.videoRight.load();
            }
            if (elems.placeholderRight) elems.placeholderRight.classList.remove('hidden');
          }
          await loadOutputVideos(false);
        } else {
          await loadVideos();
        }
      } else {
        showToast(`Delete failed: ${data.detail || 'Error'}`, 'error');
      }
    } catch (e) {
      showToast(`Delete error: ${e}`, 'error');
    }
  }

  if (elems.btnClearVideos) {
    elems.btnClearVideos.addEventListener('click', async () => {
      if (state.videos.length === 0) {
        showToast('No videos to clear', 'info');
        return;
      }
      if (!confirm(`Are you sure you want to delete all ${state.videos.length} videos from the input folder?`)) return;
      try {
        const resp = await fetch('/api/videos/clear', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        const data = await resp.json();
        if (resp.ok && data.success) {
          showToast(`Cleared ${data.deleted_count} video(s)`, 'success');
          await loadVideos();
        } else {
          showToast(`Clear failed: ${data.detail || 'Error'}`, 'error');
        }
      } catch (e) {
        showToast(`Clear error: ${e}`, 'error');
      }
    });
  }

  // 7. Queue Drawer & Telemetry
  function updateQueueUI() {
    const count = state.selectedVideos.length;
    if (elems.queueCountBadge) {
      elems.queueCountBadge.textContent = `${count} Items`;
    }
    if (elems.queueStatusText) {
      elems.queueStatusText.textContent = `${count} video${count === 1 ? '' : 's'} selected for render`;
    }
    if (elems.queueItemList) {
      if (count === 0) {
        elems.queueItemList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.75rem; text-align: center; padding: 2rem 0;">No videos queued. Select videos from the list.</div>';
        return;
      }
      elems.queueItemList.innerHTML = state.selectedVideos.map((vid, i) => {
        const metaInfo = (vid.width > 0 && vid.height > 0)
          ? `${vid.width}x${vid.height} • ${vid.size_formatted}`
          : `${vid.size_formatted}`;
        return `
          <div class="config-card flex items-center justify-between" style="padding: 0.6rem 0.8rem;">
            <div class="flex items-center gap-2" style="min-width: 0;">
              <span class="font-mono text-xs" style="color: var(--accent); font-weight: 700;">#${i + 1}</span>
              <div class="truncate">
                <div class="text-xs font-semibold truncate" style="color: #f1f5f9;">${vid.name}</div>
                <div class="text-xs" style="color: var(--text-muted); font-size: 0.7rem;">${metaInfo}</div>
              </div>
            </div>
            <span class="text-xs font-mono" style="background: var(--bg-surface-3); padding: 0.15rem 0.4rem; border-radius: 0.25rem; color: #cbd5e1;">Ready</span>
          </div>
        `;
      }).join('');
    }
  }

  if (elems.btnToggleQueue) {
    elems.btnToggleQueue.addEventListener('click', () => {
      updateQueueUI();
      if (elems.queueDrawerOverlay) elems.queueDrawerOverlay.classList.remove('hidden');
    });
  }

  if (elems.btnCloseQueue) {
    elems.btnCloseQueue.addEventListener('click', () => {
      if (elems.queueDrawerOverlay) elems.queueDrawerOverlay.classList.add('hidden');
    });
  }

  if (elems.queueDrawerOverlay) {
    elems.queueDrawerOverlay.addEventListener('click', (ev) => {
      if (ev.target === elems.queueDrawerOverlay) {
        elems.queueDrawerOverlay.classList.add('hidden');
      }
    });
  }

  if (elems.btnQueueRender) {
    elems.btnQueueRender.addEventListener('click', () => {
      if (elems.queueDrawerOverlay) elems.queueDrawerOverlay.classList.add('hidden');
      if (elems.btnRender) elems.btnRender.click();
    });
  }

  // 8. 1-Window vs 2-Window / 4-Window Mode Toggle & Corner Highlights
  function setActiveWindow(target) {
    state.lastActiveViewport = target;
    if (elems.viewportLeft) {
      if (target === 'left') elems.viewportLeft.classList.add('active-window');
      else elems.viewportLeft.classList.remove('active-window');
    }
    if (elems.viewportRight) {
      if (target === 'right') elems.viewportRight.classList.add('active-window');
      else elems.viewportRight.classList.remove('active-window');
    }

    if (state.viewMode === 'single') {
      applySingleView();
    }
  }

  function setActiveMaWindow(target) {
    state.lastActiveViewportMa = target;
    const maWindows = {
      input: elems.viewportMaInput,
      mask: elems.viewportMaMask,
      greenscreen: elems.viewportMaGreenscreen,
      matte: elems.viewportMaMatte
    };
    Object.entries(maWindows).forEach(([key, el]) => {
      if (!el) return;
      if (key === target) el.classList.add('active-window');
      else el.classList.remove('active-window');
    });

    if (state.viewMode === 'single') {
      applySingleView();
    }
  }

  function applySingleView() {
    if (state.activeTab === 'matanyone') {
      if (!elems.playerStageMa2) return;
      elems.playerStageMa2.classList.add('single-view');
      const maWindows = {
        input: elems.viewportMaInput,
        mask: elems.viewportMaMask,
        greenscreen: elems.viewportMaGreenscreen,
        matte: elems.viewportMaMatte
      };
      const activeTarget = state.lastActiveViewportMa || 'input';
      Object.entries(maWindows).forEach(([key, el]) => {
        if (!el) return;
        if (key === activeTarget) {
          el.classList.remove('inactive-view');
        } else {
          el.classList.add('inactive-view');
        }
      });
    } else {
      if (!elems.playerStage) return;
      elems.playerStage.classList.add('single-view');
      if (state.lastActiveViewport === 'left') {
        if (elems.viewportLeft) elems.viewportLeft.classList.remove('inactive-view');
        if (elems.viewportRight) elems.viewportRight.classList.add('inactive-view');
      } else {
        if (elems.viewportRight) elems.viewportRight.classList.remove('inactive-view');
        if (elems.viewportLeft) elems.viewportLeft.classList.add('inactive-view');
      }
    }
  }

  function applySplitView() {
    if (state.activeTab === 'matanyone') {
      if (!elems.playerStageMa2) return;
      elems.playerStageMa2.classList.remove('single-view');
      [elems.viewportMaInput, elems.viewportMaMask, elems.viewportMaGreenscreen, elems.viewportMaMatte].forEach(el => {
        if (el) el.classList.remove('inactive-view');
      });
    } else {
      if (!elems.playerStage) return;
      elems.playerStage.classList.remove('single-view');
      if (elems.viewportLeft) elems.viewportLeft.classList.remove('inactive-view');
      if (elems.viewportRight) elems.viewportRight.classList.remove('inactive-view');
    }
  }

  if (elems.btnViewSplit) {
    elems.btnViewSplit.addEventListener('click', () => {
      state.viewMode = 'split';
      elems.btnViewSplit.classList.add('active');
      if (elems.btnViewSingle) elems.btnViewSingle.classList.remove('active');
      applySplitView();
    });
  }

  if (elems.btnViewSingle) {
    elems.btnViewSingle.addEventListener('click', () => {
      state.viewMode = 'single';
      elems.btnViewSingle.classList.add('active');
      if (elems.btnViewSplit) elems.btnViewSplit.classList.remove('active');
      applySingleView();
    });
  }

  if (elems.viewportLeft) {
    elems.viewportLeft.addEventListener('click', () => setActiveWindow('left'));
  }
  if (elems.viewportRight) {
    elems.viewportRight.addEventListener('click', () => setActiveWindow('right'));
  }

  if (elems.viewportMaInput) {
    elems.viewportMaInput.addEventListener('click', () => setActiveMaWindow('input'));
  }
  if (elems.viewportMaMask) {
    elems.viewportMaMask.addEventListener('click', () => setActiveMaWindow('mask'));
  }
  if (elems.viewportMaGreenscreen) {
    elems.viewportMaGreenscreen.addEventListener('click', () => setActiveMaWindow('greenscreen'));
  }
  if (elems.viewportMaMatte) {
    elems.viewportMaMatte.addEventListener('click', () => setActiveMaWindow('matte'));
  }

  // 9. Render Profiles, AMT Models & Dedup Models
  function renderProfiles() {
    if (!elems.profileCards || !state.systemInfo || !state.systemInfo.profiles) return;
    const currentScale = state.upscaleConfig.scale;
    const filtered = state.systemInfo.profiles.filter(p => p.scale === currentScale);

    elems.profileCards.innerHTML = filtered.map(p => {
      const isActive = state.upscaleConfig.profile === p.name;
      const shortName = p.name.replace(/^\d+x_/, '');
      return `
        <div data-profile="${p.name}" class="config-card ${isActive ? 'active' : ''}">
          <div class="flex items-center justify-between">
            <span class="font-semibold text-sm">${shortName}</span>
            <span class="text-xs" style="color: ${isActive ? 'var(--accent)' : 'var(--text-muted)'};">${p.model_file}</span>
          </div>
          <p class="text-xs" style="color: var(--text-muted); margin-top: 0.25rem;">${p.description}</p>
        </div>
      `;
    }).join('');

    elems.profileCards.querySelectorAll('.config-card').forEach(el => {
      el.addEventListener('click', () => {
        state.upscaleConfig.profile = el.getAttribute('data-profile');
        renderProfiles();
      });
    });
  }

  function renderAmtModels() {
    if (!elems.amtModelCards || !state.systemInfo || !state.systemInfo.slow_mo_models) return;
    elems.amtModelCards.innerHTML = state.systemInfo.slow_mo_models.map(k => {
      const isActive = state.slowmoConfig.model_key === k.key;
      return `
        <div data-key="${k.key}" class="config-card ${isActive ? 'active' : ''}">
          <div class="font-semibold text-sm">${k.key}</div>
          <div class="text-xs" style="color: var(--text-muted); margin-top: 0.25rem;">${k.description}</div>
        </div>
      `;
    }).join('');

    elems.amtModelCards.querySelectorAll('.config-card').forEach(el => {
      el.addEventListener('click', () => {
        state.slowmoConfig.model_key = el.getAttribute('data-key');
        renderAmtModels();
        updateTargetInfo();
      });
    });
  }

  function renderDedupModels() {
    if (!elems.dedupModelCards || !state.systemInfo || !state.systemInfo.dedup_models) return;
    elems.dedupModelCards.innerHTML = state.systemInfo.dedup_models.map(m => {
      const isActive = state.dedupConfig.model === m.key;
      return `
        <div data-key="${m.key}" class="config-card ${isActive ? 'active' : ''}">
          <div class="font-semibold text-sm">${m.name}</div>
          <div class="text-xs" style="color: var(--text-muted); margin-top: 0.25rem;">${m.desc}</div>
        </div>
      `;
    }).join('');

    elems.dedupModelCards.querySelectorAll('.config-card').forEach(el => {
      el.addEventListener('click', () => {
        state.dedupConfig.model = el.getAttribute('data-key');
        renderDedupModels();
        updateTargetInfo();
      });
    });
  }

  // 10. Tab Navigation
  function setStageForTab(tabName) {
    const ma2 = tabName === 'matanyone';
    if (elems.playerStage) elems.playerStage.classList.toggle('hidden', ma2);
    if (elems.playerStageMa2) elems.playerStageMa2.classList.toggle('hidden', !ma2);
    if (elems.transportBar) elems.transportBar.classList.remove('hidden');

    if (elems.btnViewSplit) {
      if (ma2) {
        elems.btnViewSplit.title = 'Grid View (4 Windows)';
        elems.btnViewSplit.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><line x1="12" x2="12" y1="3" y2="21"/><line x1="3" x2="21" y1="12" y2="12"/></svg>';
      } else {
        elems.btnViewSplit.title = 'Dual Split View (2 Windows)';
        elems.btnViewSplit.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><line x1="12" x2="12" y1="3" y2="21"/></svg>';
      }
    }

    if (state.viewMode === 'single') {
      applySingleView();
    } else {
      applySplitView();
    }
  }

  function setTab(tabName) {
    state.activeTab = tabName;
    setStageForTab(tabName);

    [elems.tabUpscale, elems.tabSlowmo, elems.tabDedup, elems.tabMatanyone].forEach(btn => {
      if (btn) btn.classList.remove('active');
    });

    if (tabName === 'upscale') {
      if (elems.tabUpscale) elems.tabUpscale.classList.add('active');
      if (elems.panelUpscale) elems.panelUpscale.classList.remove('hidden');
      if (elems.panelSlowmo) elems.panelSlowmo.classList.add('hidden');
      if (elems.panelDedup) elems.panelDedup.classList.add('hidden');
      if (elems.panelMatanyone) elems.panelMatanyone.classList.add('hidden');
      if (elems.transportUnified) elems.transportUnified.classList.remove('hidden');
      if (elems.transportDual) elems.transportDual.classList.add('hidden');
    } else if (tabName === 'slowmo') {
      if (elems.tabSlowmo) elems.tabSlowmo.classList.add('active');
      if (elems.panelSlowmo) elems.panelSlowmo.classList.remove('hidden');
      if (elems.panelUpscale) elems.panelUpscale.classList.add('hidden');
      if (elems.panelDedup) elems.panelDedup.classList.add('hidden');
      if (elems.panelMatanyone) elems.panelMatanyone.classList.add('hidden');
      if (elems.transportUnified) elems.transportUnified.classList.add('hidden');
      if (elems.transportDual) elems.transportDual.classList.remove('hidden');
    } else if (tabName === 'dedup') {
      if (elems.tabDedup) elems.tabDedup.classList.add('active');
      if (elems.panelDedup) elems.panelDedup.classList.remove('hidden');
      if (elems.panelUpscale) elems.panelUpscale.classList.add('hidden');
      if (elems.panelSlowmo) elems.panelSlowmo.classList.add('hidden');
      if (elems.panelMatanyone) elems.panelMatanyone.classList.add('hidden');
      if (elems.transportUnified) elems.transportUnified.classList.add('hidden');
      if (elems.transportDual) elems.transportDual.classList.remove('hidden');
    } else if (tabName === 'matanyone') {
      if (elems.tabMatanyone) elems.tabMatanyone.classList.add('active');
      if (elems.panelMatanyone) elems.panelMatanyone.classList.remove('hidden');
      if (elems.panelUpscale) elems.panelUpscale.classList.add('hidden');
      if (elems.panelSlowmo) elems.panelSlowmo.classList.add('hidden');
      if (elems.panelDedup) elems.panelDedup.classList.add('hidden');
      if (elems.transportUnified) elems.transportUnified.classList.remove('hidden');
      if (elems.transportDual) elems.transportDual.classList.add('hidden');
      updateMaRenderButton();
    }
    updateRenderButtonLabel();
    updateTargetInfo();
    applyTransforms();
  }

  if (elems.tabUpscale) elems.tabUpscale.addEventListener('click', () => setTab('upscale'));
  if (elems.tabSlowmo) elems.tabSlowmo.addEventListener('click', () => setTab('slowmo'));
  if (elems.tabDedup) elems.tabDedup.addEventListener('click', () => setTab('dedup'));
  if (elems.tabMatanyone) elems.tabMatanyone.addEventListener('click', () => setTab('matanyone'));

  // Scale buttons
  document.querySelectorAll('.scale-btn:not(.slowmo-factor-btn):not(.dedup-factor-btn)').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.scale-btn:not(.slowmo-factor-btn):not(.dedup-factor-btn)').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const scale = parseInt(btn.getAttribute('data-scale'), 10) || 2;
      state.upscaleConfig.scale = scale;
      state.upscaleConfig.profile = `${scale}x_Balanced`;
      renderProfiles();
      updateTargetInfo();
    });
  });

  // Slowmo Factor buttons
  document.querySelectorAll('.slowmo-factor-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.slowmo-factor-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.slowmoConfig.factor = parseInt(btn.getAttribute('data-factor'), 10) || 2;
      updateTargetInfo();
    });
  });

  // Dedup Factor buttons
  document.querySelectorAll('.dedup-factor-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.dedup-factor-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.dedupConfig.factor = parseInt(btn.getAttribute('data-dedup-factor'), 10) || 2;
      updateTargetInfo();
    });
  });

  // Cadence buttons
  document.querySelectorAll('.cadence-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cadence-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.dedupConfig.npass = parseInt(btn.getAttribute('data-npass'), 10) || 0;
    });
  });

  // 11. Drag and Drop & File Upload
  if (elems.btnRefreshVideos) {
    elems.btnRefreshVideos.addEventListener('click', loadVideos);
  }

  if (elems.dropzone && elems.fileInput) {
    elems.dropzone.addEventListener('click', (e) => {
      if (e.target !== elems.fileInput) {
        elems.fileInput.click();
      }
    });
  }

  if (elems.btnStageBrowse && elems.fileInput) {
    elems.btnStageBrowse.addEventListener('click', () => {
      elems.fileInput.click();
    });
  }

  let dragCounter = 0;
  window.addEventListener('dragenter', (ev) => {
    ev.preventDefault();
    dragCounter++;
    if (elems.dragOverlay) elems.dragOverlay.classList.add('active');
  });

  window.addEventListener('dragleave', (ev) => {
    ev.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      if (elems.dragOverlay) elems.dragOverlay.classList.remove('active');
    }
  });

  window.addEventListener('dragover', (ev) => {
    ev.preventDefault();
  });

  window.addEventListener('drop', async (ev) => {
    ev.preventDefault();
    dragCounter = 0;
    if (elems.dragOverlay) elems.dragOverlay.classList.remove('active');
    if (ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files.length > 0) {
      await handleFileUpload(ev.dataTransfer.files);
    }
  });

  if (elems.fileInput) {
    elems.fileInput.addEventListener('change', async () => {
      if (elems.fileInput.files && elems.fileInput.files.length > 0) {
        await handleFileUpload(elems.fileInput.files);
        elems.fileInput.value = '';
      }
    });
  }

  async function handleFileUpload(files) {
    if (state.isUploading) return;
    state.isUploading = true;

    if (elems.uploadProgressBarContainer) {
      elems.uploadProgressBarContainer.classList.remove('hidden');
      if (elems.dropzoneLabel) elems.dropzoneLabel.classList.add('hidden');
      if (elems.dropzoneSub) elems.dropzoneSub.classList.add('hidden');
    }

    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (elems.uploadProgressText) {
          elems.uploadProgressText.textContent = `Uploading ${file.name}... (${i + 1}/${files.length})`;
        }
        if (elems.uploadProgressBarFill) {
          elems.uploadProgressBarFill.style.width = `${Math.round(((i + 0.5) / files.length) * 100)}%`;
        }

        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch('/api/videos/upload', {
          method: 'POST',
          body: formData
        });

        if (resp.ok) {
          const data = await resp.json();
          if (data.success) {
            await loadVideos();
            showToast(`Uploaded: ${file.name}`, 'success');
          } else {
            showToast(`Upload failed: ${data.detail || 'Unknown error'}`, 'error');
          }
        } else {
          const errText = await resp.text();
          let msg = 'Upload failed';
          try {
            const errJson = JSON.parse(errText);
            msg = errJson.detail || msg;
          } catch (_) {}
          showToast(msg, 'error');
        }
      }
    } catch (e) {
      console.error('File upload failed:', e);
      showToast('Network error during upload', 'error');
    } finally {
      state.isUploading = false;
      if (elems.uploadProgressBarContainer) {
        elems.uploadProgressBarContainer.classList.add('hidden');
        if (elems.dropzoneLabel) elems.dropzoneLabel.classList.remove('hidden');
        if (elems.dropzoneSub) elems.dropzoneSub.classList.remove('hidden');
      }
    }
  }

  // 12. Zoom & Pan Matrix Engine
  function applyTransforms() {
    if (elems.wrapperLeft) {
      const { panX, panY, scale } = state.zoomState.left;
      elems.wrapperLeft.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }
    if (elems.wrapperRight) {
      const { panX, panY, scale } = state.zoomState.right;
      elems.wrapperRight.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }
    if (elems.wrapperMaInput) {
      const { panX, panY, scale } = state.zoomState.maInput;
      elems.wrapperMaInput.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }
    if (elems.maMaskStage) {
      const { panX, panY, scale } = state.zoomState.maMask;
      elems.maMaskStage.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }
    if (elems.wrapperMaGreenscreen) {
      const { panX, panY, scale } = state.zoomState.maGreenscreen;
      elems.wrapperMaGreenscreen.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }
    if (elems.wrapperMaMatte) {
      const { panX, panY, scale } = state.zoomState.maMatte;
      elems.wrapperMaMatte.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
    }

    if (elems.zoomBadge) {
      let currentScale = state.zoomState.left.scale;
      if (state.activeTab === 'matanyone') {
        const maKey = state.lastActiveViewportMa === 'mask' ? 'maMask'
          : state.lastActiveViewportMa === 'greenscreen' ? 'maGreenscreen'
          : state.lastActiveViewportMa === 'matte' ? 'maMatte' : 'maInput';
        currentScale = state.zoomState[maKey]?.scale || 1.0;
      }
      elems.zoomBadge.textContent = `${Math.round(currentScale * 100)}%`;
    }
  }

  function handleWheel(ev, viewportName) {
    ev.preventDefault();
    if (viewportName.startsWith('ma')) {
      const maKey = viewportName === 'ma-mask' ? 'mask'
        : viewportName === 'ma-greenscreen' ? 'greenscreen'
        : viewportName === 'ma-matte' ? 'matte' : 'input';
      setActiveMaWindow(maKey);
    } else {
      setActiveWindow(viewportName);
    }

    const factor = ev.deltaY < 0 ? 1.15 : 0.87;
    const isUpscale = state.activeTab === 'upscale';

    if (isUpscale) {
      const newScale = Math.min(Math.max(0.5, state.zoomState.left.scale * factor), 10.0);
      state.zoomState.left.scale = newScale;
      state.zoomState.right.scale = newScale;
    } else if (state.activeTab === 'matanyone') {
      const keyMap = {
        'ma-input': 'maInput',
        'ma-mask': 'maMask',
        'ma-greenscreen': 'maGreenscreen',
        'ma-matte': 'maMatte'
      };
      const key = keyMap[viewportName] || 'maInput';
      const curr = state.zoomState[key];
      if (curr) curr.scale = Math.min(Math.max(0.5, curr.scale * factor), 10.0);
    } else {
      const curr = state.zoomState[viewportName];
      if (curr) curr.scale = Math.min(Math.max(0.5, curr.scale * factor), 10.0);
    }
    applyTransforms();
  }

  function handleMouseDown(ev, viewportName) {
    if (ev.button !== 0) return;
    if (viewportName.startsWith('ma')) {
      const maKey = viewportName === 'ma-mask' ? 'mask'
        : viewportName === 'ma-greenscreen' ? 'greenscreen'
        : viewportName === 'ma-matte' ? 'matte' : 'input';
      setActiveMaWindow(maKey);
    } else {
      setActiveWindow(viewportName);
    }

    state.zoomState.isDragging = true;
    state.zoomState.targetViewport = viewportName;
    state.zoomState.dragStart = { x: ev.clientX, y: ev.clientY };
    const vpMap = {
      'left': elems.viewportLeft,
      'right': elems.viewportRight,
      'ma-input': elems.viewportMaInput,
      'ma-mask': elems.viewportMaMask,
      'ma-greenscreen': elems.viewportMaGreenscreen,
      'ma-matte': elems.viewportMaMatte
    };
    const vp = vpMap[viewportName];
    if (vp) vp.classList.add('grabbing');
  }

  function handleMouseMove(ev) {
    if (!state.zoomState.isDragging) return;
    const dx = ev.clientX - state.zoomState.dragStart.x;
    const dy = ev.clientY - state.zoomState.dragStart.y;
    state.zoomState.dragStart = { x: ev.clientX, y: ev.clientY };

    const isUpscale = state.activeTab === 'upscale';
    if (isUpscale) {
      state.zoomState.left.panX += dx;
      state.zoomState.left.panY += dy;
      state.zoomState.right.panX += dx;
      state.zoomState.right.panY += dy;
    } else {
      const target = state.zoomState.targetViewport;
      const keyMap = {
        'left': 'left',
        'right': 'right',
        'ma-input': 'maInput',
        'ma-mask': 'maMask',
        'ma-greenscreen': 'maGreenscreen',
        'ma-matte': 'maMatte'
      };
      const key = keyMap[target] || target;
      if (key && state.zoomState[key]) {
        state.zoomState[key].panX += dx;
        state.zoomState[key].panY += dy;
      }
    }
    applyTransforms();
  }

  function handleMouseUp() {
    if (state.zoomState.isDragging) {
      state.zoomState.isDragging = false;
      [elems.viewportLeft, elems.viewportRight, elems.viewportMaInput, elems.viewportMaMask, elems.viewportMaGreenscreen, elems.viewportMaMatte].forEach(vp => {
        if (vp) vp.classList.remove('grabbing');
      });
    }
  }

  if (elems.viewportLeft) {
    elems.viewportLeft.addEventListener('wheel', ev => handleWheel(ev, 'left'));
    elems.viewportLeft.addEventListener('mousedown', ev => handleMouseDown(ev, 'left'));
  }
  if (elems.viewportRight) {
    elems.viewportRight.addEventListener('wheel', ev => handleWheel(ev, 'right'));
    elems.viewportRight.addEventListener('mousedown', ev => handleMouseDown(ev, 'right'));
  }
  if (elems.viewportMaInput) {
    elems.viewportMaInput.addEventListener('wheel', ev => handleWheel(ev, 'ma-input'));
    elems.viewportMaInput.addEventListener('mousedown', ev => handleMouseDown(ev, 'ma-input'));
  }
  if (elems.viewportMaGreenscreen) {
    elems.viewportMaGreenscreen.addEventListener('wheel', ev => handleWheel(ev, 'ma-greenscreen'));
    elems.viewportMaGreenscreen.addEventListener('mousedown', ev => handleMouseDown(ev, 'ma-greenscreen'));
  }
  if (elems.viewportMaMatte) {
    elems.viewportMaMatte.addEventListener('wheel', ev => handleWheel(ev, 'ma-matte'));
    elems.viewportMaMatte.addEventListener('mousedown', ev => handleMouseDown(ev, 'ma-matte'));
  }
  window.addEventListener('mousemove', handleMouseMove);
  window.addEventListener('mouseup', handleMouseUp);

  function resetView() {
    state.zoomState.left = { scale: 1.0, panX: 0, panY: 0 };
    state.zoomState.right = { scale: 1.0, panX: 0, panY: 0 };
    state.zoomState.maInput = { scale: 1.0, panX: 0, panY: 0 };
    state.zoomState.maMask = { scale: 1.0, panX: 0, panY: 0 };
    state.zoomState.maGreenscreen = { scale: 1.0, panX: 0, panY: 0 };
    state.zoomState.maMatte = { scale: 1.0, panX: 0, panY: 0 };
    applyTransforms();
  }

  if (elems.btnResetView) elems.btnResetView.addEventListener('click', resetView);

  if (elems.btnZoomIn) {
    elems.btnZoomIn.addEventListener('click', () => {
      if (state.activeTab === 'matanyone') {
        const maKey = state.lastActiveViewportMa === 'mask' ? 'maMask'
          : state.lastActiveViewportMa === 'greenscreen' ? 'maGreenscreen'
          : state.lastActiveViewportMa === 'matte' ? 'maMatte' : 'maInput';
        const curr = state.zoomState[maKey];
        if (curr) curr.scale = Math.min(10.0, curr.scale * 1.25);
      } else {
        const newScale = Math.min(10.0, state.zoomState.left.scale * 1.25);
        state.zoomState.left.scale = newScale;
        state.zoomState.right.scale = newScale;
      }
      applyTransforms();
    });
  }

  if (elems.btnZoomOut) {
    elems.btnZoomOut.addEventListener('click', () => {
      if (state.activeTab === 'matanyone') {
        const maKey = state.lastActiveViewportMa === 'mask' ? 'maMask'
          : state.lastActiveViewportMa === 'greenscreen' ? 'maGreenscreen'
          : state.lastActiveViewportMa === 'matte' ? 'maMatte' : 'maInput';
        const curr = state.zoomState[maKey];
        if (curr) curr.scale = Math.max(0.5, curr.scale / 1.25);
      } else {
        const newScale = Math.max(0.5, state.zoomState.left.scale / 1.25);
        state.zoomState.left.scale = newScale;
        state.zoomState.right.scale = newScale;
      }
      applyTransforms();
    });
  }

  // 12b. App & GPU VRAM Reset Button (Full App State & Video Preview Reset)
  if (elems.btnResetApp) {
    elems.btnResetApp.addEventListener('click', async () => {
      if (!confirm('Clear GPU memory (VRAM) and reset application state?')) return;
      try {
        const resp = await fetch('/api/system/reset', { method: 'POST' });
        const data = await resp.json();
        
        // 1. Unload & clear all video players, target mask canvas & brush controls completely
        clearPreview(true);

        // 2. Reset state selections
        state.selectedVideos = [];
        state.selectedVideo = null;
        state.selectedOutputVideo = null;
        state.anchorVideoIndex = -1;
        state.activeJob = null;

        // 3. Reset view mode to default split and reset active windows
        state.viewMode = 'split';
        if (elems.btnViewSplit) elems.btnViewSplit.classList.add('active');
        if (elems.btnViewSingle) elems.btnViewSingle.classList.remove('active');
        setActiveWindow('left');
        setActiveMaWindow('input');
        applySplitView();

        // 4. Reset zoom and view
        resetView();

        // 5. Reset UI indicators, banners, progress cards and queue
        resetRenderUI();
        if (elems.renderCompletedBanner) elems.renderCompletedBanner.classList.add('hidden');

        // 6. Reload lists to restore initial startup state
        await loadSystemInfo();
        await loadVideos();
        if (state.isOutputDropdownOpen) {
          await loadOutputVideos(false);
        }

        const vramMsg = data.vram && data.vram.free_gb ? ` (${data.vram.free_gb} GB VRAM free)` : '';
        showToast(`App and server state reset!${vramMsg}`, 'success');
      } catch (e) {
        showToast(`Reset error: ${e}`, 'error');
      }
    });
  }

  // 13. Video Playback & Synchronized Controls (Robust Seeking + Dynamic Progress Fill)
  let isScrubbing = false;
  let isScrubbingLeft = false;
  let isScrubbingRight = false;

  function getVideoDuration(videoElem) {
    if (videoElem && !isNaN(videoElem.duration) && isFinite(videoElem.duration) && videoElem.duration > 0) {
      return videoElem.duration;
    }
    if (state.selectedVideo && state.selectedVideo.duration > 0) {
      return state.selectedVideo.duration;
    }
    return 0;
  }

  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0 || !isFinite(seconds)) return '00:00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const f = Math.floor((seconds % 1) * 24);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}:${String(f).padStart(2, '0')}`;
  }

  function formatShortTime(seconds) {
    if (isNaN(seconds) || seconds < 0 || !isFinite(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  function setSliderProgress(slider, pct) {
    if (!slider) return;
    const clamped = Math.max(0, Math.min(100, pct));
    slider.value = clamped;
    slider.style.setProperty('--progress', `${clamped}%`);
  }

  function togglePlayPause() {
    if (state.activeTab === 'matanyone') {
      const mainVid = elems.videoMaInput;
      if (!mainVid) return;
      if (mainVid.paused) {
        mainVid.play().catch(() => {});
        if (elems.videoMaGreenscreen && elems.videoMaGreenscreen.src) elems.videoMaGreenscreen.play().catch(() => {});
        if (elems.videoMaMatte && elems.videoMaMatte.src) elems.videoMaMatte.play().catch(() => {});
        if (elems.playIcon) {
          elems.playIcon.innerHTML = '<rect width="4" height="16" x="6" y="4"/><rect width="4" height="16" x="14" y="4"/>';
        }
      } else {
        mainVid.pause();
        if (elems.videoMaGreenscreen) elems.videoMaGreenscreen.pause();
        if (elems.videoMaMatte) elems.videoMaMatte.pause();
        if (elems.playIcon) {
          elems.playIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"/>';
        }
      }
      return;
    }

    if (!elems.videoLeft) return;
    if (elems.videoLeft.paused) {
      elems.videoLeft.play().catch(() => {});
      if (elems.videoRight && elems.videoRight.src) elems.videoRight.play().catch(() => {});
      if (elems.playIcon) {
        elems.playIcon.innerHTML = '<rect width="4" height="16" x="6" y="4"/><rect width="4" height="16" x="14" y="4"/>';
      }
    } else {
      elems.videoLeft.pause();
      if (elems.videoRight && elems.videoRight.src) elems.videoRight.pause();
      if (elems.playIcon) {
        elems.playIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"/>';
      }
    }
  }

  if (elems.btnPlayPause) elems.btnPlayPause.addEventListener('click', togglePlayPause);

  function syncMaPlayback() {
    if (state.activeTab !== 'matanyone' || !elems.videoMaInput) return;
    const cur = elems.videoMaInput.currentTime || 0;
    const dur = getVideoDuration(elems.videoMaInput);

    if (elems.videoMaGreenscreen && elems.videoMaGreenscreen.src && Math.abs(elems.videoMaGreenscreen.currentTime - cur) > 0.08) {
      elems.videoMaGreenscreen.currentTime = cur;
    }
    if (elems.videoMaMatte && elems.videoMaMatte.src && Math.abs(elems.videoMaMatte.currentTime - cur) > 0.08) {
      elems.videoMaMatte.currentTime = cur;
    }

    if (!isScrubbing) {
      if (elems.timecodeCurrent) {
        elems.timecodeCurrent.textContent = formatTime(cur);
      }
      if (dur > 0 && elems.timelineScrubber) {
        const pct = (cur / dur) * 100;
        setSliderProgress(elems.timelineScrubber, pct);
      }
      if (dur > 0 && elems.timecodeTotal) {
        elems.timecodeTotal.textContent = formatTime(dur);
      }
    }
  }

  if (elems.videoMaInput) {
    elems.videoMaInput.addEventListener('loadedmetadata', () => {
      const dur = getVideoDuration(elems.videoMaInput);
      if (state.activeTab === 'matanyone') {
        if (elems.timecodeTotal) elems.timecodeTotal.textContent = formatTime(dur);
        setSliderProgress(elems.timelineScrubber, 0);
      }
      if (elems.videoMaInput.videoWidth > 0 && elems.videoMaInput.videoHeight > 0) {
        const fpsStr = (state.selectedVideo && state.selectedVideo.fps > 0) ? ` • ${state.selectedVideo.fps}fps` : '';
        if (elems.maInputMetaBadge) {
          elems.maInputMetaBadge.textContent = `${elems.videoMaInput.videoWidth}x${elems.videoMaInput.videoHeight}${fpsStr}`;
        }
      }
    });

    elems.videoMaInput.addEventListener('timeupdate', syncMaPlayback);

    elems.videoMaInput.addEventListener('play', () => {
      if (state.activeTab === 'matanyone' && elems.playIcon) {
        elems.playIcon.innerHTML = '<rect width="4" height="16" x="6" y="4"/><rect width="4" height="16" x="14" y="4"/>';
      }
    });

    elems.videoMaInput.addEventListener('pause', () => {
      if (state.activeTab === 'matanyone' && elems.playIcon) {
        elems.playIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"/>';
      }
    });

    elems.videoMaInput.addEventListener('ended', () => {
      if (state.activeTab === 'matanyone') {
        if (elems.btnLoop && elems.btnLoop.classList.contains('active')) {
          elems.videoMaInput.currentTime = 0;
          elems.videoMaInput.play().catch(() => {});
          if (elems.videoMaGreenscreen && elems.videoMaGreenscreen.src) {
            elems.videoMaGreenscreen.currentTime = 0;
            elems.videoMaGreenscreen.play().catch(() => {});
          }
          if (elems.videoMaMatte && elems.videoMaMatte.src) {
            elems.videoMaMatte.currentTime = 0;
            elems.videoMaMatte.play().catch(() => {});
          }
        } else {
          if (elems.playIcon) {
            elems.playIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"/>';
          }
        }
      }
    });
  }

  if (elems.videoLeft) {
    elems.videoLeft.addEventListener('loadedmetadata', () => {
      const dur = getVideoDuration(elems.videoLeft);
      if (state.activeTab !== 'matanyone') {
        if (elems.timecodeTotal) elems.timecodeTotal.textContent = formatTime(dur);
        setSliderProgress(elems.timelineScrubber, 0);
        setSliderProgress(elems.scrubberLeft, 0);
      }

      if (elems.videoLeft.videoWidth > 0 && elems.videoLeft.videoHeight > 0) {
        const fpsStr = (state.selectedVideo && state.selectedVideo.fps > 0) ? ` • ${state.selectedVideo.fps}fps` : '';
        if (elems.beforeMetaBadge) {
          elems.beforeMetaBadge.textContent = `${elems.videoLeft.videoWidth}x${elems.videoLeft.videoHeight}${fpsStr}`;
        }
      }
    });

    elems.videoLeft.addEventListener('timeupdate', () => {
      if (state.activeTab === 'matanyone') return;
      const cur = elems.videoLeft.currentTime || 0;
      const dur = getVideoDuration(elems.videoLeft);

      if (state.activeTab === 'upscale') {
        if (elems.videoRight && elems.videoRight.src && Math.abs(elems.videoRight.currentTime - cur) > 0.08) {
          elems.videoRight.currentTime = cur;
        }
        if (!isScrubbing) {
          if (elems.timecodeCurrent) {
            elems.timecodeCurrent.textContent = formatTime(cur);
          }
          if (dur > 0 && elems.timelineScrubber) {
            const pct = (cur / dur) * 100;
            setSliderProgress(elems.timelineScrubber, pct);
          }
          if (dur > 0 && elems.timecodeTotal) {
            elems.timecodeTotal.textContent = formatTime(dur);
          }
        }
      } else {
        if (!isScrubbingLeft) {
          if (elems.timecodeLeft) {
            elems.timecodeLeft.textContent = formatShortTime(cur);
          }
          if (dur > 0 && elems.scrubberLeft) {
            const pct = (cur / dur) * 100;
            setSliderProgress(elems.scrubberLeft, pct);
          }
        }
      }
    });

    elems.videoLeft.addEventListener('play', () => {
      if (state.activeTab !== 'matanyone' && elems.playIcon) {
        elems.playIcon.innerHTML = '<rect width="4" height="16" x="6" y="4"/><rect width="4" height="16" x="14" y="4"/>';
      }
    });

    elems.videoLeft.addEventListener('pause', () => {
      if (state.activeTab !== 'matanyone' && elems.playIcon) {
        elems.playIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"/>';
      }
    });

    elems.videoLeft.addEventListener('ended', () => {
      if (state.activeTab !== 'matanyone') {
        if (elems.btnLoop && elems.btnLoop.classList.contains('active')) {
          elems.videoLeft.currentTime = 0;
          elems.videoLeft.play().catch(() => {});
          if (elems.videoRight && elems.videoRight.src) {
            elems.videoRight.currentTime = 0;
            elems.videoRight.play().catch(() => {});
          }
        } else {
          if (elems.playIcon) {
            elems.playIcon.innerHTML = '<polygon points="6 3 20 12 6 21 6 3"/>';
          }
        }
      }
    });
  }

  if (elems.videoRight) {
    elems.videoRight.addEventListener('loadedmetadata', () => {
      setSliderProgress(elems.scrubberRight, 0);
      if (elems.videoRight.videoWidth > 0 && elems.videoRight.videoHeight > 0) {
        const fpsStr = (state.selectedOutputVideo && state.selectedOutputVideo.fps > 0) ? ` • ${state.selectedOutputVideo.fps}fps` : '';
        if (elems.afterMetaBadge && state.selectedOutputVideo) {
          elems.afterMetaBadge.textContent = `${elems.videoRight.videoWidth}x${elems.videoRight.videoHeight}${fpsStr}`;
        }
      }
    });

    elems.videoRight.addEventListener('timeupdate', () => {
      if (state.activeTab !== 'upscale' && state.activeTab !== 'matanyone' && !isScrubbingRight) {
        const cur = elems.videoRight.currentTime || 0;
        const dur = elems.videoRight.duration || 0;
        if (elems.timecodeRight) {
          elems.timecodeRight.textContent = formatShortTime(cur);
        }
        if (dur > 0 && elems.scrubberRight) {
          const pct = (cur / dur) * 100;
          setSliderProgress(elems.scrubberRight, pct);
        }
      }
    });
  }

  // Unified Timeline Scrubber (Skip & Seek)
  if (elems.timelineScrubber) {
    const onScrubStart = () => {
      isScrubbing = true;
    };

    const onScrub = () => {
      const pct = parseFloat(elems.timelineScrubber.value) || 0;
      setSliderProgress(elems.timelineScrubber, pct);

      if (state.activeTab === 'matanyone') {
        const dur = getVideoDuration(elems.videoMaInput);
        if (dur > 0) {
          const targetTime = (pct / 100) * dur;
          if (elems.timecodeCurrent) {
            elems.timecodeCurrent.textContent = formatTime(targetTime);
          }
          if (elems.videoMaInput) elems.videoMaInput.currentTime = targetTime;
          if (elems.videoMaGreenscreen && elems.videoMaGreenscreen.src) elems.videoMaGreenscreen.currentTime = targetTime;
          if (elems.videoMaMatte && elems.videoMaMatte.src) elems.videoMaMatte.currentTime = targetTime;
        }
      } else if (elems.videoLeft) {
        const dur = getVideoDuration(elems.videoLeft);
        if (dur > 0) {
          const targetTime = (pct / 100) * dur;
          if (elems.timecodeCurrent) {
            elems.timecodeCurrent.textContent = formatTime(targetTime);
          }
          elems.videoLeft.currentTime = targetTime;
          if (elems.videoRight && elems.videoRight.src) {
            elems.videoRight.currentTime = targetTime;
          }
        }
      }
    };

    const onScrubEnd = () => {
      isScrubbing = false;
      const pct = parseFloat(elems.timelineScrubber.value) || 0;
      setSliderProgress(elems.timelineScrubber, pct);

      if (state.activeTab === 'matanyone') {
        const dur = getVideoDuration(elems.videoMaInput);
        if (dur > 0) {
          const targetTime = (pct / 100) * dur;
          if (elems.videoMaInput) elems.videoMaInput.currentTime = targetTime;
          if (elems.videoMaGreenscreen && elems.videoMaGreenscreen.src) elems.videoMaGreenscreen.currentTime = targetTime;
          if (elems.videoMaMatte && elems.videoMaMatte.src) elems.videoMaMatte.currentTime = targetTime;
        }
      } else if (elems.videoLeft) {
        const dur = getVideoDuration(elems.videoLeft);
        if (dur > 0) {
          const targetTime = (pct / 100) * dur;
          elems.videoLeft.currentTime = targetTime;
          if (elems.videoRight && elems.videoRight.src) {
            elems.videoRight.currentTime = targetTime;
          }
        }
      }
    };

    elems.timelineScrubber.addEventListener('pointerdown', onScrubStart);
    elems.timelineScrubber.addEventListener('mousedown', onScrubStart);
    elems.timelineScrubber.addEventListener('touchstart', onScrubStart, { passive: true });

    elems.timelineScrubber.addEventListener('input', onScrub);
    elems.timelineScrubber.addEventListener('change', onScrubEnd);

    elems.timelineScrubber.addEventListener('pointerup', onScrubEnd);
    elems.timelineScrubber.addEventListener('mouseup', onScrubEnd);
    elems.timelineScrubber.addEventListener('touchend', onScrubEnd);
  }

  // Dual Transport Left Scrubber
  if (elems.scrubberLeft && elems.videoLeft) {
    const onScrubLeft = () => {
      const pct = parseFloat(elems.scrubberLeft.value) || 0;
      setSliderProgress(elems.scrubberLeft, pct);
      const dur = getVideoDuration(elems.videoLeft);
      if (dur > 0) {
        const targetTime = (pct / 100) * dur;
        elems.videoLeft.currentTime = targetTime;
        if (elems.timecodeLeft) elems.timecodeLeft.textContent = formatShortTime(targetTime);
      }
    };

    elems.scrubberLeft.addEventListener('pointerdown', () => { isScrubbingLeft = true; });
    elems.scrubberLeft.addEventListener('mousedown', () => { isScrubbingLeft = true; });
    elems.scrubberLeft.addEventListener('input', onScrubLeft);
    elems.scrubberLeft.addEventListener('change', () => { isScrubbingLeft = false; onScrubLeft(); });
    elems.scrubberLeft.addEventListener('pointerup', () => { isScrubbingLeft = false; });
    elems.scrubberLeft.addEventListener('mouseup', () => { isScrubbingLeft = false; });
  }

  // Dual Transport Right Scrubber
  if (elems.scrubberRight && elems.videoRight) {
    const onScrubRight = () => {
      const pct = parseFloat(elems.scrubberRight.value) || 0;
      setSliderProgress(elems.scrubberRight, pct);
      const dur = elems.videoRight.duration || 0;
      if (dur > 0) {
        const targetTime = (pct / 100) * dur;
        elems.videoRight.currentTime = targetTime;
        if (elems.timecodeRight) elems.timecodeRight.textContent = formatShortTime(targetTime);
      }
    };

    elems.scrubberRight.addEventListener('pointerdown', () => { isScrubbingRight = true; });
    elems.scrubberRight.addEventListener('mousedown', () => { isScrubbingRight = true; });
    elems.scrubberRight.addEventListener('input', onScrubRight);
    elems.scrubberRight.addEventListener('change', () => { isScrubbingRight = false; onScrubRight(); });
    elems.scrubberRight.addEventListener('pointerup', () => { isScrubbingRight = false; });
    elems.scrubberRight.addEventListener('mouseup', () => { isScrubbingRight = false; });
  }

  // Frame Stepping
  if (elems.btnStepBack) {
    elems.btnStepBack.addEventListener('click', () => {
      const fps = (state.selectedVideo && state.selectedVideo.fps) || 24;
      if (state.activeTab === 'matanyone' && elems.videoMaInput) {
        const newTime = Math.max(0, (elems.videoMaInput.currentTime || 0) - 1 / fps);
        elems.videoMaInput.currentTime = newTime;
        if (elems.videoMaGreenscreen && elems.videoMaGreenscreen.src) elems.videoMaGreenscreen.currentTime = newTime;
        if (elems.videoMaMatte && elems.videoMaMatte.src) elems.videoMaMatte.currentTime = newTime;
        const dur = getVideoDuration(elems.videoMaInput);
        if (dur > 0) {
          setSliderProgress(elems.timelineScrubber, (newTime / dur) * 100);
        }
      } else if (elems.videoLeft) {
        const newTime = Math.max(0, (elems.videoLeft.currentTime || 0) - 1 / fps);
        elems.videoLeft.currentTime = newTime;
        if (elems.videoRight && elems.videoRight.src) elems.videoRight.currentTime = newTime;
        const dur = getVideoDuration(elems.videoLeft);
        if (dur > 0) {
          setSliderProgress(elems.timelineScrubber, (newTime / dur) * 100);
        }
      }
    });
  }

  if (elems.btnStepFwd) {
    elems.btnStepFwd.addEventListener('click', () => {
      const fps = (state.selectedVideo && state.selectedVideo.fps) || 24;
      if (state.activeTab === 'matanyone' && elems.videoMaInput) {
        const dur = getVideoDuration(elems.videoMaInput);
        const newTime = Math.min(dur || 99999, (elems.videoMaInput.currentTime || 0) + 1 / fps);
        elems.videoMaInput.currentTime = newTime;
        if (elems.videoMaGreenscreen && elems.videoMaGreenscreen.src) elems.videoMaGreenscreen.currentTime = newTime;
        if (elems.videoMaMatte && elems.videoMaMatte.src) elems.videoMaMatte.currentTime = newTime;
        if (dur > 0) {
          setSliderProgress(elems.timelineScrubber, (newTime / dur) * 100);
        }
      } else if (elems.videoLeft) {
        const dur = getVideoDuration(elems.videoLeft);
        const newTime = Math.min(dur || 99999, (elems.videoLeft.currentTime || 0) + 1 / fps);
        elems.videoLeft.currentTime = newTime;
        if (elems.videoRight && elems.videoRight.src) elems.videoRight.currentTime = newTime;
        if (dur > 0) {
          setSliderProgress(elems.timelineScrubber, (newTime / dur) * 100);
        }
      }
    });
  }

  // Loop Toggle
  if (elems.btnLoop) {
    elems.btnLoop.addEventListener('click', () => {
      elems.btnLoop.classList.toggle('active');
      const isLoop = elems.btnLoop.classList.contains('active');
      elems.btnLoop.style.color = isLoop ? 'var(--accent)' : 'var(--text-muted)';
      showToast(isLoop ? 'Loop playback enabled' : 'Loop playback disabled', 'info');
    });
  }

  // Mute Toggle
  if (elems.btnMute) {
    elems.btnMute.addEventListener('click', () => {
      const currentMuted = (state.activeTab === 'matanyone' && elems.videoMaInput)
        ? elems.videoMaInput.muted
        : (elems.videoLeft ? elems.videoLeft.muted : false);
      const isMuted = !currentMuted;

      if (elems.videoLeft) elems.videoLeft.muted = isMuted;
      if (elems.videoRight) elems.videoRight.muted = isMuted;
      if (elems.videoMaInput) elems.videoMaInput.muted = isMuted;
      if (elems.videoMaGreenscreen) elems.videoMaGreenscreen.muted = isMuted;
      if (elems.videoMaMatte) elems.videoMaMatte.muted = isMuted;

      elems.btnMute.style.color = isMuted ? 'var(--text-muted)' : 'var(--accent)';
      if (elems.muteIcon) {
        if (isMuted) {
          elems.muteIcon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" x2="1" y1="1" y2="23" stroke="currentColor" stroke-width="2"/>';
        } else {
          elems.muteIcon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>';
        }
      }
    });
  }

  // Dual Play Buttons
  if (elems.btnPlayLeft && elems.videoLeft) {
    elems.btnPlayLeft.addEventListener('click', () => {
      if (elems.videoLeft.paused) elems.videoLeft.play().catch(() => {});
      else elems.videoLeft.pause();
    });
  }

  if (elems.btnPlayRight && elems.videoRight) {
    elems.btnPlayRight.addEventListener('click', () => {
      if (elems.videoRight.paused) elems.videoRight.play().catch(() => {});
      else elems.videoRight.pause();
    });
  }

  // 14. Render Submission (Single & Batch) & Progress
  function updateRenderButtonLabel() {
    if (!elems.btnRender) return;
    const count = state.selectedVideos.length;
    if (state.activeTab === 'matanyone') {
      const hasMask = !!(window.__maMask && !window.__maMask.isEmpty());
      elems.btnRender.disabled = !(count > 0 && hasMask && !state.activeJob);
      elems.btnRender.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg><span>Render Easy Mask</span>`;
    } else {
      elems.btnRender.disabled = (count === 0 || !!state.activeJob);
      if (count > 1) {
        elems.btnRender.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg><span>Render Batch (${count} Videos)</span>`;
      } else {
        elems.btnRender.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg><span>Render Video</span>`;
      }
    }
  }

  function updateTargetInfo() {
    if (!state.selectedVideo) {
      if (elems.afterMetaBadge) elems.afterMetaBadge.textContent = 'Enhanced';
      return;
    }
    if (state.activeTab === 'upscale') {
      const scale = state.upscaleConfig.scale;
      const targetW = state.selectedVideo.width ? state.selectedVideo.width * scale : 0;
      const targetH = state.selectedVideo.height ? state.selectedVideo.height * scale : 0;
      if (targetW > 0 && targetH > 0 && elems.afterMetaBadge) {
        elems.afterMetaBadge.textContent = `Target: ${targetW}x${targetH} • ${state.selectedVideo.fps}fps (${state.upscaleConfig.profile})`;
      }
    } else if (state.activeTab === 'slowmo') {
      const factor = state.slowmoConfig.factor;
      const targetFps = state.selectedVideo.fps ? (state.selectedVideo.fps * factor).toFixed(1) : 0;
      if (elems.afterMetaBadge) {
        elems.afterMetaBadge.textContent = `Target: ${factor}x Slow-mo • ${targetFps}fps (${state.slowmoConfig.model_key})`;
      }
    } else if (state.activeTab === 'dedup') {
      const factor = state.dedupConfig.factor;
      const targetFps = state.selectedVideo.fps ? (state.selectedVideo.fps * factor).toFixed(1) : 0;
      if (elems.afterMetaBadge) {
        elems.afterMetaBadge.textContent = `Target: ${factor}x Interpolation • ${targetFps}fps (${state.dedupConfig.model.toUpperCase()})`;
      }
    } else if (state.activeTab === 'matanyone') {
      if (elems.afterMetaBadge) elems.afterMetaBadge.textContent = 'Target: Easy Mask separation';
    }
  }

  function previewVideo(vid) {
    if (!vid) {
      clearPreview();
      return;
    }
    if (elems.videoLeft) {
      elems.videoLeft.src = `/api/stream/video?path=${encodeURIComponent(vid.path)}`;
      elems.videoLeft.load();
      if (elems.placeholderLeft) elems.placeholderLeft.classList.add('hidden');
    }
    if (elems.videoMaInput) {
      elems.videoMaInput.src = `/api/stream/video?path=${encodeURIComponent(vid.path)}`;
      elems.videoMaInput.load();
      if (elems.placeholderMaInput) elems.placeholderMaInput.classList.add('hidden');
      if (elems.maInputMetaBadge && vid.width > 0) {
        elems.maInputMetaBadge.textContent = `${vid.width}x${vid.height} • ${vid.fps}fps`;
      }
    }
    updateTargetInfo();
    updateMaRenderButton();
  }

  function clearPreview(clearInput = true) {
    if (clearInput) {
      if (elems.videoLeft) {
        elems.videoLeft.pause();
        elems.videoLeft.removeAttribute('src');
        elems.videoLeft.load();
        if (elems.placeholderLeft) elems.placeholderLeft.classList.remove('hidden');
      }
      if (elems.videoMaInput) {
        elems.videoMaInput.pause();
        elems.videoMaInput.removeAttribute('src');
        elems.videoMaInput.load();
        if (elems.placeholderMaInput) elems.placeholderMaInput.classList.remove('hidden');
      }
    }
    if (elems.videoRight) {
      elems.videoRight.pause();
      elems.videoRight.removeAttribute('src');
      elems.videoRight.load();
      if (elems.placeholderRight) elems.placeholderRight.classList.remove('hidden');
    }
    // Fully reset MatAnyone2 target mask canvas & brush controls
    if (elems.maMaskStage) {
      elems.maMaskStage.innerHTML = '';
    }
    window.__maMask = null;
    maTargetVideoName = null;
    maDetectPoints = [];
    if (elems.placeholderMaMask) {
      elems.placeholderMaMask.classList.remove('hidden');
    }
    const brushControls = document.getElementById('ma-brush-controls');
    if (brushControls) {
      brushControls.classList.add('hidden');
    }

    if (elems.videoMaGreenscreen) {
      elems.videoMaGreenscreen.pause();
      elems.videoMaGreenscreen.removeAttribute('src');
      elems.videoMaGreenscreen.load();
      if (elems.placeholderMaGreenscreen) elems.placeholderMaGreenscreen.classList.remove('hidden');
    }
    if (elems.videoMaMatte) {
      elems.videoMaMatte.pause();
      elems.videoMaMatte.removeAttribute('src');
      elems.videoMaMatte.load();
      if (elems.placeholderMaMatte) elems.placeholderMaMatte.classList.remove('hidden');
    }
    if (elems.beforeMetaBadge) elems.beforeMetaBadge.textContent = 'Original';
    if (elems.afterMetaBadge) elems.afterMetaBadge.textContent = 'Enhanced';
    if (elems.maInputMetaBadge) elems.maInputMetaBadge.textContent = 'Original';
    if (elems.maGsMetaBadge) elems.maGsMetaBadge.textContent = 'Result';
    if (elems.maMatteMetaBadge) elems.maMatteMetaBadge.textContent = 'Result';
    if (elems.timecodeCurrent) elems.timecodeCurrent.textContent = '00:00:00';
    if (elems.timecodeTotal) elems.timecodeTotal.textContent = '00:00:00';
    setSliderProgress(elems.timelineScrubber, 0);
    updateMaRenderButton();
  }

  // Render Execution
  if (elems.btnRender) {
    elems.btnRender.addEventListener('click', async () => {
      if (state.selectedVideos.length === 0) {
        showToast('Please select at least one video to render', 'error');
        return;
      }
      if (state.activeJob) {
        showToast('A render job is already in progress', 'error');
        return;
      }

      if (state.activeTab === 'matanyone') {
        startMatanyoneJob();
        return;
      }

      let action = 'Upscale';
      let params = {};

      if (state.activeTab === 'upscale') {
        action = 'Upscale';
        params = { profile: state.upscaleConfig.profile };
      } else if (state.activeTab === 'slowmo') {
        action = 'Slow-motion';
        params = {
          model_key: state.slowmoConfig.model_key,
          factor: state.slowmoConfig.factor
        };
      } else if (state.activeTab === 'dedup') {
        action = 'Interpolate';
        params = {
          model: state.dedupConfig.model,
          factor: state.dedupConfig.factor,
          npass: state.dedupConfig.npass
        };
      }

      const videoNames = state.selectedVideos.map(v => v.name);
      const outputDir = (elems.outputDirInput && elems.outputDirInput.value) || 'output';

      try {
        if (elems.btnRender) elems.btnRender.classList.add('hidden');
        if (elems.btnCancel) elems.btnCancel.classList.remove('hidden');
        if (elems.renderProgressCard) elems.renderProgressCard.classList.remove('hidden');
        if (elems.renderCompletedBanner) elems.renderCompletedBanner.classList.add('hidden');

        const resp = await fetch('/api/jobs/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: action,
            video_names: videoNames,
            output_dir: outputDir,
            params: params
          })
        });

        const data = await resp.json();
        if (data.success) {
          state.activeJob = data.job;
          showToast(`Started ${action} job (${videoNames.length} file${videoNames.length === 1 ? '' : 's'})`, 'info');
        } else {
          showToast(`Error: ${data.detail || 'Could not start render job'}`, 'error');
          resetRenderUI();
        }
      } catch (e) {
        showToast(`Error starting job: ${e}`, 'error');
        resetRenderUI();
      }
    });
  }

  if (elems.btnCancel) {
    elems.btnCancel.addEventListener('click', async () => {
      if (!state.activeJob) return;
      try {
        const resp = await fetch(`/api/jobs/cancel/${state.activeJob.job_id}`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
          showToast('Job cancellation requested', 'info');
        }
      } catch (e) {
        showToast('Error cancelling job', 'error');
      }
    });
  }

  if (elems.btnCloseBanner) {
    elems.btnCloseBanner.addEventListener('click', () => {
      if (elems.renderCompletedBanner) elems.renderCompletedBanner.classList.add('hidden');
    });
  }

  function resetRenderUI() {
    if (elems.btnRender) elems.btnRender.classList.remove('hidden');
    if (elems.btnCancel) elems.btnCancel.classList.remove('hidden');
    if (elems.renderProgressCard) elems.renderProgressCard.classList.add('hidden');
  }

  // 14b. MatAnyone2 first-frame mask canvas & job wiring
  let maTargetVideoName = null;
  let maDetectPoints = [];

  function updateMaRenderButton() {
    updateRenderButtonLabel();
  }

  async function loadMaTargetFrame() {
    const video = state.selectedVideos[0];
    if (!video) { showToast('Select a video first', 'error'); return; }
    maTargetVideoName = video.name;
    maDetectPoints = [];
    const frameUrl = `/api/videos/frame?path=${encodeURIComponent(video.path)}&n=0&max=1280`;
    if (!window.__maMask) {
      window.__maMask = window.MaskCanvas.create(
        document.getElementById('ma-mask-stage'), frameUrl, { onDetect: detectSubjectAt });
    } else {
      window.__maMask.resetForImage(frameUrl);
    }
    window.__maMask.togglePreview(true);
    if (elems.placeholderMaMask) elems.placeholderMaMask.classList.add('hidden');
    document.getElementById('ma-brush-controls').classList.remove('hidden');
    updateMaRenderButton();
  }

  async function detectSubjectAt(point) {
    const video = state.selectedVideos.find(v => v.name === maTargetVideoName);
    const frameImg = document.querySelector('#ma-mask-stage .ma-frame');
    if (!video || !frameImg || !frameImg.naturalWidth) return;
    maDetectPoints.push([point.x, point.y]);
    const hintEl = document.getElementById('ma-hint');
    const prevHint = hintEl ? hintEl.textContent : '';
    if (hintEl) hintEl.textContent = `Detecting subject… (${maDetectPoints.length} click${maDetectPoints.length > 1 ? 's' : ''})`;
    document.getElementById('ma-mask-stage').classList.add('ma-detecting');
    try {
      const resp = await fetch('/api/matanyone2/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          path: video.path,
          points: maDetectPoints,
          view_width: frameImg.naturalWidth,
          view_height: frameImg.naturalHeight
        })
      });
      const data = await resp.json();
      if (!resp.ok) {
        maDetectPoints.pop();
        showToast(data.detail || 'Auto-detect failed', 'error');
        return;
      }
      await new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => { window.__maMask.setFromAlphaImage(image); resolve(); };
        image.onerror = reject;
        image.src = data.mask_png;
      });
      updateMaRenderButton();
      showToast(
        (data.engine === 'sam' ? 'Subject detected (SAM)' : 'Subject detected (fallback)') +
        ' - click again to refine, or use Add / Remove', 'info');
    } catch (e) {
      showToast('Auto-detect failed: ' + e, 'error');
    } finally {
      if (hintEl) hintEl.textContent = prevHint;
      document.getElementById('ma-mask-stage').classList.remove('ma-detecting');
    }
  }

  function populateMaResultWindows(outputFiles) {
    const gs = (outputFiles || []).find(f => f.endsWith('_greenscreen.mp4'));
    const matte = (outputFiles || []).find(f => f.endsWith('_matte.mp4'));
    if (gs && elems.videoMaGreenscreen) {
      elems.videoMaGreenscreen.src = '/api/stream/video?path=' + encodeURIComponent(gs);
      elems.videoMaGreenscreen.load();
      if (elems.placeholderMaGreenscreen) elems.placeholderMaGreenscreen.classList.add('hidden');
      if (elems.maGsMetaBadge) elems.maGsMetaBadge.textContent = 'Rendered';
    }
    if (matte && elems.videoMaMatte) {
      elems.videoMaMatte.src = '/api/stream/video?path=' + encodeURIComponent(matte);
      elems.videoMaMatte.load();
      if (elems.placeholderMaMatte) elems.placeholderMaMatte.classList.add('hidden');
      if (elems.maMatteMetaBadge) elems.maMatteMetaBadge.textContent = 'Rendered';
    }
  }

  function setMaMode(mode) {
    if (window.__maMask) window.__maMask.setMode(mode);
    const buttons = {
      detect: document.getElementById('ma-autodetect'),
      add: document.getElementById('ma-add'),
      remove: document.getElementById('ma-remove')
    };
    Object.values(buttons).forEach(b => b && b.classList.remove('active'));
    const active = buttons[mode] || buttons.add;
    if (active) active.classList.add('active');
  }

  function collectMatanyoneParams() {
    const outputs = [];
    if (document.getElementById('ma-out-matte').checked) outputs.push('matte');
    if (document.getElementById('ma-out-fg').checked) outputs.push('greenscreen');
    if (document.getElementById('ma-out-transparent').checked) outputs.push('transparent');
    if (outputs.length === 0) { showToast('Select at least one output', 'error'); return null; }
    const mask = window.__maMask ? window.__maMask.getMaskPngB64() : null;
    if (!mask) { showToast('Detect or paint the target on the first frame first', 'error'); return null; }
    return {
      mask_png: mask,
      backend: document.getElementById('ma-backend').value,
      precision: document.getElementById('ma-precision').value,
      max_size: parseInt(document.getElementById('ma-maxsize').value, 10),
      erode: parseInt(document.getElementById('ma-erode').value, 10),
      dilate: parseInt(document.getElementById('ma-dilate').value, 10),
      warmup: 10,
      outputs
    };
  }

  async function startMatanyoneJob() {
    const params = collectMatanyoneParams();
    if (params === null) return;
    if (state.activeJob) { showToast('A job is already running', 'error'); return; }
    try {
      if (elems.btnRender) elems.btnRender.classList.add('hidden');
      if (elems.btnCancel) elems.btnCancel.classList.remove('hidden');
      if (elems.renderProgressCard) elems.renderProgressCard.classList.remove('hidden');
      if (elems.renderCompletedBanner) elems.renderCompletedBanner.classList.add('hidden');

      const resp = await fetch('/api/jobs/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'MatAnyone2',
          video_names: [maTargetVideoName],
          output_dir: (elems.outputDirInput && elems.outputDirInput.value) || 'output',
          params
        })
      });
      const data = await resp.json();
      if (data.success) {
        state.activeJob = data.job;
        showToast('Easy Mask job started', 'info');
      } else {
        showToast('Error: ' + (data.detail || 'Could not start render job'), 'error');
        resetRenderUI();
        updateMaRenderButton();
      }
    } catch (e) {
      showToast('Error starting job: ' + e, 'error');
      resetRenderUI();
      updateMaRenderButton();
    }
  }

  if (document.getElementById('ma-load-target')) {
    document.getElementById('ma-load-target').addEventListener('click', loadMaTargetFrame);
  }
  const maMaskStage = document.getElementById('ma-mask-stage');
  if (maMaskStage) maMaskStage.addEventListener('pointerup', updateMaRenderButton);

  const maAddBtn = document.getElementById('ma-add');
  const maRemoveBtn = document.getElementById('ma-remove');
  const maAutoDetectBtn = document.getElementById('ma-autodetect');
  if (maAddBtn) maAddBtn.addEventListener('click', () => setMaMode('add'));
  if (maRemoveBtn) maRemoveBtn.addEventListener('click', () => setMaMode('remove'));
  if (maAutoDetectBtn) maAutoDetectBtn.addEventListener('click', () => setMaMode('detect'));
  const maClearBtn = document.getElementById('ma-clear');
  if (maClearBtn) maClearBtn.addEventListener('click', () => {
    maDetectPoints = [];
    if (window.__maMask) { window.__maMask.clear(); updateMaRenderButton(); }
  });
  const maUndoBtn = document.getElementById('ma-undo');
  if (maUndoBtn) maUndoBtn.addEventListener('click', () => {
    if (window.__maMask) { window.__maMask.undo(); updateMaRenderButton(); }
  });
  const maRedoBtn = document.getElementById('ma-redo');
  if (maRedoBtn) maRedoBtn.addEventListener('click', () => {
    if (window.__maMask) { window.__maMask.redo(); updateMaRenderButton(); }
  });
  // Erode & Dilate live value displays
  const maErodeSlider = document.getElementById('ma-erode');
  const maErodeVal = document.getElementById('ma-erode-val');
  if (maErodeSlider && maErodeVal) {
    maErodeSlider.addEventListener('input', () => {
      maErodeVal.textContent = maErodeSlider.value;
    });
  }

  const maDilateSlider = document.getElementById('ma-dilate');
  const maDilateVal = document.getElementById('ma-dilate-val');
  if (maDilateSlider && maDilateVal) {
    maDilateSlider.addEventListener('input', () => {
      maDilateVal.textContent = maDilateSlider.value;
    });
  }

  // Brush Size selector pills
  document.querySelectorAll('.ma-brush-size-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.ma-brush-size-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const size = btn.getAttribute('data-size') || 'medium';
      if (window.__maMask) window.__maMask.setBrushSize(size);
    });
  });
  if (document.getElementById('ma-preview')) {
    document.getElementById('ma-preview').addEventListener('change', (ev) => {
      if (window.__maMask) window.__maMask.togglePreview(ev.target.checked);
    });
  }
  updateMaRenderButton();

  // 15. WebSocket Progress Broadcasting
  function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/api/ws/progress`);

    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === 'job_progress' || data.type === 'job_started') {
          const job = data.job;
          state.activeJob = job;
          if (elems.renderProgressCard) elems.renderProgressCard.classList.remove('hidden');
          if (elems.btnRender) elems.btnRender.classList.add('hidden');
          if (elems.btnCancel) elems.btnCancel.classList.remove('hidden');
          if (elems.progressPercent) elems.progressPercent.textContent = `${job.percent}%`;
          if (elems.progressBarFill) elems.progressBarFill.style.width = `${job.percent}%`;
          if (elems.progressStage) elems.progressStage.textContent = job.stage;
          if (elems.progressFileInfo) elems.progressFileInfo.textContent = `[${job.current_file_index}/${job.total_files}] ${job.current_file_name || ''}`;
          if (elems.progressElapsed) elems.progressElapsed.textContent = job.elapsed_formatted;
          if (elems.progressEta) elems.progressEta.textContent = job.eta_formatted;
        } else if (data.type === 'job_completed') {
          const job = data.job;
          state.activeJob = null;
          resetRenderUI();
          updateMaRenderButton();
          if (job.status === 'completed') {
            if (elems.renderCompletedBanner) elems.renderCompletedBanner.classList.remove('hidden');
            if (elems.renderTotalTime) elems.renderTotalTime.textContent = `Render completed in ${job.elapsed_formatted}`;
            showToast(`Render finished in ${job.elapsed_formatted}!`, 'success');
            if (job.output_files && job.output_files.length > 0 && elems.videoRight) {
              let outPath = job.output_files[0];
              if (state.activeTab === 'matanyone') {
                const gs = job.output_files.find(f => f.endsWith('_greenscreen.mp4'));
                const matte = job.output_files.find(f => f.endsWith('_matte.mp4'));
                outPath = gs || matte || outPath;
                populateMaResultWindows(job.output_files);
              }
              elems.videoRight.src = `/api/stream/video?path=${encodeURIComponent(outPath)}`;
              elems.videoRight.load();
              if (elems.placeholderRight) elems.placeholderRight.classList.add('hidden');
            }
            if (state.isOutputDropdownOpen) {
              loadOutputVideos(false);
            }
          } else {
            showToast(`Render failed: ${job.error_message || 'Unknown error'}`, 'error');
          }
        }
      } catch (e) {
        console.warn('WS parse error:', e);
      }
    };

    ws.onclose = () => setTimeout(connectWebSocket, 2000);
  }

  // Initialize
  loadSystemInfo();
  loadVideos();
  connectWebSocket();
});
