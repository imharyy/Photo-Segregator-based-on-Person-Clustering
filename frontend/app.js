/* ═══════════════════════════════════════════════════════════
   Photo Segregator — Frontend Application
   Single-page app logic, API communication, and interactivity
   ═══════════════════════════════════════════════════════════ */

const App = {
  currentPage: 'dashboard',
  sseSource: null,
  configData: null,
  reviewItems: [],
  reviewIndex: 0,

  // ── Initialization ─────────────────────────────────────
  init() {
    this.setupNavigation();
    this.setupUpload();
    this.setupLightbox();
    this.loadDashboard();
    // Restore page from hash
    const hash = location.hash.replace('#', '');
    if (hash) this.navigate(hash, false);
  },

  // ── Navigation ─────────────────────────────────────────
  setupNavigation() {
    document.querySelectorAll('.nav-item[data-page]').forEach(btn => {
      btn.addEventListener('click', () => this.navigate(btn.dataset.page));
    });
  },

  navigate(page, updateHash = true) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    // Show target
    const target = document.getElementById(`page-${page}`);
    if (target) {
      target.classList.add('active');
      this.currentPage = page;
    }
    const navBtn = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (navBtn) navBtn.classList.add('active');
    if (updateHash) location.hash = page;
    // Load page data
    this.onPageLoad(page);
  },

  onPageLoad(page) {
    switch (page) {
      case 'dashboard': this.loadDashboard(); break;
      case 'gallery': this.loadGallery(); break;
      case 'review': this.loadReview(); break;
      case 'heatmap': this.loadHeatmap(); break;
      case 'config': this.loadConfig(); break;
      case 'logs': this.loadLogs(); break;
    }
  },

  // ── Dashboard ──────────────────────────────────────────
  async loadDashboard() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      const s = data.stats || {};
      this.animateCounter('statInputPhotos', s.n_input_photos || 0);
      this.animateCounter('statFaces', s.n_faces || 0);
      this.animateCounter('statClusters', s.n_clusters || 0);
      this.animateCounter('statReview', s.n_review || 0);
      // Update nav badges
      document.getElementById('navClusterCount').textContent = s.n_clusters || 0;
      document.getElementById('navReviewCount').textContent = s.n_review || 0;
      // Pipeline status
      const statusCard = document.getElementById('dashboardPipelineStatus');
      if (data.status === 'running') {
        statusCard.style.display = 'block';
        document.getElementById('dashboardProgressArea').innerHTML = `
          <div class="progress-container">
            <div class="progress-bar-track">
              <div class="progress-bar-fill" style="width:${data.progress}%"></div>
            </div>
            <div class="progress-info">
              <span class="step-name">${data.current_step || ''}</span>
              <span class="step-pct">${data.progress}%</span>
            </div>
          </div>
          <p style="color:var(--text-muted);font-size:0.82rem;margin-top:8px">${data.message || ''}</p>`;
      } else if (data.status === 'complete') {
        statusCard.style.display = 'block';
        const elapsed = data.end_time && data.start_time ? ((data.end_time - data.start_time)).toFixed(1) : '?';
        document.getElementById('dashboardProgressArea').innerHTML = `
          <p style="color:var(--green);font-size:0.9rem">✅ Pipeline complete in ${elapsed}s</p>`;
      } else {
        statusCard.style.display = 'none';
      }
    } catch (e) {
      console.error('Dashboard load error:', e);
    }
  },

  animateCounter(id, target) {
    const el = document.getElementById(id);
    if (!el) return;
    const start = parseInt(el.textContent) || 0;
    if (start === target) { el.textContent = target; return; }
    const duration = 600;
    const startTime = performance.now();
    const tick = (now) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.round(start + (target - start) * eased);
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  },

  // ── Upload ─────────────────────────────────────────────
  setupUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');
    if (!zone || !input) return;

    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('drag-over');
      if (e.dataTransfer.files.length) this.uploadFiles(e.dataTransfer.files);
    });
    input.addEventListener('change', () => {
      if (input.files.length) this.uploadFiles(input.files);
    });
  },

  async uploadFiles(files) {
    const formData = new FormData();
    for (const f of files) formData.append('files', f);

    const list = document.getElementById('fileList');
    list.innerHTML = Array.from(files).map(f => `
      <div class="file-item">
        <span>📷</span>
        <span class="file-name">${f.name}</span>
        <span class="file-size">${(f.size / 1024).toFixed(0)} KB</span>
      </div>`).join('');

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      this.toast(`Uploaded ${data.count} photo(s)`, 'success');
      this.loadDashboard();
    } catch (e) {
      this.toast('Upload failed: ' + e.message, 'error');
    }
  },

  // ── Run Pipeline ───────────────────────────────────────
  async runPipeline() {
    const btn = document.getElementById('runPipelineBtn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Starting...';

    const incremental = document.getElementById('incrementalToggle').checked;
    const card = document.getElementById('pipelineProgressCard');
    card.style.display = 'block';

    // Build step indicators
    const STEPS = [
      'Initialize', 'Discover', 'Detect & Embed', '', '', 'Cache',
      'Semi-supervised', 'UMAP', 'Thresholds', 'Cluster',
      'Refine', 'Heatmap', 'Write Output'
    ];
    document.getElementById('progressSteps').innerHTML = STEPS
      .filter(s => s)
      .map((s, i) => `<div class="progress-step" data-step="${i+1}">${s}</div>`)
      .join('');

    try {
      await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ incremental }),
      });

      // Listen to SSE
      this.connectSSE();
    } catch (e) {
      this.toast('Failed to start pipeline: ' + e.message, 'error');
      btn.disabled = false;
      btn.innerHTML = '🚀 Run Pipeline';
    }
  },

  connectSSE() {
    if (this.sseSource) this.sseSource.close();
    this.sseSource = new EventSource('/api/progress');
    this.sseSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.updateProgress(data);
      if (data.status === 'complete' || data.status === 'error') {
        this.sseSource.close();
        this.sseSource = null;
        const btn = document.getElementById('runPipelineBtn');
        btn.disabled = false;
        btn.innerHTML = '🚀 Run Pipeline';
        if (data.status === 'complete') {
          this.toast('Pipeline complete! 🎉', 'success');
          this.loadDashboard();
        } else {
          this.toast('Pipeline error: ' + (data.error || ''), 'error');
        }
      }
    };
    this.sseSource.onerror = () => {
      // Reconnect logic handled by browser
    };
  },

  updateProgress(data) {
    const fill = document.getElementById('progressBarFill');
    const stepName = document.getElementById('progressStepName');
    const pct = document.getElementById('progressPct');
    const msg = document.getElementById('progressMessage');

    if (fill) fill.style.width = `${data.progress || 0}%`;
    if (stepName) stepName.textContent = data.current_step || '';
    if (pct) pct.textContent = `${data.progress || 0}%`;
    if (msg) msg.textContent = data.message || '';

    // Update step indicators
    const stepNum = data.step_number || 0;
    document.querySelectorAll('.progress-step').forEach(el => {
      const s = parseInt(el.dataset.step);
      el.classList.remove('active', 'done');
      if (s < stepNum) el.classList.add('done');
      else if (s === stepNum) el.classList.add('active');
    });

    // Also update dashboard if visible
    if (this.currentPage === 'dashboard') this.loadDashboard();
  },

  // ── Gallery ────────────────────────────────────────────
  async loadGallery() {
    const container = document.getElementById('galleryContent');
    try {
      const res = await fetch('/api/clusters');
      const data = await res.json();
      const clusters = data.clusters || {};
      const names = Object.keys(clusters);

      if (names.length === 0) {
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">👥</div><h3>No clusters yet</h3>
            <p>Run the pipeline first to detect and group faces.</p>
            <button class="btn btn-primary" onclick="App.navigate('upload')">📤 Upload & Run</button>
          </div>`;
        return;
      }

      let html = '<div class="gallery-grid">';
      for (const name of names) {
        const info = clusters[name];
        const photos = (info.photo_files || []).slice(0, 4);
        const crops = (info.crop_files || []).slice(0, 4);
        const avgConf = info.avg_confidence || 0;
        const confClass = avgConf > 0.65 ? 'high' : avgConf > 0.35 ? 'medium' : 'low';
        const confPct = (avgConf * 100).toFixed(0);

        // Use crops as thumbnails if available, else photos
        const thumbs = crops.length > 0
          ? crops.map(c => `/api/photos/${name}/crops/${c}`)
          : photos.map(p => `/api/photos/${name}/${p}`);

        // Pad to 4
        while (thumbs.length < 4) thumbs.push('');

        html += `
          <div class="person-card" onclick="App.showPerson('${name}')">
            <div class="photo-mosaic">
              ${thumbs.map(t => t
                ? `<img src="${t}" alt="face" loading="lazy">`
                : `<div style="background:var(--bg-input)"></div>`
              ).join('')}
            </div>
            <div class="card-info">
              <div>
                <div class="person-name">${name.replace('_', ' ')}</div>
                <div class="person-meta">${info.face_count || 0} face(s) • ${(info.photo_files || []).length} photo(s)</div>
              </div>
              <span class="confidence-badge ${confClass}">${confPct}%</span>
            </div>
          </div>`;
      }
      html += '</div>';
      container.innerHTML = html;
    } catch (e) {
      container.innerHTML = `<div class="empty-state"><h3>Failed to load clusters</h3><p>${e.message}</p></div>`;
    }
  },

  async showPerson(clusterId) {
    const container = document.getElementById('galleryContent');
    try {
      const res = await fetch(`/api/cluster/${clusterId}/photos`);
      const data = await res.json();

      let html = `
        <div class="person-detail-header">
          <div>
            <h3 style="font-size:1.2rem;font-weight:700">${clusterId.replace('_', ' ')}</h3>
            <p style="color:var(--text-secondary);font-size:0.85rem">${data.photos.length} photo(s) • ${data.crops.length} face crop(s)</p>
          </div>
          <button class="btn btn-secondary btn-sm" onclick="App.loadGallery()">← Back to Gallery</button>
        </div>`;

      if (data.photos.length > 0) {
        html += `<h4 style="margin:16px 0 12px;font-size:0.9rem;color:var(--text-secondary)">📷 Photos</h4>`;
        html += '<div class="photos-grid">';
        for (const p of data.photos) {
          html += `
            <div class="photo-thumb" onclick="App.openLightbox('${p.url}')">
              <img src="${p.url}" alt="${p.name}" loading="lazy">
            </div>`;
        }
        html += '</div>';
      }

      if (data.crops.length > 0) {
        html += `<h4 style="margin:24px 0 12px;font-size:0.9rem;color:var(--text-secondary)">✂️ Face Crops</h4>`;
        html += '<div class="photos-grid">';
        for (const c of data.crops) {
          html += `
            <div class="photo-thumb" onclick="App.openLightbox('${c.url}')">
              <img src="${c.url}" alt="${c.name}" loading="lazy">
            </div>`;
        }
        html += '</div>';
      }

      container.innerHTML = html;
    } catch (e) {
      this.toast('Failed to load person: ' + e.message, 'error');
    }
  },

  // ── Review ─────────────────────────────────────────────
  async loadReview() {
    const container = document.getElementById('reviewContent');
    try {
      const res = await fetch('/api/review');
      const data = await res.json();
      this.reviewItems = data.items || [];
      this.reviewIndex = 0;

      if (this.reviewItems.length === 0) {
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">✅</div><h3>No faces to review</h3>
            <p>All face assignments look good!</p>
          </div>`;
        return;
      }

      this.renderReviewItem(container);
    } catch (e) {
      container.innerHTML = `<div class="empty-state"><h3>Failed to load review queue</h3><p>${e.message}</p></div>`;
    }
  },

  renderReviewItem(container) {
    if (this.reviewIndex >= this.reviewItems.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">🎉</div><h3>All done!</h3>
          <p>You've reviewed all uncertain faces. Run the pipeline again to apply corrections.</p>
        </div>`;
      return;
    }

    const item = this.reviewItems[this.reviewIndex];
    const conf = item.confidence || 0;
    const confPct = (conf * 100).toFixed(1);
    const qualPct = ((item.quality_score || 0) * 100).toFixed(1);
    const confColor = conf > 0.65 ? 'var(--green)' : conf > 0.35 ? 'var(--amber)' : 'var(--red)';
    const clusterName = item.current_label >= 0 ? `Person ${String(item.current_label).padStart(3, '0')}` : 'Noise';

    container.innerHTML = `
      <div class="review-counter">Face ${this.reviewIndex + 1} of ${this.reviewItems.length}</div>
      <div class="review-card">
        <div class="review-crop">
          ${item.crop_url ? `<img src="${item.crop_url}" alt="Face crop">` : '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted)">No crop</div>'}
        </div>
        <div class="review-details">
          <h3>${item.face_id}</h3>
          <dl class="review-meta">
            <dt>Cluster</dt><dd>${clusterName}</dd>
            <dt>Quality</dt><dd>${qualPct}%</dd>
            <dt>Confidence</dt><dd>${confPct}%</dd>
            <dt>Source</dt><dd style="word-break:break-all;font-size:0.75rem">${(item.image_path || '').split('\\\\').pop().split('/').pop()}</dd>
          </dl>
          <div class="confidence-meter">
            <div class="meter-fill" style="width:${confPct}%;background:${confColor}"></div>
          </div>
          <div class="btn-group">
            <button class="btn btn-success btn-sm" onclick="App.reviewAction('${item.face_id}','accept')">✅ Accept</button>
            <button class="btn btn-secondary btn-sm" onclick="App.reviewMovePrompt('${item.face_id}')">📦 Move</button>
            <button class="btn btn-secondary btn-sm" onclick="App.reviewAction('${item.face_id}','new')">➕ New Cluster</button>
            <button class="btn btn-danger btn-sm" onclick="App.reviewAction('${item.face_id}','discard')">🗑️ Discard</button>
            <button class="btn btn-secondary btn-sm" onclick="App.reviewSkip()">⏭️ Skip</button>
          </div>
        </div>
      </div>`;
  },

  async reviewAction(faceId, action, targetCluster) {
    try {
      const body = { action };
      if (targetCluster !== undefined) body.target_cluster = targetCluster;
      await fetch(`/api/review/${faceId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      this.toast(`Face ${action === 'accept' ? 'accepted' : action === 'discard' ? 'discarded' : 'updated'}`, 'success');
      this.reviewIndex++;
      this.renderReviewItem(document.getElementById('reviewContent'));
    } catch (e) {
      this.toast('Review action failed: ' + e.message, 'error');
    }
  },

  reviewMovePrompt(faceId) {
    const target = prompt('Enter target cluster number:');
    if (target !== null && !isNaN(parseInt(target))) {
      this.reviewAction(faceId, 'move', parseInt(target));
    }
  },

  reviewSkip() {
    this.reviewIndex++;
    this.renderReviewItem(document.getElementById('reviewContent'));
  },

  // ── Heatmap ────────────────────────────────────────────
  async loadHeatmap() {
    const container = document.getElementById('heatmapContent');
    try {
      const res = await fetch('/api/heatmap');
      if (res.ok) {
        container.innerHTML = `
          <div class="heatmap-container">
            <img src="/api/heatmap?t=${Date.now()}" alt="Cluster confidence heatmap" onclick="App.openLightbox('/api/heatmap')">
          </div>`;
      } else {
        container.innerHTML = `
          <div class="empty-state">
            <div class="empty-icon">🗺️</div><h3>No heatmap available</h3>
            <p>Run the pipeline to generate the confidence heatmap.</p>
          </div>`;
      }
    } catch (e) {
      container.innerHTML = `<div class="empty-state"><h3>Failed to load heatmap</h3></div>`;
    }
  },

  // ── Config ─────────────────────────────────────────────
  async loadConfig() {
    const container = document.getElementById('configContent');
    try {
      const res = await fetch('/api/config');
      this.configData = await res.json();
      container.innerHTML = this.buildConfigUI(this.configData);
    } catch (e) {
      container.innerHTML = `<div class="empty-state"><h3>Failed to load config</h3><p>${e.message}</p></div>`;
    }
  },

  buildConfigUI(config, prefix = '') {
    let html = '';
    for (const [key, value] of Object.entries(config)) {
      const path = prefix ? `${prefix}.${key}` : key;
      if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
        html += `<div class="config-section">
          <div class="config-section-title">${key.replace(/_/g, ' ')}</div>
          ${this.buildConfigUI(value, path)}
        </div>`;
      } else {
        html += this.buildConfigField(key, value, path);
      }
    }
    return html;
  },

  buildConfigField(key, value, path) {
    const label = key.replace(/_/g, ' ');
    let input;
    if (typeof value === 'boolean') {
      input = `<label class="toggle-switch">
        <input type="checkbox" ${value ? 'checked' : ''} data-config-path="${path}" data-config-type="boolean">
        <span class="slider"></span>
      </label>`;
    } else if (typeof value === 'number') {
      const step = Number.isInteger(value) ? '1' : '0.01';
      input = `<input type="number" value="${value}" step="${step}" data-config-path="${path}" data-config-type="number">`;
    } else if (value === null) {
      input = `<input type="text" value="" placeholder="null (auto)" data-config-path="${path}" data-config-type="nullable">`;
    } else if (Array.isArray(value)) {
      input = `<input type="text" value="${JSON.stringify(value)}" data-config-path="${path}" data-config-type="array">`;
    } else {
      input = `<input type="text" value="${value || ''}" data-config-path="${path}" data-config-type="string">`;
    }
    return `<div class="config-field"><label>${label}</label>${input}</div>`;
  },

  async saveConfig() {
    if (!this.configData) return;
    // Read values from inputs
    document.querySelectorAll('[data-config-path]').forEach(input => {
      const path = input.dataset.configPath.split('.');
      const type = input.dataset.configType;
      let value;
      if (type === 'boolean') value = input.checked;
      else if (type === 'number') value = parseFloat(input.value);
      else if (type === 'nullable') value = input.value === '' ? null : input.value;
      else if (type === 'array') {
        try { value = JSON.parse(input.value); } catch { value = input.value; }
      } else value = input.value;

      // Set nested value
      let obj = this.configData;
      for (let i = 0; i < path.length - 1; i++) {
        obj = obj[path[i]];
      }
      obj[path[path.length - 1]] = value;
    });

    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.configData),
      });
      this.toast('Configuration saved!', 'success');
    } catch (e) {
      this.toast('Failed to save config: ' + e.message, 'error');
    }
  },

  // ── Logs ───────────────────────────────────────────────
  async loadLogs() {
    const viewer = document.getElementById('logViewer');
    try {
      const res = await fetch('/api/logs');
      const data = await res.json();
      if (!data.lines || data.lines.length === 0) {
        viewer.innerHTML = '<div class="empty-state"><div class="empty-icon">📋</div><h3>No logs yet</h3></div>';
        return;
      }
      viewer.innerHTML = data.lines.map(line => {
        let levelClass = '';
        if (line.includes('| INFO')) levelClass = 'log-level-INFO';
        else if (line.includes('| WARNING')) levelClass = 'log-level-WARNING';
        else if (line.includes('| ERROR')) levelClass = 'log-level-ERROR';
        return `<div class="log-line"><span class="${levelClass}">${this.escapeHtml(line)}</span></div>`;
      }).join('');
      viewer.scrollTop = viewer.scrollHeight;
    } catch (e) {
      viewer.innerHTML = `<div class="empty-state"><h3>Failed to load logs</h3></div>`;
    }
  },

  // ── Lightbox ───────────────────────────────────────────
  setupLightbox() {
    const lb = document.getElementById('lightbox');
    lb.addEventListener('click', (e) => {
      if (e.target === lb) this.closeLightbox();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.closeLightbox();
    });
  },

  openLightbox(src) {
    document.getElementById('lightboxImg').src = src;
    document.getElementById('lightbox').classList.add('open');
  },

  closeLightbox() {
    document.getElementById('lightbox').classList.remove('open');
    document.getElementById('lightboxImg').src = '';
  },

  // ── Toast Notifications ────────────────────────────────
  toast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    container.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3500);
  },

  // ── Helpers ────────────────────────────────────────────
  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },
};

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
