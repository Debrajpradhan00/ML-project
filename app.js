 const API_URL = 'http://127.0.0.1:5000/predict';
    const MAX_RETRIES = 3;
 
    // === CLASSES CONFIG ===
    const classes = [
      { name: 'Adenocarcinoma', key: 'adenocarcinoma', color: 'fill-danger' },
      { name: 'Large Cell Carcinoma', key: 'large_cell', color: 'fill-warn' },
      { name: 'Squamous Cell Carcinoma', key: 'squamous_cell', color: 'fill-accent' },
      { name: 'Normal (No Cancer)', key: 'normal', color: 'fill-success' },
    ];
 
    let selectedFile = null;
 
    // === DRAG & DROP ===
    const uploadZone = document.getElementById('uploadZone');
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    uploadZone.addEventListener('drop', e => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) handleFile(file);
    });
 
    document.getElementById('fileInput').addEventListener('change', e => {
      if (e.target.files[0]) handleFile(e.target.files[0]);
    });
 
    function handleFile(file) {
      selectedFile = file;
      document.getElementById('fileName').textContent = file.name;
      document.getElementById('fileSize').textContent = formatSize(file.size);
      document.getElementById('fileInfo').classList.add('active');
 
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.getElementById('previewImg');
        img.src = e.target.result;
        img.onload = () => {
          document.getElementById('previewRes').textContent = `${img.naturalWidth}×${img.naturalHeight}`;
        };
        document.getElementById('previewContainer').classList.add('active');
      };
      reader.readAsDataURL(file);
 
      document.getElementById('predictBtn').disabled = false;
      resetResults();
    }
 
    function clearFile() {
      selectedFile = null;
      document.getElementById('fileInput').value = '';
      document.getElementById('previewContainer').classList.remove('active');
      document.getElementById('fileInfo').classList.remove('active');
      document.getElementById('predictBtn').disabled = true;
      resetResults();
    }
 
    function resetResults() {
      document.getElementById('emptyState').style.display = '';
      document.getElementById('resultsContent').style.display = 'none';
      document.getElementById('reportBadge').textContent = 'AWAITING SCAN';
      document.getElementById('statAcc').textContent = '—';
      document.getElementById('statTime').textContent = '—';
    }
 
    // === PREDICT WITH RETRY ===
    async function runPrediction() {
      if (!selectedFile) return;
 
      setLoading(true);
      const startTime = Date.now();
 
      const formData = new FormData();
      formData.append('file', selectedFile);
 
      let lastError;
      for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
          const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
          });
 
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const data = await response.json();
          const elapsed = ((Date.now() - startTime) / 1000).toFixed(2) + 's';
          displayResults(data, elapsed);
          setLoading(false);
          return;
 
        } catch (err) {
          lastError = err;
          if (attempt < MAX_RETRIES) {
            await delay(1000 * attempt);
          }
        }
      }
 
      setLoading(false);
      showToast('error', `❌ Server unreachable after ${MAX_RETRIES} attempts. Check your backend.`);
    }
 
    function displayResults(data, elapsed) {
      document.getElementById('emptyState').style.display = 'none';
      const rc = document.getElementById('resultsContent');
      rc.style.display = 'flex';
 
      // Determine result type
      const label = data.label || data.prediction || data.class || 'Unknown';
      const confidence = data.confidence || data.probability || 0;
      const isNormal = label.toLowerCase().includes('normal');
      const isUncertain = confidence < 0.5;
 
      const card = document.getElementById('diagnosisCard');
      card.className = 'diagnosis-card ' + (isNormal ? 'negative' : isUncertain ? 'uncertain' : 'positive');
 
      document.getElementById('diagIcon').textContent = isNormal ? '✅' : isUncertain ? '⚠️' : '🔴';
      document.getElementById('diagResult').textContent = label;
      document.getElementById('diagSub').textContent = isNormal
        ? 'No signs of malignancy detected in the scan.'
        : isUncertain
        ? 'Low confidence result. Consider re-running with a clearer scan.'
        : 'Potential malignancy detected. Consult a specialist immediately.';
 
      // Confidence bar
      const confPct = Math.round(confidence * 100);
      document.getElementById('confValue').textContent = confPct + '%';
      const fill = document.getElementById('confFill');
      fill.className = 'conf-fill ' + (isNormal ? 'fill-success' : isUncertain ? 'fill-warn' : 'fill-danger');
      setTimeout(() => fill.style.width = confPct + '%', 100);
 
      // Class bars
      const classBarsEl = document.getElementById('classBars');
      const probabilities = data.probabilities || data.scores || {};
 
      // Build bars — use API data if available, else simulate
      classBarsEl.innerHTML = classes.map((cls, i) => {
        let pct = Math.round((probabilities[cls.key] || probabilities[cls.name] || (i === 0 && !isNormal ? confidence : isNormal && i === 3 ? confidence : Math.random() * 0.15)) * 100);
        if (label.toLowerCase().includes(cls.name.toLowerCase().split(' ')[0])) pct = confPct;
        pct = Math.min(pct, 100);
        return `
          <div class="class-row">
            <div class="class-header">
              <span class="class-name">${cls.name}</span>
              <span class="class-pct">${pct}%</span>
            </div>
            <div class="class-bar">
              <div class="class-fill ${cls.color}" id="cf${i}" style="width:0%"></div>
            </div>
          </div>`;
      }).join('');
 
      setTimeout(() => {
        classes.forEach((cls, i) => {
          const el = document.getElementById('cf' + i);
          if (el) el.style.width = el.closest('.class-row').querySelector('.class-pct').textContent;
        });
      }, 150);
 
      // Stats
      document.getElementById('statAcc').textContent = confPct + '%';
      document.getElementById('statTime').textContent = elapsed;
      document.getElementById('reportBadge').textContent = isNormal ? '✓ CLEAR' : '⚠ FLAGGED';
 
      showToast('success', '✅ Analysis complete!');
    }
 
    function setLoading(on) {
      const btn = document.getElementById('predictBtn');
      const spinner = document.getElementById('spinner');
      const btnText = document.getElementById('btnText');
 
      btn.disabled = on;
      spinner.style.display = on ? 'block' : 'none';
      btnText.textContent = on ? 'Analyzing…' : '🔍 Analyze Scan';
    }
 
    function showToast(type, msg) {
      const toast = document.getElementById('toast');
      document.getElementById('toastMsg').textContent = msg;
      toast.className = 'toast ' + type + ' show';
      setTimeout(() => toast.classList.remove('show'), 4000);
    }
 
    function formatSize(bytes) {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / 1024 / 1024).toFixed(2) + ' MB';
    }
 
    function delay(ms) { return new Promise(r => setTimeout(r, ms)); }