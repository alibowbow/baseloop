/* BaseLoop Studio — mixer, continuity and interaction layer */
(() => {
    'use strict';

    const VERSION = '20260821.1';
    const STORAGE_KEY = 'baseloop-enhanced-audio-v1';
    const DRAFT_KEY = 'baseloop-session-draft-v1';
    const DEFAULTS = Object.freeze({
        master: 82, bass: 90, drums: 62, click: 34,
        muted: false, drumsOn: false, metroOn: false,
        realBassOn: true, mixerCollapsed: true,
    });
    const runtime = {
        messageTimer: 0, restartTimer: 0, draftTimer: 0, swingTimer: 0,
        tapTimes: [], playbackBusy: false, generationBusy: false,
        quickDockObserver: null, controls: Object.create(null),
    };

    function clamp(value, min, max) {
        const n = Number(value);
        return Math.min(max, Math.max(min, Number.isFinite(n) ? n : min));
    }
    function parseJSON(value, fallback) {
        try { return JSON.parse(value); } catch (_) { return fallback; }
    }
    function loadState() {
        let saved = {};
        try { saved = parseJSON(localStorage.getItem(STORAGE_KEY), {}) || {}; } catch (_) {}
        return {
            master: clamp(saved.master ?? DEFAULTS.master, 0, 100),
            bass: clamp(saved.bass ?? DEFAULTS.bass, 0, 100),
            drums: clamp(saved.drums ?? DEFAULTS.drums, 0, 100),
            click: clamp(saved.click ?? DEFAULTS.click, 0, 100),
            muted: Boolean(saved.muted ?? DEFAULTS.muted),
            drumsOn: Boolean(saved.drumsOn ?? DEFAULTS.drumsOn),
            metroOn: Boolean(saved.metroOn ?? DEFAULTS.metroOn),
            realBassOn: Boolean(saved.realBassOn ?? DEFAULTS.realBassOn),
            mixerCollapsed: typeof saved.mixerCollapsed === 'boolean' ? saved.mixerCollapsed : DEFAULTS.mixerCollapsed,
        };
    }
    const state = loadState();
    function saveState() {
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (_) {}
    }
    function sourceGain(value, baseline) {
        return value <= 0 ? 0 : Math.pow(value / baseline, 1.35);
    }
    function outputGain(value) {
        return value <= 0 ? 0 : Math.pow(value / DEFAULTS.master, 1.25);
    }
    function gainToDb(gain) {
        return gain <= 0.00001 ? -80 : 20 * Math.log10(gain);
    }
    function rampParam(param, value, seconds = 0.035) {
        if (!param) return;
        try {
            if (typeof param.rampTo === 'function') param.rampTo(value, seconds);
            else param.value = value;
        } catch (_) { try { param.value = value; } catch (_) {} }
    }
    function getDestination() {
        if (typeof Tone === 'undefined') return null;
        try { return typeof Tone.getDestination === 'function' ? Tone.getDestination() : Tone.Destination; }
        catch (_) { return null; }
    }
    function applyMix() {
        const destination = getDestination();
        const out = outputGain(state.master);
        if (destination) {
            try { destination.mute = Boolean(state.muted || out <= 0); } catch (_) {}
            if (destination.volume) rampParam(destination.volume, gainToDb(out), 0.04);
        }
        try {
            if (typeof __toneNodes === 'undefined' || !__toneNodes) return;
            const bass = sourceGain(state.bass, DEFAULTS.bass);
            const drums = sourceGain(state.drums, DEFAULTS.drums);
            const click = sourceGain(state.click, DEFAULTS.click);
            if (__toneNodes.bassBus?.userGain) rampParam(__toneNodes.bassBus.userGain.gain, bass);
            if (__toneNodes.bassPlayer?.volume) rampParam(__toneNodes.bassPlayer.volume, 4 + gainToDb(bass));
            if (__toneNodes.drums?._gain) rampParam(__toneNodes.drums._gain.gain, drums);
            if (__toneNodes.metro?._gain) rampParam(__toneNodes.metro._gain.gain, click);
        } catch (error) { console.debug('[BaseLoop enhanced] mix update skipped', error); }
    }
    function vibrate(pattern = 7) {
        try {
            if (navigator.vibrate && !matchMedia('(prefers-reduced-motion: reduce)').matches) navigator.vibrate(pattern);
        } catch (_) {}
    }

    function ensureToastStack() {
        let stack = document.querySelector('.bl-toast-stack');
        if (stack) return stack;
        stack = document.createElement('div');
        stack.className = 'bl-toast-stack';
        stack.setAttribute('aria-live', 'polite');
        document.body.appendChild(stack);
        return stack;
    }
    function showToast(message, type = 'info', duration = 3200) {
        if (!message) return;
        const stack = ensureToastStack();
        const toast = document.createElement('div');
        toast.className = `bl-toast ${type}`;
        toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
        const icon = document.createElement('span');
        icon.className = 'bl-toast__icon';
        icon.textContent = type === 'success' ? '✓' : type === 'error' ? '!' : type === 'warning' ? '△' : '♪';
        const text = document.createElement('span');
        text.className = 'bl-toast__text';
        text.textContent = String(message).replace(/^\s*[🎉🎶🎸🎚️⚠️🚨❗💾🎹📝🎼🔗🔒🎲✨■▶]+\s*/u, '');
        toast.append(icon, text);
        stack.appendChild(toast);
        while (stack.children.length > 3) stack.firstElementChild?.remove();
        const close = () => {
            if (!toast.isConnected) return;
            toast.classList.add('is-leaving');
            setTimeout(() => toast.remove(), 190);
        };
        toast.addEventListener('click', close, { once: true });
        setTimeout(close, duration);
    }
    function shouldToast(message, type) {
        const text = String(message || '');
        if (!text || /재생을 (시작|정지)|준비됨|로딩/.test(text)) return false;
        return ['error', 'warning', 'success'].includes(type) || /복사|저장|생성/.test(text);
    }

    const genreNames = { rock: 'Rock', funk: 'Funk', pop: 'Pop', jazz: 'Jazz', blues: 'Blues', reggae: 'Reggae', hiphop: 'Hip-hop', random: 'Random' };
    function currentSummary() {
        const bpm = Math.round(clamp(document.getElementById('bpm_input')?.value || 120, 30, 240));
        const genre = genreNames[document.getElementById('genre_input')?.value] || 'Groove';
        const swing = Math.round(clamp(document.getElementById('swing_input')?.value || 0, 0, 80));
        const tone = document.getElementById('bassToneBtn')?.textContent?.trim() || 'Bass';
        return `${bpm} BPM · ${genre}${swing ? ` · Swing ${swing}%` : ''} · ${tone}`;
    }
    function updateSummary() {
        const summary = currentSummary();
        const session = document.querySelector('.bl-session-summary');
        if (session) session.textContent = summary;
        const title = document.querySelector('.bl-quick-copy strong');
        if (title) title.textContent = summary;
        const sub = document.querySelector('.bl-quick-copy small');
        if (sub) sub.textContent = state.muted ? '출력 음소거' : (typeof __isPlaying !== 'undefined' && __isPlaying ? '그루브 재생 중' : '탭하면 바로 재생');
        try {
            if ('mediaSession' in navigator && navigator.mediaSession.metadata) {
                navigator.mediaSession.metadata = new MediaMetadata({ title: 'BaseLoop Studio', artist: 'Bass groove workstation', album: summary });
            }
        } catch (_) {}
    }
    function updateRange(input) {
        if (!input) return;
        const min = Number(input.min) || 0;
        const max = Number(input.max) || 100;
        const pct = ((Number(input.value) - min) / Math.max(1, max - min)) * 100;
        input.style.setProperty('--bl-range-pct', `${clamp(pct, 0, 100)}%`);
        const output = input.closest('.bl-mix-control')?.querySelector('output');
        if (output) output.textContent = `${Math.round(Number(input.value))}%`;
    }
    function syncMuteUI() {
        [document.getElementById('blMuteBtn'), document.getElementById('blQuickMute')].filter(Boolean).forEach(button => {
            button.setAttribute('aria-pressed', String(state.muted));
            button.title = state.muted ? '출력 음소거 해제' : '전체 출력 음소거';
            button.textContent = state.muted ? '🔇' : '🔊';
        });
        updateSummary();
    }
    function setMuted(next, announce = true) {
        state.muted = Boolean(next);
        saveState(); applyMix(); syncMuteUI(); vibrate(7);
        if (announce) showToast(state.muted ? '전체 출력을 음소거했습니다.' : '출력 음소거를 해제했습니다.', 'info', 1800);
    }
    function resetMixer() {
        Object.assign(state, { master: DEFAULTS.master, bass: DEFAULTS.bass, drums: DEFAULTS.drums, click: DEFAULTS.click, muted: false });
        saveState();
        ['master', 'bass', 'drums', 'click'].forEach(key => {
            const input = runtime.controls[key];
            if (input) { input.value = state[key]; updateRange(input); }
        });
        applyMix(); syncMuteUI(); vibrate([7, 22, 7]);
        showToast('스튜디오 믹서를 권장 밸런스로 되돌렸습니다.', 'success');
    }
    function mixControl(key, label) {
        return `<label class="bl-mix-control" for="blMix-${key}"><span>${label}</span><output>${state[key]}%</output><input type="range" id="blMix-${key}" data-mix-key="${key}" min="0" max="100" step="1" value="${state[key]}" aria-label="${label} 볼륨"></label>`;
    }
    function installMixer() {
        const host = document.querySelector('.bl-transport__main');
        if (!host || document.getElementById('blAudioPanel')) return;
        host.insertAdjacentHTML('beforeend', `<section class="bl-audio-panel" id="blAudioPanel" aria-label="스튜디오 믹서">
            <div class="bl-audio-panel__head"><div class="bl-audio-title"><span class="bl-audio-title__mark" aria-hidden="true">≋</span><span class="bl-audio-title__copy"><strong>Studio Mix</strong><small>저역 헤드룸 · 부드러운 드럼 · 피로도 낮은 클릭</small></span></div><div class="bl-audio-actions"><button type="button" class="bl-audio-icon-btn" id="blMuteBtn" aria-label="전체 출력 음소거">🔊</button><button type="button" class="bl-audio-icon-btn" id="blResetMix" aria-label="믹서 초기화" title="권장 밸런스로 초기화">↺</button><button type="button" class="bl-panel-toggle" id="blMixerToggle" aria-expanded="true">접기</button></div></div>
            <div class="bl-audio-panel__body"><div class="bl-mix-grid">${mixControl('master', 'Output')}${mixControl('bass', 'Bass')}${mixControl('drums', 'Drums')}${mixControl('click', 'Click')}</div><div class="bl-audio-panel__foot"><button type="button" class="bl-tap-btn" id="blTapTempo">TAP</button><span class="bl-session-summary">${currentSummary()}</span><span class="bl-output-meter" role="meter" aria-label="출력 레벨" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i class="bl-output-meter__bar"></i></span></div></div>
        </section>`);
        document.querySelectorAll('[data-mix-key]').forEach(input => {
            const key = input.dataset.mixKey;
            runtime.controls[key] = input;
            updateRange(input);
            input.addEventListener('input', () => {
                state[key] = clamp(input.value, 0, 100);
                updateRange(input); saveState(); applyMix();
            });
            input.addEventListener('change', () => vibrate(5));
        });
        document.getElementById('blMuteBtn')?.addEventListener('click', () => setMuted(!state.muted));
        document.getElementById('blResetMix')?.addEventListener('click', resetMixer);
        document.getElementById('blTapTempo')?.addEventListener('click', tapTempo);
        document.getElementById('blMixerToggle')?.addEventListener('click', () => {
            const panel = document.getElementById('blAudioPanel');
            state.mixerCollapsed = !panel.classList.contains('is-collapsed');
            saveState(); syncMixerMode(); vibrate(5);
        });
        syncMuteUI(); syncMixerMode();
    }
    function syncMixerMode() {
        const panel = document.getElementById('blAudioPanel');
        const toggle = document.getElementById('blMixerToggle');
        if (!panel || !toggle) return;
        const collapsed = typeof state.mixerCollapsed === 'boolean' ? state.mixerCollapsed : DEFAULTS.mixerCollapsed;
        panel.classList.toggle('is-collapsed', collapsed);
        toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.textContent = collapsed ? '펼치기' : '접기';
    }
    function tapTempo() {
        const now = performance.now();
        runtime.tapTimes = runtime.tapTimes.filter(time => now - time < 2400);
        runtime.tapTimes.push(now);
        if (runtime.tapTimes.length > 6) runtime.tapTimes.shift();
        const button = document.getElementById('blTapTempo');
        button?.classList.add('is-tapping'); setTimeout(() => button?.classList.remove('is-tapping'), 120); vibrate(5);
        if (runtime.tapTimes.length < 2) { showToast('박자에 맞춰 TAP을 몇 번 눌러주세요.', 'info', 1500); return; }
        const intervals = runtime.tapTimes.slice(1).map((time, index) => time - runtime.tapTimes[index]).sort((a, b) => a - b);
        const useful = intervals.length > 3 ? intervals.slice(1, -1) : intervals;
        const bpm = Math.round(clamp(60000 / (useful.reduce((sum, value) => sum + value, 0) / useful.length), 30, 240));
        const input = document.getElementById('bpm_input');
        if (input) input.value = bpm;
        updateSummary(); scheduleDraft(); scheduleRestart(130);
        try { setTransportStatus(`♩ ${bpm} BPM · 탭 템포`); } catch (_) {}
    }

    function installQuickDock() {
        if (document.querySelector('.bl-quick-dock')) return;
        const dock = document.createElement('div');
        dock.className = 'bl-quick-dock';
        dock.innerHTML = `<button type="button" class="bl-quick-play" id="blQuickPlay" aria-label="재생/정지">▶</button><span class="bl-quick-copy"><strong>${currentSummary()}</strong><small>탭하면 바로 재생</small></span><button type="button" class="bl-quick-action" id="blQuickGenerate" aria-label="악보 자동 생성" title="악보 자동 생성">✨</button><button type="button" class="bl-quick-action" id="blQuickMute" aria-label="전체 출력 음소거">🔊</button>`;
        document.body.appendChild(dock);
        document.getElementById('blQuickPlay')?.addEventListener('click', () => togglePlayback());
        document.getElementById('blQuickGenerate')?.addEventListener('click', () => generateNotes());
        document.getElementById('blQuickMute')?.addEventListener('click', () => setMuted(!state.muted));
        syncQuickPlay(); syncMuteUI();
        const transport = document.querySelector('.bl-transport');
        if (!transport || typeof IntersectionObserver === 'undefined') return;
        runtime.quickDockObserver = new IntersectionObserver(entries => {
            const visible = matchMedia('(max-width: 680px)').matches && !entries[0].isIntersecting;
            dock.classList.toggle('is-visible', visible);
            document.body.classList.toggle('bl-quick-dock-active', visible);
        }, { threshold: 0.08 });
        runtime.quickDockObserver.observe(transport);
    }
    function syncQuickPlay() {
        const button = document.getElementById('blQuickPlay');
        if (!button) return;
        const playing = typeof __isPlaying !== 'undefined' && __isPlaying;
        button.textContent = playing ? '⏸' : '▶';
        button.setAttribute('aria-label', playing ? '재생 정지' : '재생 시작');
        updateSummary();
    }
    function syncToggleAria() {
        ['realBassToggle', 'metronomeToggle', 'drumsToggle', 'learnToggle', 'fretToggle'].forEach(id => {
            const button = document.getElementById(id);
            if (button) button.setAttribute('aria-pressed', String(button.classList.contains('on')));
        });
        const play = document.getElementById('playAudioBtn');
        if (play) play.setAttribute('aria-pressed', String(typeof __isPlaying !== 'undefined' && __isPlaying));
    }
    function restoreAudioToggles() {
        const desired = [
            ['drumsToggle', state.drumsOn, () => toggleDrums()],
            ['metronomeToggle', state.metroOn, () => toggleMetronome()],
            ['realBassToggle', state.realBassOn, () => toggleRealBass()],
        ];
        desired.forEach(([id, wanted, action]) => {
            const button = document.getElementById(id);
            if (button && button.classList.contains('on') !== Boolean(wanted)) action();
        });
        syncToggleAria();
    }

    function scheduleRestart(delay = 260) {
        clearTimeout(runtime.restartTimer);
        runtime.restartTimer = setTimeout(() => {
            try { if (typeof __isPlaying !== 'undefined' && __isPlaying) restartIfPlaying(); } catch (_) {}
        }, delay);
    }
    function scheduleDraft() {
        clearTimeout(runtime.draftTimer);
        runtime.draftTimer = setTimeout(saveDraft, 420);
    }
    function saveDraft() {
        try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ savedAt: Date.now(), version: VERSION, state: collectState() })); } catch (_) {}
    }
    function restoreDraft() {
        if (location.hash.includes('state=')) return false;
        let draft;
        try { draft = parseJSON(localStorage.getItem(DRAFT_KEY), null); } catch (_) { return false; }
        if (!draft?.state || Date.now() - Number(draft.savedAt || 0) > 2592000000) return false;
        try {
            applyState(draft.state); onGenerationModeChange();
            const swing = document.getElementById('swing_input');
            const label = document.getElementById('swingValue');
            if (swing && label) label.textContent = `${swing.value}%`;
            renderSheetMusic(document.getElementById('notes_sequence_input')?.value || '', parseInt(document.getElementById('length_input')?.value || '4', 10), parseInt(document.getElementById('bpm_input')?.value || '120', 10));
            updateSummary(); return true;
        } catch (_) { return false; }
    }
    function installContinuity() {
        const form = document.getElementById('bassGeneratorForm');
        if (!form) return;
        form.addEventListener('input', event => {
            updateSummary(); scheduleDraft();
            if (event.target?.id === 'bpm_input') scheduleRestart(280);
            else if (event.target?.id === 'notes_sequence_input') scheduleRestart(620);
        });
        form.addEventListener('change', event => {
            updateSummary(); scheduleDraft();
            if (['genre_input', 'key_note_input', 'octave_input', 'length_input'].includes(event.target?.id)) scheduleRestart(180);
        });
        document.getElementById('bpm_input')?.addEventListener('blur', event => {
            event.target.value = Math.round(clamp(event.target.value || 120, 30, 240)); updateSummary();
        });
    }
    function flashScore() {
        const score = document.getElementById('notation-area');
        if (!score) return;
        score.classList.remove('bl-score-updated'); void score.offsetWidth;
        score.classList.add('bl-score-updated'); setTimeout(() => score.classList.remove('bl-score-updated'), 720);
    }
    function installKeyboard() {
        document.addEventListener('keydown', event => {
            const target = event.target;
            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target?.tagName?.toUpperCase?.()) || target?.isContentEditable || event.ctrlKey || event.metaKey || event.altKey) return;
            const key = event.key.toLowerCase();
            if (key === 'm') { event.preventDefault(); setMuted(!state.muted); }
            else if (key === 'd') { event.preventDefault(); toggleDrums(); }
            else if (key === 'c') { event.preventDefault(); toggleMetronome(); }
            else if (key === 'r') { event.preventDefault(); toggleRealBass(); }
        });
    }
    function installMediaSession() {
        if (!('mediaSession' in navigator)) return;
        try {
            navigator.mediaSession.metadata = new MediaMetadata({ title: 'BaseLoop Studio', artist: 'Bass groove workstation', album: currentSummary() });
            navigator.mediaSession.setActionHandler('play', () => { if (!__isPlaying) togglePlayback(); });
            navigator.mediaSession.setActionHandler('pause', () => { if (__isPlaying) togglePlayback(); });
            navigator.mediaSession.setActionHandler('stop', () => stopPlayback());
        } catch (_) {}
    }

    const nativeShowMessage = showMessage;
    showMessage = function enhancedShowMessage(message, type = 'info') {
        const result = nativeShowMessage.apply(this, arguments);
        clearTimeout(runtime.messageTimer);
        const area = document.getElementById('messageArea');
        if (shouldToast(message, type)) showToast(message, type, type === 'error' ? 5200 : 3200);
        if (message && type !== 'error') runtime.messageTimer = setTimeout(() => {
            if (area && area.textContent === String(message)) { area.style.display = 'none'; area.textContent = ''; }
        }, type === 'warning' ? 5600 : 4300);
        return result;
    };
    const nativeSetLoading = setLoadingState;
    setLoadingState = function enhancedLoading(elementId, loading) {
        const result = nativeSetLoading.apply(this, arguments);
        const map = typeof SPINNER_TO_BUTTON !== 'undefined' ? SPINNER_TO_BUTTON : {};
        document.getElementById(map[elementId] || '')?.setAttribute('aria-busy', String(Boolean(loading)));
        if (elementId === 'loadingSpinnerAudio') document.body.classList.toggle('bl-transport-busy', Boolean(loading));
        return result;
    };
    const nativeGenerate = generateNotes;
    generateNotes = async function enhancedGenerate() {
        if (runtime.generationBusy) return;
        runtime.generationBusy = true;
        document.getElementById('generateNotesBtn')?.setAttribute('aria-busy', 'true');
        try {
            const result = await nativeGenerate.apply(this, arguments);
            flashScore(); updateSummary(); scheduleDraft(); vibrate([8, 24, 8]); return result;
        } finally {
            runtime.generationBusy = false;
            document.getElementById('generateNotesBtn')?.setAttribute('aria-busy', 'false');
        }
    };
    const nativeTogglePlayback = togglePlayback;
    togglePlayback = async function enhancedTogglePlayback() {
        if (runtime.playbackBusy) return;
        runtime.playbackBusy = true;
        document.body.classList.add('bl-transport-busy');
        document.getElementById('playAudioBtn')?.setAttribute('aria-busy', 'true');
        try {
            const result = await nativeTogglePlayback.apply(this, arguments);
            const playing = typeof __isPlaying !== 'undefined' && __isPlaying;
            document.body.classList.toggle('bl-audio-playing', playing);
            syncQuickPlay(); syncToggleAria(); applyMix(); installMediaSession(); vibrate(playing ? 9 : 5);
            try { if ('mediaSession' in navigator) navigator.mediaSession.playbackState = playing ? 'playing' : 'paused'; } catch (_) {}
            return result;
        } finally {
            runtime.playbackBusy = false;
            document.body.classList.remove('bl-transport-busy');
            document.getElementById('playAudioBtn')?.setAttribute('aria-busy', 'false');
        }
    };
    const nativeStop = stopPlayback;
    stopPlayback = function enhancedStop() {
        const result = nativeStop.apply(this, arguments);
        document.body.classList.remove('bl-audio-playing', 'bl-transport-busy'); syncQuickPlay(); syncToggleAria();
        try { if ('mediaSession' in navigator) navigator.mediaSession.playbackState = 'paused'; } catch (_) {}
        return result;
    };
    const nativePlayIcon = setPlayIcon;
    setPlayIcon = function enhancedPlayIcon(playing) {
        const result = nativePlayIcon.apply(this, arguments);
        document.body.classList.toggle('bl-audio-playing', Boolean(playing)); syncQuickPlay(); syncToggleAria(); return result;
    };
    const nativeStatus = setTransportStatus;
    setTransportStatus = function enhancedStatus() { const result = nativeStatus.apply(this, arguments); updateSummary(); return result; };
    const nativeDrums = toggleDrums;
    toggleDrums = function enhancedDrums() { const result = nativeDrums.apply(this, arguments); state.drumsOn = document.getElementById('drumsToggle')?.classList.contains('on') || false; saveState(); syncToggleAria(); updateSummary(); vibrate(6); return result; };
    const nativeMetro = toggleMetronome;
    toggleMetronome = function enhancedMetro() { const result = nativeMetro.apply(this, arguments); state.metroOn = document.getElementById('metronomeToggle')?.classList.contains('on') || false; saveState(); syncToggleAria(); vibrate(6); return result; };
    const nativeRealBass = toggleRealBass;
    toggleRealBass = function enhancedRealBass() { const result = nativeRealBass.apply(this, arguments); state.realBassOn = document.getElementById('realBassToggle')?.classList.contains('on') || false; saveState(); syncToggleAria(); updateSummary(); vibrate(6); return result; };
    const nativeTone = cycleBassTone;
    cycleBassTone = function enhancedTone() { const result = nativeTone.apply(this, arguments); updateSummary(); vibrate(6); return result; };
    applySwingLive = function enhancedSwing() { updateSummary(); scheduleDraft(); clearTimeout(runtime.swingTimer); runtime.swingTimer = setTimeout(() => scheduleRestart(0), 260); };

    window.BaseLoopEnhanced = { VERSION, state, DEFAULTS, clamp, sourceGain, outputGain, gainToDb, rampParam, saveState, applyMix, showToast, updateSummary };

    document.addEventListener('DOMContentLoaded', () => {
        document.documentElement.dataset.baseloopEnhanced = VERSION;
        document.getElementById('messageArea')?.setAttribute('aria-live', 'polite');
        document.getElementById('transportStatus')?.setAttribute('aria-live', 'polite');
        installMixer(); installQuickDock(); restoreAudioToggles(); applyMix();
        const restored = restoreDraft();
        installContinuity(); installKeyboard(); installMediaSession(); syncToggleAria(); updateSummary();
        if (restored) showToast('지난 작업 상태를 이어서 불러왔습니다.', 'info', 2600);
        addEventListener('resize', () => {
            syncMixerMode(); updateSummary();
            if (!matchMedia('(max-width: 680px)').matches) {
                document.querySelector('.bl-quick-dock')?.classList.remove('is-visible');
                document.body.classList.remove('bl-quick-dock-active');
            }
        }, { passive: true });
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible' && typeof __isPlaying !== 'undefined' && __isPlaying && typeof Tone !== 'undefined') { try { Tone.start(); } catch (_) {} }
        });
        addEventListener('pagehide', saveDraft, { capture: true });
        window.dispatchEvent(new CustomEvent('baseloop:enhanced-ready', { detail: { version: VERSION } }));
        console.info(`[BaseLoop enhanced] core ${VERSION} ready`);
    });
})();
