/* BaseLoop Studio — audible metronome calibration after the Beatbox mix layer */
(() => {
    'use strict';

    const BL = window.BaseLoopEnhanced;
    if (!BL || typeof Tone === 'undefined') return;

    const VERSION = '20260820.4';
    const CLICK_MAKEUP = 1.12;
    const TONE_VOLUME_DB = -9;
    const TICK_VOLUME_DB = -18;

    const disposeNodes = nodes => nodes.forEach(node => {
        try { node?.dispose?.(); } catch (_) {}
    });

    // The v3 correction reduced the click by more than 10 dB and also removed
    // most of its transient information. Restore presence with two quiet layers:
    // a short triangle tone for pitch and a tiny filtered-noise tick for audibility
    // on phone speakers. The user's Click fader remains the final level control.
    makeMetronome = function presentMetronome(destination) {
        const sum = new Tone.Gain(1);
        const userGain = new Tone.Gain(BL.sourceGain(BL.state.click, BL.DEFAULTS.click));
        const fixedGain = new Tone.Gain(CLICK_MAKEUP);
        const limiter = new Tone.Limiter(-3);

        const highpass = new Tone.Filter({
            type: 'highpass',
            frequency: 620,
            rolloff: -12,
            Q: 0.28,
        });
        const presence = new Tone.Filter({
            type: 'peaking',
            frequency: 2600,
            Q: 0.82,
            gain: 3.2,
        });
        const tone = new Tone.Synth({
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.001, decay: 0.032, sustain: 0, release: 0.016 },
            volume: TONE_VOLUME_DB,
        });
        tone.chain(highpass, presence, sum);

        const tickFilter = new Tone.Filter({
            type: 'bandpass',
            frequency: 3600,
            Q: 0.78,
            rolloff: -12,
        });
        const tick = new Tone.NoiseSynth({
            noise: { type: 'white', playbackRate: 1.08 },
            envelope: { attack: 0.001, decay: 0.012, sustain: 0, release: 0.006 },
            volume: TICK_VOLUME_DB,
        });
        tick.chain(tickFilter, sum);

        sum.chain(userGain, fixedGain, limiter, destination);

        let disposed = false;
        return {
            _gain: userGain,
            triggerAttackRelease(note, _duration, time) {
                const accent = String(note) === 'C6';
                tone.triggerAttackRelease(
                    accent ? 'A6' : 'E6',
                    accent ? '32n' : '64n',
                    time,
                    accent ? 0.86 : 0.60,
                );
                tick.triggerAttackRelease(
                    accent ? 0.016 : 0.011,
                    time,
                    accent ? 0.34 : 0.22,
                );
            },
            dispose() {
                if (disposed) return;
                disposed = true;
                disposeNodes([
                    tone, highpass, presence,
                    tick, tickFilter,
                    sum, userGain, fixedGain, limiter,
                ]);
            },
        };
    };

    document.addEventListener('DOMContentLoaded', () => {
        document.documentElement.dataset.baseloopMetronomePresence = VERSION;
        const copy = document.querySelector('.bl-audio-title__copy small');
        if (copy) copy.textContent = 'Beatbox 레이어 드럼 · 선명한 보조 클릭 · 리얼베이스 공간 분리';
        window.dispatchEvent(new CustomEvent('baseloop:metronome-calibrated', {
            detail: { version: VERSION, makeup: CLICK_MAKEUP },
        }));
    });

    window.BaseLoopMetronomePresence = Object.freeze({
        VERSION,
        CLICK_MAKEUP,
        TONE_VOLUME_DB,
        TICK_VOLUME_DB,
    });

    console.info(`[BaseLoop metronome] ${VERSION} ready`);
})();
