/* BaseLoop Studio — mix-balance correction for real bass, drums and click */
(() => {
    'use strict';

    const BL = window.BaseLoopEnhanced;
    if (!BL || typeof Tone === 'undefined') return;

    const VERSION = '20260820.2';
    const BASS_TRIM = 0.63;       // Keep the real bass present without masking the rhythm section.
    const DRUM_OUTPUT = 1.36;     // Restore the drum bus to the foreground after the darker v1 voicing.
    const CLICK_OUTPUT = 2.35;    // A click must remain audible above the bass at normal listening levels.

    const disposeNodes = nodes => nodes.forEach(node => {
        try { node?.dispose?.(); } catch (_) {}
    });

    function ramp(param, value, seconds = 0.025) {
        if (!param) return;
        try {
            if (typeof param.rampTo === 'function') param.rampTo(value, seconds);
            else param.value = value;
        } catch (_) {
            try { param.value = value; } catch (_) {}
        }
    }

    // -----------------------------------------------------------------
    // Bass: fixed calibration trim plus a dedicated ducking gain.
    // The user-facing Bass slider remains untouched and keeps its range.
    // -----------------------------------------------------------------
    const nativeBassBus = makeBassBus;
    makeBassBus = function balancedBassBus(destination, tone) {
        const duck = new Tone.Gain(1);
        const trim = new Tone.Gain(BASS_TRIM);
        duck.chain(trim, destination);

        const bus = nativeBassBus(duck, tone);
        bus._duckGain = duck;
        bus._balanceTrim = trim;
        bus.nodes = [...(bus.nodes || []), duck, trim];
        return bus;
    };

    // -----------------------------------------------------------------
    // Drums: preserve the upgraded kit, but restore punch and articulation.
    // The old enhanced bus heavily attenuated the upper band, which made the
    // snare and hats disappear behind the real-bass samples.
    // -----------------------------------------------------------------
    const nativeDrumKit = makeDrumKit;
    makeDrumKit = function balancedDrumKit(destination) {
        const input = new Tone.Gain(1);
        const subCut = new Tone.Filter({ type: 'highpass', frequency: 38, rolloff: -24, Q: 0.5 });
        const kickPunch = new Tone.Filter({ type: 'peaking', frequency: 96, Q: 0.82, gain: 3.4 });
        const snarePresence = new Tone.Filter({ type: 'peaking', frequency: 1850, Q: 0.78, gain: 4.6 });
        const hatAir = new Tone.Filter({ type: 'highshelf', frequency: 4200, gain: 5.4 });
        const output = new Tone.Gain(DRUM_OUTPUT);
        input.chain(subCut, kickPunch, snarePresence, hatAir, output, destination);

        const kit = nativeDrumKit(input);
        const extras = [input, subCut, kickPunch, snarePresence, hatAir, output];
        let extrasDisposed = false;
        const disposeExtras = () => {
            if (extrasDisposed) return;
            extrasDisposed = true;
            disposeNodes(extras);
        };

        Object.values(kit).forEach(part => {
            if (!part || typeof part.dispose !== 'function') return;
            const nativeDispose = part.dispose.bind(part);
            part.dispose = () => {
                try { nativeDispose(); } finally { disposeExtras(); }
            };
        });
        Object.defineProperty(kit, '_balanceNodes', { value: extras, enumerable: false });
        return kit;
    };

    // Short ducking on kick/snare lets their attack pass without making the
    // bass audibly pump. It affects only the hidden calibration gain.
    const nativeTriggerDrum = triggerDrum;
    triggerDrum = function balancedTriggerDrum(kit, encoded, time) {
        const result = nativeTriggerDrum.apply(this, arguments);
        const type = String(encoded || '').split('|')[0];
        if (type !== 'k' && type !== 's') return result;

        try {
            const param = __toneNodes?.bassBus?._duckGain?.gain;
            if (!param) return result;
            const at = Math.max(Number(time) || 0, Tone.now());
            const depth = type === 'k' ? 0.68 : 0.84;
            const release = type === 'k' ? 0.115 : 0.075;
            if (typeof param.cancelAndHoldAtTime === 'function') param.cancelAndHoldAtTime(at);
            else param.cancelScheduledValues(at);
            param.setValueAtTime(1, at);
            param.linearRampToValueAtTime(depth, at + 0.004);
            param.exponentialRampToValueAtTime(1, at + release);
        } catch (_) {}
        return result;
    };

    // -----------------------------------------------------------------
    // Metronome: restore an articulate, short click. A triangle wave carries
    // useful harmonics, unlike the previous sine click that vanished in bass.
    // -----------------------------------------------------------------
    makeMetronome = function balancedMetronome(destination) {
        const userGain = new Tone.Gain(BL.sourceGain(BL.state.click, BL.DEFAULTS.click));
        const fixedGain = new Tone.Gain(CLICK_OUTPUT);
        const highpass = new Tone.Filter({ type: 'highpass', frequency: 680, rolloff: -12, Q: 0.35 });
        const presence = new Tone.Filter({ type: 'peaking', frequency: 2350, Q: 0.9, gain: 4.2 });
        const limiter = new Tone.Limiter(-2.5);
        const synth = new Tone.Synth({
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.001, decay: 0.032, sustain: 0, release: 0.018 },
            volume: -7,
        });
        synth.chain(highpass, presence, userGain, fixedGain, limiter, destination);

        let disposed = false;
        return {
            _gain: userGain,
            triggerAttackRelease(note, _duration, time) {
                const accent = String(note) === 'C6';
                synth.triggerAttackRelease(accent ? 'A6' : 'E6', '64n', time, accent ? 0.96 : 0.72);
            },
            dispose() {
                if (disposed) return;
                disposed = true;
                disposeNodes([synth, highpass, presence, userGain, fixedGain, limiter]);
            },
        };
    };

    // -----------------------------------------------------------------
    // Master bus: the previous glue compressor was reacting too strongly to
    // bass fundamentals and pulling the entire rhythm section down. Keep only
    // gentle peak control so drum/click transients remain in front.
    // -----------------------------------------------------------------
    function retuneMastering() {
        try {
            const mastering = __toneNodes?._blMastering;
            if (!mastering?.length) return;
            const [, balance, glue] = mastering;
            if (balance) {
                ramp(balance.low, -1.5);
                ramp(balance.mid, 0.8);
                ramp(balance.high, 0.65);
            }
            if (glue) {
                ramp(glue.threshold, -7);
                ramp(glue.ratio, 1.3);
                ramp(glue.attack, 0.008);
                ramp(glue.release, 0.095);
            }
        } catch (error) {
            console.debug('[BaseLoop mix balance] mastering retune skipped', error);
        }
    }

    const nativeStartEngine = startEngine;
    startEngine = async function balancedStartEngine() {
        const result = await nativeStartEngine.apply(this, arguments);
        if (result) {
            retuneMastering();
            BL.applyMix();
        }
        return result;
    };

    document.addEventListener('DOMContentLoaded', () => {
        document.documentElement.dataset.baseloopMixBalance = VERSION;
        const copy = document.querySelector('.bl-audio-title__copy small');
        if (copy) copy.textContent = '리얼 베이스 · 드럼 · 클릭 분리 밸런스';
        window.dispatchEvent(new CustomEvent('baseloop:mix-balanced', { detail: { version: VERSION } }));
    });

    console.info(`[BaseLoop mix balance] ${VERSION} ready`);
})();
