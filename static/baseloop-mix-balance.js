/* BaseLoop Studio — Beatbox-referenced rhythm mix for real-bass playback */
(() => {
    'use strict';

    const BL = window.BaseLoopEnhanced;
    if (!BL || typeof Tone === 'undefined') return;

    const VERSION = '20260820.3';
    const BEATBOX_REFERENCE = 'alibowbow/beatbox acoustic layered synthesis';

    // Real-bass samples carry considerably more sustained low-frequency energy
    // than the procedural fallback. Calibrate that path before it reaches the
    // shared master instead of trying to overpower it with the click track.
    const BASS_REAL_TRIM = 0.50;
    const BASS_FALLBACK_TRIM = 0.72;
    const DRUM_MAKEUP = 1.85;
    const CLICK_OUTPUT = 0.58;

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

    function realBassEnabled() {
        try { return typeof __useRealBass !== 'undefined' && Boolean(__useRealBass); }
        catch (_) { return true; }
    }

    function variantScale(variant, amount = 0.012) {
        return variant === 'b' ? 1 - amount : 1 + amount * 0.35;
    }

    // -----------------------------------------------------------------
    // Bass calibration: persistent trim + a very short rhythm duck.
    // The user-facing Bass fader still controls the native bus gain.
    // -----------------------------------------------------------------
    const nativeBassBus = makeBassBus;
    makeBassBus = function beatboxBalancedBassBus(destination, tone) {
        const duck = new Tone.Gain(1);
        const trim = new Tone.Gain(realBassEnabled() ? BASS_REAL_TRIM : BASS_FALLBACK_TRIM);
        duck.chain(trim, destination);

        const bus = nativeBassBus(duck, tone);
        bus._duckGain = duck;
        bus._balanceTrim = trim;
        bus.nodes = [...(bus.nodes || []), duck, trim];
        return bus;
    };

    function duckBassForHit(type, time) {
        if (type !== 'k' && type !== 's') return;
        try {
            const param = __toneNodes?.bassBus?._duckGain?.gain;
            if (!param) return;
            const at = Math.max(Number(time) || 0, Tone.now());
            const depth = type === 'k' ? 0.52 : 0.78;
            const hold = type === 'k' ? 0.010 : 0.006;
            const release = type === 'k' ? 0.125 : 0.082;

            if (typeof param.cancelAndHoldAtTime === 'function') param.cancelAndHoldAtTime(at);
            else param.cancelScheduledValues(at);
            param.setValueAtTime(1, at);
            param.linearRampToValueAtTime(depth, at + 0.0035);
            param.setValueAtTime(depth, at + hold);
            param.exponentialRampToValueAtTime(1, at + release);
        } catch (_) {}
    }

    // -----------------------------------------------------------------
    // Beatbox-inspired acoustic kit.
    // Reference structure:
    //   kick  = beater click + swept shell body + sub resonance
    //   snare = two head modes + two separate wire/noise bands
    //   hats  = inharmonic metal bank + white-noise air
    // Body and transients use parallel buses so sustained real bass cannot
    // swallow the attacks before they reach the master limiter.
    // -----------------------------------------------------------------
    function createBeatboxDrumKit(destination) {
        const bodyInput = new Tone.Gain(1);
        const transientInput = new Tone.Gain(1);

        const bodyHighpass = new Tone.Filter({ type: 'highpass', frequency: 30, rolloff: -24, Q: 0.45 });
        const bodySaturation = new Tone.Distortion({ distortion: 0.16, oversample: '2x', wet: 0.24 });
        const bodyCompressor = new Tone.Compressor({ threshold: -14, ratio: 4, attack: 0.003, release: 0.22, knee: 18 });
        const bodyMakeup = new Tone.Gain(1.16);

        const transientHighpass = new Tone.Filter({ type: 'highpass', frequency: 720, rolloff: -12, Q: 0.35 });
        const transientPresence = new Tone.Filter({ type: 'peaking', frequency: 3200, Q: 0.72, gain: 3.2 });
        const transientMakeup = new Tone.Gain(1.28);

        const sum = new Tone.Gain(1);
        const drumEq = new Tone.EQ3({ low: 1.7, mid: 1.3, high: 2.2, lowFrequency: 118, highFrequency: 4300 });
        const userGain = new Tone.Gain(BL.sourceGain(BL.state.drums, BL.DEFAULTS.drums));
        const fixedMakeup = new Tone.Gain(DRUM_MAKEUP);
        const limiter = new Tone.Limiter(-2.2);

        bodyInput.chain(bodyHighpass, bodySaturation, bodyCompressor, bodyMakeup, sum);
        transientInput.chain(transientHighpass, transientPresence, transientMakeup, sum);
        sum.chain(drumEq, userGain, fixedMakeup, limiter, destination);

        // Kick — Beatbox acoustic recipe: 3.2 kHz beater, 165→58 Hz shell,
        // and a longer 60→42 Hz sub body. Tone.MembraneSynth supplies the
        // pitch sweep while a filtered white-noise layer supplies the beater.
        const kickBody = new Tone.MembraneSynth({
            pitchDecay: 0.035,
            octaves: 2.4,
            oscillator: { type: 'sine' },
            envelope: { attack: 0.001, decay: 0.23, sustain: 0.01, release: 0.08 },
            volume: -3.5,
        }).connect(bodyInput);
        const kickSub = new Tone.MembraneSynth({
            pitchDecay: 0.12,
            octaves: 0.9,
            oscillator: { type: 'sine' },
            envelope: { attack: 0.001, decay: 0.48, sustain: 0.008, release: 0.12 },
            volume: -6.5,
        }).connect(bodyInput);
        const kickClickFilter = new Tone.Filter({ type: 'bandpass', frequency: 3200, Q: 0.8, rolloff: -12 });
        const kickClick = new Tone.NoiseSynth({
            noise: { type: 'white', playbackRate: 1.05 },
            envelope: { attack: 0.001, decay: 0.012, sustain: 0, release: 0.006 },
            volume: -8.5,
        }).connect(kickClickFilter);
        kickClickFilter.connect(transientInput);

        // Snare — two membrane modes (about 186/349 Hz) and two differently
        // filtered white-noise layers for the wires and upper snap.
        const snareHeadLow = new Tone.MembraneSynth({
            pitchDecay: 0.012,
            octaves: 0.35,
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.001, decay: 0.15, sustain: 0, release: 0.035 },
            volume: -7,
        }).connect(bodyInput);
        const snareHeadHigh = new Tone.MembraneSynth({
            pitchDecay: 0.008,
            octaves: 0.22,
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.001, decay: 0.09, sustain: 0, release: 0.025 },
            volume: -11,
        }).connect(bodyInput);
        const snareWireFilter = new Tone.Filter({ type: 'highpass', frequency: 1300, Q: 0.45, rolloff: -12 });
        const snareWire = new Tone.NoiseSynth({
            noise: { type: 'white', playbackRate: 1.08 },
            envelope: { attack: 0.001, decay: 0.16, sustain: 0, release: 0.03 },
            volume: -5.5,
        }).connect(snareWireFilter);
        snareWireFilter.connect(transientInput);
        const snareAirFilter = new Tone.Filter({ type: 'bandpass', frequency: 5200, Q: 0.6, rolloff: -12 });
        const snareAir = new Tone.NoiseSynth({
            noise: { type: 'white', playbackRate: 1.16 },
            envelope: { attack: 0.001, decay: 0.23, sustain: 0, release: 0.04 },
            volume: -8.5,
        }).connect(snareAirFilter);
        snareAirFilter.connect(transientInput);

        // Hats — Tone.MetalSynth provides the Beatbox-style inharmonic bank;
        // a high-passed white-noise layer keeps the attack readable on phones.
        const hasMetalSynth = typeof Tone.MetalSynth === 'function';
        const closedMetal = hasMetalSynth ? new Tone.MetalSynth({
            frequency: 230,
            harmonicity: 5.1,
            modulationIndex: 28,
            resonance: 10500,
            octaves: 1.35,
            envelope: { attack: 0.001, decay: 0.055, release: 0.018 },
            volume: -13,
        }).connect(transientInput) : null;
        const openMetal = hasMetalSynth ? new Tone.MetalSynth({
            frequency: 220,
            harmonicity: 5.1,
            modulationIndex: 30,
            resonance: 9800,
            octaves: 1.55,
            envelope: { attack: 0.001, decay: 0.24, release: 0.11 },
            volume: -16,
        }).connect(transientInput) : null;
        const hatNoiseFilter = new Tone.Filter({ type: 'highpass', frequency: 9000, Q: 0.3, rolloff: -12 });
        const closedNoise = new Tone.NoiseSynth({
            noise: { type: 'white', playbackRate: 1.42 },
            envelope: { attack: 0.001, decay: 0.048, sustain: 0, release: 0.012 },
            volume: -17,
        }).connect(hatNoiseFilter);
        const openNoise = new Tone.NoiseSynth({
            noise: { type: 'white', playbackRate: 1.34 },
            envelope: { attack: 0.001, decay: 0.22, sustain: 0.01, release: 0.08 },
            volume: -19,
        }).connect(hatNoiseFilter);
        hatNoiseFilter.connect(transientInput);

        let disposed = false;
        const allNodes = [
            kickBody, kickSub, kickClick, kickClickFilter,
            snareHeadLow, snareHeadHigh, snareWire, snareWireFilter, snareAir, snareAirFilter,
            closedMetal, openMetal, closedNoise, openNoise, hatNoiseFilter,
            bodyInput, transientInput, bodyHighpass, bodySaturation, bodyCompressor,
            bodyMakeup, transientHighpass, transientPresence, transientMakeup,
            sum, drumEq, userGain, fixedMakeup, limiter,
        ];
        const disposeAll = () => {
            if (disposed) return;
            disposed = true;
            disposeNodes(allNodes);
        };

        const kit = {
            kick: {
                trigger(time, velocity, variant = 'a') {
                    const pitch = variantScale(variant, 0.009);
                    kickClick.triggerAttackRelease(0.014, time, velocity * 0.82);
                    kickBody.triggerAttackRelease(58 * pitch, '8n', time, velocity);
                    kickSub.triggerAttackRelease(42 * pitch, '4n', time, velocity * 0.82);
                },
                dispose: disposeAll,
            },
            snare: {
                trigger(time, velocity, ghost = false, variant = 'a') {
                    const pitch = variantScale(variant, 0.008);
                    const v = ghost ? velocity * 0.48 : velocity;
                    snareHeadLow.triggerAttackRelease(186 * pitch, ghost ? '64n' : '32n', time, v * 0.78);
                    if (!ghost) snareHeadHigh.triggerAttackRelease(349 * pitch, '64n', time, v * 0.52);
                    snareWire.triggerAttackRelease(ghost ? 0.06 : 0.17, time, v * (ghost ? 0.62 : 1));
                    if (!ghost) snareAir.triggerAttackRelease(0.24, time, v * 0.82);
                },
                dispose: disposeAll,
            },
            hat: {
                trigger(time, velocity, open = false) {
                    if (!open) {
                        try { openMetal?.triggerRelease(time); } catch (_) {}
                        try { openNoise.triggerRelease(time); } catch (_) {}
                        closedMetal?.triggerAttackRelease(0.065, time, velocity * 0.92);
                        closedNoise.triggerAttackRelease(0.052, time, velocity * 0.72);
                    } else {
                        openMetal?.triggerAttackRelease(0.28, time, velocity * 0.82);
                        openNoise.triggerAttackRelease(0.24, time, velocity * 0.68);
                    }
                },
                dispose: disposeAll,
            },
        };

        Object.defineProperty(kit, '_gain', { value: userGain, enumerable: false });
        Object.defineProperty(kit, '_makeup', { value: fixedMakeup, enumerable: false });
        Object.defineProperty(kit, '_beatboxReference', { value: BEATBOX_REFERENCE, enumerable: false });
        return kit;
    }

    makeDrumKit = createBeatboxDrumKit;
    triggerDrum = function beatboxTriggerDrum(kit, encoded, time) {
        const [type = 'h', rawVelocity = '.5', variant = 'a'] = String(encoded).split('|');
        const velocity = BL.clamp(rawVelocity, 0.08, 1);

        if (type === 'k') kit.kick.trigger(time, velocity, variant);
        else if (type === 's') kit.snare.trigger(time, velocity, false, variant);
        else if (type === 'g') kit.snare.trigger(time, velocity, true, variant);
        else if (type === 'o') kit.hat.trigger(time, velocity, true);
        else kit.hat.trigger(time, velocity, false);

        duckBassForHit(type, time);
    };

    // -----------------------------------------------------------------
    // Metronome: deliberately quieter than the drum backbeat. It remains a
    // timing guide, not the loudest instrument in the mix.
    // -----------------------------------------------------------------
    makeMetronome = function restrainedMetronome(destination) {
        const userGain = new Tone.Gain(BL.sourceGain(BL.state.click, BL.DEFAULTS.click));
        const fixedGain = new Tone.Gain(CLICK_OUTPUT);
        const highpass = new Tone.Filter({ type: 'highpass', frequency: 900, rolloff: -12, Q: 0.28 });
        const softPresence = new Tone.Filter({ type: 'peaking', frequency: 2200, Q: 0.82, gain: 1.2 });
        const limiter = new Tone.Limiter(-4);
        const synth = new Tone.Synth({
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.001, decay: 0.026, sustain: 0, release: 0.014 },
            volume: -13,
        });
        synth.chain(highpass, softPresence, userGain, fixedGain, limiter, destination);

        let disposed = false;
        return {
            _gain: userGain,
            triggerAttackRelease(note, _duration, time) {
                const accent = String(note) === 'C6';
                synth.triggerAttackRelease(accent ? 'G6' : 'D6', '64n', time, accent ? 0.68 : 0.40);
            },
            dispose() {
                if (disposed) return;
                disposed = true;
                disposeNodes([synth, highpass, softPresence, userGain, fixedGain, limiter]);
            },
        };
    };

    // Disable the shared glue compression. Bass sustain was making it reduce
    // the whole mix, so even a boosted drum bus could disappear behind it.
    // Peak safety remains handled by the existing limiter plus the drum limiter.
    function retuneMastering() {
        try {
            const mastering = __toneNodes?._blMastering;
            if (!mastering?.length) return;
            const [, balance, glue] = mastering;
            if (balance) {
                ramp(balance.low, -1.6);
                ramp(balance.mid, 0.65);
                ramp(balance.high, 0.35);
            }
            if (glue) {
                ramp(glue.threshold, 0);
                ramp(glue.ratio, 1);
                ramp(glue.attack, 0.012);
                ramp(glue.release, 0.09);
            }
        } catch (error) {
            console.debug('[BaseLoop Beatbox mix] mastering retune skipped', error);
        }
    }

    const nativeStartEngine = startEngine;
    startEngine = async function beatboxBalancedStartEngine() {
        const result = await nativeStartEngine.apply(this, arguments);
        if (result) {
            retuneMastering();
            BL.applyMix();
        }
        return result;
    };

    document.addEventListener('DOMContentLoaded', () => {
        document.documentElement.dataset.baseloopMixBalance = VERSION;
        document.documentElement.dataset.baseloopDrumEngine = 'beatbox-layered';
        const copy = document.querySelector('.bl-audio-title__copy small');
        if (copy) copy.textContent = 'Beatbox 레이어 드럼 · 낮춘 클릭 · 리얼베이스 공간 분리';
        window.dispatchEvent(new CustomEvent('baseloop:mix-balanced', {
            detail: { version: VERSION, drumEngine: 'beatbox-layered' },
        }));
    });

    window.BaseLoopBeatboxMix = Object.freeze({
        VERSION,
        BEATBOX_REFERENCE,
        BASS_REAL_TRIM,
        DRUM_MAKEUP,
        CLICK_OUTPUT,
    });

    console.info(`[BaseLoop Beatbox mix] ${VERSION} ready`);
})();
