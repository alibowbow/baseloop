/* BaseLoop Studio — deterministic groove and upgraded Tone.js signal path */
(() => {
    'use strict';

    const BL = window.BaseLoopEnhanced;
    if (!BL) return;
    const { state, DEFAULTS, clamp, sourceGain } = BL;
    let meter = null;
    let meterFrame = 0;

    function hashString(value) {
        let hash = 2166136261;
        for (const char of String(value)) {
            hash ^= char.charCodeAt(0);
            hash = Math.imul(hash, 16777619);
        }
        return hash >>> 0;
    }
    function noiseAt(seed, index) {
        let x = (seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0;
        x ^= x >>> 16; x = Math.imul(x, 0x7feb352d);
        x ^= x >>> 15; x = Math.imul(x, 0x846ca68b);
        x ^= x >>> 16;
        return (x >>> 0) / 4294967295;
    }
    function profileFor(genre) {
        return {
            rock: { jitter: .0032, duration: .91, laidBack: 0 },
            funk: { jitter: .0024, duration: .79, laidBack: -.001 },
            pop: { jitter: .002, duration: .94, laidBack: 0 },
            jazz: { jitter: .0055, duration: .92, laidBack: .004 },
            blues: { jitter: .0044, duration: .92, laidBack: .003 },
            reggae: { jitter: .0048, duration: .84, laidBack: .005 },
            hiphop: { jitter: .0058, duration: .88, laidBack: .006 },
            random: { jitter: .0038, duration: .9, laidBack: .001 },
        }[genre] || { jitter: .0032, duration: .91, laidBack: 0 };
    }

    function enhancedBuildLoopData() {
        const sequence = document.getElementById('notes_sequence_input')?.value || '';
        const bpm = Math.round(clamp(document.getElementById('bpm_input')?.value || 120, 30, 240));
        const secPerBeat = 60 / bpm;
        const swing = currentSwing();
        const genre = document.getElementById('genre_input')?.value || 'rock';
        const notes = parseNotes(sequence).filter(Boolean);
        const measures = groupIntoMeasures(notes);
        const profile = profileFor(genre);
        const seed = hashString(`${sequence}|${bpm}|${genre}|${swing.toFixed(3)}`);
        const swingTime = beat => {
            const fraction = beat - Math.floor(beat);
            return (beat + (Math.abs(fraction - .5) < .001 ? swing / 6 : 0)) * secPerBeat;
        };
        const velocityFor = (beat, index, noteName) => {
            const inMeasure = ((beat % 4) + 4) % 4;
            let velocity = Math.abs(inMeasure) < .001 ? .98 : Math.abs(beat % 1) < .001 ? .86 : Math.abs((beat % 1) - .5) < .001 ? .76 : .66;
            if (genre === 'funk' && beat % 1 !== 0) velocity += .035;
            if (genre === 'reggae' && Math.floor(inMeasure) % 2 === 1) velocity += .025;
            velocity += ((hashString(noteName) % 11) - 5) * .0015;
            velocity += (noiseAt(seed, index * 5 + 1) - .5) * .07;
            return clamp(velocity, .5, 1);
        };
        let beat = 0;
        let index = 0;
        const bassEvents = [];
        measures.forEach(measure => measure.forEach(note => {
            if (!note.isRest) {
                const jitter = index === 0 ? 0 : (noiseAt(seed, index * 5) - .5) * 2 * profile.jitter;
                const durationVariation = .965 + noiseAt(seed, index * 5 + 2) * .07;
                bassEvents.push({
                    time: Math.max(0, swingTime(beat) + jitter + (index ? profile.laidBack : 0)),
                    note: note.name + note.octave,
                    dur: Math.max(.045, note.duration * secPerBeat * profile.duration * durationVariation),
                    vel: velocityFor(beat, index, note.name), beat, index,
                });
                index += 1;
            }
            beat += note.duration;
        }));
        return { bassEvents, measures, loopBeats: beat, secPerBeat, bpm, swing, swingTime, genre };
    }

    const PATTERNS = Object.freeze({
        rock: { k: [[0,.98],[7,.56],[8,.9],[10,.56]], s: [[4,.88],[12,.94]], h: [[0,.55],[2,.44],[4,.57],[6,.43],[8,.55],[10,.43],[12,.58],[14,.46]], o: [[14,.36]] },
        pop: { k: [[0,.92],[8,.86],[11,.48]], s: [[4,.8],[12,.86]], h: [[0,.43],[2,.36],[4,.46],[6,.35],[8,.43],[10,.35],[12,.47],[14,.38]], o: [[14,.28]] },
        funk: { k: [[0,.94],[3,.48],[6,.7],[10,.78],[15,.46]], s: [[4,.78],[12,.88]], g: [[7,.3],[10,.26],[15,.3]], h: [[0,.48],[2,.36],[4,.5],[5,.3],[6,.38],[8,.48],[10,.38],[12,.52],[14,.4]], o: [[7,.34],[15,.4]] },
        jazz: { k: [[0,.42],[10,.3]], s: [[4,.3],[12,.34]], g: [[7,.2],[15,.22]], h: [[0,.3],[4,.28],[8,.31],[12,.3]], o: [[2,.34],[6,.3],[10,.35],[14,.31]] },
        blues: { k: [[0,.82],[8,.72]], s: [[4,.68],[12,.76]], g: [[10,.24]], h: [[0,.42],[2,.34],[4,.45],[6,.33],[8,.42],[10,.34],[12,.46],[14,.36]], o: [[14,.3]] },
        reggae: { k: [[8,.78]], s: [[8,.72]], g: [[4,.24],[12,.25]], h: [[2,.34],[6,.36],[10,.34],[14,.38]], o: [[14,.3]] },
        hiphop: { k: [[0,.94],[6,.66],[10,.82],[15,.44]], s: [[4,.78],[12,.9]], g: [[11,.25]], h: [[0,.34],[2,.3],[4,.36],[6,.3],[8,.34],[9,.25],[10,.31],[11,.25],[12,.38],[14,.32],[15,.28]], o: [[7,.22]] },
        random: { k: [[0,.9],[8,.76]], s: [[4,.76],[12,.82]], h: [[0,.4],[2,.34],[4,.42],[6,.34],[8,.4],[10,.34],[12,.43],[14,.35]] },
    });
    const encodeHit = (type, velocity, variant = '') => `${type}|${clamp(velocity, .08, 1).toFixed(3)}|${variant}`;
    function enhancedBuildDrumEvents(data) {
        const pattern = PATTERNS[data.genre] || PATTERNS.rock;
        const seed = hashString(`${data.genre}|${data.bpm}|${data.measures.length}|drums`);
        const events = [];
        for (let measure = 0; measure < data.measures.length; measure += 1) {
            const base = measure * 4;
            Object.entries(pattern).forEach(([type, hits]) => hits.forEach(([step, velocity], hitIndex) => {
                const drift = (noiseAt(seed, measure * 97 + step * 7 + hitIndex) - .5) * .036;
                events.push({ time: data.swingTime(base + step * .25), type: encodeHit(type, velocity + drift, measure % 2 ? 'b' : 'a') });
            }));
        }
        return events.sort((a, b) => a.time - b.time);
    }
    function disposeNodes(nodes) {
        nodes.forEach(node => { try { node?.dispose?.(); } catch (_) {} });
    }

    function enhancedMakeDrumKit(destination) {
        const gain = new Tone.Gain(sourceGain(state.drums, DEFAULTS.drums));
        const eq = new Tone.EQ3({ low: -1.5, mid: -.8, high: -4.5, lowFrequency: 115, highFrequency: 4200 });
        const compressor = new Tone.Compressor({ threshold: -18, ratio: 2.6, attack: .008, release: .12, knee: 10 });
        gain.chain(eq, compressor, destination);
        const kick = new Tone.MembraneSynth({ pitchDecay: .035, octaves: 5.1, oscillator: { type: 'sine' }, envelope: { attack: .001, decay: .24, sustain: .015, release: .08 }, volume: -7 }).connect(gain);
        const snareFilter = new Tone.Filter({ type: 'bandpass', frequency: 1850, Q: .72, rolloff: -12 });
        const snareNoise = new Tone.NoiseSynth({ noise: { type: 'pink', playbackRate: 1.18 }, envelope: { attack: .001, decay: .115, sustain: 0, release: .025 }, volume: -16 }).connect(snareFilter);
        snareFilter.connect(gain);
        const snareBody = new Tone.MembraneSynth({ pitchDecay: .016, octaves: 1.7, oscillator: { type: 'triangle' }, envelope: { attack: .001, decay: .095, sustain: 0, release: .025 }, volume: -18 }).connect(gain);
        const hatFilter = new Tone.Filter({ type: 'highpass', frequency: 5200, rolloff: -24, Q: .35 });
        const hat = new Tone.NoiseSynth({ noise: { type: 'pink', playbackRate: 1.5 }, envelope: { attack: .001, decay: .026, sustain: 0, release: .012 }, volume: -25 }).connect(hatFilter);
        const openHat = new Tone.NoiseSynth({ noise: { type: 'pink', playbackRate: 1.45 }, envelope: { attack: .001, decay: .18, sustain: .012, release: .08 }, volume: -27 }).connect(hatFilter);
        hatFilter.connect(gain);
        let disposed = false;
        const disposeAll = () => {
            if (disposed) return;
            disposed = true;
            disposeNodes([kick, snareNoise, snareBody, snareFilter, hat, openHat, hatFilter, gain, eq, compressor]);
        };
        const kit = {
            kick: { trigger: (time, velocity) => kick.triggerAttackRelease('C1', '8n', time, velocity), dispose: disposeAll },
            snare: { trigger: (time, velocity, ghost = false) => { const v = ghost ? velocity * .58 : velocity; snareNoise.triggerAttackRelease(ghost ? .055 : .12, time, v); snareBody.triggerAttackRelease(ghost ? 'D2' : 'C2', ghost ? '64n' : '32n', time, v * .72); }, dispose: disposeAll },
            hat: { trigger: (time, velocity, open = false) => (open ? openHat : hat).triggerAttackRelease(open ? .2 : .035, time, velocity), dispose: disposeAll },
        };
        Object.defineProperty(kit, '_gain', { value: gain, enumerable: false });
        return kit;
    }
    function enhancedTriggerDrum(kit, encoded, time) {
        const [type = 'h', rawVelocity = '.5'] = String(encoded).split('|');
        const velocity = clamp(rawVelocity, .08, 1);
        if (type === 'k') kit.kick.trigger(time, velocity);
        else if (type === 's') kit.snare.trigger(time, velocity, false);
        else if (type === 'g') kit.snare.trigger(time, velocity, true);
        else if (type === 'o') kit.hat.trigger(time, velocity, true);
        else kit.hat.trigger(time, velocity, false);
    }
    function enhancedMakeMetronome(destination) {
        const gain = new Tone.Gain(sourceGain(state.click, DEFAULTS.click));
        const filter = new Tone.Filter({ type: 'lowpass', frequency: 2350, rolloff: -12, Q: .45 });
        const synth = new Tone.Synth({ oscillator: { type: 'sine' }, envelope: { attack: .001, decay: .025, sustain: 0, release: .018 }, volume: -12 });
        synth.chain(filter, gain, destination);
        let disposed = false;
        return {
            _gain: gain,
            triggerAttackRelease(note, _duration, time) { const accent = String(note) === 'C6'; synth.triggerAttackRelease(accent ? 'G5' : 'E5', '64n', time, accent ? .7 : .44); },
            dispose() { if (disposed) return; disposed = true; disposeNodes([synth, filter, gain]); },
        };
    }
    function enhancedMakeBassBus(destination, tone) {
        const current = tone || currentTone();
        const userGain = new Tone.Gain(sourceGain(state.bass, DEFAULTS.bass));
        const rumbleCut = new Tone.Filter({ type: 'highpass', frequency: 28, rolloff: -24, Q: .55 });
        const velocityFilter = new Tone.Filter({ type: 'lowpass', frequency: current.fcMax, rolloff: -24, Q: .72 });
        const body = new Tone.Filter({ type: 'peaking', frequency: __bassTone === 'pick' ? 640 : (__bassTone === 'mute' ? 190 : 260), Q: __bassTone === 'pick' ? .9 : .72, gain: __bassTone === 'pick' ? 1.8 : (__bassTone === 'mute' ? 2.4 : 1.2) });
        const eq = new Tone.EQ3({ low: current.eq.low - 1.2, mid: current.eq.mid, high: current.eq.high - .8, lowFrequency: current.eq.lowF, highFrequency: current.eq.highF });
        const compressor = new Tone.Compressor({ threshold: current.comp.th, ratio: Math.max(2.5, current.comp.ratio - .4), attack: Math.max(.004, current.comp.at), release: Math.max(.12, current.comp.rel), knee: 12 });
        const saturation = new Tone.Distortion({ distortion: Math.max(.025, current.sat.dist * .82), oversample: '2x', wet: Math.min(.44, current.sat.wet * .82) });
        const cabinet = new Tone.Filter({ type: 'lowpass', frequency: __bassTone === 'pick' ? 6900 : (__bassTone === 'mute' ? 2600 : 5200), rolloff: -12, Q: .35 });
        userGain.chain(rumbleCut, velocityFilter, body, eq, compressor, saturation, cabinet, destination);
        return { input: userGain, userGain, vfilt: velocityFilter, nodes: [userGain, rumbleCut, velocityFilter, body, eq, compressor, saturation, cabinet] };
    }

    function meterValue() {
        if (!meter) return 0;
        try {
            let value = meter.getValue();
            if (Array.isArray(value)) value = Math.max(...value.map(Number));
            value = Number(value);
            if (!Number.isFinite(value)) return 0;
            return clamp(value < 0 ? Math.pow(10, value / 20) : value, 0, 1);
        } catch (_) { return 0; }
    }
    function setMeter(value) {
        const element = document.querySelector('.bl-output-meter');
        const bar = document.querySelector('.bl-output-meter__bar');
        const percent = Math.round(clamp(value, 0, 1) * 100);
        if (bar) bar.style.width = `${Math.max(2, percent)}%`;
        if (element) element.setAttribute('aria-valuenow', String(percent));
    }
    function stopMeter() {
        if (meterFrame) cancelAnimationFrame(meterFrame);
        meterFrame = 0; meter = null; setMeter(0);
    }
    function startMeter() {
        if (meterFrame) cancelAnimationFrame(meterFrame);
        const frame = () => {
            setMeter(state.muted ? 0 : meterValue());
            meterFrame = requestAnimationFrame(frame);
        };
        meterFrame = requestAnimationFrame(frame);
    }
    function installMasteringChain() {
        if (!__toneNodes || __toneNodes._blMastering) return;
        try {
            const rumble = new Tone.Filter({ type: 'highpass', frequency: 24, rolloff: -24, Q: .45 });
            const balance = new Tone.EQ3({ low: -.7, mid: .15, high: -.35, lowFrequency: 110, highFrequency: 5200 });
            const glue = new Tone.Compressor({ threshold: -15, ratio: 2.1, attack: .028, release: .18, knee: 14 });
            __toneNodes.master.disconnect();
            __toneNodes.master.chain(rumble, balance, glue, __toneNodes.limiter);
            __toneNodes._blMastering = [rumble, balance, glue];
            try {
                meter = new Tone.Meter({ smoothing: .84, normalRange: true });
                glue.connect(meter); __toneNodes._blMeter = meter; startMeter();
            } catch (_) {}
        } catch (error) { console.warn('[BaseLoop enhanced] mastering fallback', error); }
    }

    function processOfflineBuffer(toneBuffer) {
        let buffer;
        try { buffer = typeof toneBuffer?.get === 'function' ? toneBuffer.get() : toneBuffer; } catch (_) { return toneBuffer; }
        if (!buffer?.getChannelData) return toneBuffer;
        const gain = BL.outputGain(state.master);
        const channels = [];
        let peak = 0;
        for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
            const data = buffer.getChannelData(channel); channels.push(data);
            for (let index = 0; index < data.length; index += 1) {
                const sample = Math.tanh(data[index] * gain * 1.035) / Math.tanh(1.035);
                data[index] = sample; peak = Math.max(peak, Math.abs(sample));
            }
        }
        const ceiling = .89125;
        if (peak > ceiling) {
            const scale = ceiling / peak;
            channels.forEach(data => { for (let index = 0; index < data.length; index += 1) data[index] *= scale; });
        }
        return toneBuffer;
    }
    function enhancedMp3(buffer) {
        const sampleRate = buffer.sampleRate;
        const channels = Math.min(2, Math.max(1, buffer.numberOfChannels));
        const leftFloat = buffer.getChannelData(0);
        const rightFloat = channels > 1 ? buffer.getChannelData(1) : leftFloat;
        const left = new Int16Array(buffer.length);
        const right = channels > 1 ? new Int16Array(buffer.length) : null;
        for (let index = 0; index < buffer.length; index += 1) {
            const l = clamp(leftFloat[index], -1, 1); left[index] = l < 0 ? l * 0x8000 : l * 0x7fff;
            if (right) { const r = clamp(rightFloat[index], -1, 1); right[index] = r < 0 ? r * 0x8000 : r * 0x7fff; }
        }
        const encoder = new lamejs.Mp3Encoder(channels, sampleRate, 192);
        const output = [];
        for (let index = 0; index < left.length; index += 1152) {
            const l = left.subarray(index, index + 1152);
            const encoded = channels > 1 ? encoder.encodeBuffer(l, right.subarray(index, index + 1152)) : encoder.encodeBuffer(l);
            if (encoded.length) output.push(new Int8Array(encoded));
        }
        const tail = encoder.flush();
        if (tail.length) output.push(new Int8Array(tail));
        return new Blob(output, { type: 'audio/mpeg' });
    }

    buildLoopData = enhancedBuildLoopData;
    buildDrumEvents = enhancedBuildDrumEvents;
    makeDrumKit = enhancedMakeDrumKit;
    triggerDrum = enhancedTriggerDrum;
    makeMetronome = enhancedMakeMetronome;
    makeBassBus = enhancedMakeBassBus;

    const nativeDispose = disposeEngine;
    disposeEngine = function enhancedDispose() {
        stopMeter();
        try {
            __toneNodes?._blMeter?.dispose();
            if (__toneNodes?._blMastering) disposeNodes(__toneNodes._blMastering);
        } catch (_) {}
        return nativeDispose.apply(this, arguments);
    };
    const nativeStart = startEngine;
    startEngine = async function enhancedStart() {
        const result = await nativeStart.apply(this, arguments);
        if (result) { installMasteringChain(); BL.applyMix(); }
        return result;
    };
    const nativeRender = renderToBuffer;
    renderToBuffer = async function enhancedRender() {
        return processOfflineBuffer(await nativeRender.apply(this, arguments));
    };
    audioBufferToMp3 = enhancedMp3;

    console.info(`[BaseLoop enhanced] audio ${BL.VERSION} ready`);
})();
