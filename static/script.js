/**
 * NEUROPULSE EEG MONITOR
 * Live + Filtered (delayed) display
 */

const CONFIG = {
    SAMPLING_RATE: 256,
    CHUNK_DURATION: 0.2,
    BUFFER_SECONDS: 12,
    LAG_CHUNKS: 3,
    NOISE_LEVEL: 0.8
};

class SignalEngine {
    constructor() {
        this.isActive = true;
        this.buffer = {
            time: [],
            raw: [],
            filteredTime: [],
            filtered: []
        };
        this.filteredQueue = [];
        this.globalTime = 0;
        this.filteredTime = 0;
        this.listeners = [];
        this.noiseMultiplier = 1.0;
        this.loop();
    }

    subscribe(callback) { this.listeners.push(callback); }
    emit(data) { this.listeners.forEach(cb => cb(data)); }

    injectNoiseBurst() {
        this.noiseMultiplier = 6.0;
        setTimeout(() => this.noiseMultiplier = 1.0, 2000);
    }

    async loop() {
        if (!this.isActive) return;
        const start = performance.now();
        try { await this.cycle(); } catch (e) { console.warn(e); }
        const elapsed = performance.now() - start;
        const nextDelay = Math.max(0, 200 - elapsed);
        setTimeout(() => this.loop(), nextDelay);
    }

    async cycle() {
        const t0 = performance.now();
        // 1. Generate clean signal
        const res = await fetch('http://localhost:5000/api/generate-signal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                duration: CONFIG.CHUNK_DURATION,
                sampling_rate: CONFIG.SAMPLING_RATE
            })
        });
        if (!res.ok) throw new Error('Signal generation failed');
        const data = await res.json();
        const base = data.signal;
        const newTime = data.time.map(t => t + this.globalTime);
        this.globalTime += CONFIG.CHUNK_DURATION;

        // 2. Add noise via backend to get metrics
        const noiseRes = await fetch('http://localhost:5000/api/add-noise', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                signal: base,
                noise_type: 'gaussian',
                noise_level: CONFIG.NOISE_LEVEL * this.noiseMultiplier
            })
        });
        let noisy, metrics = {};
        if (noiseRes.ok) {
            const noiseData = await noiseRes.json();
            noisy = noiseData.noisy_signal;
            metrics = noiseData;
        } else {
            // fallback to local noise if backend fails
            noisy = base.map(v => v + (Math.random() - 0.5) * CONFIG.NOISE_LEVEL * this.noiseMultiplier);
        }

        this.appendToBuffer(newTime, noisy, this.buffer.time, this.buffer.raw);

        // 3. Filter signal
        const filteredResult = await this.filterSignal(noisy, base);
        const filteredLatency = filteredResult.latencyMs;

        this.filteredQueue.push({ time: newTime, filtered: filteredResult.signal });
        if (this.filteredQueue.length > CONFIG.LAG_CHUNKS) {
            const delayed = this.filteredQueue.shift();
            this.appendToBuffer(delayed.time, delayed.filtered, this.buffer.filteredTime, this.buffer.filtered);
            if (delayed.time.length > 0) this.filteredTime = delayed.time[delayed.time.length - 1];
        }

        const liveLatency = Math.round(performance.now() - t0);
        const spectrum = this.mockSpectrum();

        // 4. Emit all metrics for UI
        this.emit({
            ...metrics,
            spectrum,
            latencyLive: liveLatency,
            latencyFiltered: filteredLatency
        });
    }

    appendToBuffer(timeArr, signalArr, timeBuffer, signalBuffer) {
        const maxPoints = CONFIG.BUFFER_SECONDS * CONFIG.SAMPLING_RATE;
        timeBuffer.push(...timeArr);
        signalBuffer.push(...signalArr);
        if (timeBuffer.length > maxPoints) timeBuffer.splice(0, timeBuffer.length - maxPoints);
        if (signalBuffer.length > maxPoints) signalBuffer.splice(0, signalBuffer.length - maxPoints);
    }

    async filterSignal(noisy, clean) {
        const t0 = performance.now();
        try {
            const res = await fetch('http://localhost:5000/api/filter-signal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ noisy_signal: noisy, clean_signal: clean })
            });
            if (!res.ok) throw new Error('Filter failed');
            const data = await res.json();
            return {
                signal: data.filtered_signal || this.movingAverage(noisy, 5),
                latencyMs: Math.round(performance.now() - t0)
            };
        } catch (e) {
            return {
                signal: this.movingAverage(noisy, 5),
                latencyMs: Math.round(performance.now() - t0)
            };
        }
    }

    movingAverage(values, windowSize) {
        const result = [];
        let sum = 0;
        for (let i = 0; i < values.length; i++) {
            sum += values[i];
            if (i >= windowSize) sum -= values[i - windowSize];
            const denom = i < windowSize ? i + 1 : windowSize;
            result.push(sum / denom);
        }
        return result;
    }

    calculateSNR(clean, noisy) {
        let signalPower = 0;
        let noisePower = 0;
        for (let i = 0; i < clean.length; i++) {
            signalPower += clean[i] * clean[i];
            const n = noisy[i] - clean[i];
            noisePower += n * n;
        }
        const snr = 10 * Math.log10((signalPower + 1e-9) / (noisePower + 1e-9));
        return snr;
    }

    mockSpectrum() {
        return {
            'Delta': Math.random(),
            'Theta': Math.random(),
            'Alpha': this.noiseMultiplier > 2 ? 0.1 : Math.random() * 2,
            'Beta': this.noiseMultiplier > 2 ? 5.0 : Math.random(),
            'Gamma': Math.random() * 0.5
        };
    }
}

class GraphController {
    constructor(engine) {
        this.engine = engine;
        this.markers = [];
        this.initPlots();
        this.animate();
    }

    addMarker() {
        const now = this.engine.globalTime;
        this.markers.push({
            type: 'line', x0: now, x1: now,
            y0: 0, y1: 1, xref: 'x', yref: 'paper',
            line: { color: '#ff4d4d', width: 2, dash: 'dot' }
        });
        if (this.markers.length > 6) this.markers.shift();
    }

    initPlots() {
        this.initSinglePlot('plot-live', '#2ef2a3', 'RAW');
        this.initSinglePlot('plot-filtered', '#5ac8ff', 'FILTERED');
    }

    initSinglePlot(containerId, color, label) {
        const trace = {
            x: [0],
            y: [0],
            type: 'scattergl',
            mode: 'lines',
            line: { color, width: 1.5 },
            name: label
        };

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, r: 10, b: 30, l: 45 },
            showlegend: false,
            xaxis: { showgrid: true, gridcolor: '#1b2430', color: '#7a8796' },
            yaxis: { showgrid: true, gridcolor: '#1b2430', range: [-5, 5], fixedrange: true, color: '#7a8796' },
            shapes: []
        };

        Plotly.newPlot(containerId, [trace], layout, { displayModeBar: false, responsive: true });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        if (this.engine.buffer.time.length < 10) return;

        const nowLive = this.engine.globalTime;
        const windowSec = 10;
        const liveRange = [nowLive - windowSec, nowLive];

        Plotly.update('plot-live',
            { x: [this.engine.buffer.time], y: [this.engine.buffer.raw] },
            { 'xaxis.range': liveRange, shapes: this.markers }
        );

        const nowFiltered = this.engine.filteredTime || nowLive;
        const filteredRange = [nowFiltered - windowSec, nowFiltered];

        Plotly.update('plot-filtered',
            { x: [this.engine.buffer.filteredTime], y: [this.engine.buffer.filtered] },
            { 'xaxis.range': filteredRange, shapes: this.markers }
        );
    }
}

class UIManager {
    constructor(engine, graph) {
        this.engine = engine;
        this.graph = graph;
        this.bindEvents();
        this.engine.subscribe(data => this.render(data));
        this.updateTime();
    }

    bindEvents() {
        document.getElementById('btn-mark').addEventListener('click', () => {
            this.graph.addMarker();
        });

        document.getElementById('btn-interfere').addEventListener('click', () => {
            this.engine.injectNoiseBurst();
        });
    }


    render(data) {
        // SNR
        if (typeof data.snr !== 'undefined')
            document.getElementById('val-snr').textContent = data.snr.toFixed(2);
        // PSNR
        if (typeof data.psnr !== 'undefined')
            document.getElementById('val-psnr').textContent = data.psnr.toFixed(2);
        // SSIM
        if (typeof data.ssim !== 'undefined')
            document.getElementById('val-ssim').textContent = data.ssim.toFixed(3);
        // MSE
        if (typeof data.mse !== 'undefined')
            document.getElementById('val-mse').textContent = data.mse.toExponential(2);
        // Efficiency
        if (typeof data.efficiency !== 'undefined')
            document.getElementById('val-eff').textContent = data.efficiency.toFixed(3);

        const noiseRatio = (typeof data.snr !== 'undefined') ? Math.pow(10, -data.snr / 10) : 0;
        document.getElementById('val-nsr').textContent = noiseRatio.toFixed(3);
        document.getElementById('latency-live').textContent = `Latency: ${data.latencyLive} ms`;
        document.getElementById('latency-filtered').textContent = `Latency: ${data.latencyFiltered} ms`;

        if (data.spectrum) {
            let pwrMax = 0; let dom = '';
            for (let [k, v] of Object.entries(data.spectrum)) {
                document.getElementById(`pwr-${k.toLowerCase()}`).textContent = v.toFixed(2);
                if (v > pwrMax) { pwrMax = v; dom = k; }
            }
            document.getElementById('val-band').textContent = dom.toUpperCase();

            let state = 'IDLE';
            if (typeof data.snr !== 'undefined' && data.snr < 0) state = 'ARTIFACT';
            else if (dom === 'Alpha') state = 'RELAXED';
            else if (dom === 'Beta') state = 'FOCUSED';

            const el = document.getElementById('val-state');
            el.textContent = state;
            el.style.color = state === 'ARTIFACT' ? '#ff4d4d' : '#ffffff';
        }
    }

    updateTime() {
        setInterval(() => {
            const d = new Date();
            document.getElementById('time-display').textContent = d.toLocaleTimeString();
        }, 1000);
    }
}

const sys = new SignalEngine();
const graph = new GraphController(sys);
const ui = new UIManager(sys, graph);
