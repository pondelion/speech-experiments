// 試聴ページ: site/manifest.json(ベースモデル → exp の一覧と、条件の表示名)と、
// 各 exp の results/samples/index.json(試聴用の MP3 の一覧)を読み、同じ文章の音声を条件ごとに並べる。
// 新しい exp は manifest.json に 1 項目足すだけで表示される(index.json は各 exp の評価スクリプトが作る)。
"use strict";

const state = { manifest: null, cache: {} };
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (v) => (v == null ? "" : `${(v * 100).toFixed(1)}%`);

function storeGet(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } }
function storeSet(key, v) { try { localStorage.setItem(key, JSON.stringify(v)); } catch { /* 保存できなくても表示はできる */ } }

function findExp(baseId, expId) {
  const base = state.manifest.bases.find((b) => b.id === baseId);
  return base ? { base, exp: base.exps.find((e) => e.id === expId) } : {};
}

function renderNav(route) {
  const html = state.manifest.bases.map((b) => {
    const exps = b.exps.map((e) =>
      `<a class="exp${route.exp === e.id && route.base === b.id ? " active" : ""}" href="#${b.id}/${e.id}">${esc(e.id)}</a>`).join("");
    return `<h2>${esc(b.title)}</h2><a class="base${route.base === b.id && !route.exp ? " active" : ""}" href="#${b.id}">概要</a>${exps}`;
  }).join("");
  $("#nav").innerHTML = `<a class="${route.base ? "" : "active"}" href="#">トップ</a>` + html;
}

function renderHome() {
  const m = state.manifest;
  $("#main").innerHTML = `<h1>${esc(m.title)}</h1><p class="lead">${esc(m.description)}</p>` +
    m.bases.map((b) => `<h2>${esc(b.title)}</h2><p class="lead">${esc(b.summary)}</p><div class="cards">` +
      b.exps.map((e) => `<a class="card" href="#${b.id}/${e.id}"><h3>${esc(e.title)}</h3><p>${esc(e.summary)}</p></a>`).join("") +
      "</div>").join("");
}

function renderBase(base) {
  $("#main").innerHTML = `<h1>${esc(base.title)}</h1><p class="lead">${esc(base.summary)}</p>` +
    `<p class="links"><a href="${state.manifest.repo_url}/tree/main/${base.dir}">README(GitHub)</a></p><div class="cards">` +
    base.exps.map((e) => `<a class="card" href="#${base.id}/${e.id}"><h3>${esc(e.title)}</h3><p>${esc(e.summary)}</p></a>`).join("") +
    "</div>";
}

async function loadSamples(exp) {
  if (!state.cache[exp.id]) {
    const res = await fetch(`${exp.samples}/index.json`);
    if (!res.ok) throw new Error(`${exp.samples}/index.json を読めない(${res.status})`);
    state.cache[exp.id] = await res.json();
  }
  return state.cache[exp.id];
}

// manifest に書いた条件(表示名・CER・既定で選ぶか)と、index.json にだけある条件を合わせた一覧
function conditionList(exp, rows) {
  const groups = (exp.condition_groups || []).map((g) => ({ title: g.title, conditions: g.conditions.slice() }));
  const known = new Set(groups.flatMap((g) => g.conditions.map((c) => c.id)));
  const extra = [...new Set(rows.map((r) => r.condition))].filter((c) => !known.has(c));
  if (extra.length) groups.push({ title: "その他", conditions: extra.map((id) => ({ id, label: id })) });
  return groups;
}

async function renderExp(base, exp) {
  const main = $("#main");
  main.innerHTML = `<h1>${esc(exp.title)}</h1><p class="lead">${esc(exp.summary)}</p>` +
    `<p class="links"><a href="${state.manifest.repo_url}/tree/main/${exp.dir}">README(GitHub)</a></p><p class="muted">読み込み中…</p>`;
  let rows;
  try { rows = await loadSamples(exp); } catch (err) { main.querySelector("p.muted").textContent = err.message; return; }

  const groups = conditionList(exp, rows);
  const all = groups.flatMap((g) => g.conditions);
  const key = `selected:${base.id}/${exp.id}`;
  let selected = storeGet(key) || all.filter((c) => c.default).map((c) => c.id);
  let showSpec = storeGet("showSpec") !== false;  // 既定は表示する(自分で OFF にしたときだけ消す)

  const picker = document.createElement("section");
  picker.className = "picker";
  const tables = document.createElement("div");
  main.querySelector("p.muted").replaceWith(picker, tables);

  const draw = () => {
    picker.innerHTML = `<div class="picker-head"><strong>比べる条件</strong><span class="picker-actions">` +
      `<label class="chip${showSpec ? " on" : ""}"><input type="checkbox" data-spec${showSpec ? " checked" : ""}>スペクトログラム</label>` +
      `<button data-act="default">既定に戻す</button><button data-act="all">全て</button><button data-act="none">なし</button></span></div>` +
      groups.map((g) => `<div class="cgroup"><div class="cgroup-title">${esc(g.title)}</div><div class="chips">` +
        g.conditions.map((c) => {
          const on = selected.includes(c.id);
          const cer = c.cer && c.cer.test != null ? `<span class="cer">${pct(c.cer.test)}</span>` : "";
          return `<label class="chip${on ? " on" : ""}"><input type="checkbox" value="${esc(c.id)}"${on ? " checked" : ""}>${esc(c.label)}${cer}</label>`;
        }).join("") + "</div></div>").join("") +
      `<p class="muted" style="font-size:12px;margin:8px 0 0">数値はテスト 100 文の読み CER(ASR の文字誤り率。低いほど正しく読めている)` +
      (showSpec ? "。スペクトログラムは MP3 からブラウザで計算した log-Mel(80 帯域、下が低い周波数、横幅は音声の長さによらず同じ)" : "") + "</p>";
    tables.innerHTML = Object.entries(state.manifest.sample_groups)
      .map(([gid, g]) => renderTable(gid, g, rows, all, selected, exp.samples, showSpec)).join("");
    if (showSpec) attachSpectrograms(tables);
  };
  picker.addEventListener("change", (ev) => {
    if (ev.target.hasAttribute("data-spec")) {
      showSpec = ev.target.checked;
      storeSet("showSpec", showSpec);
      return draw();
    }
    const id = ev.target.value;
    selected = ev.target.checked ? [...selected, id] : selected.filter((s) => s !== id);
    selected = all.map((c) => c.id).filter((c) => selected.includes(c));  // 表示の順番を一覧の順にそろえる
    storeSet(key, selected);
    draw();
  });
  picker.addEventListener("click", (ev) => {
    const act = ev.target.dataset && ev.target.dataset.act;
    if (!act) return;
    selected = act === "all" ? all.map((c) => c.id) : act === "none" ? [] : all.filter((c) => c.default).map((c) => c.id);
    storeSet(key, selected);
    draw();
  });
  draw();
}

// 文章ごとの行 × 選んだ条件の列の表(その文章の音声が無い条件は「-」)
function renderTable(gid, g, rows, all, selected, samplesDir, showSpec) {
  const inGroup = rows.filter((r) => r.group === gid);
  if (!inGroup.length) return "";
  const present = new Set(inGroup.map((r) => r.condition));
  const cols = all.filter((c) => selected.includes(c.id) && present.has(c.id));
  const head = `<h2>${esc(g.title)}</h2><p class="group-note muted">${esc(g.note || "")}</p>`;
  if (!cols.length) return head + `<p class="muted">選んだ条件には、この文章の音声がない。</p>`;
  const texts = [];
  const byKey = {};
  for (const r of inGroup) {
    if (!byKey[r.id]) { byKey[r.id] = {}; texts.push(r); }
    byKey[r.id][r.condition] = r.file;
  }
  const th = cols.map((c) => {
    const cer = c.cer && c.cer[gid] != null ? `<span class="cer">CER ${pct(c.cer[gid])}</span>` : "";
    return `<th>${esc(c.label)}${cer}</th>`;
  }).join("");
  const body = texts.map((t) => `<tr><td class="text"><span class="id">${esc(t.id)}${t.category && t.category !== "jsut_test" ? ` · ${esc(t.category)}` : ""}</span>${esc(t.text)}</td>` +
    cols.map((c) => {
      const f = byKey[t.id][c.id];
      if (!f) return `<td class="none">-</td>`;
      const src = esc(`${samplesDir}/${f}`);
      const caption = esc(`${c.label} | ${t.id}`);
      const spec = showSpec
        ? `<canvas class="spec" data-src="${src}" data-caption="${caption}" width="1" height="${N_MELS}" title="クリックで拡大"></canvas>` : "";
      return `<td class="audio"><audio controls preload="none" src="${src}"></audio>${spec}</td>`;
    }).join("") + "</tr>").join("");
  return head + `<div class="table-wrap"><table><thead><tr><th>文章</th>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}

// ---- スペクトログラム: MP3 をブラウザで波形に戻し、exp の Mel と同じ設定で log-Mel を計算して canvas に描く ----
const SR = 24000, N_FFT = 1024, HOP = 256, N_MELS = 80, FMIN = 0, FMAX = 12000;  // tts/lib_common/audio.py と同じ
const specCache = new Map();  // 音声のパス → 計算結果の Promise(条件を選び直して表を描き直しても再計算しない)
let melBank = null;

// Slaney 型の Mel 尺度(librosa の既定と同じ): 1kHz までは線形、それより上は対数
const hzToMel = (f) => (f < 1000 ? (3 * f) / 200 : 15 + (27 * Math.log(f / 1000)) / Math.log(6.4));
const melToHz = (m) => (m < 15 ? (200 * m) / 3 : 1000 * Math.exp(((m - 15) * Math.log(6.4)) / 27));

// 三角形の Mel フィルタ N_MELS 個(各帯域の面積をそろえる Slaney の正規化つき)。各 [N_FFT/2 + 1]
function melFilterbank() {
  if (melBank) return melBank;
  const nBins = N_FFT / 2 + 1, lo = hzToMel(FMIN), hi = hzToMel(FMAX);
  const hz = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (N_MELS + 1)));
  melBank = hz.slice(0, N_MELS).map((_, m) => {
    const w = new Float32Array(nBins), norm = 2 / (hz[m + 2] - hz[m]);
    for (let k = 0; k < nBins; k++) {
      const f = (k * SR) / N_FFT;
      w[k] = Math.max(0, Math.min((f - hz[m]) / (hz[m + 1] - hz[m]), (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]))) * norm;
    }
    return w;
  });
  return melBank;
}

// 長さ N(2 のべき乗)の FFT(その場で書き換える、基数 2 の Cooley-Tukey)
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {  // ビット反転の順に並べ替える
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
      }
    }
  }
}

// 波形 → log-Mel(フレームごとに Hann 窓をかけて FFT し、振幅を Mel フィルタでまとめて log を取る)
function logMel(y) {
  const pad = N_FFT / 2, frames = Math.floor(y.length / HOP) + 1, bank = melFilterbank();
  const win = Float64Array.from({ length: N_FFT }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT));
  const out = new Float32Array(frames * N_MELS);  // [フレーム, 帯域]
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), mag = new Float64Array(N_FFT / 2 + 1);
  for (let t = 0; t < frames; t++) {
    for (let n = 0; n < N_FFT; n++) {  // 前後に 0 を詰めて、フレームの中心が t × HOP に来るようにする
      const i = t * HOP + n - pad;
      re[n] = i >= 0 && i < y.length ? y[i] * win[n] : 0;
      im[n] = 0;
    }
    fft(re, im);
    for (let k = 0; k < mag.length; k++) mag[k] = Math.hypot(re[k], im[k]);
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      for (let k = 0; k < mag.length; k++) s += bank[m][k] * mag[k];
      out[t * N_MELS + m] = Math.log(Math.max(s, 1e-5));
    }
  }
  return { mel: out, frames };
}

function getSpec(src) {
  if (!specCache.has(src)) {
    specCache.set(src, fetch(src).then((r) => r.arrayBuffer()).then((buf) =>
      new OfflineAudioContext(1, 1, SR).decodeAudioData(buf)).then((audio) => logMel(audio.getChannelData(0))));
  }
  return specCache.get(src);
}

// 色: magma に近い 5 色を線形につなぐ(暗い = 弱い、明るい = 強い)
const PALETTE = [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]];
function color(v) {
  const x = Math.min(Math.max(v, 0), 1) * (PALETTE.length - 1), i = Math.min(Math.floor(x), PALETTE.length - 2), f = x - i;
  return PALETTE[i].map((c, j) => c + (PALETTE[i + 1][j] - c) * f);
}

function drawSpec(canvas, { mel, frames }) {
  canvas.width = frames;
  canvas.height = N_MELS;
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(frames, N_MELS);
  const max = mel.reduce((a, b) => Math.max(a, b), -Infinity), lo = max - 8;  // 一番強い所から約 70dB 下までを色にする
  for (let t = 0; t < frames; t++) {
    for (let m = 0; m < N_MELS; m++) {
      const [r, g, b] = color((mel[t * N_MELS + m] - lo) / (max - lo));
      const p = ((N_MELS - 1 - m) * frames + t) * 4;  // 下が低い周波数
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// 画面に入った canvas から順に計算して描く(表の全部を一度に計算しない)
function attachSpectrograms(root) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      getSpec(e.target.dataset.src).then((s) => drawSpec(e.target, s))
        .catch(() => { e.target.title = "スペクトログラムを計算できなかった"; });
    }
  }, { rootMargin: "200px" });
  root.querySelectorAll("canvas.spec").forEach((c) => io.observe(c));
}

// スペクトログラムをクリックすると、大きく描き直した図と、その音声の再生ボタンを画面の中央に出す
function openZoom(canvas) {
  const src = canvas.dataset.src;
  const overlay = document.createElement("div");
  overlay.className = "zoom";
  overlay.innerHTML = `<div class="zoom-box" role="dialog" aria-label="スペクトログラムの拡大">` +
    `<div class="zoom-head"><span>${esc(canvas.dataset.caption)}</span><button aria-label="閉じる">×</button></div>` +
    `<canvas class="zoom-spec"></canvas><audio controls src="${esc(src)}"></audio>` +
    `<p class="muted">log-Mel(80 帯域、下が低い周波数 0Hz・上が 12kHz、横が時間)</p></div>`;
  const close = () => { overlay.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (ev) => { if (ev.key === "Escape") close(); };
  overlay.addEventListener("click", (ev) => { if (ev.target === overlay || ev.target.closest(".zoom-head button")) close(); });
  document.addEventListener("keydown", onKey);
  document.body.appendChild(overlay);
  getSpec(src).then((s) => drawSpec(overlay.querySelector("canvas"), s));
}

document.addEventListener("click", (ev) => {
  if (ev.target.matches && ev.target.matches("canvas.spec")) openZoom(ev.target);
});

// 1 つ再生したら他は止める(聞き比べで音が重ならないように)
document.addEventListener("play", (ev) => {
  document.querySelectorAll("audio").forEach((a) => { if (a !== ev.target) a.pause(); });
}, true);

function route() {
  const [baseId, expId] = location.hash.replace(/^#/, "").split("/");
  const r = { base: baseId || null, exp: expId || null };
  renderNav(r);
  $("#nav").classList.remove("open");
  if (!r.base) return renderHome();
  const { base, exp } = findExp(r.base, r.exp);
  if (!base) return renderHome();
  if (!exp) return renderBase(base);
  renderExp(base, exp);
}

$(".nav-toggle").addEventListener("click", () => {
  const open = $("#nav").classList.toggle("open");
  $(".nav-toggle").setAttribute("aria-expanded", String(open));
});

fetch("site/manifest.json")
  .then((r) => r.json())
  .then((m) => { state.manifest = m; window.addEventListener("hashchange", route); route(); })
  .catch((err) => { $("#main").innerHTML = `<p class="muted">site/manifest.json を読めない: ${esc(err.message)}</p>`; });
