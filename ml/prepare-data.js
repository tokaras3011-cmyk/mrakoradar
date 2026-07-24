"use strict";
// Dekoduje CCSN dataset a pro každý obrázek spočítá ručně navržený vektor příznaků
// (barva/textura/hrany po mřížce) místo syrových pixelů — plně propojená síť na to trénuje
// řádově rychleji a lépe generalizuje (viz poznámka v train.js, proč ne konvoluce/syrové pixely).
const fs = require("fs");
const path = require("path");
const jpeg = require("jpeg-js");

const DATA_DIR = path.join(__dirname, "data", "CCSN_v2");
const OUT_DIR = path.join(__dirname, "cache");
const WORK = 64;   // pracovní rozlišení pro výpočet příznaků
const GRID = 8;    // 8x8 buněk -> buňka 8x8 px
const CELL = WORK / GRID;
const FEAT_DIM = GRID * GRID * 5;   // meanR, meanG, meanB, std jasu, hustota hran na buňku

if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);

const classes = fs.readdirSync(DATA_DIR).filter(f => fs.statSync(path.join(DATA_DIR, f)).isDirectory()).sort();
console.log("Třídy:", classes.join(", "));

function resizeBilinear(rgba, srcW, srcH, dstSize) {
  const out = new Float32Array(dstSize * dstSize * 3);
  const sx = (srcW - 1) / dstSize, sy = (srcH - 1) / dstSize;
  for (let oy = 0; oy < dstSize; oy++) {
    const fy = oy * sy, y0 = Math.floor(fy), y1 = Math.min(y0 + 1, srcH - 1), wy = fy - y0;
    for (let ox = 0; ox < dstSize; ox++) {
      const fx = ox * sx, x0 = Math.floor(fx), x1 = Math.min(x0 + 1, srcW - 1), wx = fx - x0;
      const i00 = (y0 * srcW + x0) * 4, i10 = (y0 * srcW + x1) * 4;
      const i01 = (y1 * srcW + x0) * 4, i11 = (y1 * srcW + x1) * 4;
      const oi = (oy * dstSize + ox) * 3;
      for (let c = 0; c < 3; c++) {
        const top = rgba[i00 + c] * (1 - wx) + rgba[i10 + c] * wx;
        const bot = rgba[i01 + c] * (1 - wx) + rgba[i11 + c] * wx;
        out[oi + c] = (top * (1 - wy) + bot * wy) / 255;
      }
    }
  }
  return out;
}

// vektor příznaků: pro každou buňku mřížky průměr R,G,B, směrodatná odchylka jasu (textura)
// a hustota hran (součet |gradientu| v buňce) — vše invariantní vůči přesné pozici pixelů
// v rámci buňky, na rozdíl od syrových pixelů
function extractFeatures(px, size) {
  const lum = new Float32Array(size * size);
  for (let i = 0; i < size * size; i++) lum[i] = 0.299 * px[i * 3] + 0.587 * px[i * 3 + 1] + 0.114 * px[i * 3 + 2];
  const feats = new Float32Array(FEAT_DIM);
  let fi = 0;
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      let sumR = 0, sumG = 0, sumB = 0, sumL = 0, sumL2 = 0, edgeSum = 0, n = 0;
      for (let y = gy * CELL; y < (gy + 1) * CELL; y++) {
        for (let x = gx * CELL; x < (gx + 1) * CELL; x++) {
          const i = y * size + x;
          sumR += px[i * 3]; sumG += px[i * 3 + 1]; sumB += px[i * 3 + 2];
          sumL += lum[i]; sumL2 += lum[i] * lum[i];
          if (x < size - 1) edgeSum += Math.abs(lum[i + 1] - lum[i]);
          if (y < size - 1) edgeSum += Math.abs(lum[i + size] - lum[i]);
          n++;
        }
      }
      const meanL = sumL / n;
      const varL = Math.max(0, sumL2 / n - meanL * meanL);
      feats[fi++] = sumR / n; feats[fi++] = sumG / n; feats[fi++] = sumB / n;
      feats[fi++] = Math.sqrt(varL); feats[fi++] = edgeSum / (2 * n);
    }
  }
  return feats;
}

const records = [];
classes.forEach((cls, ci) => {
  const dir = path.join(DATA_DIR, cls);
  const files = fs.readdirSync(dir).filter(f => /\.jpe?g$/i.test(f));
  let ok = 0, fail = 0;
  for (const f of files) {
    try {
      const buf = fs.readFileSync(path.join(dir, f));
      const img = jpeg.decode(buf, { useTArray: true, formatAsRGBA: true });
      const px = resizeBilinear(img.data, img.width, img.height, WORK);
      records.push({ feats: extractFeatures(px, WORK), label: ci });
      ok++;
    } catch (e) { fail++; }
  }
  console.log(`${cls}: ${ok} ok, ${fail} chyb`);
});
console.log("Celkem záznamů:", records.length);

function shuffle(arr, seed) {
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
}
const byClass = classes.map(() => []);
records.forEach((r, i) => byClass[r.label].push(i));
byClass.forEach(idxs => shuffle(idxs, 42));
const trainIdx = [], valIdx = [];
byClass.forEach(idxs => {
  const nVal = Math.max(1, Math.round(idxs.length * 0.15));
  valIdx.push(...idxs.slice(0, nVal));
  trainIdx.push(...idxs.slice(nVal));
});
shuffle(trainIdx, 7); shuffle(valIdx, 7);
console.log(`train: ${trainIdx.length}, val: ${valIdx.length}`);

function writeSplit(name, idxs) {
  const X = new Float32Array(idxs.length * FEAT_DIM);
  const y = new Int32Array(idxs.length);
  idxs.forEach((ri, i) => { X.set(records[ri].feats, i * FEAT_DIM); y[i] = records[ri].label; });
  fs.writeFileSync(path.join(OUT_DIR, `${name}_X.bin`), Buffer.from(X.buffer));
  fs.writeFileSync(path.join(OUT_DIR, `${name}_y.json`), JSON.stringify(Array.from(y)));
  console.log(`uloženo ${name}: ${idxs.length} vzorků`);
}
writeSplit("train", trainIdx);
writeSplit("val", valIdx);
fs.writeFileSync(path.join(OUT_DIR, "classes.json"), JSON.stringify(classes));
fs.writeFileSync(path.join(OUT_DIR, "meta.json"), JSON.stringify({ featDim: FEAT_DIM, grid: GRID, work: WORK, numClasses: classes.length }));
console.log("HOTOVO");
