"use strict";
// Místo ručně navržených příznaků (barva/textura/hrany po mřížce, viz prepare-data.js) tady
// každý obrázek jednou proženeme přes MobileNetV2 (předtrénovaný na ImageNetu) a uložíme jeho
// 1280rozměrný "bottleneck" vektor — mnohem bohatší popis textury a tvaru než ruční statistiky.
// Jen dopředný průchod (žádná zpětná propagace), proto to jde i na WASM CPU backendu rychle.
const fs = require("fs");
const path = require("path");
const jpeg = require("jpeg-js");
const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-backend-wasm");

const DATA_DIR = path.join(__dirname, "data", "CCSN_v2");
const OUT_DIR = path.join(__dirname, "cache_mn");
const MOBILENET_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2";
const SIZE = 224;

if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);

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
        out[oi + c] = (top * (1 - wy) + bot * wy) / 255;   // MobileNetV2 feature_vector čeká [0,1]
      }
    }
  }
  return out;
}
function flipHoriz(px, size) {
  const out = new Float32Array(px.length);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) for (let c = 0; c < 3; c++)
    out[(y * size + (size - 1 - x)) * 3 + c] = px[(y * size + x) * 3 + c];
  return out;
}

function shuffle(arr, seed) {
  let s = seed;
  const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
}

async function main() {
  await tf.setBackend("wasm");
  await tf.ready();
  console.log("backend:", tf.getBackend());
  console.log("Načítám MobileNetV2 (feature_vector)…");
  const mobilenet = await tf.loadGraphModel(MOBILENET_URL, { fromTFHub: true });
  console.log("MobileNet načten.");

  function embed(px) {
    return tf.tidy(() => {
      const t = tf.tensor4d(px, [1, SIZE, SIZE, 3]);
      const out = mobilenet.predict(t);
      return out.dataSync().slice();   // Float32Array, 1280 čísel
    });
  }

  const classes = fs.readdirSync(DATA_DIR).filter(f => fs.statSync(path.join(DATA_DIR, f)).isDirectory()).sort();
  console.log("Třídy:", classes.join(", "));

  // seznam souborů podle třídy, rozdělený na train/val PŘED extrakcí (aby augmentace nikdy
  // neprosákla stejný zdrojový obrázek do obou splitů)
  const byClass = classes.map(cls => {
    const dir = path.join(DATA_DIR, cls);
    const files = fs.readdirSync(dir).filter(f => /\.jpe?g$/i.test(f)).map(f => path.join(dir, f));
    shuffle(files, 42);
    return files;
  });

  const trainRecs = [], valRecs = [];
  byClass.forEach((files, ci) => {
    const nVal = Math.max(1, Math.round(files.length * 0.15));
    files.slice(0, nVal).forEach(f => valRecs.push({ file: f, label: ci }));
    files.slice(nVal).forEach(f => trainRecs.push({ file: f, label: ci }));
  });
  shuffle(trainRecs, 7); shuffle(valRecs, 7);
  console.log(`train: ${trainRecs.length} obrázků, val: ${valRecs.length} obrázků`);

  let dim = null;
  function processSplit(recs, withFlip) {
    const xs = [], ys = [];
    let done = 0, fail = 0;
    const t0 = Date.now();
    for (const r of recs) {
      try {
        const buf = fs.readFileSync(r.file);
        const img = jpeg.decode(buf, { useTArray: true, formatAsRGBA: true });
        const px = resizeBilinear(img.data, img.width, img.height, SIZE);
        const e1 = embed(px);
        if (!dim) dim = e1.length;
        xs.push(e1); ys.push(r.label);
        if (withFlip) {
          const e2 = embed(flipHoriz(px, SIZE));
          xs.push(e2); ys.push(r.label);
        }
      } catch (e) { fail++; }
      done++;
      if (done % 200 === 0) {
        const dt = (Date.now() - t0) / 1000;
        console.log(`  ${done}/${recs.length}  (${dt.toFixed(0)}s, ${(dt / done * 1000).toFixed(0)} ms/obr)`);
      }
    }
    console.log(`hotovo: ${xs.length} vektorů, ${fail} chyb dekódování`);
    return { xs, ys };
  }

  console.log("Extrahuji TRAIN (s vodorovným zrcadlením)…");
  const train = processSplit(trainRecs, true);
  console.log("Extrahuji VAL (bez augmentace — validace musí odrážet realitu)…");
  const val = processSplit(valRecs, false);

  function writeSplit(name, { xs, ys }) {
    const X = new Float32Array(xs.length * dim);
    xs.forEach((v, i) => X.set(v, i * dim));
    fs.writeFileSync(path.join(OUT_DIR, `${name}_X.bin`), Buffer.from(X.buffer));
    fs.writeFileSync(path.join(OUT_DIR, `${name}_y.json`), JSON.stringify(ys));
    console.log(`uloženo ${name}: ${ys.length} vzorků, dim=${dim}`);
  }
  writeSplit("train", train);
  writeSplit("val", val);
  fs.writeFileSync(path.join(OUT_DIR, "classes.json"), JSON.stringify(classes));
  fs.writeFileSync(path.join(OUT_DIR, "meta.json"), JSON.stringify({
    featDim: dim, numClasses: classes.length, mobilenetUrl: MOBILENET_URL, inputSize: SIZE,
  }));
  console.log("HOTOVO");
}
main().catch(e => { console.error("CHYBA:", e && e.stack || e); process.exit(1); });
