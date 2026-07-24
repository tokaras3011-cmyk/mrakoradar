"use strict";
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");
// Pozn.: Conv2D backprop je na čistě-JS CPU backendu (bez WASM/native — WASM nemá gradientní
// jádro pro Conv2D a tfjs-node nejde zkompilovat bez Python/MSVC) neúnosně pomalý. Trénink na
// syrových pixelech přes plně propojenou síť taky nefungoval (uvízlo na ~20 % přesnosti).
// Řešení: ručně navržené příznaky (barva/textura/hrany po mřížce, viz prepare-data.js) —
// menší, informativnější vstup, na kterém se malá síť trénuje rychle i dobře.

const CACHE = path.join(__dirname, "cache");
const MODEL_OUT = path.join(__dirname, "..", "model");
const EPOCHS = parseInt(process.argv[2] || "60", 10);
const USE_FLIP = process.argv[3] !== "noflip";

function loadSplit(name) {
  const X = new Float32Array(fs.readFileSync(path.join(CACHE, `${name}_X.bin`)).buffer);
  const y = JSON.parse(fs.readFileSync(path.join(CACHE, `${name}_y.json`)));
  return { X, y, n: y.length };
}
// zrcadlové převrácení ve fea­turovém prostoru: přehodit pořadí sloupců mřížky (řádky/hodnoty
// uvnitř buňky zůstávají stejné - průměr barvy a textura buňky se otočením obrázku nemění)
function flipFeatures(X, n, grid, cellDim) {
  const out = new Float32Array(X.length);
  const rowDim = grid * cellDim;
  for (let i = 0; i < n; i++) {
    const base = i * rowDim;
    for (let gy = 0; gy < grid; gy++) {
      for (let gx = 0; gx < grid; gx++) {
        const src = base + (gy * grid + gx) * cellDim;
        const dst = base + (gy * grid + (grid - 1 - gx)) * cellDim;
        for (let c = 0; c < cellDim; c++) out[dst + c] = X[src + c];
      }
    }
  }
  return out;
}

async function main() {
  console.log("backend:", tf.getBackend());
  const meta = JSON.parse(fs.readFileSync(path.join(CACHE, "meta.json")));
  const classes = JSON.parse(fs.readFileSync(path.join(CACHE, "classes.json")));
  const { featDim, grid, numClasses } = meta;
  const cellDim = featDim / (grid * grid);
  console.log("featDim:", featDim, "tříd:", numClasses, "epochs:", EPOCHS);

  const train = loadSplit("train");
  const val = loadSplit("val");

  let trainXAll, trainYAll;
  if (USE_FLIP) {
    const flipped = flipFeatures(train.X, train.n, grid, cellDim);
    trainXAll = new Float32Array(train.X.length * 2);
    trainXAll.set(train.X, 0); trainXAll.set(flipped, train.X.length);
    trainYAll = train.y.concat(train.y);
  } else { trainXAll = train.X; trainYAll = train.y; }
  const nTrain = trainYAll.length;
  console.log(`train (${USE_FLIP ? "s flipem" : "bez flipu"}):`, nTrain, "val:", val.n);

  // normalizace příznaků na nulovou střední hodnotu / jednotkový rozptyl (z-score) - pomáhá
  // konvergenci, protože barvy (0..1) a hrany (jiné měřítko) mají jinak velmi rozdílný rozsah
  const mean = new Float32Array(featDim), std = new Float32Array(featDim);
  for (let j = 0; j < featDim; j++) {
    let s = 0; for (let i = 0; i < nTrain; i++) s += trainXAll[i * featDim + j];
    mean[j] = s / nTrain;
  }
  for (let j = 0; j < featDim; j++) {
    let s = 0; for (let i = 0; i < nTrain; i++) { const d = trainXAll[i * featDim + j] - mean[j]; s += d * d; }
    std[j] = Math.sqrt(s / nTrain) || 1;
  }
  function normalize(X, n) {
    const out = new Float32Array(X.length);
    for (let i = 0; i < n; i++) for (let j = 0; j < featDim; j++) out[i * featDim + j] = (X[i * featDim + j] - mean[j]) / std[j];
    return out;
  }
  const trainXNorm = normalize(trainXAll, nTrain);
  const valXNorm = normalize(val.X, val.n);

  const xsTrain = tf.tensor2d(trainXNorm, [nTrain, featDim]);
  const ysTrain = tf.oneHot(tf.tensor1d(trainYAll, "int32"), numClasses);
  const xsVal = tf.tensor2d(valXNorm, [val.n, featDim]);
  const ysVal = tf.oneHot(tf.tensor1d(val.y, "int32"), numClasses);

  const l2 = tf.regularizers.l2({ l2: 2e-4 });
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [featDim], units: 96, activation: "relu", kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.4 }));
  model.add(tf.layers.dense({ units: 48, activation: "relu", kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
  model.compile({ optimizer: tf.train.adam(0.0015), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  model.summary();

  const t0 = Date.now();
  const PATIENCE = 30;
  let bestValAcc = 0, bestWeights = null, bestEpoch = -1, noImprove = 0;
  await model.fit(xsTrain, ysTrain, {
    epochs: EPOCHS,
    batchSize: 32,
    shuffle: true,
    validationData: [xsVal, ysVal],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const dt = ((Date.now() - t0) / 1000).toFixed(1);
        console.log(`[${dt}s] epoch ${epoch + 1}/${EPOCHS}  loss=${logs.loss.toFixed(3)} acc=${logs.acc.toFixed(3)}  val_loss=${logs.val_loss.toFixed(3)} val_acc=${logs.val_acc.toFixed(3)}`);
        if (logs.val_acc > bestValAcc) {
          bestValAcc = logs.val_acc; bestEpoch = epoch + 1; noImprove = 0;
          if (bestWeights) bestWeights.forEach(t => t.dispose());
          bestWeights = model.getWeights().map(t => t.clone());
        } else if (++noImprove >= PATIENCE) {
          console.log(`Early stopping po epoše ${epoch + 1} (${PATIENCE} epoch bez zlepšení)`);
          model.stopTraining = true;
        }
      },
    },
  });
  if (bestWeights) {
    model.setWeights(bestWeights);   // ulož nejlepší validační checkpoint, ne poslední (přeučenou) epochu
    console.log(`Obnoveny váhy z nejlepší epochy ${bestEpoch} (val_acc=${bestValAcc.toFixed(4)})`);
  }

  const evalRes = model.evaluate(xsVal, ysVal);
  const finalValAcc = (await evalRes[1].data())[0];
  console.log("FINAL_VAL_ACC:", finalValAcc.toFixed(4), " BEST_VAL_ACC:", bestValAcc.toFixed(4));

  const predsT = model.predict(xsVal);
  const preds = await predsT.argMax(-1).data();
  const trueLabels = val.y;
  const confusion = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
  for (let i = 0; i < preds.length; i++) confusion[trueLabels[i]][preds[i]]++;
  console.log("Matice záměn (řádek = skutečnost, sloupec = predikce):");
  console.log("       " + classes.map(c => c.padStart(5)).join(" "));
  confusion.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    const correct = row[i];
    console.log(classes[i].padEnd(6) + row.map(v => String(v).padStart(5)).join(" ") + `   (${correct}/${total} = ${(100 * correct / total).toFixed(0)}%)`);
  });

  if (!fs.existsSync(MODEL_OUT)) fs.mkdirSync(MODEL_OUT, { recursive: true });
  await model.save(tf.io.withSaveHandler(async artifacts => {
    fs.writeFileSync(path.join(MODEL_OUT, "weights.bin"), Buffer.from(artifacts.weightData));
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      weightsManifest: [{ paths: ["weights.bin"], weights: artifacts.weightSpecs }],
    };
    fs.writeFileSync(path.join(MODEL_OUT, "model.json"), JSON.stringify(modelJson));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
  }));
  // normalizace (mean/std) a parametry příznaků musí appka replikovat stejně při inferenci
  fs.writeFileSync(path.join(MODEL_OUT, "labels.json"), JSON.stringify({
    classes, featDim, grid, work: meta.work, valAccuracy: finalValAcc,
    norm: { mean: Array.from(mean), std: Array.from(std) },
  }));
  console.log("Model uložen do", MODEL_OUT);
}
main().catch(e => { console.error(e); process.exit(1); });
