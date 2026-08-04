"use strict";
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");

const CACHE = path.join(__dirname, "cache_mn");
const MODEL_OUT = path.join(__dirname, "..", "model");
const EPOCHS = parseInt(process.argv[2] || "150", 10);

function loadSplit(name) {
  const X = new Float32Array(fs.readFileSync(path.join(CACHE, `${name}_X.bin`)).buffer);
  const y = JSON.parse(fs.readFileSync(path.join(CACHE, `${name}_y.json`)));
  return { X, y, n: y.length };
}

async function main() {
  console.log("backend:", tf.getBackend());
  const meta = JSON.parse(fs.readFileSync(path.join(CACHE, "meta.json")));
  const classes = JSON.parse(fs.readFileSync(path.join(CACHE, "classes.json")));
  const { featDim, numClasses } = meta;
  console.log("featDim:", featDim, "tříd:", numClasses, "epochs:", EPOCHS);

  const train = loadSplit("train");
  const val = loadSplit("val");
  console.log("train:", train.n, "val:", val.n);

  // normalizace příznaků na nulovou střední hodnotu / jednotkový rozptyl (z-score)
  const mean = new Float32Array(featDim), std = new Float32Array(featDim);
  for (let j = 0; j < featDim; j++) {
    let s = 0; for (let i = 0; i < train.n; i++) s += train.X[i * featDim + j];
    mean[j] = s / train.n;
  }
  for (let j = 0; j < featDim; j++) {
    let s = 0; for (let i = 0; i < train.n; i++) { const d = train.X[i * featDim + j] - mean[j]; s += d * d; }
    std[j] = Math.sqrt(s / train.n) || 1;
  }
  function normalize(X, n) {
    const out = new Float32Array(X.length);
    for (let i = 0; i < n; i++) for (let j = 0; j < featDim; j++) out[i * featDim + j] = (X[i * featDim + j] - mean[j]) / std[j];
    return out;
  }
  const trainXNorm = normalize(train.X, train.n);
  const valXNorm = normalize(val.X, val.n);

  const xsTrain = tf.tensor2d(trainXNorm, [train.n, featDim]);
  const ysTrain = tf.oneHot(tf.tensor1d(train.y, "int32"), numClasses);
  const xsVal = tf.tensor2d(valXNorm, [val.n, featDim]);
  const ysVal = tf.oneHot(tf.tensor1d(val.y, "int32"), numClasses);

  const l2 = tf.regularizers.l2({ l2: 3e-4 });
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [featDim], units: 128, activation: "relu", kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.5 }));
  model.add(tf.layers.dense({ units: 64, activation: "relu", kernelRegularizer: l2 }));
  model.add(tf.layers.dropout({ rate: 0.35 }));
  model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
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
    model.setWeights(bestWeights);
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
    fs.writeFileSync(path.join(MODEL_OUT, "head_weights.bin"), Buffer.from(artifacts.weightData));
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      weightsManifest: [{ paths: ["head_weights.bin"], weights: artifacts.weightSpecs }],
    };
    fs.writeFileSync(path.join(MODEL_OUT, "head_model.json"), JSON.stringify(modelJson));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
  }));
  fs.writeFileSync(path.join(MODEL_OUT, "labels.json"), JSON.stringify({
    classes, featDim, valAccuracy: finalValAcc,
    mobilenetUrl: meta.mobilenetUrl, inputSize: meta.inputSize,
    norm: { mean: Array.from(mean), std: Array.from(std) },
  }));
  console.log("Model uložen do", MODEL_OUT, "(head_model.json / head_weights.bin / labels.json)");
}
main().catch(e => { console.error("CHYBA:", e && e.stack || e); process.exit(1); });
