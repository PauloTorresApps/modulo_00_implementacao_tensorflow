import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import readline from 'readline';

// --- Função customizada: Top-K Accuracy ---
function topKAccuracy(yTrue, yPred, k) {
    return tf.tidy(() => {
        const topKPreds = tf.topk(yPred, k).indices;
        const trueIndices = tf.argMax(yTrue, 1).expandDims(1);
        const matches = tf.equal(topKPreds, tf.broadcastTo(trueIndices, topKPreds.shape));
        const anyMatch = tf.any(matches, 1);
        return tf.mean(tf.cast(anyMatch, 'float32')).arraySync();
    });
}

// --- Label Smoothing ---
function applyLabelSmoothing(labels, numClasses, smoothing = 0.1) {
    return tf.tidy(() => {
        const confident = 1.0 - smoothing;
        const low = smoothing / (numClasses - 1);
        return labels.mul(confident - low).add(low);
    });
}

// --- Loss customizada com Label Smoothing embutida ---
function categoricalCrossentropyWithSmoothing(smoothing, numClasses) {
    return (yTrue, yPred) => {
        return tf.tidy(() => {
            const confident = 1.0 - smoothing;
            const low = smoothing / (numClasses - 1);
            const smoothedLabels = yTrue.mul(confident - low).add(low);
            const clipped = yPred.clipByValue(1e-7, 1 - 1e-7);
            return smoothedLabels.mul(clipped.log()).sum(-1).neg().mean();
        });
    };
}

async function run() {
    console.log("1. Carregando e processando os dados (streaming)...");

    const numClasses = 169;

    // --- PASSO 1: Ler header para saber o número de features ---
    const headerLine = fs.readFileSync('treinamento_ia_onehot.csv', 'utf-8').split('\n')[0];
    const allColumns = headerLine.split(',');
    const numFeatures = allColumns.length - 1; // Exclui id_promotoria
    console.log(`Features: ${numFeatures}`);

    // --- PASSO 2: Contar linhas e coletar class counts (1a passagem leve) ---
    console.log("Contando registros e calculando class weights...");
    const classCounts = new Array(numClasses).fill(0);
    let totalRows = 0;

    const rl1 = readline.createInterface({
        input: fs.createReadStream('treinamento_ia_onehot.csv'),
        crlfDelay: Infinity
    });

    let isHeader = true;
    for await (const line of rl1) {
        if (isHeader) { isHeader = false; continue; }
        const id = parseInt(line.split(',', 1)[0]);
        if (id >= 0 && id < numClasses) classCounts[id]++;
        totalRows++;
    }

    console.log(`Total de registros: ${totalRows}`);

    // Class weights
    const classWeight = {};
    for (let i = 0; i < numClasses; i++) {
        classWeight[i] = classCounts[i] > 0
            ? totalRows / (numClasses * classCounts[i])
            : 0;
    }
    console.log("Class weights calculados.");

    // --- PASSO 3: Ler dados direto em Float32Array (2a passagem) ---
    console.log("Carregando dados em memória otimizada...");

    // Pré-aloca arrays tipados (usam ~4x menos memória que arrays JS)
    const xsFlat = new Float32Array(totalRows * numFeatures);
    const labels = new Int32Array(totalRows); // Só o índice da classe

    const rl2 = readline.createInterface({
        input: fs.createReadStream('treinamento_ia_onehot.csv'),
        crlfDelay: Infinity
    });

    let isHeader2 = true;
    let rowIdx = 0;

    for await (const line of rl2) {
        if (isHeader2) { isHeader2 = false; continue; }
        const parts = line.split(',');
        const promotoriaId = parseInt(parts[0]);
        labels[rowIdx] = promotoriaId;

        const offset = rowIdx * numFeatures;
        for (let i = 1; i < parts.length; i++) {
            xsFlat[offset + (i - 1)] = parseInt(parts[i]);
        }

        rowIdx++;
        if (rowIdx % 20000 === 0) {
            console.log(`  Lidas ${rowIdx}/${totalRows} linhas...`);
        }
    }

    console.log(`${rowIdx} registros carregados.`);

    // --- PASSO 4: Embaralhar os índices ---
    console.log("Embaralhando dados...");
    const indices = new Uint32Array(totalRows);
    for (let i = 0; i < totalRows; i++) indices[i] = i;
    for (let i = totalRows - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }

    // --- PASSO 5: Separar treino/validação e criar tensores ---
    const splitIndex = Math.floor(totalRows * 0.8);
    const trainCount = splitIndex;
    const valCount = totalRows - splitIndex;

    console.log(`Treino: ${trainCount} amostras | Validação: ${valCount} amostras`);
    console.log("Construindo tensores...");

    // Features de treino
    const trainXsFlat = new Float32Array(trainCount * numFeatures);
    const trainLabels = new Int32Array(trainCount);
    for (let i = 0; i < trainCount; i++) {
        const srcIdx = indices[i];
        trainLabels[i] = labels[srcIdx];
        const srcOffset = srcIdx * numFeatures;
        const dstOffset = i * numFeatures;
        for (let j = 0; j < numFeatures; j++) {
            trainXsFlat[dstOffset + j] = xsFlat[srcOffset + j];
        }
    }

    // Features de validação
    const valXsFlat = new Float32Array(valCount * numFeatures);
    const valLabels = new Int32Array(valCount);
    for (let i = 0; i < valCount; i++) {
        const srcIdx = indices[splitIndex + i];
        valLabels[i] = labels[srcIdx];
        const srcOffset = srcIdx * numFeatures;
        const dstOffset = i * numFeatures;
        for (let j = 0; j < numFeatures; j++) {
            valXsFlat[dstOffset + j] = xsFlat[srcOffset + j];
        }
    }

    // Liberar arrays originais
    // (serão coletados pelo GC quando não mais referenciados)

    // Criar tensores
    const xsTrain = tf.tensor2d(trainXsFlat, [trainCount, numFeatures]);
    const xsVal = tf.tensor2d(valXsFlat, [valCount, numFeatures]);

    // Labels one-hot
    const ysValOneHot = tf.oneHot(tf.tensor1d(valLabels, 'int32'), numClasses).cast('float32');

    // Labels de treino com smoothing
    const SMOOTHING = 0.1;
    const ysTrainOneHot = tf.oneHot(tf.tensor1d(trainLabels, 'int32'), numClasses).cast('float32');
    const ysTrain = applyLabelSmoothing(ysTrainOneHot, numClasses, SMOOTHING);
    ysTrainOneHot.dispose();

    console.log(`Label Smoothing: ${SMOOTHING}`);

    // 3. Definir a Arquitetura do Modelo
    console.log("\n2. Criando o modelo neural...");
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [numFeatures],
        units: 512,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: 0.3 }));

    model.add(tf.layers.dense({
        units: 256,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: 0.3 }));

    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));

    model.add(tf.layers.dense({
        units: numClasses,
        activation: 'softmax'
    }));

    const initialLearningRate = 0.001;
    const decayRate = 0.98;

    model.compile({
        optimizer: tf.train.adam(initialLearningRate),
        loss: categoricalCrossentropyWithSmoothing(SMOOTHING, numClasses),
        metrics: ['accuracy']
    });

    model.summary();

    // 4. Treinar o Modelo
    console.log("\n3. Iniciando o treinamento...");

    let bestValAcc = 0;
    const patience = 25;
    let wait = 0;

    await model.fit(xsTrain, ysTrain, {
        epochs: 200,
        batchSize: 32,
        validationData: [xsVal, ysValOneHot],
        classWeight,
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                const newLr = initialLearningRate * Math.pow(decayRate, epoch);
                model.optimizer.learningRate = newLr;

                if ((epoch + 1) % 5 === 0 || epoch === 0) {
                    console.log(
                        `Época ${epoch + 1} | Erro: ${logs.loss.toFixed(4)} | ` +
                        `Precisão Treino: ${(logs.acc * 100).toFixed(2)}% | ` +
                        `Precisão Validação: ${(logs.val_acc * 100).toFixed(2)}% | ` +
                        `LR: ${newLr.toFixed(6)}`
                    );
                }

                if (logs.val_acc > bestValAcc) {
                    bestValAcc = logs.val_acc;
                    wait = 0;
                    await model.save('file://./modelo_promotoria');
                } else {
                    wait++;
                    if (wait >= patience) {
                        model.stopTraining = true;
                        console.log(`\nEarly stopping na época ${epoch + 1}. Melhor val_acc: ${(bestValAcc * 100).toFixed(2)}%`);
                    }
                }
            }
        }
    });

    // 5. Avaliar Top-K Accuracy
    console.log("\n4. Avaliando Top-K Accuracy na validação...");

    const bestModel = await tf.loadLayersModel('file://./modelo_promotoria/model.json');
    const predictions = bestModel.predict(xsVal);

    const top1 = topKAccuracy(ysValOneHot, predictions, 1);
    const top3 = topKAccuracy(ysValOneHot, predictions, 3);
    const top5 = topKAccuracy(ysValOneHot, predictions, 5);

    console.log("╔══════════════════════════════════════╗");
    console.log("║       RESULTADO FINAL                ║");
    console.log("╠══════════════════════════════════════╣");
    console.log(`║  Top-1 Accuracy: ${(top1 * 100).toFixed(2)}%`.padEnd(39) + "║");
    console.log(`║  Top-3 Accuracy: ${(top3 * 100).toFixed(2)}%`.padEnd(39) + "║");
    console.log(`║  Top-5 Accuracy: ${(top5 * 100).toFixed(2)}%`.padEnd(39) + "║");
    console.log("╚══════════════════════════════════════╝");

    console.log("\nModelo salvo em ./modelo_promotoria");

    predictions.dispose();
    xsTrain.dispose();
    ysTrain.dispose();
    xsVal.dispose();
    ysValOneHot.dispose();
}

run().catch(console.error);
