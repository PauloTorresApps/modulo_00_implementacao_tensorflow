import * as tf from '@tensorflow/tfjs-node';
import express from 'express';
import fs from 'fs';
import { parse } from 'csv-parse/sync';
import path from 'path';
import { fileURLToPath } from 'url';
import crypto from 'crypto';
import { initChromaDB, isConnected, getCollections } from './chromadb-client.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// --- Variáveis globais carregadas na inicialização ---
let model;
let embeddingModel;   // Sub-modelo para extrair embeddings 128-dim
let columnNames;      // Nomes das colunas one-hot (sem id_promotoria)
let promotoriasMap;   // Map: id_promotoria → nome

// --- Carrega o mapeamento de promotorias ---
function loadPromotorias() {
    const csv = fs.readFileSync(path.join(__dirname, 'promotorias.csv'), 'utf-8');
    const records = parse(csv, { columns: true, skip_empty_lines: true });
    const map = {};
    records.forEach(row => {
        map[row.id_promotoria] = row.nome;
    });
    return map;
}

// --- Extrai os nomes das colunas do CSV one-hot (apenas o header) ---
function loadColumnNames() {
    const headerLine = fs.readFileSync(
        path.join(__dirname, 'treinamento_ia_onehot.csv'), 'utf-8'
    ).split('\n')[0];
    const allColumns = headerLine.split(',');
    // Remove id_promotoria, mantém apenas as features
    return allColumns.filter(col => col !== 'id_promotoria');
}

// --- Extrai as opções válidas para cada campo a partir dos nomes das colunas ---
function extractOptions(columns) {
    const options = {
        cod_competencia: [],
        id_assunto_principal: [],
        id_localidade_judicial: [],
        id_orgao_juizo: []
    };

    columns.forEach(col => {
        for (const prefix of Object.keys(options)) {
            if (col.startsWith(prefix + '_')) {
                const value = col.slice(prefix.length + 1);
                options[prefix].push(value);
                break;
            }
        }
    });

    // Ordena numericamente
    for (const key of Object.keys(options)) {
        options[key].sort((a, b) => Number(a) - Number(b));
    }

    return options;
}

// --- Monta vetor one-hot a partir dos valores informados ---
function buildInputVector(params) {
    const vector = new Array(columnNames.length).fill(0);

    const mappings = {
        cod_competencia: params.cod_competencia,
        id_assunto_principal: params.id_assunto_principal,
        id_localidade_judicial: params.id_localidade_judicial,
        id_orgao_juizo: params.id_orgao_juizo
    };

    for (const [prefix, value] of Object.entries(mappings)) {
        const colName = `${prefix}_${value}`;
        const index = columnNames.indexOf(colName);
        if (index !== -1) {
            vector[index] = 1;
        }
    }

    return vector;
}

// --- Endpoints da API ---

// Retorna opções dos dropdowns
app.get('/api/opcoes', (req, res) => {
    const options = extractOptions(columnNames);
    res.json(options);
});

// Métricas de treinamento para o dashboard
app.get('/api/training-metrics', (req, res) => {
    const metricsPath = path.join(__dirname, 'logs', 'training_metrics.json');
    if (!fs.existsSync(metricsPath)) {
        return res.json({ status: 'idle' });
    }
    try {
        const data = JSON.parse(fs.readFileSync(metricsPath, 'utf-8'));
        res.json(data);
    } catch {
        res.json({ status: 'idle' });
    }
});

// Classifica um processo
app.post('/api/classificar', async (req, res) => {
    const { cod_competencia, id_assunto_principal, id_localidade_judicial, id_orgao_juizo } = req.body;

    if (!cod_competencia || !id_assunto_principal || !id_localidade_judicial || !id_orgao_juizo) {
        return res.status(400).json({ erro: 'Todos os 4 campos são obrigatórios.' });
    }

    const inputVector = buildInputVector(req.body);

    // Verifica se pelo menos um campo foi mapeado
    const onesCount = inputVector.filter(v => v === 1).length;
    if (onesCount === 0) {
        return res.status(400).json({ erro: 'Nenhum dos valores informados foi encontrado nas colunas do modelo.' });
    }

    const inputTensor = tf.tensor2d([inputVector]);
    const prediction = model.predict(inputTensor);

    // Extrair embedding 128-dim
    const embeddingTensor = embeddingModel.predict(inputTensor);
    const embeddingArray = await embeddingTensor.array();
    const embedding = embeddingArray[0];

    // Top-5
    const topK = tf.topk(prediction, 5);
    const indices = await topK.indices.array();
    const values = await topK.values.array();

    const resultados = indices[0].map((id, i) => ({
        posicao: i + 1,
        id_promotoria: id,
        nome: promotoriasMap[id] || `Promotoria ID ${id} (não encontrada)`,
        confianca: (values[0][i] * 100).toFixed(2) + '%'
    }));

    // Buscar processos similares e armazenar no ChromaDB
    let processos_similares = [];
    if (isConnected()) {
        try {
            const { embeddingsCollection, historyCollection, trainingCollection } = getCollections();
            const inputKey = `${cod_competencia}_${id_assunto_principal}_${id_localidade_judicial}_${id_orgao_juizo}`;
            const timestamp = new Date().toISOString();
            const top1 = resultados[0];
            const confiancaNum = parseFloat(top1.confianca);
            const sentToCentral = confiancaNum < 75;

            // Upsert em process_embeddings (deduplica por combinacao de inputs)
            await embeddingsCollection.upsert({
                ids: [inputKey],
                embeddings: [embedding],
                metadatas: [{
                    cod_competencia, id_assunto_principal,
                    id_localidade_judicial, id_orgao_juizo,
                    predicted_promotoria_id: top1.id_promotoria,
                    predicted_promotoria_name: top1.nome,
                    confidence: confiancaNum,
                    created_at: timestamp
                }],
                documents: [`Comp:${cod_competencia} Assunto:${id_assunto_principal} Local:${id_localidade_judicial} Orgao:${id_orgao_juizo} -> ${top1.nome} (${top1.confianca})`]
            });

            // Add em classification_history (cada chamada eh unica)
            await historyCollection.add({
                ids: [crypto.randomUUID()],
                embeddings: [embedding],
                metadatas: [{
                    cod_competencia, id_assunto_principal,
                    id_localidade_judicial, id_orgao_juizo,
                    top1_id: top1.id_promotoria,
                    top1_name: top1.nome,
                    top1_confidence: confiancaNum,
                    top5_ids: JSON.stringify(resultados.map(r => r.id_promotoria)),
                    top5_confidences: JSON.stringify(resultados.map(r => r.confianca)),
                    sent_to_central: sentToCentral,
                    timestamp
                }],
                documents: [`Classificacao em ${timestamp}: ${top1.nome} (${top1.confianca})`]
            });

            // Buscar processos similares no treinamento
            const simResults = await trainingCollection.query({
                queryEmbeddings: [embedding],
                nResults: 5
            });
            if (simResults.metadatas && simResults.metadatas[0]) {
                processos_similares = simResults.metadatas[0].map((meta, i) => ({
                    ...meta,
                    distancia: simResults.distances[0][i]
                }));
            }
        } catch (chromaErr) {
            console.error('ChromaDB erro (nao-fatal):', chromaErr.message);
        }
    }

    // Limpar tensores
    inputTensor.dispose();
    prediction.dispose();
    embeddingTensor.dispose();
    topK.indices.dispose();
    topK.values.dispose();

    res.json({ resultados, processos_similares });
});

// Busca processos similares por embedding
app.post('/api/similares', async (req, res) => {
    const { cod_competencia, id_assunto_principal, id_localidade_judicial, id_orgao_juizo } = req.body;

    if (!cod_competencia || !id_assunto_principal || !id_localidade_judicial || !id_orgao_juizo) {
        return res.status(400).json({ erro: 'Todos os 4 campos são obrigatórios.' });
    }
    if (!isConnected()) {
        return res.status(503).json({ erro: 'ChromaDB indisponivel.' });
    }

    const inputVector = buildInputVector(req.body);
    const inputTensor = tf.tensor2d([inputVector]);
    const embeddingTensor = embeddingModel.predict(inputTensor);
    const embedding = (await embeddingTensor.array())[0];
    inputTensor.dispose();
    embeddingTensor.dispose();

    const nResults = parseInt(req.query.n) || 10;
    const { trainingCollection } = getCollections();

    try {
        const results = await trainingCollection.query({
            queryEmbeddings: [embedding],
            nResults
        });

        const similares = results.metadatas[0].map((meta, i) => ({
            ...meta,
            distancia: results.distances[0][i],
            descricao: results.documents[0][i]
        }));

        res.json({ similares });
    } catch (err) {
        res.status(500).json({ erro: 'Erro ao buscar similares: ' + err.message });
    }
});

// Historico de classificacoes
app.get('/api/historico', async (req, res) => {
    if (!isConnected()) {
        return res.status(503).json({ erro: 'ChromaDB indisponivel.' });
    }

    const limit = parseInt(req.query.limit) || 50;
    const { historyCollection } = getCollections();

    try {
        const results = await historyCollection.get({ limit });

        const historico = results.metadatas.map((meta, i) => ({
            ...meta,
            id: results.ids[i]
        }));

        // Ordenar por timestamp decrescente
        historico.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        res.json({ historico: historico.slice(0, limit) });
    } catch (err) {
        res.status(500).json({ erro: 'Erro ao buscar historico: ' + err.message });
    }
});

// --- Inicialização ---
async function init() {
    console.log('Carregando modelo...');
    model = await tf.loadLayersModel('file://./modelo_promotoria/model.json');
    console.log('Modelo carregado com sucesso.');

    // Criar sub-modelo para extrair embeddings da camada dense_Dense3 (128-dim)
    const embeddingLayer = model.getLayer('dense_Dense3');
    embeddingModel = tf.model({
        inputs: model.input,
        outputs: embeddingLayer.output
    });
    console.log('Modelo de embeddings criado (128-dim).');

    console.log('Carregando promotorias...');
    promotoriasMap = loadPromotorias();
    console.log(`${Object.keys(promotoriasMap).length} promotorias carregadas.`);

    console.log('Carregando colunas do dataset...');
    columnNames = loadColumnNames();
    console.log(`${columnNames.length} features identificadas.`);

    // Conectar ao ChromaDB (nao-bloqueante se indisponivel)
    console.log('Conectando ao ChromaDB...');
    await initChromaDB();

    const PORT = parseInt(process.env.PORT) || 3000;
    app.listen(PORT, () => {
        console.log(`\nServidor rodando em http://localhost:${PORT}`);
    });
}

init().catch(console.error);
