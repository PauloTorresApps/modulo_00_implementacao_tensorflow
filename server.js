import * as tf from '@tensorflow/tfjs-node';
import express from 'express';
import fs from 'fs';
import { parse } from 'csv-parse/sync';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// --- Variáveis globais carregadas na inicialização ---
let model;
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

    // Limpar tensores
    inputTensor.dispose();
    prediction.dispose();
    topK.indices.dispose();
    topK.values.dispose();

    res.json({ resultados });
});

// --- Inicialização ---
async function init() {
    console.log('Carregando modelo...');
    model = await tf.loadLayersModel('file://./modelo_promotoria/model.json');
    console.log('Modelo carregado com sucesso.');

    console.log('Carregando promotorias...');
    promotoriasMap = loadPromotorias();
    console.log(`${Object.keys(promotoriasMap).length} promotorias carregadas.`);

    console.log('Carregando colunas do dataset...');
    columnNames = loadColumnNames();
    console.log(`${columnNames.length} features identificadas.`);

    const PORT = 3000;
    app.listen(PORT, () => {
        console.log(`\nServidor rodando em http://localhost:${PORT}`);
    });
}

init().catch(console.error);
