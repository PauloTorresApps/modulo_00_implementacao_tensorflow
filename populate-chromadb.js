import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import readline from 'readline';
import path from 'path';
import { fileURLToPath } from 'url';
import { initChromaDB, getCollections } from './chromadb-client.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Arquivo CSV raw com os dados de treinamento
const INPUT_FILE = process.env.INPUT_FILE || process.argv[2];
const BATCH_SIZE = 256;

async function main() {
    console.log('=== Populando ChromaDB com embeddings do treinamento ===\n');

    // 1. Carregar modelo e criar embedding model
    console.log('Carregando modelo...');
    const model = await tf.loadLayersModel('file://./modelo_promotoria/model.json');
    const embeddingLayer = model.getLayer('dense_Dense3');
    const embeddingModel = tf.model({
        inputs: model.input,
        outputs: embeddingLayer.output
    });
    console.log('Modelo de embeddings criado (128-dim).\n');

    // 2. Carregar nomes das colunas do CSV one-hot para construir vetores
    const headerLine = fs.readFileSync(
        path.join(__dirname, 'treinamento_ia_onehot.csv'), 'utf-8'
    ).split('\n')[0];
    const allColumns = headerLine.split(',');
    const columnNames = allColumns.filter(col => col !== 'id_promotoria');
    console.log(`${columnNames.length} features identificadas.\n`);

    // 3. Carregar mapeamento de promotorias
    const { parse } = await import('csv-parse/sync');
    const promotorias = parse(
        fs.readFileSync(path.join(__dirname, 'promotorias.csv'), 'utf-8'),
        { columns: true, skip_empty_lines: true }
    );
    const promotoriasMap = {};
    promotorias.forEach(row => { promotoriasMap[row.id_promotoria] = row.nome; });

    // 4. Conectar ao ChromaDB
    const result = await initChromaDB();
    if (!result) {
        console.error('Nao foi possivel conectar ao ChromaDB. Verifique se o servidor esta rodando.');
        process.exit(1);
    }
    const { trainingCollection } = getCollections();

    // Verificar contagem atual
    const currentCount = await trainingCollection.count();
    if (currentCount > 0) {
        console.log(`Collection training_processes ja possui ${currentCount} registros.`);
        console.log('Limpando collection para repopular...');
        // Delete all by getting all ids
        const existing = await trainingCollection.get({ limit: currentCount });
        if (existing.ids.length > 0) {
            await trainingCollection.delete({ ids: existing.ids });
        }
        console.log('Collection limpa.\n');
    }

    // 5. Funcao para construir vetor one-hot a partir dos valores raw
    function buildInputVector(cod_comp, assunto, localidade, orgao) {
        const vector = new Array(columnNames.length).fill(0);
        const mappings = {
            cod_competencia: cod_comp,
            id_assunto_principal: assunto,
            id_localidade_judicial: localidade,
            id_orgao_juizo: orgao
        };
        for (const [prefix, value] of Object.entries(mappings)) {
            const colName = `${prefix}_${value}`;
            const index = columnNames.indexOf(colName);
            if (index !== -1) vector[index] = 1;
        }
        return vector;
    }

    // 6. Ler CSV e processar em batches
    console.log(`Lendo ${INPUT_FILE}...\n`);

    const rl = readline.createInterface({
        input: fs.createReadStream(path.join(__dirname, INPUT_FILE)),
        crlfDelay: Infinity
    });

    let isHeader = true;
    let batch = [];
    let totalProcessed = 0;

    async function processBatch(rows) {
        const vectors = rows.map(r => buildInputVector(r.cod_comp, r.assunto, r.localidade, r.orgao));
        const inputTensor = tf.tensor2d(vectors);
        const embeddingTensor = embeddingModel.predict(inputTensor);
        const embeddings = await embeddingTensor.array();
        inputTensor.dispose();
        embeddingTensor.dispose();

        const ids = [];
        const metas = [];
        const docs = [];
        const embeds = [];

        rows.forEach((row, i) => {
            ids.push(`train_${totalProcessed - rows.length + i}`);
            embeds.push(embeddings[i]);
            metas.push({
                cod_competencia: row.cod_comp,
                id_assunto_principal: row.assunto,
                id_localidade_judicial: row.localidade,
                id_orgao_juizo: row.orgao,
                id_promotoria: parseInt(row.id_promotoria),
                promotoria_name: promotoriasMap[row.id_promotoria] || `ID ${row.id_promotoria}`
            });
            docs.push(`Comp:${row.cod_comp} Assunto:${row.assunto} Local:${row.localidade} Orgao:${row.orgao} -> ${promotoriasMap[row.id_promotoria] || row.id_promotoria}`);
        });

        await trainingCollection.add({
            ids,
            embeddings: embeds,
            metadatas: metas,
            documents: docs
        });
    }

    for await (const line of rl) {
        if (isHeader) { isHeader = false; continue; }
        const parts = line.replace(/"/g, '').split(',');
        if (parts.length < 5) continue;

        batch.push({
            id_promotoria: parts[0].trim(),
            cod_comp: parts[1].trim(),
            assunto: parts[2].trim(),
            localidade: parts[3].trim(),
            orgao: parts[4].trim()
        });

        totalProcessed++;

        if (batch.length >= BATCH_SIZE) {
            await processBatch(batch);
            batch = [];
            if (totalProcessed % 5000 === 0) {
                console.log(`  Processados ${totalProcessed} registros...`);
            }
        }
    }

    // Processar batch restante
    if (batch.length > 0) {
        await processBatch(batch);
    }

    const finalCount = await trainingCollection.count();
    console.log(`\nSUCESSO! ${finalCount} registros inseridos em training_processes.`);
    console.log('ChromaDB populado com embeddings do treinamento.');
}

main().catch(err => {
    console.error('Erro:', err);
    process.exit(1);
});
