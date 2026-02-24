import fs from 'fs';
import readline from 'readline';

const INPUT_FILE = 'treinamento_ia_202602240943.csv';
const OUTPUT_FILE = 'treinamento_ia_onehot.csv';

// --- PASSO 1: Primeira leitura — descobrir valores únicos ---
console.log(`Passo 1: Lendo ${INPUT_FILE} para descobrir valores únicos...`);

const uniqueSets = {
    cod_competencia: new Set(),
    id_assunto_principal: new Set(),
    id_localidade_judicial: new Set(),
    id_orgao_juizo: new Set(),
};

let totalLines = 0;
async function firstPass() {
    const rl = readline.createInterface({
        input: fs.createReadStream(INPUT_FILE),
        crlfDelay: Infinity
    });

    let isHeader = true;
    for await (const line of rl) {
        if (isHeader) { isHeader = false; continue; }
        const parts = line.replace(/"/g, '').split(',');
        if (parts.length < 5) continue;

        const cod_comp = Number(parts[1]);
        const assunto = Number(parts[2]);
        const localidade = Number(parts[3]);
        const orgao = Number(parts[4]);

        uniqueSets.cod_competencia.add(cod_comp);
        uniqueSets.id_assunto_principal.add(assunto);
        uniqueSets.id_localidade_judicial.add(localidade);
        uniqueSets.id_orgao_juizo.add(orgao);
        totalLines++;
    }
}

await firstPass();
console.log(`${totalLines} registros encontrados.`);

// Ordena e cria índices
const unique = {};
const indexes = {};
const offsets = {};
let offset = 0;

for (const key of Object.keys(uniqueSets)) {
    unique[key] = [...uniqueSets[key]].sort((a, b) => a - b);
    indexes[key] = new Map();
    unique[key].forEach((val, i) => indexes[key].set(val, i));
    offsets[key] = offset;
    offset += unique[key].length;
    console.log(`  ${key}: ${unique[key].length} valores únicos`);
}

const totalCols = offset;
console.log(`Total de features: ${totalCols} colunas`);

// Montar o cabeçalho
const header = ['id_promotoria'];
for (const key of Object.keys(unique)) {
    unique[key].forEach(val => header.push(`${key}_${val}`));
}

// --- PASSO 2: Segunda leitura — gerar one-hot streaming ---
console.log(`\nPasso 2: Gerando ${OUTPUT_FILE} (streaming)...`);

const writeStream = fs.createWriteStream(OUTPUT_FILE);
writeStream.write(header.join(',') + '\n');

const rl2 = readline.createInterface({
    input: fs.createReadStream(INPUT_FILE),
    crlfDelay: Infinity
});

let isHeader2 = true;
let processed = 0;

for await (const line of rl2) {
    if (isHeader2) { isHeader2 = false; continue; }
    const parts = line.replace(/"/g, '').split(',');
    if (parts.length < 5) continue;

    const id_promotoria = parts[0].trim();
    const values = {
        cod_competencia: Number(parts[1]),
        id_assunto_principal: Number(parts[2]),
        id_localidade_judicial: Number(parts[3]),
        id_orgao_juizo: Number(parts[4]),
    };

    // Vetor de zeros
    const vec = new Array(totalCols).fill(0);

    // Ativa as 4 posições
    for (const key of Object.keys(indexes)) {
        const idx = indexes[key].get(values[key]);
        if (idx !== undefined) {
            vec[offsets[key] + idx] = 1;
        }
    }

    writeStream.write(id_promotoria + ',' + vec.join(',') + '\n');

    processed++;
    if (processed % 10000 === 0) {
        console.log(`  Processadas ${processed}/${totalLines} linhas...`);
    }
}

await new Promise(resolve => writeStream.end(resolve));
console.log(`\nSUCESSO! Arquivo "${OUTPUT_FILE}" gerado com ${processed} linhas.`);
