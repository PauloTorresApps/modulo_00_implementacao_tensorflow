import fs from 'fs';
import readline from 'readline';

const INPUT_FILE = 'treinamento_ia_202602240943.csv';

console.log(`Analisando conflitos no dataset: ${INPUT_FILE}\n`);

const combos = new Map(); // chave: "comp,assunto,local,orgao" → Map<promotoria, count>
let total = 0;

const rl = readline.createInterface({
    input: fs.createReadStream(INPUT_FILE),
    crlfDelay: Infinity
});

let isHeader = true;
for await (const line of rl) {
    if (isHeader) { isHeader = false; continue; }
    const parts = line.replace(/"/g, '').split(',');
    if (parts.length < 5) continue;

    const promotoria = parts[0].trim();
    const key = `${parts[1].trim()},${parts[2].trim()},${parts[3].trim()},${parts[4].trim()}`;

    if (!combos.has(key)) combos.set(key, new Map());
    const counts = combos.get(key);
    counts.set(promotoria, (counts.get(promotoria) || 0) + 1);

    total++;
}

// Análise
let totalConflitos = 0;      // registros em combinações com mais de 1 promotoria
let registrosConflitantes = 0; // registros que NÃO são a classe majoritária
let combosComConflito = 0;
let combosUnicas = 0;
let acertosTeoricos = 0;      // se sempre escolher a classe majoritária

for (const [key, counts] of combos) {
    const totalCombo = [...counts.values()].reduce((a, b) => a + b, 0);
    const maxCount = Math.max(...counts.values());

    acertosTeoricos += maxCount;

    if (counts.size > 1) {
        combosComConflito++;
        totalConflitos += totalCombo;
        registrosConflitantes += totalCombo - maxCount;
    } else {
        combosUnicas++;
    }
}

const acuraciaMaxima = (acertosTeoricos / total * 100).toFixed(2);
const pctConflito = (totalConflitos / total * 100).toFixed(2);

console.log("╔══════════════════════════════════════════════════════╗");
console.log("║         DIAGNÓSTICO DO DATASET                      ║");
console.log("╠══════════════════════════════════════════════════════╣");
console.log(`║  Total de registros:        ${String(total).padStart(8)}                ║`);
console.log(`║  Combinações únicas:        ${String(combos.size).padStart(8)}                ║`);
console.log(`║  Combinações sem conflito:  ${String(combosUnicas).padStart(8)}                ║`);
console.log(`║  Combinações COM conflito:  ${String(combosComConflito).padStart(8)}                ║`);
console.log(`║  Registros em conflito:     ${String(totalConflitos).padStart(8)} (${pctConflito}%)       ║`);
console.log(`║  Registros impossíveis:     ${String(registrosConflitantes).padStart(8)}                ║`);
console.log(`║                                                      ║`);
console.log(`║  ACURÁCIA MÁXIMA TEÓRICA:   ${acuraciaMaxima.padStart(7)}%               ║`);
console.log("╚══════════════════════════════════════════════════════╝");

console.log("\n--- Top 20 combinações mais conflitantes ---\n");

const conflitos = [];
for (const [key, counts] of combos) {
    if (counts.size > 1) {
        const totalCombo = [...counts.values()].reduce((a, b) => a + b, 0);
        const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
        conflitos.push({ key, totalCombo, numClasses: counts.size, sorted });
    }
}

conflitos.sort((a, b) => b.totalCombo - a.totalCombo);

for (let i = 0; i < Math.min(20, conflitos.length); i++) {
    const c = conflitos[i];
    console.log(`${i + 1}. Features: [${c.key}] — ${c.totalCombo} registros, ${c.numClasses} promotorias:`);
    for (const [prom, count] of c.sorted.slice(0, 5)) {
        const pct = (count / c.totalCombo * 100).toFixed(1);
        console.log(`   Promotoria ${prom}: ${count}x (${pct}%)`);
    }
    console.log();
}
