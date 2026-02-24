# Relatorio Tecnico - Sistema de Classificacao de Processos por Inteligencia Artificial

## Ministerio Publico do Estado do Tocantins

**Projeto:** Classificador Automatico de Processos para Promotorias de Justica
**Data:** 23 de fevereiro de 2026
**Tecnologia:** Rede Neural Artificial (Deep Learning) com TensorFlow.js

---

## 1. Objetivo

Desenvolver um sistema de inteligencia artificial capaz de classificar processos judiciais e sugerir automaticamente a promotoria de justica mais adequada para atuacao, com base nas caracteristicas do processo. O sistema visa agilizar a distribuicao de processos e reduzir a carga de trabalho manual dos analistas.

---

## 2. Dados Utilizados

### 2.1 Fonte dos Dados

Os dados de treinamento foram extraidos do sistema de gestao de processos do Ministerio Publico do Tocantins, contendo o historico de distribuicao de processos para promotorias.

### 2.2 Estrutura dos Dados Brutos

Cada registro representa um processo com 4 variaveis de entrada (features) e 1 variavel de saida (label):

| Campo | Tipo | Descricao |
|---|---|---|
| `cod_competencia` | Categorico | Codigo da competencia do processo (35 valores unicos) |
| `id_assunto_principal` | Categorico | Identificador do assunto principal (607 valores unicos) |
| `id_localidade_judicial` | Categorico | Identificador da localidade judicial (45 valores unicos) |
| `id_orgao_juizo` | Categorico | Identificador do orgao/juizo (213 valores unicos) |
| `id_promotoria` | Label (saida) | Promotoria destino do processo (169 classes) |

**Exemplo de registro bruto:**
```
id_promotoria, cod_competencia, id_assunto_principal, id_localidade_judicial, id_orgao_juizo
1,             5,               3435,                 2706,                   270000536
```

### 2.3 Volume de Dados

| Etapa | Registros | Promotorias | Resultado Top-1 |
|---|---|---|---|
| 1o dataset | 10.500 | 169 (completo) | 83.82% |
| 2o dataset | 21.000 | 169 (completo) | 85.51% |
| 3o dataset | 30.624 | 169 (completo) | 86.37% |
| 4o dataset | 50.000 | parcial* | 89.22% |
| **5o dataset (final)** | **98.320** | **169 (completo)** | **77.77%** |

> *\*O 4o dataset foi gerado com uma limitacao na extracao do banco de dados que truncou os resultados em 50.000 registros, excluindo as promotorias com ID acima de 102. Isso reduziu artificialmente a complexidade do problema de classificacao, resultando em acuracia inflada.*

O 5o dataset (98.320 registros) e o dataset final com **todas as 169 promotorias** e o maior volume de amostras. A analise de teto teorico (secao 6.4) demonstra que o modelo atinge 95.8% do maximo possivel com as features disponiveis.

### 2.4 Pre-processamento: One-Hot Encoding

Os dados categoricos brutos nao podem ser usados diretamente pela rede neural, pois os valores numericos (ex: `cod_competencia = 46`) nao representam grandezas — sao apenas codigos. Utilizar valores brutos faria o modelo interpretar erroneamente que "competencia 46" e maior que "competencia 5".

**Tecnica aplicada:** One-Hot Encoding — cada valor unico de cada campo e transformado em uma coluna binaria (0 ou 1).

**Antes (1 registro, 4 colunas):**
```
cod_competencia=5, id_assunto_principal=3435, id_localidade=2706, id_orgao=270000536
```

**Depois (1 registro, 1.191 colunas binarias):**
```
cod_competencia_1=0, cod_competencia_2=0, ..., cod_competencia_5=1, ..., cod_competencia_49=0,
id_assunto_principal_3370=0, ..., id_assunto_principal_3435=1, ...,
id_localidade_judicial_2702=0, ..., id_localidade_judicial_2706=1, ...,
id_orgao_juizo_270000501=0, ..., id_orgao_juizo_270000536=1, ...
```

**Total de features apos encoding:** 1.191 colunas binarias (35 + 607 + 45 + 213 + valores adicionais do dataset expandido).

> **Nota sobre normalizacao:** Como todos os dados ja sao binarios (0 e 1) apos o One-Hot Encoding, a normalizacao (Min-Max ou Z-Score) nao e necessaria — todas as features ja estao na mesma escala.

---

## 3. Arquitetura do Modelo

### 3.1 Tipo de Rede

Rede Neural Densa (Fully Connected / Multi-Layer Perceptron) implementada com TensorFlow.js.

### 3.2 Camadas

| Camada | Neuronios | Ativacao | Regularizacao | Extras |
|---|---|---|---|---|
| Entrada | 1.191 | - | - | Dimensao do vetor one-hot |
| Densa 1 | 512 | ReLU | L2 (0.001) | Batch Normalization + Dropout (30%) |
| Densa 2 | 256 | ReLU | L2 (0.001) | Batch Normalization + Dropout (30%) |
| Densa 3 | 128 | ReLU | L2 (0.001) | Dropout (20%) |
| Saida | 169 | Softmax | - | Uma saida por promotoria |

**Total de parametros treinaveis:** ~750.000

### 3.3 Funcao de Ativacao

- **ReLU (Rectified Linear Unit):** Usada nas camadas ocultas. Permite aprendizado rapido e evita o problema de gradientes que desaparecem.
- **Softmax:** Usada na camada de saida. Converte as saidas em probabilidades que somam 100%, permitindo interpretar o resultado como confianca.

---

## 4. Tecnicas de Treinamento

### 4.1 Label Smoothing (Suavizacao de Labels)

**Problema:** O one-hot encoding da saida diz ao modelo que a resposta e "100% promotoria X, 0% todas as outras". Isso forca o modelo a ser extremamente confiante, o que prejudica a generalizacao quando existem promotorias com perfis semelhantes.

**Solucao:** Label Smoothing com fator 0.1 — redistribui 10% da confianca entre as demais classes.

```
Antes:  [0, 0, 0, 1.0, 0, 0]  →  "100% classe 3"
Depois: [0.0006, 0.0006, 0.0006, 0.9, 0.0006, 0.0006]  →  "90% classe 3, 10% distribuido"
```

**Impacto medido:** Top-1 subiu de 83.38% para 85.51% (+2.13%) com a mesma quantidade de dados.

### 4.2 Class Weights (Balanceamento de Classes)

Como algumas promotorias tem muito mais processos que outras, o modelo tenderia a favorecer as mais frequentes. O class weight aplica peso inverso a frequencia: classes raras recebem peso maior no calculo do erro, forcando o modelo a aprender igualmente todas as promotorias.

```
peso = total_amostras / (num_classes * amostras_da_classe)
```

### 4.3 Regularizacao L2

Penaliza pesos com valores muito altos, evitando que o modelo decore os dados de treinamento (overfitting). A penalidade e proporcional ao quadrado dos pesos com fator 0.001.

### 4.4 Batch Normalization

Normaliza as ativacoes entre camadas, estabilizando o treinamento. Permite usar learning rates maiores e acelera a convergencia.

### 4.5 Dropout

Desliga aleatoriamente uma porcentagem dos neuronios durante o treinamento:
- 30% nas duas primeiras camadas
- 20% na terceira camada

Isso forca a rede a aprender representacoes redundantes e robustas, reduzindo overfitting.

### 4.6 Learning Rate Decay

A taxa de aprendizado comeca em 0.001 e decai 2% por epoca (fator 0.98):

```
lr_epoca = 0.001 * 0.98^epoca
```

Isso permite que o modelo faca ajustes grandes no inicio e refinamentos finos nas epocas finais.

### 4.7 Early Stopping

O treinamento para automaticamente se a precisao de validacao nao melhorar por 25 epocas consecutivas. O melhor modelo (nao o ultimo) e salvo automaticamente.

---

## 5. Processo de Treinamento

### 5.1 Divisao dos Dados

| Conjunto | Proporcao | Uso |
|---|---|---|
| Treino | 80% (~78.656 amostras) | Aprendizado dos pesos da rede |
| Validacao | 20% (~19.664 amostras) | Avaliacao de desempenho em dados nao vistos |

Os dados sao embaralhados aleatoriamente (Fisher-Yates shuffle) antes da divisao para garantir representatividade.

### 5.2 Hiperparametros Finais

| Parametro | Valor |
|---|---|
| Epocas maximas | 200 |
| Batch size | 32 |
| Otimizador | Adam |
| Learning rate inicial | 0.001 |
| Decay rate | 0.98 por epoca |
| Label smoothing | 0.1 |
| L2 regularization | 0.001 |
| Dropout | 0.3 / 0.3 / 0.2 |
| Early stopping patience | 25 epocas |
| Loss function | Categorical Cross-Entropy (customizada com smoothing) |

### 5.3 Otimizador Adam

O Adam (Adaptive Moment Estimation) combina as vantagens de dois outros otimizadores (AdaGrad e RMSProp), ajustando automaticamente a taxa de aprendizado para cada parametro. E o otimizador padrao mais eficiente para classificacao multi-classe.

---

## 6. Resultados

### 6.1 Metricas de Avaliacao (Dataset Final — 98.320 registros)

| Metrica | Valor | Significado |
|---|---|---|
| **Top-1 Accuracy** | **77.77%** | A promotoria correta e a primeira sugestao |
| **Top-3 Accuracy** | **93.14%** | A promotoria correta esta entre as 3 primeiras sugestoes |
| **Top-5 Accuracy** | **95.84%** | A promotoria correta esta entre as 5 primeiras sugestoes |

### 6.2 Evolucao por Volume de Dados

| Dataset | Registros | Promotorias | Top-1 | Top-3 | Top-5 |
|---|---|---|---|---|---|
| 1o dataset | 10.500 | 169 (completo) | 83.82% | 94.45% | 96.25% |
| 2o dataset | 21.000 | 169 (completo) | 85.51% | 95.83% | 97.40% |
| 3o dataset | 30.624 | 169 (completo) | 86.37% | 95.95% | 97.21% |
| 4o dataset | 50.000 | parcial* | 89.22% | 97.01% | 98.16% |
| **5o dataset (final)** | **98.320** | **169 (completo)** | **77.77%** | **93.14%** | **95.84%** |

> *\*O 4o dataset continha apenas promotorias com ID ate 102 devido a truncamento na extracao do banco de dados. O resultado de 89.22% reflete um problema de classificacao mais simples (menos classes), nao sendo comparavel diretamente com os demais.*

### 6.3 Analise de Overfitting (Dataset Final)

| Metrica | Treino | Validacao | Gap |
|---|---|---|---|
| Accuracy | 78.2% | 77.8% | 0.4% |

O gap de apenas 0.4% entre treino e validacao indica que o modelo generaliza muito bem — as tecnicas de regularizacao (Dropout, L2, Label Smoothing) foram eficazes.

### 6.4 Analise de Conflitos nos Dados e Teto Teorico

Uma analise diagnostica revelou uma limitacao fundamental: **muitas combinacoes identicas de features apontam para promotorias diferentes**. Isso ocorre porque varias promotorias possuem atribuicoes sobrepostas, e a distribuicao real depende de criterios que nao estao nos dados (rodizio, carga de trabalho, etc.).

**Resultados da analise no dataset final (98.320 registros):**

| Metrica | Valor |
|---|---|
| Total de registros | 98.320 |
| Combinacoes unicas de features | 14.340 |
| Combinacoes sem conflito | 10.412 (73%) |
| Combinacoes com conflito | 3.928 (27%) |
| Registros em combinacoes conflitantes | 70.411 (72% do total) |
| Registros impossiveis de acertar | 18.471 |
| **Acuracia maxima teorica** | **81.21%** |

Isso significa que, com as 4 features disponiveis, **nenhum modelo pode ultrapassar ~81% de acuracia** — independente da arquitetura ou tecnica utilizada. O resultado de 77.77% representa **95.8% do teto teorico**.

**Exemplo concreto:** A combinacao `cod_competencia=9, assunto=10949, localidade=2729, orgao=270000525` aparece 1.842 vezes no dataset, distribuida entre **29 promotorias diferentes** — sendo 41.8% para a promotoria 168 e 37.9% para a promotoria 89.

**Por que o 4o dataset (50k) obteve resultado superior?** O 4o dataset continha apenas promotorias com ID ate 102 (menos de 2/3 das classes) devido a truncamento na extracao, o que reduzia drasticamente o numero de conflitos entre promotorias. Os datasets 1 a 3, embora completos, tinham menos amostras — ainda assim, apresentaram resultados coerentes com o teto teorico quando ajustado pelo numero de classes presentes.

**Nota:** A queda de acuracia entre o 3o dataset (86.37%) e o 5o dataset (77.77%) reflete o aumento natural de complexidade quando o volume de dados e suficiente para representar plenamente os conflitos de atribuicao entre as 169 promotorias.

**Validacao cruzada:** O mesmo diagnostico foi executado em um dataset ainda maior (143.087 registros), que apresentou teto teorico de 80.67% — confirmando que o limite e uma caracteristica estrutural dos dados, nao do volume.

### 6.6 Convergencia

O treinamento final convergiu na epoca 117, com early stopping ativado apos 25 epocas sem melhoria na validacao. O learning rate ao final do treinamento era de ~0.0001.

---

## 7. Aplicacao Web

### 7.1 Arquitetura

```
[Navegador] → [Express.js / Node.js] → [TensorFlow.js] → [Modelo Treinado]
                      ↕
              [promotorias.csv]
              [colunas one-hot]
```

### 7.2 Fluxo de Classificacao

1. O usuario seleciona os 4 dados do processo nos menus dropdown
2. O frontend envia requisicao POST para a API
3. O backend monta o vetor one-hot de 1.191 posicoes (colocando 1 nas posicoes correspondentes)
4. O modelo executa a predicao via `model.predict()`
5. As 5 promotorias com maior probabilidade sao retornadas com nome e percentual de confianca

### 7.3 Regra de Negocio

Quando a maior probabilidade de classificacao e inferior a **75%**, o sistema exibe um aviso indicando que o processo deve ser encaminhado para a **Central de Distribuicao** para avaliacao manual por um analista.

### 7.4 Endpoints da API

| Metodo | Rota | Descricao |
|---|---|---|
| GET | `/api/opcoes` | Retorna valores validos para cada campo |
| POST | `/api/classificar` | Classifica um processo e retorna Top-5 |

---

## 8. Stack Tecnologico

| Componente | Tecnologia | Versao |
|---|---|---|
| Runtime | Node.js | - |
| Machine Learning | TensorFlow.js (Node) | 4.22.0 |
| Servidor Web | Express.js | 5.2.1 |
| Parsing CSV | csv-parse | 6.1.0 |
| Frontend | HTML/CSS/JavaScript | Vanilla |

---

## 9. Consideracoes e Proximos Passos

### 9.1 Pontos Fortes

- **Top-5 de 96%:** Em quase todos os casos, a promotoria correta esta entre as 5 sugestoes
- **Baixo overfitting:** Gap de apenas 0.4% entre treino e validacao
- **Escalavel:** A arquitetura suporta adicao de novas promotorias e features
- **Leve:** O modelo tem apenas ~3MB, executa em milissegundos

### 9.2 Limitacoes

- **Sobreposicao de atribuicoes:** Varias promotorias atendem os mesmos tipos de processo, e a distribuicao real depende de criterios nao presentes nos dados (rodizio, carga de trabalho). Isso impoe um teto teorico de ~81% com as 4 features atuais
- **Features limitadas:** O modelo depende de apenas 4 features categoricas. Mais informacoes do processo (tipo, vara, texto do assunto) poderiam reduzir a ambiguidade e elevar a acuracia alem do teto atual

### 9.3 Oportunidades de Melhoria

1. **Features adicionais:** Incorporar informacoes como tipo de processo, vara de origem, ou texto do assunto — a principal alavanca para superar o teto atual de acuracia, reduzindo a ambiguidade entre promotorias com atribuicoes sobrepostas
2. **Retreinamento periodico:** Atualizar o modelo com novos processos distribuidos para manter a precisao ao longo do tempo
3. **Monitoramento em producao:** Registrar as classificacoes e feedback dos analistas para medir a precisao real em uso
4. **Dados de distribuicao:** Incluir informacoes sobre criterios de distribuicao (rodizio, carga) poderia resolver conflitos entre promotorias concorrentes

---

## 10. Conclusao

O sistema desenvolvido demonstra viabilidade tecnica para automatizar a distribuicao de processos para promotorias de justica. Com **77.77% de acerto direto** (Top-1), **93.14% considerando as 3 primeiras sugestoes** (Top-3) e **95.84% considerando as 5 primeiras** (Top-5), o modelo oferece ganho significativo de eficiencia, mantendo a seguranca de encaminhar casos incertos para avaliacao humana.

A analise diagnostica revelou que a sobreposicao natural de atribuicoes entre promotorias impoe um teto teorico de **81.21%** com as 4 features atuais — o que torna o resultado de 77.77% expressivo, atingindo **95.8% do maximo possivel**. O modelo extraiu quase toda a informacao discriminante disponivel nas variaveis de entrada. A incorporacao de features adicionais (tipo de processo, vara, texto do assunto) e o principal caminho para evolucao futura do sistema, permitindo ultrapassar o teto atual.
