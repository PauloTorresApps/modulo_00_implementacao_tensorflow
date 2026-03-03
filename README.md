# Relatorio Tecnico - Sistema de Classificacao de Processos por Inteligencia Artificial

## Ministerio Publico do Estado do Tocantins

**Projeto:** Classificador Automatico de Processos para Promotorias de Justica
**Data:** 23 de fevereiro de 2026
**Tecnologia:** Rede Neural Artificial (Deep Learning) com TensorFlow.js

---

## Guia Rapido de Uso

### Pre-requisitos

- **Node.js 22** (ou superior)
- **Docker** e **Docker Compose** (para ChromaDB e deploy containerizado)

### Instalacao

```bash
git clone <url-do-repositorio>
cd projeto-promotoria-ia
npm install
```

### Fluxo completo (passo a passo)

**1. Configurar variaveis de ambiente**

Crie o arquivo `.env` na raiz do projeto com o caminho do CSV de treinamento bruto:

```
INPUT_FILE=./fontes_de_dados/treinamento_ia-2026-02-23_30k.csv
ONEHOT_FILE=treinamento_ia_onehot.csv
```

**2. Gerar CSV one-hot a partir dos dados brutos**

```bash
npm run gerar-csv
```

Transforma o CSV bruto (4 colunas categoricas) em um CSV com XXXX (depende do arquivo de dados fornecido) colunas binarias (one-hot encoding). O arquivo gerado (`treinamento_ia_onehot.csv`) sera usado no treinamento.

**3. Treinar o modelo**

```bash
npm run train
```

Treina a rede neural. O melhor modelo e salvo automaticamente em `modelo_promotoria/`. O progresso pode ser acompanhado pelo dashboard (passo 5). O treinamento para sozinho via early stopping quando a precisao estabiliza.

**4. Subir os containers (ChromaDB + aplicacao)**

```bash
npm run docker:up
```

Inicia tres containers: a aplicacao (porta 3000), o ChromaDB (porta 8000) e a interface administrativa do ChromaDB (porta 3001).

**5. Popular o ChromaDB com embeddings do treinamento**

```bash
npm run docker:populate
```

Gera embeddings de 128 dimensoes para cada registro do treinamento e armazena no ChromaDB. Necessario apenas uma vez (ou apos retreinamento do modelo).

### Uso da aplicacao

- **Interface de classificacao:** `http://localhost:3000` — selecione competencia, assunto, localidade e orgao para obter as 5 promotorias mais provaveis
- **Dashboard de treinamento:** `http://localhost:3000/dashboard.html` — acompanhe metricas de loss, accuracy e learning rate em tempo real
- **Admin ChromaDB:** `http://localhost:3001` — visualize collections, embeddings e metadados armazenados

### Execucao local (sem Docker)

Para rodar a aplicacao sem Docker (sem ChromaDB):

```bash
npm start
```

A classificacao funciona normalmente. As funcionalidades de busca por similaridade e historico ficam indisponiveis.

### Retreinamento com novos dados

1. Atualize o `INPUT_FILE` no `.env` apontando para o novo CSV bruto
2. Execute `npm run gerar-csv` para gerar o novo one-hot
3. Execute `npm run train` para retreinar o modelo
4. Execute `npm run docker:populate` para atualizar os embeddings no ChromaDB

### Comandos disponiveis

| Comando | Descricao |
|---|---|
| `npm run gerar-csv` | Gera CSV one-hot a partir dos dados brutos |
| `npm run train` | Treina o modelo (8GB de heap) |
| `npm run start` | Inicia o servidor localmente |
| `npm run diagnostico` | Analisa conflitos no dataset |
| `npm run docker:up` | Inicia containers (app + ChromaDB + admin) |
| `npm run docker:down` | Para os containers |
| `npm run docker:logs` | Exibe logs dos containers |
| `npm run docker:populate` | Popula ChromaDB com embeddings |

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

O sistema foi treinado com 5 datasets progressivamente maiores. O dataset final contem **98.320 registros** com todas as **169 promotorias**. A evolucao detalhada dos resultados por dataset esta na secao 6.2.

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

### 4.8 Penalizacao por Localidade Invalida

**Problema:** O modelo original tratava as 169 promotorias como classes independentes, sem considerar que cada promotoria pertence a uma localidade judicial especifica. Isso fazia com que, em predicoes de menor confianca (Top-3 a Top-5), o modelo sugerisse promotorias de localidades diferentes da informada na entrada — algo impossivel no mundo real.

**Solucao:** A funcao de perda foi modificada para incorporar uma penalizacao estrutural baseada na relacao promotoria-localidade:

1. **Mapeamento automatico:** Na primeira passagem dos dados de treinamento, o sistema identifica quais localidades cada promotoria atende, construindo um mapeamento `promotoria → {localidades}`

2. **Matriz de validade:** Uma matriz `[169 x 169]` e construida onde `validityMatrix[i][j] = 1` se as promotorias `i` e `j` compartilham pelo menos uma localidade. Promotorias sem dados de localidade sao tratadas como validas (nao penalizadas)

3. **Loss com penalizacao:** A funcao de perda combina a cross-entropy padrao (com label smoothing) com um termo de penalidade:

```text
loss_total = cross_entropy + 5.0 * media(probabilidade_em_classes_invalidas)
```

Para cada amostra do batch, o modelo identifica a classe verdadeira, consulta a matriz de validade para obter quais promotorias sao validas para aquela localidade, e penaliza a probabilidade atribuida a promotorias invalidas.

**Diferencial:** Diferentemente de um filtro pos-processamento (que apenas esconde os erros na inferencia), esta abordagem ensina o modelo durante o treinamento que cruzar localidades e proibido. O modelo aprende internamente a restricao, resultando em predicoes estruturalmente corretas.

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
| Loss function | Categorical Cross-Entropy (customizada com smoothing + penalizacao por localidade) |
| Penalizacao por localidade | Peso 5.0 sobre predicoes fora da localidade correta |

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

**Por que o 4o dataset (50k) obteve resultado superior?** O 4o dataset continha apenas promotorias com ID ate 102 (menos de 2/3 das classes) devido a truncamento na extracao, o que reduzia drasticamente o numero de conflitos entre promotorias.

**Por que os datasets 1 a 3 (completos) tambem tiveram acuracia alta?** Com menos amostras (10k a 30k registros), os conflitos de atribuicao entre promotorias estavam sub-representados nos dados. Quanto menor o volume, menor a chance de capturar a variabilidade natural da distribuicao (rodizio, carga de trabalho, etc.), resultando em um teto teorico mais alto e, consequentemente, acuracia superior. O 5o dataset (98k), com volume suficiente para representar plenamente esses conflitos, revela o desempenho real do modelo.

**Validacao cruzada:** O mesmo diagnostico foi executado em um dataset ainda maior (143.087 registros), que apresentou teto teorico de 80.67% — confirmando que o limite e uma caracteristica estrutural dos dados, nao do volume.

### 6.5 Convergencia

O treinamento final convergiu na epoca 117, com early stopping ativado apos 25 epocas sem melhoria na validacao. O learning rate ao final do treinamento era de ~0.0001.

### 6.6 Peso Relativo das Features nas Decisoes do Modelo

Embora a rede neural nao atribua um unico "peso" a cada feature (os pesos sao distribuidos por centenas de milhares de conexoes entre camadas), e possivel inferir a influencia relativa de cada parametro a partir da estrutura dos dados, da funcao de perda e da logica de negocio.

**Hierarquia de influencia (da mais determinante para a menos):**

| Feature | Colunas One-Hot | Papel na Decisao |
|---|---|---|
| `id_localidade_judicial` | 45 | **Filtro eliminatorio.** Promotorias sao vinculadas a comarcas — uma promotoria de Palmas nao recebe processos de Araguaina. Reduz as 169 classes para um subconjunto de ~5 a 30 promotorias possiveis |
| `cod_competencia` | 35 | **Refinamento por tipo de atuacao.** Dentro de uma localidade, define se o processo e civel, criminal, infancia, etc., restringindo ainda mais as promotorias elegiveis |
| `id_orgao_juizo` | 213 | **Correlacao com localidade e competencia.** Cada vara pertence a uma comarca e atua em competencias especificas, reforçando os sinais dos dois parametros anteriores |
| `id_assunto_principal` | 607 | **Desempate em casos ambiguos.** Apesar de ocupar mais da metade do vetor one-hot (607 de 1.191 colunas), a alta granularidade dilui o sinal: muitos assuntos distintos levam as mesmas promotorias |

**Por que `id_localidade_judicial` domina?**

1. **Penalizacao explicita na funcao de perda:** E o unico parametro com penalizacao estrutural dedicada (peso 5.0 na loss customizada, secao 4.8). O modelo e punido 5x mais por atribuir probabilidade a promotorias fora da localidade correta do que por errar entre promotorias da mesma localidade

2. **Matriz de validade construida sobre localidade:** A restricao rigida do modelo (secao 4.8) e definida exclusivamente pelo mapeamento `promotoria → {localidades}`. Nenhum outro parametro possui restricao equivalente

3. **Maior poder de reducao do espaco de classes:** Com 45 localidades distribuindo 169 promotorias, cada localidade contem em media ~4 promotorias. Apos o filtro por localidade, os demais parametros precisam discriminar entre poucas opcoes, nao entre 169

**Por que `id_assunto_principal` tem menor influencia apesar do maior numero de colunas?**

O assunto principal possui 607 valores unicos, mas a maioria e atendida por multiplas promotorias com atribuicoes sobrepostas. E justamente o parametro que mais contribui para os conflitos identificados na secao 6.4: combinacoes identicas de features apontando para promotorias diferentes. O volume de colunas one-hot nao se traduz em poder discriminante proporcional.

---

## 7. Aplicacao Web

### 7.1 Arquitetura

```
[Navegador] → [Express.js / Node.js] → [TensorFlow.js] → [Modelo Treinado]
                      ↕                       ↓
              [promotorias.csv]        [Embedding Model]
              [colunas one-hot]       (camada Dense3, 128-dim)
                                            ↓
                                       [ChromaDB]
                                  ┌────────┼────────┐
                          process_    classification_  training_
                          embeddings  history          processes
```

### 7.2 Fluxo de Classificacao

1. O usuario seleciona os 4 dados do processo nos menus dropdown
2. O frontend envia requisicao POST para a API
3. O backend monta o vetor one-hot de 1.191 posicoes (colocando 1 nas posicoes correspondentes)
4. O modelo executa a predicao via `model.predict()`
5. O sub-modelo de embeddings extrai o vetor de 128 dimensoes da camada `dense_Dense3`
6. O embedding e armazenado no ChromaDB (collections `process_embeddings` e `classification_history`)
7. O ChromaDB e consultado para recuperar os 5 processos de treinamento mais similares via busca vetorial
8. As 5 promotorias com maior probabilidade sao retornadas com nome, percentual de confianca e processos similares

### 7.3 Regra de Negocio

Quando a maior probabilidade de classificacao e inferior a **75%**, o sistema exibe um aviso indicando que o processo deve ser encaminhado para a **Central de Distribuicao** para avaliacao manual por um analista.

### 7.4 Endpoints da API

| Metodo | Rota | Descricao |
|---|---|---|
| GET | `/api/opcoes` | Retorna valores validos para cada campo |
| POST | `/api/classificar` | Classifica um processo, retorna Top-5 e processos similares |
| POST | `/api/similares` | Busca N processos similares via embedding no ChromaDB |
| GET | `/api/historico` | Retorna historico recente de classificacoes |
| GET | `/api/training-metrics` | Retorna metricas do treinamento em andamento ou concluido |

---

## 8. Armazenamento Vetorial com ChromaDB

### 8.1 Conceito

Alem da classificacao por rede neural, o sistema integra o **ChromaDB** — um banco de dados vetorial — para armazenar e consultar representacoes aprendidas (embeddings) dos processos. Isso habilita tres capacidades complementares a classificacao:

1. **Busca por similaridade:** Encontrar processos com caracteristicas proximas usando distancia vetorial
2. **Historico de classificacoes:** Registrar todas as classificacoes realizadas para auditoria e analise
3. **Contexto para decisao (RAG):** Exibir processos similares do treinamento junto ao resultado, fornecendo contexto adicional ao analista

### 8.2 Extracao de Embeddings

A rede neural, ao aprender a classificar processos, desenvolve internamente representacoes compactas das caracteristicas de cada processo. A **camada Dense 3** (128 neuronios) funciona como um "resumo aprendido" — ela comprime as 1.191 features one-hot em um vetor denso de 128 dimensoes que captura as relacoes relevantes entre competencia, assunto, localidade e orgao.

Para extrair esses embeddings sem retreinamento, um sub-modelo e criado a partir do modelo principal usando `tf.model()`, compartilhando os mesmos pesos:

```
Modelo Principal:    Entrada (1191) → Dense1 (512) → Dense2 (256) → Dense3 (128) → Saida (169)
Sub-modelo Embedding: Entrada (1191) → Dense1 (512) → Dense2 (256) → Dense3 (128) ← SAIDA AQUI
```

O sub-modelo reutiliza os pesos ja treinados — nenhum treinamento adicional e necessario.

### 8.3 Collections no ChromaDB

O sistema utiliza tres collections independentes:

| Collection | Finalidade | Chave | Conteudo |
|---|---|---|---|
| `process_embeddings` | Embeddings unicos por combinacao de features | Concatenacao dos 4 valores de entrada | Embedding 128-dim + metadata da classificacao |
| `classification_history` | Log de todas as classificacoes | UUID unico por chamada | Embedding 128-dim + Top-5 resultados + timestamp |
| `training_processes` | Embeddings dos dados de treinamento | Indice sequencial | Embedding 128-dim + promotoria real (ground truth) |

- **`process_embeddings`** usa upsert: classificacoes repetidas para a mesma combinacao de features atualizam o registro existente
- **`classification_history`** usa insert: cada classificacao gera um registro unico, permitindo rastrear o volume e padroes de uso
- **`training_processes`** e populada uma unica vez (ou apos retreinamento) pelo script `populate-chromadb.js`

### 8.4 Busca por Similaridade

A busca vetorial no ChromaDB permite encontrar processos "proximos" no espaco de embeddings aprendido pela rede neural. Dois processos com embeddings proximos compartilham padroes semelhantes — mesmo que suas features brutas sejam diferentes.

**Exemplo pratico:** Ao classificar um novo processo, o sistema consulta a collection `training_processes` e retorna os 5 processos de treinamento mais similares, exibindo a promotoria real de cada um. Isso da ao analista uma referencia concreta: "processos parecidos com este foram historicamente distribuidos para estas promotorias".

### 8.5 Degradacao Graceful

A integracao com o ChromaDB foi projetada para nao comprometer a funcionalidade principal. Se o ChromaDB estiver indisponivel:

- A classificacao por rede neural continua funcionando normalmente
- Os endpoints de similaridade e historico retornam erro 503 (Service Unavailable)
- O frontend oculta as secoes de processos similares e historico
- Nenhum erro e propagado para o usuario na interface de classificacao

### 8.6 Populacao Inicial e Retreinamento

O script `populate-chromadb.js` processa o dataset de treinamento em batches de 256 registros:

1. Le o CSV de treinamento linha a linha (streaming)
2. Constroi o vetor one-hot para cada registro
3. Executa o sub-modelo de embeddings em batch para obter vetores 128-dim
4. Insere os embeddings no ChromaDB com metadata (features + promotoria real)

> **Nota:** Apos cada retreinamento, e necessario re-executar este script para atualizar os embeddings.

---

## 9. Stack Tecnologico

| Componente | Tecnologia | Versao |
|---|---|---|
| Runtime | Node.js | 22 |
| Machine Learning | TensorFlow.js (Node) | 4.22.0 |
| Banco de Dados Vetorial | ChromaDB + chromadb (npm) | 3.3.1 |
| Servidor Web | Express.js | 5.2.1 |
| Parsing CSV | csv-parse | 6.1.0 |
| Frontend | HTML/CSS/JavaScript | Vanilla |
| Admin UI | ChromaDB Admin (Next.js) | - |
| Containerizacao | Docker + Docker Compose | - |

---

## 10. Containerizacao com Docker

### 10.1 Visao Geral

O sistema utiliza **Docker Compose** para orquestrar tres containers em uma unica operacao:

```text
docker-compose.yml
├── app (classificador-promotorias)
│   ├── Node.js 22 + TensorFlow.js
│   ├── Modelo treinado (modelo_promotoria/)
│   ├── Express.js (porta 3000)
│   └── Cliente ChromaDB
├── chromadb
│   ├── ChromaDB Server (porta 8000)
│   └── Volume persistente (chromadb_data)
└── chromadb-admin
    ├── Interface administrativa web (porta 3001)
    └── Visualizacao de collections, embeddings e metadados
```

### 10.2 Dockerfile

A imagem da aplicacao utiliza build **multi-stage** para otimizar o tamanho final:

1. **Estagio build:** Instala dependencias com `npm ci --omit=dev` (apenas producao)
2. **Estagio producao:** Copia `node_modules` ja resolvidos, codigo-fonte, modelo treinado e dados essenciais (`promotorias.csv`, `treinamento_ia_onehot.csv`)

A imagem base `node:22-slim` minimiza o tamanho do container mantendo compatibilidade com o TensorFlow.js.

### 10.3 Orquestracao

O `docker-compose.yml` define:

- **Dependencia com healthcheck:** O container `app` so inicia apos o ChromaDB responder ao endpoint `/api/v2/heartbeat`, garantindo que a conexao esteja disponivel na inicializacao
- **Interface administrativa:** O container `chromadb-admin` disponibiliza uma interface web (porta 3001) para visualizar collections, inspecionar embeddings e metadados armazenados no ChromaDB
- **Volume persistente:** Os dados do ChromaDB sao armazenados no volume `chromadb_data`, preservados entre reinicializacoes e atualizacoes dos containers
- **Restart automatico:** Todos os containers reiniciam automaticamente em caso de falha (`unless-stopped`)
- **Variaveis de ambiente:** Host e porta do ChromaDB sao configurados via environment, permitindo flexibilidade na implantacao

### 10.4 Operacao

| Comando | Descricao |
|---|---|
| `npm run docker:up` | Constroi imagens e inicia app + ChromaDB |
| `npm run docker:down` | Para e remove os containers |
| `npm run docker:logs` | Acompanha logs de ambos os containers |
| `npm run docker:populate` | Popula ChromaDB com embeddings do treinamento (dentro do container) |

---

## 11. Dashboard de Treinamento (tfjs-vis)

### 11.1 Visao Geral

O sistema inclui um dashboard web para monitoramento em tempo real do treinamento da rede neural, acessivel em `http://localhost:3000/dashboard.html`. O dashboard utiliza a biblioteca **TensorFlow.js Visualization (tfjs-vis)** para renderizar graficos interativos diretamente no navegador.

### 11.2 Arquitetura

```text
[train.js] → grava metricas → [logs/training_metrics.json] ← leitura ← [server.js /api/training-metrics]
                                                                              ↑
                                                                   [dashboard.html]
                                                                   polling a cada 3s
                                                                   renderiza com tfjs-vis
```

O desacoplamento via arquivo JSON permite que o treinamento (Node.js) e o dashboard (browser) operem de forma independente. O diretorio `logs/` e montado como volume Docker compartilhado, permitindo que o treinamento execute localmente enquanto o dashboard e servido pelo container.

### 11.3 Metricas Visualizadas

| Grafico | Descricao |
|---|---|
| Loss (Treino vs Validacao) | Evolucao da funcao de perda por epoca, com duas series para comparacao |
| Accuracy (Treino vs Validacao) | Evolucao da acuracia percentual por epoca |
| Learning Rate | Decaimento da taxa de aprendizado ao longo das epocas |

### 11.4 Indicadores de Status

O dashboard exibe em tempo real:

- **Status do treinamento:** Preparando / Treinando / Concluido
- **Epoca atual / total:** Progresso do treinamento
- **Melhor acuracia de validacao:** Atualizada a cada melhoria
- **Resultados Top-K:** Exibidos ao concluir o treinamento (Top-1, Top-3, Top-5)

### 11.5 Gravacao de Metricas

O script `train.js` grava metricas no arquivo `logs/training_metrics.json` em tres momentos:

1. **Inicio:** Status `preparing` com timestamp e configuracao de epocas
2. **A cada epoca:** Acumula loss, accuracy, val_loss, val_accuracy e learning rate
3. **Ao concluir:** Status `completed` com resultados Top-K e timestamp de finalizacao

---

## 12. Consideracoes e Proximos Passos

### 12.1 Pontos Fortes

- **Top-5 de 96%:** Em quase todos os casos, a promotoria correta esta entre as 5 sugestoes
- **Baixo overfitting:** Gap de apenas 0.4% entre treino e validacao (secao 6.3)
- **Escalavel:** A arquitetura suporta adicao de novas promotorias e features
- **Leve:** O modelo tem apenas ~3MB, executa em milissegundos
- **Busca vetorial e rastreabilidade** via ChromaDB (secao 8)
- **Resiliencia:** Degradacao graceful quando o ChromaDB esta indisponivel (secao 8.5)
- **Portabilidade:** Deploy completo via Docker Compose com um unico comando (secao 10)
- **Restricao por localidade** integrada a funcao de perda (secao 4.8)
- **Monitoramento de treinamento** em tempo real via dashboard (secao 11)

### 12.2 Limitacoes

- **Sobreposicao de atribuicoes:** Conforme detalhado na secao 6.4, a ambiguidade entre promotorias com atribuicoes sobrepostas impoe um teto teorico de ~81% com as 4 features atuais
- **Features limitadas:** Informacoes adicionais do processo (tipo, vara, texto do assunto) poderiam reduzir essa ambiguidade

### 12.3 Oportunidades de Melhoria

1. **Features adicionais:** Incorporar tipo de processo, vara de origem ou texto do assunto — principal alavanca para superar o teto teorico atual
2. **Dados de distribuicao:** Incluir criterios de rodizio e carga de trabalho para resolver conflitos entre promotorias concorrentes
3. **Retreinamento periodico:** Atualizar o modelo com novos processos e re-executar `populate-chromadb.js` para manter embeddings sincronizados
4. **Feedback dos analistas:** Integrar confirmacao ou correcao da promotoria sugerida para medir a precisao real em producao
5. **Analise de clusters:** Utilizar embeddings para identificar agrupamentos naturais de processos e padroes nao evidentes nas features brutas

---

## 13. Conclusao

O sistema demonstra viabilidade tecnica para automatizar a distribuicao de processos para promotorias de justica. Com **95.84% de acuracia Top-5** e resultado Top-1 que atinge **95.8% do teto teorico** imposto pela ambiguidade dos dados (secao 6.4), o modelo extrai quase toda a informacao discriminante disponivel nas features atuais.

A integracao com ChromaDB (busca por similaridade, historico e RAG) fornece contexto complementar ao analista, enquanto a containerizacao com Docker Compose garante implantacao com um unico comando. A incorporacao de features adicionais (secao 12.3) e o principal caminho para evolucao futura do sistema.
