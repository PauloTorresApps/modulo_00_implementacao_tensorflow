# ---- Build stage ----
FROM node:22-slim AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# ---- Production stage ----
FROM node:22-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends libc6 libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /app/node_modules ./node_modules
COPY package.json ./
COPY server.js chromadb-client.js ./
COPY public/ ./public/
COPY promotorias.csv treinamento_ia_onehot.csv ./
COPY modelo_promotoria/ ./modelo_promotoria/

# Scripts auxiliares (treinamento e populacao - usados sob demanda)
COPY train.js gerar_csv.js diagnostico.js populate-chromadb.js ./

ENV NODE_ENV=production
ENV CHROMADB_HOST=chromadb
ENV CHROMADB_PORT=8000

EXPOSE 3000

CMD ["node", "server.js"]
