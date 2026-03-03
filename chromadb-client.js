import { ChromaClient } from 'chromadb';

let client;
let embeddingsCollection;
let historyCollection;
let trainingCollection;
let connected = false;

export async function initChromaDB(
    host = process.env.CHROMADB_HOST || 'localhost',
    port = parseInt(process.env.CHROMADB_PORT) || 8000
) {
    try {
        client = new ChromaClient({ host, port });
        await client.heartbeat();

        embeddingsCollection = await client.getOrCreateCollection({
            name: 'process_embeddings',
            embeddingFunction: null,
            metadata: { description: 'Embeddings 128-dim por combinacao unica de features' }
        });

        historyCollection = await client.getOrCreateCollection({
            name: 'classification_history',
            embeddingFunction: null,
            metadata: { description: 'Log de todas as classificacoes realizadas' }
        });

        trainingCollection = await client.getOrCreateCollection({
            name: 'training_processes',
            embeddingFunction: null,
            metadata: { description: 'Embeddings dos dados de treinamento' }
        });

        connected = true;
        console.log('ChromaDB conectado com sucesso.');
        return { embeddingsCollection, historyCollection, trainingCollection };
    } catch (err) {
        connected = false;
        console.warn('ChromaDB indisponivel - sistema continua sem armazenamento vetorial:', err.message);
        return null;
    }
}

export function isConnected() {
    return connected;
}

export function getCollections() {
    return { embeddingsCollection, historyCollection, trainingCollection };
}
