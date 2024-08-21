/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { embed, EmbedderArgument } from '@genkit-ai/ai/embedder';
import {
  CommonRetrieverOptionsSchema,
  defineIndexer,
  defineRetriever,
  Document,
  indexerRef,
  retrieverRef,
} from '@genkit-ai/ai/retriever';
import { genkitPlugin, PluginProvider } from '@genkit-ai/core';
import { MilvusClient, ClientConfig } from '@zilliz/milvus2-sdk-node';
import * as z from 'zod';

/**
 * Verify the data of indices and values in a pair
 * TODO: Decide whether to use sparse or not
 */
const SparseVectorSchema = z
  .object({
    indices: z.number().array(),
    values: z.number().array(),
  })
  .refine(
    (input) => {
      return input.indices.length === input.values.length;
    },
    {
      message: 'Indices and values must be of the same length',
    }
  );

/**
 * Define the retriever and indexer options schema
 * TODO: check the parameters
 */
const MilvusRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().max(1000),
  collectionName: z.string().optional(),
  filter: z.record(z.string(), z.any()).optional(),
  sparseVector: SparseVectorSchema.optional(),
});

const MilvusIndexerOptionsSchema = z.object({
  collectionName: z.string().optional(),
});

const TEXT_KEY = '_content';

export const milvusRetrieverRef = (params: {
  collectionName: string;
  displayName?: string;
}) => {
  return retrieverRef({
    name: `milvus/${params.collectionName}`,
    info: {
      label: params.displayName ?? `Milvus - ${params.collectionName}`,
    },
    configSchema: MilvusRetrieverOptionsSchema.optional(),
  });
};

export const milvusIndexerRef = (params: {
  collectionName: string;
  displayName?: string;
}) => {
  return indexerRef({
    name: `milvus/${params.collectionName}`,
    info: {
      label: params.displayName ?? `Milvus - ${params.collectionName}`,
    },
    configSchema: MilvusIndexerOptionsSchema.optional(),
  });
};

/**
 * Milvus plugin that provides a milvus retriever and indexer.
 */
export function milvus<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: {
    clientParams?: ClientConfig;
    collectionName: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }[]
): PluginProvider {
  const plugin = genkitPlugin(
    'milvus',
    async (
      params: {
        clientParams?: ClientConfig;
        collectionName: string;
        embedder: EmbedderArgument<EmbedderCustomOptions>;
        embedderOptions?: z.infer<EmbedderCustomOptions>;
      }[]
    ) => ({
      retrievers: params.map((i) => configureMilvusRetriever(i)),
      indexers: params.map((i) => configureMilvusIndexer(i)),
    })
  );
  return plugin(params);
}

export default milvus;

/**
 * Configures a Milvus vector store retriever.
 */
export function configureMilvusRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(params: {
  collectionName: string;
  clientParams?: ClientConfig;
  textKey?: string;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}) {
  const { collectionName, embedder, embedderOptions } = {
    ...params,
  };
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const milvus = new MilvusClient(milvusConfig);
  const textKey = params.textKey ?? TEXT_KEY;

  return defineRetriever(
    {
      name: `milvus/${collectionName}`,
      configSchema: MilvusRetrieverOptionsSchema,
    },
    async (content, options) => {
      const queryEmbeddings = await embed({
        embedder,
        content,
        options: embedderOptions,
      });

      const response = await milvus.search({
        collection_name: collectionName,
        vector: queryEmbeddings,
        limit: options.limit,
        params: JSON.stringify(options.filter),
      });

      return {
        documents: response.results
          .map((result) => result.entity.value)
          .filter((m): m is RecordMetadata => !!m)
          .map((m) => {
            const metadata = m;
            const content = metadata[textKey] as string;
            delete metadata[textKey];
            return Document.fromText(content, metadata).toJSON();
          }),
      };
    }
  );
}

/**
 * Configures a Milvus indexer.
 */
export function configureMilvusIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(params: {
  collectionName: string;
  clientParams?: ClientConfig;
  textKey?: string;
  embedder: EmbedderArgument<EmbedderCustomOptions>;
  embedderOptions?: z.infer<EmbedderCustomOptions>;
}) {
  const { collectionName, embedder, embedderOptions } = params;
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const milvus = new MilvusClient(milvusConfig);
  const textKey = params.textKey ?? TEXT_KEY;

  return defineIndexer(
    {
      name: `milvus/${collectionName}`,
      configSchema: MilvusIndexerOptionsSchema.optional(),
    },
    async (docs, options) => {
      const embeddings = await Promise.all(
        docs.map((doc) =>
          embed({
            embedder,
            content: doc,
            options: embedderOptions,
          })
        )
      );

      await milvus.insert({
        collection_name: collectionName,
        fields_data: embeddings.map((value, i) => {
          const metadata: RecordMetadata = {
            ...docs[i].metadata,
          };
          metadata[textKey] = docs[i].text();
          return {
            id: docs[i].id ?? String(i),
            vector: value,
            metadata,
          };
        }),
      });
    }
  );
}

/**
 * Helper function for creating a Milvus Collection.
 */
export async function createMilvusCollection(params: {
  clientParams?: ClientConfig;
  options: {
    collection_name: string,
    dimension: number,
    enable_dynamic_field: true,
  };
}) {
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const collectionConfig = params.options;
  const milvus = new MilvusClient(milvusConfig);
  return await milvus.createCollection(collectionConfig);
}

/**
 * Helper function to describe a Milvus Collection. Use it to check if a newly created index is ready for use.
 */
export async function describeMilvusCollection(params: {
  clientParams?: ClientConfig;
  collectionName: string;
}) {
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const milvus = new MilvusClient(milvusConfig);
  return await milvus.describeCollection({
    collection_name: params.collectionName,
  });
}

/**
 * Helper function for deleting Milvus collection.
 */
export async function deleteMilvusCollection(params: {
  clientParams?: ClientConfig;
  collectionName: string;
}) {
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const milvus = new MilvusClient(milvusConfig);

  try {
    return await milvus.dropCollection({
      collection_name: params.collectionName,
    });
  } catch (error) {
    console.error('Failed to delete Milvus collection:', error);
    throw error;
  }
}

/**
 * Get Default config.
 */
function getDefaultConfig(): ClientConfig {
  const configOrAddress = process.env.MILVUS_URI ?? "http://localhost:19530";

  return {
    address: configOrAddress,
    token: "",  // Default token is an empty string
    username: "",  // Default username is an empty string
    password: ""  // Default password is an empty string
  };
}
