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
import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';
import { Md5 } from 'ts-md5';
import * as z from 'zod';
import {
  Pinecone,
  PineconeConfiguration,
  RecordMetadata,
} from '@pinecone-database/pinecone';


/**
 * Verify the data of indices and values in a pair
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

const MilvusRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  limit: z.number().max(1000),
  namespace: z.string().optional(),
  filter: z.record(z.string(), z.any()).optional(),
  // includeValues is always false
  // includeMetadata is always true
  sparseVector: SparseVectorSchema.optional(),
});

const MilvusIndexerOptionsSchema = z.object({
  namespace: z.string().optional(),
});

const TEXT_KEY = '_content';


/**
 * Milvus plugin that provides a milvus retriever and indexer.
 */
export function milvus<EmbedderCustomOptions extends z.ZodTypeAny>(
  params: {
    clientParams?: MilvusConfiguration;
    collectio_name: string;
    indexId: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
  }[]
): PluginProvider {
  const plugin = genkitPlugin(
    'milvus',
    async (
      params: {
        clientParams?: MilvusConfiguration;
        indexId: string;
        textKey?: string;
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


/**
 * Configures a Milvus retriever.
 */

/**
 * Configures a Milvus Collection.
 * Store document data into an existing collection
 */


/**
 * Helper function for creating a Milvus Collection.
 */
export async function createMilvusCollection(params: {
  clientParams?: MilvusConfiguration;
  options: CreateCollectionOptions;
}) {
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const milvus = new MilvusClient(milvusConfig);
  return await milvus.createCollection(params.options);
}

/**
 * Helper function to describe a Milvus Collection. Use it to check if a newly created index is ready for use.
 */
export async function describeMilvusCollection(params: {
  clientParams?: MilvusConfiguration;
  collectionName: string;
}) {
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const milvus = new MilvusClient(milvusConfig);
  return await milvus.describeCollection({
    collection_name: params.collectionName
  });
}

/**
 * Helper function for deleting Milvus collection.
 */
export async function deleteMilvusCollection(params: {
  clientParams?: MilvusConfiguration;
  collectionName: string;
}) {
  const milvusConfig = params.clientParams ?? getDefaultConfig();
  const milvus = new MilvusClient(milvusConfig);

  try {
    return await milvus.dropCollection({
      collection_name: params.collectionName
    });
  } catch (error) {
    console.error('Failed to delete Milvus collection:', error);
    throw error;
  }
}

function getDefaultConfig() {
  const configOrAddress = process.env.MILVUS_URI;
  if (!configOrAddress)
    throw new Error(
      'Please pass in the address or set MILVUS_URI environment variable.\n'
    );
  return { configOrAddress: configOrAddress } as MilvusConfiguration;
}








