import {
  BedrockRuntimeClient,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
} from '@aws-sdk/client-bedrock-runtime';
import { generatePrompt } from './prompter';
import { ApiInterface } from 'generative-ai-use-cases-jp/src/utils';

const client = new BedrockRuntimeClient({
  region: process.env.MODEL_REGION,
});

// claude
// const PARAMS = {
//   max_tokens_to_sample: 3000,
//   temperature: 0.6,
//   top_k: 300,
//   top_p: 0.8,
// };

// jurassic-2
// const PARAMS = {
//   maxTokens: 8000,
//   temperature: 0.7,
//   topP: 1,
//   stopSequences: [],
//   countPenalty: {
//     scale: 0
//   },
//   presencePenalty: {
//     scale: 0
//   },
//   frequencyPenalty: {
//     scale: 0
//   }
// }

// cohere
const PARAMS = {
  max_tokens: 3000,
  temperature: 0.75,
  p: 0.01,
  k: 0,
  return_likelihoods: "NONE",
};

// llama
// const PARAMS = {
//   max_gen_len: 1900,
//   temperature: 0.5,
//   top_p: 0.9
// };

// titan
// const myObject = {
//   inputText: "",
//   textGenerationConfig: {
//     maxTokenCount: 7000,
//     stopSequences: [],
//     temperature: 0,
//     topP: 0.9
//   }
// };

const bedrockApi: ApiInterface = {
  invoke: async (messages) => {
    console.log(messages)
    console.log(PARAMS)
    const command = new InvokeModelCommand({
      modelId: process.env.MODEL_NAME,
      body: JSON.stringify({
        prompt: generatePrompt(messages),
        // ...PARAMS,
      }),
      contentType: 'application/json',
    });
    const data = await client.send(command);
    const aData = JSON.parse(data.body.transformToString());
    console.log(aData)
    return JSON.parse(data.body.transformToString()).generations[0].text;
  },
  invokeStream: async function* (messages) {
    const command = new InvokeModelWithResponseStreamCommand({
      modelId: process.env.MODEL_NAME,
      body: JSON.stringify({
        prompt: generatePrompt(messages),
        // ...PARAMS,
      }),
      contentType: 'application/json',
    });
    const res = await client.send(command);

    if (!res.body) {
      return;
    }

    for await (const streamChunk of res.body) {
      if (!streamChunk.chunk?.bytes) {
        break;
      }
      const body = JSON.parse(
        new TextDecoder('utf-8').decode(streamChunk.chunk?.bytes)
      );
      console.log(body)
      if (body.generations[0].text) {
        yield body.generations[0].text;
      }
      if (body.generations[0].finish_reason) {
        break;
      }
    }
  },
  generateImage: async (params) => {
    // 現状 StableDiffusion のみの対応です。
    // パラメータは以下を参照
    // https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage
    const command = new InvokeModelCommand({
      modelId: process.env.IMAGE_GEN_MODEL_NAME,
      body: JSON.stringify({
        text_prompts: params.textPrompt,
        cfg_scale: params.cfgScale,
        style_preset: params.stylePreset,
        seed: params.seed,
        steps: params.step,
        init_image: params.initImage,
        image_strength: params.imageStrength,
      }),
      contentType: 'application/json',
    });
    const res = await client.send(command);
    const body = JSON.parse(Buffer.from(res.body).toString('utf-8'));

    if (body.result !== 'success') {
      throw new Error('Failed to invoke model');
    }

    return body.artifacts[0].base64;
  },
};

export default bedrockApi;
