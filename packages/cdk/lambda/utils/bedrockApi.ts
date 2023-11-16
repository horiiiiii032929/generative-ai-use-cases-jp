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

const PARAMS = {
  textGenerationConfig: {
    maxTokenCount: 8192,
    temperature:0,
    topP:1
  }
}

const bedrockApi: ApiInterface = {
  invoke: async (messages) => {
    const command = new InvokeModelCommand({
      modelId: process.env.MODEL_NAME,
      body: JSON.stringify({
        inputText: generatePrompt(messages),
        ...PARAMS,
      }),
      contentType: 'application/json',
    });
    const data = await client.send(command);
    return JSON.parse(data.body.transformToString()).results[0].outputText;
  },
  invokeStream: async function* (messages) {
    const command = new InvokeModelWithResponseStreamCommand({
      modelId: process.env.MODEL_NAME,
      body: JSON.stringify({
        inputText: generatePrompt(messages),
        ...PARAMS,
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
      if (body.outputText) {
        yield body.outputText;
      }
      if (body.completionReason) {
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
