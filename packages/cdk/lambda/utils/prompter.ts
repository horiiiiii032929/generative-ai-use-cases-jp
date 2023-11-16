import { PromptTemplate, UnrecordedMessage } from 'generative-ai-use-cases-jp';

export const pt: PromptTemplate = JSON.parse(process.env.PROMPT_TEMPLATE || '');

export const generatePrompt = (messages: UnrecordedMessage[]) => {
  console.log(pt)
  const prompt =
    pt.prefix +
    messages
      .map((message) => {
        if (message.role == 'user') {
          return pt.user.replace('{}', message.content);
        } else if (message.role == 'assistant') {
          return pt.assistant.replace('{}', message.content);
        } else if (message.role === 'system') {
          return pt.system.replace('{}', message.content);
        } else {
          throw new Error(`Invalid message role: ${message.role}`);
        }
      })
      .join(pt.join) +
    pt.suffix;
  return prompt;
};
