import {
  RecordedMessage,
  ToBeRecordedMessage,
  UnrecordedMessage,
} from './message';
import { Chat } from './chat';
import {
  QueryCommandOutput,
  RetrieveCommandOutput,
} from '@aws-sdk/client-kendra';
import { GenerateImageParams } from './image';
import { MediaFormat } from '@aws-sdk/client-transcribe';

export type CreateChatResponse = {
  chat: Chat;
};

export type CreateMessagesRequest = {
  messages: ToBeRecordedMessage[];
};

export type CreateMessagesResponse = {
  messages: RecordedMessage[];
};

export type ListChatsResponse = {
  chats: Chat[];
};

export type FindChatByIdResponse = {
  chat: Chat;
};

export type ListMessagesResponse = {
  messages: RecordedMessage[];
};

export type UpdateFeedbackRequest = {
  createdDate: string;
  feedback: string;
};

export type UpdateFeedbackResponse = {
  message: RecordedMessage;
};

export type UpdateTitleRequest = {
  title: string;
};

export type UpdateTitleResponse = {
  chat: Chat;
};

export type PredictRequest = {
  messages: UnrecordedMessage[];
};

export type PredictResponse = string;

export type PredictTitleRequest = {
  chat: Chat;
  messages: UnrecordedMessage[];
};

export type PredictTitleResponse = string;

export type QueryKendraRequest = {
  query: string;
};

export type QueryKendraResponse = QueryCommandOutput;

export type RetrieveKendraRequest = {
  query: string;
};

export type RetrieveKendraResponse = RetrieveCommandOutput;

export type GenerateImageRequest = GenerateImageParams;
export type GenerateImageResponse = string;

export type GetSignedUrlRequest = {
  mediaFormat: MediaFormat;
};

export type GetSignedUrlResponse = string;

export type StartTranscriptionRequest = {
  audioUrl: string;
};

export type StartTranscriptionResponse = {
  jobName: string;
};

export type GetTranscriptionResponse = {
  status: string;
  transcript?: string;
};

export type UploadAudioRequest = {
  file: File;
};

export type SettingResponse = {
  modelType: string;
  modelRegion: string;
  modelName: string;
  promptTemplateFile: string;
  imageGenModelName: string;
};
