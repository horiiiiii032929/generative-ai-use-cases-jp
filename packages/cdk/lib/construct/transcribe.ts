import { Duration, RemovalPolicy } from 'aws-cdk-lib';
import {
  AuthorizationType,
  CognitoUserPoolsAuthorizer,
  LambdaIntegration,
  RestApi,
} from 'aws-cdk-lib/aws-apigateway';
import { UserPool } from 'aws-cdk-lib/aws-cognito';
import { Effect, PolicyStatement } from 'aws-cdk-lib/aws-iam';
import { Runtime } from 'aws-cdk-lib/aws-lambda';
import { NodejsFunction } from 'aws-cdk-lib/aws-lambda-nodejs';
import { Bucket, BucketEncryption, HttpMethods } from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface TranscribeProps {
  userPool: UserPool;
  api: RestApi;
}

export class Transcribe extends Construct {
  constructor(scope: Construct, id: string, props: TranscribeProps) {
    super(scope, id);

    const audioBucket = new Bucket(this, 'AudioBucket', {
      encryption: BucketEncryption.S3_MANAGED,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });
    audioBucket.addCorsRule({
      allowedOrigins: ['*'],
      allowedMethods: [HttpMethods.PUT],
      allowedHeaders: ['*'],
      exposedHeaders: [],
      maxAge: 3000,
    });

    const transcriptBucket = new Bucket(this, 'TranscriptBucket', {
      encryption: BucketEncryption.S3_MANAGED,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    const getSignedUrlFunction = new NodejsFunction(this, 'GetSignedUrl', {
      runtime: Runtime.NODEJS_18_X,
      entry: './lambda/getSignedUrl.ts',
      timeout: Duration.minutes(15),
      environment: {
        AUDIO_BUCKET_NAME: audioBucket.bucketName,
      },
    });
    audioBucket.grantWrite(getSignedUrlFunction);

    const startTranscriptionFunction = new NodejsFunction(
      this,
      'StartTranscription',
      {
        runtime: Runtime.NODEJS_18_X,
        entry: './lambda/startTranscription.ts',
        timeout: Duration.minutes(15),
        environment: {
          TRANSCRIPT_BUCKET_NAME: transcriptBucket.bucketName,
        },
        initialPolicy: [
          new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['transcribe:*'],
            resources: ['*'],
          }),
        ],
      }
    );
    audioBucket.grantRead(startTranscriptionFunction);
    transcriptBucket.grantWrite(startTranscriptionFunction);

    const getTranscriptionFunction = new NodejsFunction(
      this,
      'GetTranscription',
      {
        runtime: Runtime.NODEJS_18_X,
        entry: './lambda/getTranscription.ts',
        timeout: Duration.minutes(15),
        initialPolicy: [
          new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['transcribe:*'],
            resources: ['*'],
          }),
        ],
      }
    );
    transcriptBucket.grantRead(getTranscriptionFunction);

    // API Gateway
    const authorizer = new CognitoUserPoolsAuthorizer(this, 'Authorizer', {
      cognitoUserPools: [props.userPool],
    });

    const commonAuthorizerProps = {
      authorizationType: AuthorizationType.COGNITO,
      authorizer,
    };
    const transcribeResource = props.api.root.addResource('transcribe');

    // POST: /transcribe/start
    transcribeResource
      .addResource('start')
      .addMethod(
        'POST',
        new LambdaIntegration(startTranscriptionFunction),
        commonAuthorizerProps
      );

    // POST: /transcribe/url
    transcribeResource
      .addResource('url')
      .addMethod(
        'POST',
        new LambdaIntegration(getSignedUrlFunction),
        commonAuthorizerProps
      );

    // GET: /transcribe/result/{jobName}
    transcribeResource
      .addResource('result')
      .addResource('{jobName}')
      .addMethod(
        'GET',
        new LambdaIntegration(getTranscriptionFunction),
        commonAuthorizerProps
      );
  }
}
