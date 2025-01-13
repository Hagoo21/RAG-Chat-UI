import React, { useState } from 'react';
import { AudioRecorder as VoiceRecorder } from 'react-audio-voice-recorder';
import axios from 'axios';
import styles from './style.module.less';
import { Icon } from '@/components';
import { classnames } from '@/components/utils';
import { useGlobal } from '../../context';

export function AudioRecorder({ onTranscription }) {
  const [isRecording, setIsRecording] = useState(false);
  const { options } = useGlobal();

  const addAudioElement = async (blob) => {
    if (!options?.openai?.apiKey) {
      console.error('OpenAI API key is not configured');
      return;
    }

    const formData = new FormData();
    formData.append('file', blob, 'audio.webm');
    formData.append('model', 'whisper-1');

    try {
      const response = await axios.post('https://api.openai.com/v1/audio/transcriptions', formData, {
        headers: {
          'Authorization': `Bearer ${options.openai.apiKey}`,
          'Content-Type': 'multipart/form-data',
        },
      });
      onTranscription(response.data.text);
    } catch (error) {
      console.error('Transcription error:', error);
    }
  };

  return (
    <div className={styles.recorder}>
      <VoiceRecorder 
        onRecordingComplete={addAudioElement}
        audioTrackConstraints={{
          noiseSuppression: true,
          echoCancellation: true,
        }}
        onStartRecording={() => setIsRecording(true)}
        onStopRecording={() => setIsRecording(false)}
      />
      <Icon 
        type="mic" 
        className={classnames(styles.icon, isRecording && styles.recording)} 
      />
    </div>
  );
}