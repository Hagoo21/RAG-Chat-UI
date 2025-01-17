import { createParser } from "eventsource-parser";
import { setAbortController } from "./abortController.mjs";

export async function* streamAsyncIterable(stream) {
  const reader = stream.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        return;
      }
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

export const fetchBaseUrl = (baseUrl) =>
  baseUrl || "https://api.openai.com/v1/chat/completions";

export const fetchHeaders = (options = {}) => {
  const { organizationId, apiKey } = options;
  return {
    Authorization: "Bearer " + apiKey,
    "Content-Type": "application/json",
    ...(organizationId && { "OpenAI-Organization": organizationId }),
  };
};

export const throwError = async (response) => {
  if (!response.ok) {
    let errorPayload = null;
    try {
      errorPayload = await response.json();
      console.log(errorPayload);
    } catch (e) {
      // ignore
    }
  }
};

// Helper function to count tokens (rough estimate)
const estimateTokenCount = (text) => {
  // Rough estimate: 1 token ≈ 4 characters for English text
  return Math.ceil(text.length / 4);
};

// Function to limit context window
const limitContextWindow = (messages, maxTokens = 4000) => {
  let totalTokens = 0;
  const limitedMessages = [];
  
  // Process messages in reverse to keep most recent context
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    const estimatedTokens = estimateTokenCount(message.content);
    
    if (totalTokens + estimatedTokens <= maxTokens) {
      limitedMessages.unshift(message);
      totalTokens += estimatedTokens;
    } else {
      break;
    }
  }
  
  return limitedMessages;
};

export const fetchBody = ({ options = {}, messages = [] }) => {
  const { top_p, n, max_tokens, temperature, model, stream } = options;
  
  // Limit context window to prevent rate limit errors
  const limitedMessages = limitContextWindow(messages);
  
  return {
    messages: limitedMessages,
    stream,
    n: 1,
    ...(model && { model }),
    ...(temperature && { temperature }),
    ...(max_tokens && { max_tokens }),
    ...(top_p && { top_p }),
    ...(n && { n }),
  };
};

export const fetchAction = async ({
  method = "POST",
  messages = [],
  options = {},
  signal,
}) => {
  const { baseUrl, ...rest } = options;
  const url = fetchBaseUrl(baseUrl);
  const headers = fetchHeaders({ ...rest });
  const body = JSON.stringify(fetchBody({ messages, options }));
  const response = await fetch(url, {
    method,
    headers,
    body,
    signal,
  });
  return response;
};

export const fetchStream = async ({
  options,
  messages,
  onMessage,
  onEnd,
  onError,
  onStar,
}) => {
  let answer = "";
  const { controller, signal } = setAbortController();
  
  try {
    const result = await fetchAction({ options, messages, signal });
    
    if (!result.ok) {
      const error = await result.json();
      onError && onError(error);
      return;
    }

    const parser = createParser((event) => {
      if (event.type === "event") {
        if (event.data === "[DONE]") {
          return;
        }
        let data;
        try {
          data = JSON.parse(event.data);
        } catch (error) {
          return;
        }
        if ("content" in data.choices[0].delta) {
          answer += data.choices[0].delta.content;
          onMessage && onMessage(answer, controller);
        }
      }
    });

    let hasStarted = false;
    for await (const chunk of streamAsyncIterable(result.body)) {
      const str = new TextDecoder().decode(chunk);
      parser.feed(str);
      if (!hasStarted) {
        hasStarted = true;
        onStar && onStar(str, controller);
      }
    }
  } catch (error) {
    onError && onError(error, controller);
  } finally {
    await onEnd();
  }
};