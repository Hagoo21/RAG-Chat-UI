export const initState = {
  conversation: [],
  current: 0,
  chat: [
    {
      title: "Welcome",
      id: Date.now(),
      ct: new Date().toISOString(),
      messages: [],
      icon: [2, "files"],
    }
  ],
  currentChat: 0,
  options: {
    account: {
      name: "CHAT——AI",
      avatar: "",
    },
    general: {
      language: "English",
      theme: "light",
      command: "COMMAND_ENTER",
      size: "normal",
    },
    openai: {
      baseUrl: "",
      organizationId: "",
      temperature: 1,
      model: "gpt-3.5-turbo",
      apiKey: "",
      max_tokens: 2048,
      n: 1,
      stream: true,
    },
  },
  is: {
    typeing: false,
    config: false,
    fullScreen: true,
    sidebar: true,
    inputing: false,
    thinking: false,
    apps: false,
    knowledge: false
  },
  typeingMessage: { content: '' },
  version: "0.1.0",
}