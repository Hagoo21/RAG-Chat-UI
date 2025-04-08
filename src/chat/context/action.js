import { fetchStream } from "../service";
import { getContext } from "../service/api";

export default function action(state, dispatch) {
  const setState = (payload = {}) =>
    dispatch({
      type: "SET_STATE",
      payload: { ...payload },
    });

  const ensureValidChatState = () => {
    if (!state.chat || !Array.isArray(state.chat) || state.chat.length === 0) {
      setState({
        chat: [{
          title: "Welcome",
          id: Date.now(),
          ct: new Date().toISOString(),
          messages: [],
          icon: [2, "files"],
        }],
        currentChat: 0
      });
      return false;
    }
    
    if (state.currentChat >= state.chat.length) {
      setState({ currentChat: state.chat.length - 1 });
      return false;
    }
    
    return true;
  };

  // Helper to get limited message history
  const getLimitedMessageHistory = (messages, limit = 4) => {
    if (!messages || messages.length === 0) return [];
    
    // Always include the latest message
    const latestMessage = messages[messages.length - 1];
    
    // Get previous messages up to the limit
    const previousMessages = messages.slice(-limit, -1);
    
    return [...previousMessages, latestMessage];
  };

  return {
    setState,
    clearTypeing() {
      setState({ 
        typeingMessage: { content: '' },
        is: { ...state.is, typeing: false }
      });
    }, 
    async sendMessage() {
      if (!ensureValidChatState()) return;
    
      const { typeingMessage, chat, is, currentChat } = state;
      if (!typeingMessage?.content) return;
    
      try {
        // Store the original message in chat history
        const messages = [...(chat[currentChat].messages || []), {
          ...typeingMessage,
          sentTime: Date.now()
        }];

        let newChat = [...chat];
        newChat[currentChat] = { ...chat[currentChat], messages };

        // Set thinking state immediately
        setState({
          is: { ...is, thinking: true },
          typeingMessage: { content: '' },
          chat: newChat,
        });

        let answer = '';
        let currentTool = null;
        
        // Get streaming response from the chat endpoint
        await getContext(
          typeingMessage.content,
          (data) => {
            if (data.content) {
              // Raw response content (token by token)
              answer += data.content;
              if (!newChat[currentChat]) return;
              newChat[currentChat] = {
                ...chat[currentChat],
                messages: [
                  ...messages,
                  {
                    content: answer,
                    role: "assistant",
                    sentTime: Date.now(),
                    id: Date.now(),
                  },
                ],
              };
              setState({
                is: { ...is, thinking: answer.length },
                chat: newChat,
              });
            } else if (data.tool) {
              // Tool usage notification
              currentTool = data.tool;
              // You could show a loading indicator for the specific tool here
            } else if (data.tool_output) {
              // Tool output received
              currentTool = null;
              // You could show the tool output in a different format here
            } else if (data.agent_update) {
              // Agent change notification
              // You could show this in the UI if desired
            } else if (data.message) {
              // Complete message received
              answer = data.message;
              if (!newChat[currentChat]) return;
              newChat[currentChat] = {
                ...chat[currentChat],
                messages: [
                  ...messages,
                  {
                    content: answer,
                    role: "assistant",
                    sentTime: Date.now(),
                    id: Date.now(),
                  },
                ],
              };
              setState({
                is: { ...is, thinking: false },
                chat: newChat,
              });
            }
          },
          (error) => {
            console.error('Stream error:', error);
            if (newChat[currentChat]) {
              newChat[currentChat] = {
                ...chat[currentChat],
                error,
              };
              setState({
                chat: newChat,
                is: { ...is, thinking: false },
              });
            }
          },
          () => {
            setState({
              is: { ...is, thinking: false },
            });
          }
        );
      } catch (error) {
        console.error('Error sending message:', error);
        setState({
          is: { ...is, thinking: false },
        });
      }
    },

    newChat() {
      const { chat } = state;
      const chatList = [
        ...chat,
        {
          title: "New Conversation",
          id: Date.now(),
          messages: [],
          ct: new Date().toISOString(),
          icon: [2, "files"],
        },
      ];
      setState({ chat: chatList, currentChat: chatList.length - 1 });
    },

    modifyChat(arg, index) {
      if (!ensureValidChatState()) return;
      const chat = [...state.chat];
      chat[index] = { ...chat[index], ...arg };
      setState({ chat, currentEditor: null });
    },

    editChat(index, title) {
      if (!ensureValidChatState()) return;
      const chat = [...state.chat];
      chat[index] = { ...chat[index], title };
      setState({ chat });
    },
    
    removeChat(index) {
      if (!ensureValidChatState()) return;
      const chat = [...state.chat];
      chat.splice(index, 1);
      
      if (chat.length === 0) {
        chat.push({
          title: "New Conversation",
          id: Date.now(),
          messages: [],
          ct: new Date().toISOString(),
          icon: [2, "files"],
        });
      }
      
      setState({
        chat,
        currentChat: state.currentChat === index ? Math.max(0, index - 1) : state.currentChat
      });
    },

    setMessage(content) {
      const typeingMessage = content === "" ? { content: '' } : {
        role: "user",
        content,
        id: Date.now(),
      };
      setState({ 
        is: { ...state.is, typeing: content !== '' }, 
        typeingMessage 
      });
    },   

    clearMessage() {
      if (!ensureValidChatState()) return;
      const chat = [...state.chat];
      chat[state.currentChat] = { ...chat[state.currentChat], messages: [] };
      setState({ chat });
    },

    removeMessage(index) {
      if (!ensureValidChatState()) return;
      const chat = [...state.chat];
      const messages = [...chat[state.currentChat].messages];
      messages.splice(index, 1);
      chat[state.currentChat] = { ...chat[state.currentChat], messages };
      setState({ chat });
    },

    setOptions({ type, data = {} }) {
      const options = { 
        ...state.options,
        [type]: { ...state.options[type], ...data }
      };
      setState({ options });
    },

    setIs(arg) {
      setState({ is: { ...state.is, ...arg } });
    },

    currentList() {
      return ensureValidChatState() ? state.chat[state.currentChat] : null;
    },

    stopResonse() {
      setState({
        is: { ...state.is, thinking: false },
      });
    },
  };
}