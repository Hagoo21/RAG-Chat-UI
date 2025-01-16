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
      // Initialize with a default chat if invalid
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
    
    // Ensure currentChat is valid
    if (state.currentChat >= state.chat.length) {
      setState({ currentChat: state.chat.length - 1 });
      return false;
    }
    
    return true;
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
    
      const { typeingMessage, options, chat, is, currentChat } = state;
      if (!typeingMessage?.content) return;
    
      try {
        // Get context from the /chat endpoint
        const contextResponse = await getContext(typeingMessage.content);
        console.log('Context received:', contextResponse); // Debug log
        
        // Extract context from the response
        const context = contextResponse.context;
        
        // Create message with context
        const messageWithContext = `Context: ${context}\n\nQuestion: ${typeingMessage.content}`;
        console.log('Message with context:', messageWithContext); // Debug log
    
        // Store the original message in chat history (without context)
        const messages = [...(chat[currentChat].messages || []), {
          ...typeingMessage,
          sentTime: Date.now()
        }];
    
        let newChat = [...chat];
        newChat[currentChat] = { ...chat[currentChat], messages };
    
        setState({
          is: { ...is, thinking: true },
          typeingMessage: { content: '' },
          chat: newChat,
        });
    
        const controller = new AbortController();
    
        await fetchStream({
          messages: messages.map((msg, index) => {
            // Replace the last message with our context-enhanced version
            if (index === messages.length - 1) {
              return {
                role: msg.role,
                content: messageWithContext // Use the enhanced message with context
              };
            }
            return {
              role: msg.role,
              content: msg.content
            };
          }),
          options: options.openai,
          signal: controller.signal,
          onMessage(content) {
            if (!newChat[currentChat]) return;
            newChat[currentChat] = {
              ...chat[currentChat],
              messages: [
                ...messages,
                {
                  content,
                  role: "assistant",
                  sentTime: Date.now(),
                  id: Date.now(),
          },
              ],
            };
            setState({
              is: { ...is, thinking: content.length },
              chat: newChat,
            });
          },
          onStar() {},
          onEnd() {
            setState({
              is: { ...is, thinking: false },
            });
          },
          onError(res) {
            console.error('Stream error:', res);
            const { error } = res || {};
            if (error && newChat[currentChat]) {
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
        });
      } catch (error) {
        console.error('Send message error:', error);
        setState({
          is: { ...is, thinking: false },
        });
      }
    }
    ,
    

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
      
      // Ensure at least one chat remains
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