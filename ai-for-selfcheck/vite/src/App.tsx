// --- START OF FILE App.tsx (修改后) ---

import Sidebar from './components/Sidebar'
import Header from './components/Header'
import ChatDisplay from './components/ChatDisplay'
import ChatInput from './components/ChatInput'
import './App.css'
import { useEffect, useState } from 'react'
//import axios from 'axios';

// 定义消息类型
export type Message = {
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean; // 标记是否正在流式输出
  isWaiting?: boolean; // 新增：标记是否正在等待回复
};

function App() {
  // 1. 消息状态 - 从sessionStorage中加载已保存的消息
  const [messages, setMessages] = useState<Message[]>(() => {
    // 尝试从sessionStorage获取之前保存的聊天记录
    const savedMessages = sessionStorage.getItem('chatMessages');
    // 如果存在保存的消息，则解析并返回；否则返回空数组
    return savedMessages ? JSON.parse(savedMessages) : [];
  });

  // 会话ID状态管理
  const [sessionId, setSessionId] = useState<string | null>(() => {
    return sessionStorage.getItem('currentSessionId');
  });

  // 新增：控制输入框位置的状态
  const [isCentered, setIsCentered] = useState(messages.length === 0);

  // 聊天消息持久化到sessionStorage
  useEffect(() => {
    // 只有当没有消息正在流式输出时才保存
    if (!messages.some(msg => msg.isStreaming)) {
      sessionStorage.setItem('chatMessages', JSON.stringify(messages));
    }

    // 有消息时，取消居中布局
    if (messages.length > 0) {
      setIsCentered(false);
    }
  }, [messages]);

  // 2. 发送消息逻辑 - 适配后端API的流式响应
  const sendMessage = async (userContent: string) => {
    // 添加用户消息到聊天列表
    setMessages(msgs => [...msgs, { role: 'user', content: userContent }]);

    try {
      // 先添加一个空的助手消息，标记为正在等待
      setMessages(msgs => [...msgs, { role: 'assistant', content: '', isWaiting: true }]);

      // 更新消息状态为开始流式输出
      setMessages(msgs => {
        const newMsgs = [...msgs];
        const lastIndex = newMsgs.length - 1;
        newMsgs[lastIndex] = {
          ...newMsgs[lastIndex],
          isWaiting: false,
          isStreaming: true,
          content: ''
        };
        return newMsgs;
      });

      // 定义后端API URL
      const API_URL = 'http://localhost:8000/api/llm/chat';
      console.log('发送请求到:', API_URL);

      // 准备FormData
      const formData = new FormData();

      // 添加当前消息
      formData.append('message', userContent);

      // 构建上下文，提取最近的对话历史（最多5轮）
      const recentMessages = messages.slice(-10); // 获取最近的10条消息
      if (recentMessages.length > 0) {
        // 将历史消息格式化为文本上下文
        let historyContext = "以下是之前的对话内容：\n";
        recentMessages.forEach(msg => {
          const role = msg.role === 'user' ? '用户' : '助手';
          historyContext += `${role}：${msg.content}\n`;
        });
        historyContext += "\n当前问题：" + userContent;

        // 使用构建的上下文替换原始消息
        formData.set('message', historyContext);
      }

      // 如果有session_id，添加到请求中
      if (sessionId) {
        formData.append('session_id', sessionId);
        console.log('使用会话ID:', sessionId);
      }

      console.log('准备发送请求，消息内容:', userContent);

      // 发送请求
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP错误! 状态: ${response.status}, 信息: ${response.statusText}, 响应: ${errorText}`);
      }

      // 获取完整响应文本
      const responseText = await response.text();
      console.log('完整响应:', responseText);

      try {
        // 解析JSON响应
        const jsonData = JSON.parse(responseText);

        // 保存会话ID - 更新：兼容request_id和session_id字段
        const sessionIdFromResponse = jsonData.request_id || jsonData.session_id;
        if (sessionIdFromResponse) {
          setSessionId(sessionIdFromResponse);
          sessionStorage.setItem('currentSessionId', sessionIdFromResponse);
          console.log('保存会话ID:', sessionIdFromResponse);
        }

        // 提取AI回复内容
        if (jsonData?.output?.choices?.[0]?.message?.content) {
          const aiResponse = jsonData.output.choices[0].message.content;
          console.log('AI回复:', aiResponse);

          // 实现真正的逐字符流式输出
          let displayContent = '';
          const fullContent = aiResponse;

          // 先清空内容，准备流式显示
          setMessages(msgs => {
            const newMsgs = [...msgs];
            const lastIndex = newMsgs.length - 1;
            newMsgs[lastIndex] = {
              ...newMsgs[lastIndex],
              content: '',
              isStreaming: true,
              isWaiting: false
            };
            return newMsgs;
          });

          // 逐字符显示内容
          for (let i = 0; i < fullContent.length; i++) {
            await new Promise(resolve => setTimeout(resolve, 15)); // 控制显示速度

            displayContent += fullContent[i];
            // 直接使用文本内容，不进行HTML转换
            const formattedContent = displayContent;

            setMessages(msgs => {
              const newMsgs = [...msgs];
              const lastIndex = newMsgs.length - 1;
              newMsgs[lastIndex] = {
                ...newMsgs[lastIndex],
                content: formattedContent,
                isStreaming: true
              };
              return newMsgs;
            });
          }

          // 完成后更新最终状态
          setMessages(msgs => {
            const newMsgs = [...msgs];
            const lastIndex = newMsgs.length - 1;
            newMsgs[lastIndex] = {
              role: 'assistant',
              content: fullContent,
              isStreaming: false,
              isWaiting: false
            };
            return newMsgs;
          });
        } else {
          throw new Error('无法从响应中提取AI回复内容');
        }
      } catch (error) {
        console.error('解析响应时出错:', error);
        throw error;
      }
    } catch (error) {
      console.error('发送消息时出错:', error);
      // 更新最后一条消息，显示错误状态
      setMessages(msgs => {
        const newMsgs = [...msgs];
        const lastIndex = newMsgs.length - 1;
        if (lastIndex >= 0 && newMsgs[lastIndex].role === 'assistant') {
          newMsgs[lastIndex] = {
            ...newMsgs[lastIndex],
            content: `出错了: ${error instanceof Error ? error.message : String(error)}`,
            isStreaming: false,
            isWaiting: false
          };
        }
        return newMsgs;
      });
    }
  };

  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme')
    return saved === 'dark' ? 'dark' : 'light'
  })
  const toggleTheme = () => {
    setTheme(t => {
      const next = t === 'light' ? 'dark' : 'light'
      localStorage.setItem('theme', next)
      return next
    })
  }
  useEffect(() => {
    document.body.classList.toggle('dark-theme', theme === 'dark')
    document.body.classList.toggle('light-theme', theme === 'light')
  }, [theme])
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    const sidebarState = localStorage.getItem('sidebarState')
    return sidebarState !== null ? (JSON.parse(sidebarState) ? 250 : 65) : 250
  })
  const handleSidebarWidthChange = (width: number) => {
    setSidebarWidth(width)
  }
  const chatTitle = "知测";


  return (
    <div className="app-container">
      <Sidebar onWidthChange={handleSidebarWidthChange} theme={theme} />
      <div
        className="main-content"
        style={{ maxWidth: `calc(100% - ${sidebarWidth}px)` }}
      >
        <Header theme={theme} onToggleTheme={toggleTheme} title={chatTitle} />
        <div className="chat-area-container">
          {isCentered ? (
            <div className="initial-container">
              <div className="welcome-message">
                <h1 className="welcome-title">你好，我是知测</h1>
              </div>
              <ChatInput onSend={sendMessage} isCentered={true} />
            </div>
          ) : (
            <>
              <ChatDisplay messages={messages} isCentered={false} theme={theme} />
              <ChatInput onSend={sendMessage} isCentered={false} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default App