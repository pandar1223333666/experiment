// --- START OF FILE App.tsx (修改后) ---

import Sidebar from './components/Sidebar'
import Header from './components/Header'
import ChatDisplay from './components/ChatDisplay'
import ChatInput from './components/ChatInput'
import './App.css'
import { useEffect, useState } from 'react'
import axios from 'axios';

// 定义消息类型
export type Message = {
  role: 'user' | 'assistant';
  content: string;
};

function App() {
  // 1. 消息状态
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = localStorage.getItem('chatMessages');
    return saved ? JSON.parse(saved) : [];
  });

  // 聊天消息持久化
  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages));
  }, [messages]);

  // 2. 发送消息逻辑
  const sendMessage = async (userContent: string) => {
    setMessages(msgs => [...msgs, { role: 'user', content: userContent }]);
    try {
      const res = await axios.post('http://localhost:8000/chat', { message: userContent });
      setMessages(msgs => [...msgs, { role: 'assistant', content: res.data.reply }]);
    } catch (err) {
      setMessages(msgs => [...msgs, { role: 'assistant', content: 'AI回复失败，请重试。' }]);
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
      <Sidebar onWidthChange={handleSidebarWidthChange} />
      <div
        className="main-content"
        style={{ maxWidth: `calc(100% - ${sidebarWidth}px)` }}
      >
        <Header theme={theme} onToggleTheme={toggleTheme} title={chatTitle} />
        <div className="chat-area-container">
          {/* 传递消息和发送函数 */}
          <ChatDisplay messages={messages} />
          <ChatInput onSend={sendMessage} />
        </div>
      </div>
    </div>
  )
}

export default App