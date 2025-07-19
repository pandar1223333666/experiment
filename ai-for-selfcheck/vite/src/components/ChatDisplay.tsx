// --- START OF FILE ChatDisplay.tsx (修改后) ---

import { Avatar, Paper, Typography, Box } from '@mui/material';
import type { Message } from '../App';
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

interface ChatBubbleProps extends Message {
  theme?: 'light' | 'dark';
}

function ChatBubble({ role, content, theme = 'light', isWaiting, isStreaming }: ChatBubbleProps) {
  const isUser = role === 'user';
  const isDark = theme === 'dark';

  // 确保内容是字符串
  const safeContent = typeof content === 'string' ? content : '';

  return (
    <Box display="flex" justifyContent={isUser ? 'flex-end' : 'flex-start'} alignItems="flex-start" mb={2} gap={1}>
      {!isUser && (
        <Avatar
          sx={{
            width: 38,
            height: 38,
            mt: '4px',
            bgcolor: isDark ? '#3b5f8c' : '#a0d0ff',
            color: isDark ? '#a0d0ff' : '#1677ff',
            fontWeight: 'bold',
            fontSize: '14px',
            boxShadow: isDark ? '0 2px 6px rgba(22, 119, 255, 0.15)' : '0 2px 6px rgba(22, 119, 255, 0.3)',
            fontFamily: '"Noto Serif SC", serif',
            letterSpacing: '1px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: isDark
              ? 'linear-gradient(135deg, #3b5f8c 0%, #2a4365 100%)'
              : 'linear-gradient(135deg, #c2e0ff 0%, #a0d0ff 100%)',
            border: isDark ? '1px solid #4a6890' : '1px solid #d9eaff'
          }}
        >
          <span style={{ fontStyle: 'italic', transform: 'rotate(-5deg)' }}>知测</span>
        </Avatar>
      )}
      <Paper
        elevation={1}
        sx={{
          px: 2,
          py: 1.5,
          maxWidth: isUser ? '50%' : '60%',
          bgcolor: isUser
            ? (isDark ? '#1a365d' : '#e6f4ff')
            : (isDark ? '#2c3138' : 'white'),
          color: isUser
            ? (isDark ? '#e2e8f0' : '#333')
            : (isDark ? '#e2e8f0' : '#333'),
          borderRadius: isUser ? '16px 16px 0 16px' : '16px 16px 16px 0',
          boxShadow: isDark ? '0 2px 6px rgba(0,0,0,0.2)' : '0 2px 6px rgba(0,0,0,0.05)',
          border: isUser
            ? (isDark ? '1px solid #2a4365' : '1px solid #d9eaff')
            : (isDark ? '1px solid #3a3f47' : '1px solid #e0e0e0'),
        }}
      >
        {isUser ? (
          <Typography
            variant="body1"
            sx={{
              m: 0,
              whiteSpace: 'pre-wrap',
              lineHeight: 1.6,
              overflowWrap: 'break-word',
              wordBreak: 'break-word',
              fontSize: '15px'
            }}
          >
            {safeContent}
          </Typography>
        ) : (
          <div className="chat-assistant-content">
            {/* 使用ReactMarkdown替代dangerouslySetInnerHTML */}
            {!isWaiting && (
              <div className={`markdown-content ${isStreaming ? 'streaming' : ''}`}>
                <ReactMarkdown rehypePlugins={[rehypeRaw]} remarkPlugins={[remarkGfm]}>
                  {safeContent}
                </ReactMarkdown>
              </div>
            )}
            {/* 等待回复或流式输出时显示加载指示器 */}
            {(isWaiting || isStreaming) && (
              <span className="loading-spinner"></span>
            )}
          </div>
        )}
      </Paper>
    </Box>
  );
}

// 欢迎组件
function WelcomeMessage() {
  return (
    <div className="welcome-message">
      <h1 className="welcome-title">你好，我是知测</h1>
    </div>
  );
}

interface ChatDisplayProps {
  messages: Message[];
  isCentered?: boolean;
  theme?: 'light' | 'dark';
}

function ChatDisplay({ messages, isCentered = false, theme = 'light' }: ChatDisplayProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [inputHeight, setInputHeight] = useState(40); // 默认输入框高度

  // 监听输入框高度变化
  useEffect(() => {
    const handleInputResize = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail && customEvent.detail.height) {
        setInputHeight(customEvent.detail.height);
      }
    };

    document.addEventListener('chat-input-resize', handleInputResize);
    return () => {
      document.removeEventListener('chat-input-resize', handleInputResize);
    };
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current && messages.length > 0) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, inputHeight]);

  // 恢复滚动位置
  useEffect(() => {
    const saved = sessionStorage.getItem('chatScrollTop');
    if (scrollRef.current && saved) {
      scrollRef.current.scrollTop = parseInt(saved, 10);
    }
  }, []);

  // 保存滚动位置
  useEffect(() => {
    const ref = scrollRef.current;
    if (!ref) return;
    const handler = () => {
      sessionStorage.setItem('chatScrollTop', String(ref.scrollTop));
    };
    ref.addEventListener('scroll', handler);
    return () => {
      ref.removeEventListener('scroll', handler);
    };
  }, []);

  // 如果是居中模式，只显示欢迎信息
  if (isCentered) {
    return <WelcomeMessage />;
  }

  // 如果没有消息，显示欢迎界面
  if (messages.length === 0) {
    return <WelcomeMessage />;
  }

  return (
    <Box
      className="chat-display-scroll-container"
      ref={scrollRef}
      sx={{
        // 让滚动条正常显示
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        overflowY: 'auto',
        // 移除遮罩相关样式
        WebkitMaskImage: 'none',
        maskImage: 'none',
        padding: '20px 0',
      }}
    >
      <Box
        sx={{
          width: '100%',
          maxWidth: '800px',
          margin: '0 auto',
          px: 2,
          py: 3,
          // 根据输入框高度动态调整底部padding，确保聊天内容不被输入框遮挡
          paddingBottom: `${120 + (inputHeight - 40)}px`,
        }}
      >
        {messages.map((msg, idx) => (
          <ChatBubble key={idx} {...msg} theme={theme} />
        ))}
      </Box>
    </Box>
  );
}

export default ChatDisplay;