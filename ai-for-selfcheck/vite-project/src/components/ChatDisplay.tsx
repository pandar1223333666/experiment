// --- START OF FILE ChatDisplay.tsx (修改后) ---

import { Avatar, Paper, Typography, Box } from '@mui/material';
import type { Message } from '../App';
import { useEffect, useRef, useState } from 'react';

interface ChatBubbleProps extends Message { }
function ChatBubble({ role, content }: ChatBubbleProps) {
  const isUser = role === 'user';
  return (
    <Box display="flex" justifyContent={isUser ? 'flex-end' : 'flex-start'} alignItems="flex-start" mb={2} gap={1}>
      {!isUser && <Avatar src="/ai.png" sx={{ width: 32, height: 32, mt: '4px' }} />}
      <Paper elevation={0} sx={{ px: 2, py: 1, maxWidth: isUser ? '40%' : '50%', bgcolor: isUser ? 'primary.main' : 'grey.200', color: isUser ? 'white' : 'black', borderRadius: 4, }}>
        <Typography variant="body1" sx={{ m: 0, whiteSpace: 'pre-wrap', lineHeight: 1.6, overflowWrap: 'break-word', wordBreak: 'break-word', }}>
          {content}
        </Typography>
      </Paper>
      {isUser && <Avatar src="/user.png" sx={{ width: 32, height: 32, mt: '4px' }} />}
    </Box>
  );
}

interface ChatDisplayProps {
  messages: Message[];
}
function ChatDisplay({ messages }: ChatDisplayProps) {
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
    const saved = localStorage.getItem('chatScrollTop');
    if (scrollRef.current && saved) {
      scrollRef.current.scrollTop = parseInt(saved, 10);
    }
  }, []);

  // 保存滚动位置
  useEffect(() => {
    const ref = scrollRef.current;
    if (!ref) return;
    const handler = () => {
      localStorage.setItem('chatScrollTop', String(ref.scrollTop));
    };
    ref.addEventListener('scroll', handler);
    return () => {
      ref.removeEventListener('scroll', handler);
    };
  }, []);

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
          <ChatBubble key={idx} {...msg} />
        ))}
      </Box>
    </Box>
  );
}

export default ChatDisplay;