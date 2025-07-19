import { useState, useRef, useEffect } from 'react';
import { FiSend, FiPaperclip } from 'react-icons/fi';

interface ChatInputProps {
  onSend: (content: string) => void;
  isCentered?: boolean;
}

function ChatInput({ onSend, isCentered = false }: ChatInputProps) {
  const [message, setMessage] = useState(() => {
    // 从会话存储中恢复输入框内容
    return sessionStorage.getItem('chatInputDraft') || '';
  });
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 调整文本区域高度的函数
  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      // 重置高度以获取正确的scrollHeight
      textarea.style.height = 'auto';
      // 设置新高度，但最大不超过200px
      const newHeight = Math.min(textarea.scrollHeight, 200);
      textarea.style.height = `${newHeight}px`;

      // 当内容超过最大高度时显示滚动条
      textarea.style.overflowY = textarea.scrollHeight > 200 ? 'auto' : 'hidden';

      // 通知父组件输入框高度变化
      document.dispatchEvent(new CustomEvent('chat-input-resize', {
        detail: { height: newHeight }
      }));
    }
  };

  // 组件加载时调整高度
  useEffect(() => {
    adjustTextareaHeight();
  }, []);

  // 监听消息变化时调整高度并保存到会话存储
  useEffect(() => {
    adjustTextareaHeight();
    // 保存输入框内容到会话存储
    sessionStorage.setItem('chatInputDraft', message);
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSend(message);
      setMessage('');
      // 清除会话存储中的草稿
      sessionStorage.removeItem('chatInputDraft');
      // 重置输入框高度
      if (textareaRef.current) {
        textareaRef.current.style.height = '40px';
        // 重置滚动条状态
        textareaRef.current.style.overflowY = 'hidden';
        // 通知高度变化
        document.dispatchEvent(new CustomEvent('chat-input-resize', {
          detail: { height: 40 }
        }));
      }
    }
  };

  const handleFileUpload = () => {
    // TODO: Implement file upload functionality
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.click();

    fileInput.onchange = (e) => {
      const target = e.target as HTMLInputElement;
      if (target.files && target.files.length > 0) {
        const file = target.files[0];
        console.log('File selected:', file.name);
        // Implement file handling logic here
      }
    };
  };

  return (
    <div className={`chat-input ${isCentered ? 'centered-input' : ''}`}>
      <form onSubmit={handleSubmit} className="chat-input-form">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="询问任何问题->😊"
          className="chat-input-textarea"
          style={{ overflowY: 'hidden' }} // 初始状态隐藏滚动条
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
        />
        <button
          type="button"
          onClick={handleFileUpload}
          className="chat-input-button file-button"
        >
          <FiPaperclip className="button-icon" />
        </button>
        <button
          type="submit"
          disabled={!message.trim()}
          className="chat-input-button send-button"
        >
          <FiSend className="button-icon" />
        </button>
      </form>
    </div>
  );
}

export default ChatInput;
