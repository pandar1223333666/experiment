import { useState } from 'react';
import { FiSend, FiPaperclip } from 'react-icons/fi';

interface ChatInputProps {
  onSend: (content: string) => void;
}

function ChatInput({ onSend }: ChatInputProps) {
  const [message, setMessage] = useState('');
  // --- 保持滚动位置 ---
  // 记录滚动位置
  // 只需在组件挂载和卸载时处理
  // 依赖于 chat-display-scroll-container 的 class
  // 这里仅提供保存滚动位置的逻辑，实际恢复需在 ChatDisplay 组件实现

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSend(message);
      setMessage('');
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
    <div className="chat-input">
      <form onSubmit={handleSubmit} className="chat-input-form">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="询问任何问题->😊"
          className="chat-input-textarea"
          rows={1}
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
