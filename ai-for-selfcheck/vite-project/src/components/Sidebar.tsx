import { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Tooltip
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import AddIcon from '@mui/icons-material/Add';

interface SidebarProps {
  onWidthChange?: (width: number) => void;
  onNewChat?: () => void;
  onSelectChat?: (id: string) => void;
}

interface ChatItem {
  id: string;
  title: string;
}

function Sidebar({ onWidthChange, onNewChat, onSelectChat }: SidebarProps) {
  const [isOpen, setIsOpen] = useState(() => {
    const savedState = localStorage.getItem('sidebarState');
    return savedState !== null ? JSON.parse(savedState) : true;
  });

  useEffect(() => {
    localStorage.setItem('sidebarState', JSON.stringify(isOpen));
    if (onWidthChange) {
      onWidthChange(isOpen ? 250 : 65);
    }
  }, [isOpen, onWidthChange]);

  // 模拟历史对话
  const [chats, setChats] = useState<ChatItem[]>([
    { id: '1', title: '与AI聊天' },
    { id: '2', title: '项目讨论' },
    { id: '3', title: '写作助手' }
  ]);

  return (
    <Box
      sx={{
        width: isOpen ? 250 : 65,
        height: '100vh',
        flexShrink: 0,
        bgcolor: 'background.paper',
        borderRight: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
        transition: 'width 0.3s ease',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      {isOpen ? (
        <Box sx={{ p: 2, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          {/* 标题和折叠按钮 */}
          <Box sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            mb: 2
          }}>
            <Typography
              variant="h6"
              noWrap
              sx={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
            >
              导航菜单
            </Typography>
            <IconButton onClick={() => setIsOpen(false)}>
              <ChevronLeftIcon />
            </IconButton>
          </Box>

          {/* 新对话按钮 */}
          <Button
            variant="contained"
            size="small"
            startIcon={<AddIcon />}
            onClick={onNewChat}
            sx={{ mb: 2 ,whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}
            
          >
            新对话
          </Button>

          {/* 历史对话列表 */}
          <List dense sx={{ overflowY: 'auto', flexGrow: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>
            {chats.map(chat => (
              <ListItem key={chat.id} disablePadding>
                <ListItemButton onClick={() => onSelectChat?.(chat.id)}>
                  <ListItemText primary={chat.title} />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      ) : (
        <>
          <IconButton
            sx={{
              position: 'absolute',
              top: 16,
              left: 13,
              zIndex: 1
            }}
            onClick={() => setIsOpen(true)}
          >
            <MenuIcon />
          </IconButton>

          {/* 折叠状态下的新对话按钮（带 tooltip） */}
          <Tooltip title="新对话" placement="right">
            <IconButton
              onClick={onNewChat}
              sx={{ position: 'absolute', top: 65, left: 13 }}
            >
              <AddIcon />
            </IconButton>
          </Tooltip>
        </>
      )}
    </Box>
  );
}

export default Sidebar;
