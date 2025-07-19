// --- START OF MODIFIED FILE Header.tsx ---

import { useState } from 'react';
import { Box, Switch, Avatar, IconButton, Popover, MenuItem, ListItemIcon, Divider, Typography } from '@mui/material';
import { styled } from '@mui/material/styles';

// --- Icon Imports ---
import AssignmentIcon from '@mui/icons-material/Assignment';
import ContactsIcon from '@mui/icons-material/Contacts';
import LogoutIcon from '@mui/icons-material/Logout';

// --- Dark Or Light Switch (代码保持不变) ---
const MaterialUISwitch = styled(Switch)(({ theme }) => ({
  width: 48,
  height: 26,
  padding: 4,
  '& .MuiSwitch-switchBase': {
    margin: 1,
    padding: 0,
    transform: 'translateX(3px)', // 调整初始位移
    '&.Mui-checked': {
      color: '#fff',
      transform: 'translateX(22px)', // 增大右滑位移，让thumb能滑到底
      '& .MuiSwitch-thumb:before': {
        backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 20 20"><path fill="${encodeURIComponent(
          '#fff',
        )}" d="M4.2 2.5l-.7 1.8-1.8.7 1.8.7.7 1.8.6-1.8L6.7 5l-1.9-.7-.6-1.8zm15 8.3a6.7 6.7 0 11-6.6-6.6 5.8 5.8 0 006.6 6.6z"/></svg>')`,
      },
      '& + .MuiSwitch-track': {
        opacity: 1,
        backgroundColor: '#aab4be',
      },
    },
  },
  '& .MuiSwitch-thumb': {
    backgroundColor: theme.palette.mode === 'dark' ? '#003892' : '#001e3c',
    width: 22,
    height: 22,
    left: 0, // 确保thumb不被裁剪
    top: 0,
    boxSizing: 'border-box',
    '&::before': {
      content: "''",
      position: 'absolute',
      width: '100%',
      height: '100%',
      left: 0,
      top: 0,
      backgroundRepeat: 'no-repeat',
      backgroundPosition: 'center',
      backgroundSize: '18px 18px', // 缩小icon，避免被裁剪
      backgroundImage: `url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 20 20"><path fill="${encodeURIComponent(
        '#fff',
      )}" d="M9.305 1.667V3.75h1.389V1.667h-1.39zm-4.707 1.95l-.982.982L5.09 6.072l.982-.982-1.473-1.473zm10.802 0L13.927 5.09l.982.982 1.473-1.473-.982-.982zM10 5.139a4.872 4.872 0 00-4.862 4.86A4.872 4.872 0 0010 14.862 4.872 4.872 0 0014.86 10 4.872 4.872 0 0010 5.139zm0 1.389A3.462 3.462 0 0113.471 10a3.462 3.462 0 01-3.473 3.472A3.462 3.462 0 016.527 10 3.462 3.462 0 0110 6.528zM1.665 9.305v1.39h2.083v-1.39H1.666zm14.583 0v1.39h2.084v-1.39h-2.084zM5.09 13.928L3.616 15.4l.982.982 1.473-1.473-.982-.982zm9.82 0l-.982.982 1.473 1.473.982-.982-1.473-1.473zM9.305 16.25v2.083h1.389V16.25h-1.39z"/></svg>')`,
    },
  },
  '& .MuiSwitch-track': {
    opacity: 1,
    backgroundColor: '#aab4be',
    borderRadius: 20 / 2,
  },
}));

interface HeaderProps {
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
  title: string; // 新增
}

function Header({ theme, onToggleTheme, title }: HeaderProps) {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <div className="header" style={{ position: 'relative' }}>
      {/* 标题居中显示 */}
      <Box
        sx={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 0,
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          pointerEvents: 'none', // 避免遮挡右侧按钮
          zIndex: 0,
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 'bold',
            color: 'inherit',
            userSelect: 'none',
            letterSpacing: 1,
          }}
        >
          {title}
        </Typography>
      </Box>

      {/* 右侧控件 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2, zIndex: 1 }}>
        {/* 主题切换开关放回 Header */}
        <MaterialUISwitch
          checked={theme === 'dark'}
          onChange={onToggleTheme}
          aria-label="切换白天/黑夜模式"
        />

        {/* 用户头像按钮 */}
        <IconButton
          onClick={handleClick}
          size="small"
          aria-controls={open ? 'account-menu' : undefined}
          aria-haspopup="true"
          aria-expanded={open ? 'true' : undefined}
          sx={{ p: 0.5 }} // 缩小点击区域
        >
          <Avatar sx={{ width: 28, height: 28, bgcolor: '#95d59aff', fontSize: 18 }}>U</Avatar>
        </IconButton>
      </Box>

      {/* 2. Popover 菜单保持不变，但移除了内部的主题切换功能 */}
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        PaperProps={{
          sx: {
            p: 0,
            mt: 1.5,
            width: 300,
            borderRadius: '12px',
            boxShadow: '0px 5px 25px rgba(0,0,0,0.1)',
          },
        }}
      >
        {/* 用户信息区 */}
        <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
          <Avatar sx={{ width: 40, height: 40, bgcolor: '#95d59aff', fontSize: 22 }}>U</Avatar>
          <Box>
            <Typography variant="subtitle1" fontWeight="bold">
              User Name
            </Typography>
            <Typography variant="body2" color="text.secondary">
              user.email@example.com
            </Typography>
          </Box>
        </Box>

        <Divider />

        {/* 菜单项区 */}
        <Box sx={{ p: 1 }}>
          <MenuItem onClick={handleClose}>
            <ListItemIcon>
              <AssignmentIcon fontSize="small" />
            </ListItemIcon>
            个人资料
          </MenuItem>
          <MenuItem onClick={handleClose}>
            <ListItemIcon>
              <ContactsIcon fontSize="small" />
            </ListItemIcon>
            我的账户
          </MenuItem>
        </Box>

        <Divider />

        {/* 退出登录区 */}
        <Box sx={{ p: 1 }}>
          <MenuItem onClick={handleClose}>
            <ListItemIcon>
              <LogoutIcon fontSize="small" />
            </ListItemIcon>
            退出登录
          </MenuItem>
        </Box>
      </Popover>
    </div>
  );
}

export default Header;