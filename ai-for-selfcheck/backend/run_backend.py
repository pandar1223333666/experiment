"""
后端启动脚本
直接运行此文件即可启动后端服务
"""

import uvicorn
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 检查API密钥
if not os.getenv("QWEN_API_KEY"):
    print("警告: 未设置QWEN_API_KEY环境变量，请在.env文件中配置或手动设置环境变量")

if __name__ == "__main__":
    print("正在启动AI自检系统后端...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    print("服务已启动，请访问 http://127.0.0.1:8000/docs 查看API文档") 