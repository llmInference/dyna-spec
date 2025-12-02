#!/usr/bin/env python3
"""
Markdown查看器HTTP服务器

启动一个简单的HTTP服务器来查看Markdown分析报告

使用方法:
    python view_server.py [--port PORT] [--directory DIR]
    
示例:
    python view_server.py                    # 默认端口8000
    python view_server.py --port 8080        # 使用端口8080
    python view_server.py --directory ./      # 指定目录
"""

import http.server
import socketserver
import os
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class MarkdownViewHandler(http.server.SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器"""
    
    def __init__(self, *args, directory=None, **kwargs):
        self.directory = directory
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # API: 获取文件列表
        if path == '/api/files':
            self.send_file_list()
            return
        
        # API: 获取文件内容
        if path.startswith('/api/file/'):
            filename = path.replace('/api/file/', '')
            self.send_file_content(filename)
            return
        
        # 默认：返回HTML查看器
        if path == '/' or path == '/index.html':
            self.send_html_viewer()
            return
        
        # 其他文件：使用默认的文件服务
        super().do_GET()
    
    def send_file_list(self):
        """发送Markdown文件列表"""
        try:
            base_dir = Path(self.directory) if self.directory else Path.cwd()
            md_files = sorted([f.name for f in base_dir.glob('*.md')])
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(md_files).encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error: {str(e)}")
    
    def send_file_content(self, filename):
        """发送文件内容"""
        try:
            # 安全检查：防止路径遍历攻击
            if '..' in filename or '/' in filename or '\\' in filename:
                self.send_error(400, "Invalid filename")
                return
            
            base_dir = Path(self.directory) if self.directory else Path.cwd()
            file_path = base_dir / filename
            
            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "File not found")
                return
            
            # 只允许读取.md文件
            if not filename.endswith('.md'):
                self.send_error(403, "Only .md files are allowed")
                return
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-type', 'text/markdown; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error: {str(e)}")
    
    def send_html_viewer(self):
        """发送HTML查看器页面"""
        html_file = Path(__file__).parent / 'view_markdown.html'
        
        if not html_file.exists():
            self.send_error(404, "HTML viewer not found")
            return
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    parser = argparse.ArgumentParser(
        description='启动Markdown查看器HTTP服务器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python view_server.py
  python view_server.py --port 8080
  python view_server.py --directory /path/to/markdown/files
        """
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='服务器端口 (默认: 8000)'
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='/home/llminference/syq/dyna-spec/analysis/result.md',
        help='Markdown文件目录 (默认: 当前目录)'
    )
    
    args = parser.parse_args()
    
    # 创建自定义处理器类
    directory = args.directory
    
    class CustomHandler(MarkdownViewHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
    
    # 启动服务器
    with socketserver.TCPServer(("", args.port), CustomHandler) as httpd:
        print("=" * 60)
        print("Markdown查看器服务器已启动")
        print("=" * 60)
        print(f"访问地址: http://localhost:{args.port}")
        print(f"文件目录: {args.directory or os.getcwd()}")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器")
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")


if __name__ == '__main__':
    main()

