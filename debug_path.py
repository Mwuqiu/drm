import inspect
import os
import sys

try:
    from libero.libero import get_libero_path
    
    # 1. 找到这个函数定义在哪个文件
    file_path = inspect.getfile(get_libero_path)
    print(f"=== 藏身之处: {file_path} ===")
    
    # 2. 直接打印这个函数的源代码
    print("\n=== 源代码内容 ===")
    src = inspect.getsource(get_libero_path)
    print(src)
    print("====================")

    # 3. 顺便看看该文件的前几行（通常全局变量定义在这里）
    print("\n=== 文件开头的全局变量 ===")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 打印前 20 行，通常 ROOT 路径就定义在这里
        for i, line in enumerate(lines[:20]):
            print(f"{i+1}: {line.strip()}")

except Exception as e:
    print(f"侦查失败: {e}")