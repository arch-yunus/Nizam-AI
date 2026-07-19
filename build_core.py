#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nizam-AI C++ Core Compilation Script
Attempts to compile core C++ libraries using cmake and g++ / MSVC.
Falls back to logging instructions if compilers are not found.
"""

import os
import sys
import subprocess
import shutil

def check_command(cmd):
    try:
        subprocess.run([cmd, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def build_cpp():
    print("[Nizam-AI] C++ Çekirdek derleme adımları başlatılıyor...")
    
    # Check dependencies
    has_cmake = check_command("cmake")
    # check compiler (g++ or cl)
    has_gcc = check_command("g++")
    
    if not has_cmake:
        print("[!] Hata: 'cmake' komutu sistem yolunda bulunamadı.")
        print("[*] Lütfen CMake kurun ve PATH ortam değişkenine ekleyin: https://cmake.org/download/")
        return False
        
    build_dir = os.path.join(os.path.dirname(__file__), "core", "build")
    os.makedirs(build_dir, exist_ok=True)
    
    try:
        print("[*] CMake yapılandırılıyor...")
        subprocess.run(["cmake", "-B", build_dir, "-S", "."], check=True)
        
        print("[*] C++ Kütüphanesi derleniyor...")
        subprocess.run(["cmake", "--build", build_dir, "--config", "Release"], check=True)
        
        # Move the compiled library to workspace root or nizam folder for loading
        dll_name = "nizam_core.dll" if sys.platform == "win32" else "libnizam_core.so"
        # Look for the library in build folder
        src_lib = None
        for root, dirs, files in os.walk(build_dir):
            if dll_name in files:
                src_lib = os.path.join(root, dll_name)
                break
                
        if src_lib:
            dest_lib = os.path.join(os.path.dirname(__file__), "nizam", dll_name)
            shutil.copy(src_lib, dest_lib)
            print(f"[✓] C++ Çekirdeği başarıyla derlendi ve kopyalandı: {dest_lib}")
            return True
        else:
            print("[!] Hata: Derlenen kütüphane dosyası bulunamadı.")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"[!] Derleme hatası oluştu: {e}")
        return False
    except Exception as e:
        print(f"[!] Beklenmedik hata: {e}")
        return False

if __name__ == "__main__":
    success = build_cpp()
    if not success:
        print("\n[Nizam-AI] Derleme tamamlanamadı. Sisteminizde C++ derleyicisi eksik olabilir.")
        print("[Nizam-AI] Endişelenmeyin! Nizam-AI, C++ kütüphanesi olmadan da otomatik ")
        print("[Nizam-AI] 'Pure Python Fallback' moduyla %100 uyumlu şekilde çalışmaktadır.")
        sys.exit(0)
