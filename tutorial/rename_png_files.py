#!/usr/bin/env python3
import os
import csv
import re
from pathlib import Path
import shutil

def reallocate_files():
    png_dir = "/mnt/hdd5/Qiaoceng/navsim_workspace/new_output_Qiao/stage_two_worse_case"
    png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')] 
    print(f"Find {len(png_files)} PNG files")
    
    file_groups = {}
    
    for filename in png_files:
        if len(filename) >= 16:
            prefix = filename[:16]
            if re.match(r'^[a-fA-F0-9]{16}$', prefix):
                if prefix not in file_groups:
                    file_groups[prefix] = []
                file_groups[prefix].append(filename)
    
    for prefix, files in file_groups.items():
        target_dir = os.path.join(png_dir, prefix)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Build Folder：{target_dir}")
        
        for filename in files:
            src_path = os.path.join(png_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            
            try:
                shutil.move(src_path, dst_path)
                print(f"Move file：{filename} -> {prefix}/")
            except Exception as e:
                print(f"Error occurred while moving file {filename}：{e}")

    print(f"\nDone！A total of {len(file_groups)} groups were processed")
    for prefix, files in file_groups.items():
        print(f"  {prefix}: {len(files)} files")

def count_files_in_all_folders():
    """統計所有資料夾內的檔案數量"""
    stage2_folder = "/mnt/hdd5/Qiaoceng/navsim_workspace/new_output_Qiao/stage_two_worse_case"
    
    if not os.path.exists(stage2_folder):
        print(f"資料夾不存在: {stage2_folder}")
        return
    
    total_files = 0
    folder_count = 0
    folder_stats = []
    
    # 遍歷所有子資料夾
    for item in os.listdir(stage2_folder):
        item_path = os.path.join(stage2_folder, item)
        if os.path.isdir(item_path):
            folder_count += 1
            # 計算該資料夾內的檔案數量
            files_in_folder = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
            file_count = len(files_in_folder)
            total_files += file_count
            folder_stats.append((item, file_count))
    
    # 排序並顯示結果
    folder_stats.sort(key=lambda x: x[1], reverse=True)  # 按檔案數量降序排列
    
    print(f"📊 統計結果:")
    print(f"總資料夾數: {folder_count}")
    print(f"總檔案數: {total_files}")
    print(f"平均每個資料夾檔案數: {total_files/folder_count:.2f}" if folder_count > 0 else "平均: 0")
    
    print(f"\n📂 檔案數最多的前10個資料夾:")
    for i, (folder_name, file_count) in enumerate(folder_stats[:10]):
        print(f"{i+1:2d}. {folder_name}: {file_count} 個檔案")
    
    if len(folder_stats) > 10:
        print(f"\n📂 檔案數最少的後5個資料夾:")
        for folder_name, file_count in folder_stats[-5:]:
            print(f"    {folder_name}: {file_count} 個檔案")
    
    return total_files, folder_count

def rename_folders():
    stage1_folder = "/mnt/hdd5/Qiaoceng/navsim_workspace/new_output_Qiao/stage_one_worse_case"
    stage2_folder = "/mnt/hdd5/Qiaoceng/navsim_workspace/new_output_Qiao/stage_two_worse_case"
    stage1_png = [f for f in os.listdir(stage1_folder) if f.endswith('.png')]
    stage2_subfol = [f for f in os.listdir(stage2_folder)]
    
    
    

def main():
    '''
    print("=== 重新分配檔案 ===")
    reallocate_files()
    '''
    print("\n=== 統計檔案數量 ===")
    count_files_in_all_folders()
    
if __name__ == "__main__":
    main()