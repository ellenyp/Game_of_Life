#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# --- 設定參數 ---
N = 150  # 網格大小 (NxN)
steps = 300  # 模擬代數

# --- 初始化隨機網格 ---
np.random.seed(42)
grid = np.random.randint(0, 2, (N, N), dtype=np.int8)

# --- 定義卷積核 (3x3，中心為0) ---
KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.int8)

# --- 定義更新規則 ---
def update(grid):
    """依據 Game of Life 規則產生下一代"""
    # 計算每個 cell 的鄰居數
    neighbors = convolve2d(grid, KERNEL, mode='same', boundary='wrap')

    # 根據規則更新
    # (1) 若活著且有2或3個鄰居 -> 活
    # (2) 若死掉且剛好3個鄰居 -> 復活
    new_grid = ((grid == 1) & ((neighbors == 2) | (neighbors == 3))) | ((grid == 0) & (neighbors == 3))
    return new_grid.astype(np.int8)

# --- 畫面設定 ---
plt.ion()  # 開啟互動模式
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap='binary')
ax.set_title("Conway's Game of Life")
ax.axis('off')


if __name__ == "__main__":
    
    for t in range(steps):
        grid = update(grid)
        img.set_data(grid)
        ax.set_xlabel(f"Step: {t+1}")
        plt.pause(0.07)
    
    plt.ioff()
    plt.show()
