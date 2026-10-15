#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP 图例绘制脚本
=========================
基于 experiment.py 训练好的最优模型，生成论文所需的全部图例。
支持为不同因变量选择不同的 SHAP 分析模型（XGBOOST / LightGBM）。

必做图例（⭐⭐⭐）：
  图1: 模型性能验证图（三联组合图）
  图2: SHAP全局特征贡献图（Bar Plot）
  图3: SHAP蜂群图（Beeswarm Plot）
  图4: SHAP单特征依赖图（Dependence Plot）
  图5: 二维PDP部分依赖图

推荐图例（⭐⭐☆）：
  图6: 特征交互效应复合矩阵图
  图7: 主效应与交互效应对比图
  图8: SHAP成对交互散点图
  图9: SHAP热力图（Heatmap）

补充图例（从 experiment.py 迁移）：
  图10: 各模型残差图（测试集）
  图11: 最佳树模型特征重要性图
  图12: SHAP特征影响-交互网络图

使用方式：
  python plot_shap.py                    # 绘制所有图例
  python plot_shap.py --plots 1 2 3      # 只绘制指定编号的图例（1-11）
  python plot_shap.py --target 步行联系强度（%）  # 只绘制指定因变量的图例
  python plot_shap.py --plots 12         # 只绘制SHAP特征影响-交互网络图
"""

import os
import sys
import json
import argparse
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import concurrent.futures
from functools import partial

# ============ 跨平台中文字体设置 ============
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'STHeiti', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

import shap
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import statsmodels.api as sm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ============ 配置（与 experiment.py 保持一致） ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '建成环境数据.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-built-environment')
BEST_PARAMS_DIR = os.path.join(OUTPUT_DIR, 'best_params')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

TARGET_A = '步行联系强度（%）'
TARGET_B = '步行联系收缩度'
ALL_TARGETS = [TARGET_A, TARGET_B]
ALWAYS_EXCLUDE = ['FID']
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1

# ============ SHAP 分析模型配置 ============
# auto 从本轮训练摘要选用内层CV MSE最低的树模型
SHAP_MODEL_CONFIG = {
    TARGET_A: 'auto',
    TARGET_B: 'auto',
}

# SHAP 依赖图选取的 Top 特征数量
TOP_N_DEPENDENCE = 8
# 二维 PDP 的变量对数量
N_PDP_PAIRS = 5

# ============ 配色方案 ============
MODEL_COLORS = {
    'OLS': '#7f8c8d',
    'GBDT': '#3498db',
    'RF': '#2ecc71',
    'XGBOOST': '#e74c3c',
    'LightGBM': '#9b59b6',
}

# ============ TARGET 主题色与色带配置 ============
from matplotlib.colors import LinearSegmentedColormap

# 主题一：Viridis 风格（高能暖色）- 顺序色谱
# 颜色节点: 亮黄色（最高值） → 黄绿色 → 青绿色 → 深蓝色 → 深紫色（最低值）
_CMAP_VIRIDIS_SEQUENTIAL = plt.cm.viridis

# 主题二：RdBu 风格（冷暖对比）- 发散色谱
# 颜色节点: 深蓝色（最高值） → 青色 → 浅绿色 → 米黄色（中间值） → 浅橙色 → 深红色（最低值）
_CMAP_RDBU_DIVERGING = plt.cm.RdBu_r

# 主题一配置（Viridis - 用于步行联系收缩度）
THEME_VAR1 = {
    'cmap_diverging': _CMAP_VIRIDIS_SEQUENTIAL,
    'cmap_sequential': _CMAP_VIRIDIS_SEQUENTIAL,
    'primary': '#44AA99',
    'accent': '#FDB863',
    'bg_positive': '#E6F5D0',
    'bg_negative': '#FFF2CC',
    'bar_main': '#44AA99',
    'bar_interact': '#FDB863',
    'bar_importance': '#44AA99',
    'node_positive': (0.27, 0.67, 0.60),
    'node_negative': (0.99, 0.72, 0.39),
    'ratio_high': '#FDB863',
    'ratio_mid': '#B2DFDB',
    'ratio_low': '#556B2F',  # 【已修改】从 #C8D5C4 修改为深褐绿色，增加对比度
}

# 主题二配置（RdBu - 用于第二个变量）
THEME_VAR2 = {
    'cmap_diverging': _CMAP_RDBU_DIVERGING,
    'cmap_sequential': _CMAP_VIRIDIS_SEQUENTIAL,  # 交互矩阵仍用viridis保持一致性
    'primary': '#2A78B4',  # RdBu 中间偏蓝
    'accent': '#B42A2A',   # RdBu 中间偏红
    'bg_positive': '#D1E5F0',
    'bg_negative': '#FDDBC7',
    'bar_main': '#2A78B4',
    'bar_interact': '#B42A2A',
    'bar_importance': '#44AA99',
    'node_positive': (0.165, 0.471, 0.706),  # #2A78B4 RGB
    'node_negative': (0.706, 0.165, 0.165),  # #B42A2A RGB
    'ratio_high': '#B42A2A',
    'ratio_mid': '#F7F7F7',
    'ratio_low': '#2A78B4',
}

def get_theme(target_name):
    """获取指定 TARGET 的主题色配置，第一个变量用Viridis，第二个用RdBu"""
    if target_name == TARGET_A:
        return THEME_VAR2
    else:
        return THEME_VAR1


def _worker_pdp_2d(rank, idx_a, idx_b, strength, model, X_all_scaled, feature_cols, theme, scaler, sub_dir):
    """
    单个二维 PDP 图的并行绘图工作函数
    """
    from sklearn.inspection import partial_dependence
    import matplotlib.pyplot as plt
    import numpy as np

    feat_a, feat_b = feature_cols[idx_a], feature_cols[idx_b]

    # 1. 计算二维偏依赖
    pd_result = partial_dependence(
        model, X_all_scaled, features=[(idx_a, idx_b)],
        kind='average', grid_resolution=30
    )

    grid_a, grid_b = pd_result['grid_values'][0], pd_result['grid_values'][1]

    # 2. 坐标反向标准化（还原原始量纲）
    if scaler is not None:
        grid_a = grid_a * scaler.scale_[idx_a] + scaler.mean_[idx_a]
        grid_b = grid_b * scaler.scale_[idx_b] + scaler.mean_[idx_b]

    xx, yy = np.meshgrid(grid_a, grid_b)
    zz = pd_result['average'][0].T

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    contour_filled = ax.contourf(xx, yy, zz, levels=10, cmap=theme['cmap_diverging'], alpha=0.9)
    contour_lines = ax.contour(xx, yy, zz, levels=contour_filled.levels, colors='white', linewidths=0.8, alpha=0.7)
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f', colors='black')

    fig.colorbar(contour_filled, ax=ax, shrink=0.8, label='部分依赖值')
    ax.set_xlabel(feat_a, fontsize=12)
    ax.set_ylabel(feat_b, fontsize=12)
    ax.set_title(f'Rank {rank}: {feat_a} × {feat_b}\n(强度={strength:.5f})', fontsize=13, fontweight='bold')

    plt.tight_layout()

    # 4. 保存
    save_path = os.path.join(sub_dir, f'{rank:03d}_{feat_a}×{feat_b}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')  # 并行时建议200dpi，平衡速度与清晰度
    plt.close(fig)
    return f"Rank {rank} 完成"

# ============ 日志 ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('plot_shap')

import threading


def timed_continue_prompt(timeout=10):
    """
    带超时的用户确认提示
    """
    print(f"\n>>> [确认] 即将开始全量样本 SHAP 计算及 Top 100 密集绘图任务。")
    print(f">>> 输入 'n' 中止，输入其他或等待 {timeout}s 将自动开始...")

    user_input = [None]

    def get_input():
        try:
            user_input[0] = sys.stdin.readline().strip().lower()
        except:
            pass

    # 开启线程等待输入
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()

    # 等待指定的秒数
    input_thread.join(timeout)

    if user_input[0] is None:
        print(f"\n>>> [超时] 未检测到输入，默认继续执行...\n")
        return True
    elif user_input[0] == 'n':
        print("\n>>> [中止] 用户取消了绘图任务。")
        sys.exit(0)
    else:
        print("\n>>> [继续] 收到确认，开始执行任务...\n")
        return True


def _worker_interaction_scatter(rank, i, j, strength, interaction_vals, x_vals, color_vals,
                               feat_a, feat_b, theme, save_path):
    """图8的并行工作函数"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5.5))

    scatter = ax.scatter(x_vals, interaction_vals, c=color_vals, cmap=theme['cmap_diverging'],
                         alpha=0.5, s=14, edgecolors='none')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(feat_a, fontsize=11)
    ax.set_ylabel(f'SHAP 交互值\n({feat_a} × {feat_b})', fontsize=10)
    ax.set_title(f'Rank {rank}: {feat_a} × {feat_b}\n(强度={strength:.5f})', fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(feat_b, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ============ 数据加载 ============
def load_data(current_targets):
    """
    加载数据并返回特征矩阵、因变量字典、特征名。
    current_targets: 当前脚本运行需要生成的因变量列表
    """
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

    # 检查当前请求的因变量是否存在
    missing = [t for t in current_targets if t not in df.columns]
    if missing:
        log.error(f"以下因变量在数据中不存在: {missing}")
        sys.exit(1)

    # 修改逻辑：始终排除【所有】已知的因变量候选集，而不仅仅是当前分析的那个
    # 这样可以确保“分析 A 时排除 B，分析 B 时排除 A”
    exclude_cols = ALWAYS_EXCLUDE + ALL_TARGETS + list(current_targets)

    # 动态获取特征列（确保这些列确实存在于 df 中）
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 与训练流程使用相同的数据清理。
    df = df.dropna().reset_index(drop=True)
    numeric_cols = feature_cols + list(current_targets)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise ValueError('特征或目标中含有无穷值')

    # 提取特征矩阵 X
    X = df[feature_cols].values

    # 只返回当前需要的因变量数据
    y_dict = {t: df[t].values for t in current_targets}

    log.info(f"数据加载完成: {X.shape[0]} 样本, {len(feature_cols)} 特征")
    log.info(f"已从特征中排除所有因变量: {ALL_TARGETS}")
    return X, y_dict, feature_cols


def load_best_params(target_name):
    """加载已保存的最优超参数"""
    json_path = os.path.join(BEST_PARAMS_DIR, f'best_params_{target_name}.json')
    if not os.path.exists(json_path):
        log.error(f"未找到最优参数文件: {json_path}")
        log.error("请先运行 experiment.py --mode tune 进行超参数调优")
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    log.info(f"已加载最优参数: {json_path}")
    return params


def create_model_with_params(model_name, params):
    """根据模型名称和参数创建模型实例"""
    if model_name == 'GBDT':
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    elif model_name == 'RF':
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    elif model_name == 'XGBOOST':
        return XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1, **params)
    elif model_name == 'LightGBM':
        return LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=1, **params)
    else:
        raise ValueError(f"未知模型: {model_name}")


def train_all_models(X_train, y_train, X_test, y_test, all_params, feature_cols):
    """
    训练所有模型（含OLS），返回训练好的模型字典和测试集预测结果。
    返回:
        trained_models: {model_name: fitted_model}
        test_results: {model_name: {'y_pred': array, 'metrics': dict}}
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    trained_models = {}
    test_results = {}

    # OLS (始终训练，用于性能比较)
    X_train_const = sm.add_constant(X_train_s)
    X_test_const = sm.add_constant(X_test_s)
    ols_model = sm.OLS(y_train, X_train_const).fit()
    y_pred_ols = ols_model.predict(X_test_const)
    trained_models['OLS'] = ols_model
    test_results['OLS'] = {
        'y_pred': y_pred_ols,
        'metrics': _compute_metrics(y_test, y_pred_ols)
    }

    # 树模型
    for model_name in ['GBDT', 'RF', 'XGBOOST', 'LightGBM']:
        params = all_params.get(model_name)
        if params is None:
            continue
        model = create_model_with_params(model_name, params)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        trained_models[model_name] = model
        test_results[model_name] = {
            'y_pred': y_pred,
            'metrics': _compute_metrics(y_test, y_pred)
        }
        log.info(f"  {model_name} 训练完成 — 测试集 R²={test_results[model_name]['metrics']['R²']:.4f}")

    return trained_models, test_results, scaler


def _compute_metrics(y_true, y_pred):
    """计算评估指标"""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R²': r2_score(y_true, y_pred),
    }


def compute_shap_values(model, X_data, feature_cols):
    """
    计算 SHAP 值。
    返回 shap.Explanation 对象。
    """
    log.info("  正在计算 SHAP 值（TreeExplainer）...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame(X_data, columns=feature_cols))
    log.info(f"  SHAP 值计算完成: shape={shap_values.shape}")
    return shap_values, explainer


def compute_shap_interaction_values(model, X_data, feature_cols):
    """
    计算 SHAP 交互值。
    返回 shape=(n_samples, n_features, n_features) 的数组。
    """
    log.info("  正在计算 SHAP 交互值（可能需要较长时间）...")
    explainer = shap.TreeExplainer(model)
    shap_interaction = explainer.shap_interaction_values(
        pd.DataFrame(X_data, columns=feature_cols)
    )
    log.info(f"  SHAP 交互值计算完成: shape={shap_interaction.shape}")
    return shap_interaction


# =====================================================================
#  图1: 模型性能验证图（三联组合图）
# =====================================================================
def _plot_1a_r2_bar(test_results, target_name, shap_model_name, theme, save_path):
    """图1(a): 多模型 R² 对比柱状图"""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    model_names = list(test_results.keys())
    r2_values = [test_results[m]['metrics']['R²'] for m in model_names]
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in model_names]

    bars = ax.bar(model_names, r2_values, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
    for i, m in enumerate(model_names):
        if m == shap_model_name:
            bars[i].set_edgecolor(theme['accent'])
            bars[i].set_linewidth(2.5)

    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('R²', fontsize=12)
    ax.set_title(f'{target_name} — 多模型R^2对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(r2_values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_1b_joint(y_test, y_pred_test, y_train, y_pred_train, has_train,
                   shap_model_name, target_name, save_path):
    """图1(b): 最优模型 Joint Plot（带边际分布的散点图 + 底部残差图）"""
    from scipy.stats import gaussian_kde
    from matplotlib.gridspec import GridSpec

    train_color = '#6C8EBF'
    test_color = '#D4726A'

    if has_train:
        all_actual = np.concatenate([y_train, y_test])
        all_pred = np.concatenate([y_pred_train, y_pred_test])
    else:
        all_actual = y_test
        all_pred = y_pred_test

    # 计算指标
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    if has_train:
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)

    # 坐标范围
    lims = [min(all_actual.min(), all_pred.min()), max(all_actual.max(), all_pred.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(3, 2, figure=fig,
                  height_ratios=[0.8, 3, 1.2], width_ratios=[3, 0.8],
                  hspace=0.08, wspace=0.08)

    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    ax_resid = fig.add_subplot(gs[2, 0])
    ax_empty1 = fig.add_subplot(gs[0, 1])
    ax_empty2 = fig.add_subplot(gs[2, 1])
    ax_empty1.axis('off')
    ax_empty2.axis('off')

    # --- 主散点图 ---
    if has_train:
        ax_main.scatter(y_train, y_pred_train, alpha=0.3, s=14, color=train_color,
                        edgecolors='none', label='Train data', zorder=2)
    ax_main.scatter(y_test, y_pred_test, alpha=0.4, s=14, color=test_color,
                    edgecolors='none', label='Test data', zorder=3)

    # 拟合线
    coeffs = np.polyfit(all_actual, all_pred, 1)
    fit_x = np.linspace(lims[0], lims[1], 100)
    fit_y = np.polyval(coeffs, fit_x)
    ax_main.plot(fit_x, fit_y, 'k-', linewidth=1.8, label='Fitted line', zorder=4)

    # 45° 参考虚线
    ax_main.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, zorder=1)

    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)
    ax_main.set_aspect('equal')

    # 标注 R² 和 RMSE
    if has_train:
        stats_text = (f'Train $R^2$ = {r2_train:.2f}, $RMSE$ = {rmse_train:.2f}\n'
                      f'Test $R^2$ = {r2_test:.2f}, $RMSE$ = {rmse_test:.2f}')
    else:
        stats_text = f'Test $R^2$ = {r2_test:.2f}, $RMSE$ = {rmse_test:.2f}'
    ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes,
                 fontsize=10, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax_main.set_title(f'{target_name} — {shap_model_name}', fontsize=13, fontweight='bold')
    ax_main.legend(loc='center right', fontsize=8, framealpha=0.8)
    ax_main.grid(True, alpha=0.15)
    ax_main.tick_params(labelbottom=False)

    # --- 顶部边际直方图 ---
    bins_x = np.linspace(lims[0], lims[1], 35)
    if has_train:
        ax_top.hist(y_train, bins=bins_x, alpha=0.5, color=train_color,
                    edgecolor='white', linewidth=0.3, density=True)
    ax_top.hist(y_test, bins=bins_x, alpha=0.5, color=test_color,
                edgecolor='white', linewidth=0.3, density=True)
    x_kde = np.linspace(lims[0], lims[1], 200)
    if has_train:
        kde_train_x = gaussian_kde(y_train)
        ax_top.plot(x_kde, kde_train_x(x_kde), color=train_color, linewidth=1.5)
    kde_test_x = gaussian_kde(y_test)
    ax_top.plot(x_kde, kde_test_x(x_kde), color=test_color, linewidth=1.5)

    ax_top.set_xlim(lims)
    ax_top.tick_params(labelbottom=False, labelleft=False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)

    # --- 右侧边际直方图 ---
    bins_y = np.linspace(lims[0], lims[1], 35)
    if has_train:
        ax_right.hist(y_pred_train, bins=bins_y, alpha=0.5, color=train_color,
                      edgecolor='white', linewidth=0.3, density=True,
                      orientation='horizontal')
    ax_right.hist(y_pred_test, bins=bins_y, alpha=0.5, color=test_color,
                  edgecolor='white', linewidth=0.3, density=True,
                  orientation='horizontal')
    y_kde = np.linspace(lims[0], lims[1], 200)
    if has_train:
        kde_train_y = gaussian_kde(y_pred_train)
        ax_right.plot(kde_train_y(y_kde), y_kde, color=train_color, linewidth=1.5)
    kde_test_y = gaussian_kde(y_pred_test)
    ax_right.plot(kde_test_y(y_kde), y_kde, color=test_color, linewidth=1.5)

    ax_right.set_ylim(lims)
    ax_right.tick_params(labelbottom=False, labelleft=False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    # --- 底部残差图 ---
    resid_test = y_test - y_pred_test
    ax_resid.scatter(y_test, resid_test, alpha=0.3, s=10, color=test_color,
                     edgecolors='none', zorder=2)
    if has_train:
        resid_train = y_train - y_pred_train
        ax_resid.scatter(y_train, resid_train, alpha=0.2, s=10, color=train_color,
                         edgecolors='none', zorder=1)

    ax_resid.axhline(y=0, color='k', linewidth=1, zorder=3)
    ax_resid.set_xlim(lims)

    if has_train:
        mae_text = f'MAE (Train) = {mae_train:.3f}\nMAE (Test) = {mae_test:.3f}'
    else:
        mae_text = f'MAE (Test) = {mae_test:.3f}'
    ax_resid.text(0.98, 0.95, mae_text, transform=ax_resid.transAxes,
                  fontsize=9, fontweight='bold', va='top', ha='right',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax_resid.grid(True, alpha=0.15)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_1c_residual_hist(y_test, y_pred_test, shap_model_name, target_name,
                           model_color, save_path):
    """图1(c): 最优模型 残差直方图"""
    from scipy.stats import norm

    fig, ax = plt.subplots(figsize=(7, 5.5))

    residuals = y_test - y_pred_test
    ax.hist(residuals, bins=40, color=model_color, alpha=0.7, edgecolor='white', linewidth=0.5,
            density=True)
    mu, std = norm.fit(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x_range, norm.pdf(x_range, mu, std), 'k-', linewidth=1.5,
            label=f'正态拟合 (μ={mu:.3f}, σ={std:.3f})')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('残差 (实际值 - 预测值)', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'{target_name} — {shap_model_name} 残差分布', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_1_model_performance(y_test, test_results, target_name, shap_model_name='XGBOOST',
                             y_train=None, train_preds=None, plot_dir=PLOT_DIR):
    """
    模型性能验证图，拆分为3张独立图片保存到子目录：
    (a) 多模型 R² 对比柱状图
    (b) 最优模型 Joint Plot（带边际分布的散点图 + 底部残差图）
    (c) 最优模型 残差直方图
    """
    log.info(f"绘制图1: 模型性能验证图 — {target_name}（SHAP模型: {shap_model_name}）")

    theme = get_theme(target_name)
    model_color = MODEL_COLORS.get(shap_model_name, '#e74c3c')

    # 创建子目录
    sub_dir = os.path.join(plot_dir, '图1_模型性能验证')
    os.makedirs(sub_dir, exist_ok=True)

    # 准备 Joint Plot 所需数据
    y_pred_test = test_results[shap_model_name]['y_pred']
    has_train = y_train is not None and train_preds is not None and shap_model_name in train_preds
    y_pred_train = train_preds[shap_model_name] if has_train else None

    # (a) 多模型 R² 对比柱状图
    save_a = os.path.join(sub_dir, '01_R2对比.png')
    _plot_1a_r2_bar(test_results, target_name, shap_model_name, theme, save_a)
    log.info(f"  已保存: {save_a}")

    # (b) Joint Plot
    save_b = os.path.join(sub_dir, f'02_JointPlot_{shap_model_name}.png')
    _plot_1b_joint(y_test, y_pred_test, y_train, y_pred_train, has_train,
                   shap_model_name, target_name, save_b)
    log.info(f"  已保存: {save_b}")

    # (c) 残差直方图
    save_c = os.path.join(sub_dir, f'03_残差分布_{shap_model_name}.png')
    _plot_1c_residual_hist(y_test, y_pred_test, shap_model_name, target_name,
                           model_color, save_c)
    log.info(f"  已保存: {save_c}")

    log.info(f"  图1全部完成，共 3 张图，保存于: {sub_dir}")


# =====================================================================
#  图2: SHAP 全局特征贡献图（Bar Plot）
# =====================================================================
def plot_2_shap_bar(shap_values, feature_cols, target_name, plot_dir=PLOT_DIR):
    """SHAP 全局特征重要性条形图"""
    log.info(f"绘制图2: SHAP全局特征贡献图 — {target_name}")
    theme = get_theme(target_name)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=len(feature_cols), show=False, ax=ax)

    # 规范化：单一颜色选择 ratio_low
    for patch in ax.patches:
        patch.set_facecolor(theme['ratio_low'])

    for txt in ax.texts: txt.remove()
    ax.set_title(f'{target_name} — SHAP 全局特征贡献', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图2_SHAP特征贡献_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


# =====================================================================
#  图3: SHAP 蜂群图（Beeswarm Plot）
# =====================================================================
def plot_3_shap_beeswarm(shap_values, feature_cols, target_name, plot_dir=PLOT_DIR):
    """SHAP 蜂群图"""
    log.info(f"绘制图3: SHAP蜂群图 — {target_name}")

    theme = get_theme(target_name)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=len(feature_cols), show=False,
                         plot_size=None, ax=ax, color=theme['cmap_diverging'])
    ax.set_title(f'{target_name} — SHAP 蜂群图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图3_SHAP蜂群图_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


# =====================================================================
#  图4: SHAP 单特征依赖图（Dependence Plot）— LOWESS 增强版
# =====================================================================
def _find_lowess_zero_crossings(lowess_x, lowess_y):
    """
    找到 LOWESS 曲线与 y=0 的所有交叉点（线性插值）。
    返回交叉点的 x 坐标列表。
    """
    crossings = []
    for i in range(len(lowess_y) - 1):
        if lowess_y[i] * lowess_y[i + 1] < 0:  # 符号变化
            # 线性插值求交叉点
            x_cross = lowess_x[i] - lowess_y[i] * (lowess_x[i + 1] - lowess_x[i]) / (lowess_y[i + 1] - lowess_y[i])
            crossings.append(x_cross)
    return crossings


def _compute_lowess_with_ci(feat_vals, shap_vals, frac=0.5, n_bootstrap=200):
    """
    计算 LOWESS 拟合曲线及 Bootstrap 置信区间。
    返回: lowess_x, lowess_y, ci_lower, ci_upper
    """
    from scipy.interpolate import interp1d

    # 主 LOWESS 拟合
    lowess = sm.nonparametric.lowess
    fitted = lowess(shap_vals, feat_vals, frac=frac, return_sorted=True)
    lowess_x = fitted[:, 0]
    lowess_y = fitted[:, 1]

    # Bootstrap 置信区间
    bootstrap_curves = []
    n_samples = len(feat_vals)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        try:
            boot_fit = lowess(shap_vals[idx], feat_vals[idx], frac=frac, return_sorted=True)
            f_interp = interp1d(boot_fit[:, 0], boot_fit[:, 1],
                                bounds_error=False, fill_value=np.nan)
            bootstrap_curves.append(f_interp(lowess_x))
        except Exception:
            continue

    if len(bootstrap_curves) > 10:
        bootstrap_array = np.array(bootstrap_curves)
        ci_lower = np.nanpercentile(bootstrap_array, 2.5, axis=0)
        ci_upper = np.nanpercentile(bootstrap_array, 97.5, axis=0)
    else:
        ci_lower = lowess_y
        ci_upper = lowess_y

    return lowess_x, lowess_y, ci_lower, ci_upper

# 图4 修正版片段
def _plot_single_dependence(feat, feat_vals, shap_vals, theme, target_name, save_path):
    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    lowess_x, lowess_y, ci_lower, ci_upper = _compute_lowess_with_ci(feat_vals, shap_vals)

    ax_hist = ax.twinx()
    ax_hist.hist(feat_vals, bins=30, color='#C0C0C0', alpha=0.3, zorder=0)

    # 散点和基准线使用 low
    ax.scatter(feat_vals, shap_vals, color=theme['ratio_low'], alpha=0.35, s=14, edgecolors='none', zorder=2)
    ax.axhline(y=0, color=theme['ratio_low'], linestyle='--', linewidth=1, alpha=0.6, zorder=3)

    # 拟合线和区间使用 high
    ax.plot(lowess_x, lowess_y, color=theme['ratio_high'], linewidth=2.5, zorder=4, label='LOWESS')
    ax.fill_between(lowess_x, ci_lower, ci_upper, alpha=0.2, color=theme['ratio_high'], zorder=3, label='95% 置信区间')

    ax.set_xlabel(f'{feat}', fontsize=24, color='black')
    ax.set_ylabel('SHAP value', fontsize=24, color='black')
    ax.spines['top'].set_visible(True) # 确认开启
    ax.spines['right'].set_visible(True) # 确认开启
    ax.tick_params(axis='both', labelsize=21, colors='black')
    ax_hist.set_ylabel('样本数', fontsize=24, color='black')
    ax_hist.tick_params(axis='y', labelsize=21, colors='black')
    ax.legend(loc='best', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _worker_single_dependence(rank, feat, feat_vals, shap_vals, theme, target_name, save_path):
    """图4的并行工作函数：仅封装调用逻辑"""
    # 直接调用您原有的逻辑函数，保证内部逻辑 100% 一致
    _plot_single_dependence(feat, feat_vals, shap_vals, theme, target_name, save_path)
    return f"图4 - {feat} 完成"


def plot_4_shap_dependence(shap_values, X_data, feature_cols, target_name, scaler=None, plot_dir=PLOT_DIR, max_features=None):
    """并行化绘制图4"""
    log.info(f"绘制图4: SHAP单特征依赖图（并行模式）— {target_name}")
    theme = get_theme(target_name)
    sub_dir = os.path.join(plot_dir, '图4_SHAP依赖图')
    os.makedirs(sub_dir, exist_ok=True)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    if max_features is not None:
        if max_features < 1:
            raise ValueError('max_features 必须大于等于 1')
        sorted_indices = sorted_indices[:max_features]

    if scaler is not None:
        X_display = scaler.inverse_transform(X_data)
    else:
        X_display = X_data

    shap_vals_array = shap_values.values

    log.info(f"  准备并行绘制 {len(sorted_indices)} 张依赖图...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(
                _worker_single_dependence,
                rank,
                feature_cols[feat_idx],
                X_display[:, feat_idx],
                shap_vals_array[:, feat_idx],
                theme,
                target_name,
                os.path.join(sub_dir, f'{rank:02d}_{feature_cols[feat_idx]}.png')
            )
            for rank, feat_idx in enumerate(sorted_indices, start=1)
        ]

        # 实时监控进度
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if i % 5 == 0:
                log.info(f"  图4进度: {i}/{len(sorted_indices)}...")

    log.info(f"  图4并行绘制完成，保存于: {sub_dir}")


# =====================================================================
#  图5: 二维 PDP 部分依赖图
# =====================================================================
def plot_5_pdp_2d(model, X_all_scaled, feature_cols, target_name,
                  shap_interaction=None, scaler=None, plot_dir=PLOT_DIR):
    """
    修改版：并行化绘制全域交互强度最高的 Top 100 对特征的二维 PDP。
    """
    log.info(f"绘制图5: 二维PDP部分依赖图（并行模式 - Top 100）— {target_name}")
    theme = get_theme(target_name)

    sub_dir = os.path.join(plot_dir, '图5_二维PDP')
    os.makedirs(sub_dir, exist_ok=True)

    n_features = len(feature_cols)

    # 1. 计算/提取交互矩阵（逻辑不变）
    if shap_interaction is not None:
        interaction_matrix = np.abs(shap_interaction).mean(axis=0)
    else:
        explainer = shap.TreeExplainer(model)
        shap_inter = explainer.shap_interaction_values(pd.DataFrame(X_all_scaled, columns=feature_cols))
        interaction_matrix = np.abs(shap_inter).mean(axis=0)

    # 2. 获取特征对并排序
    all_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            all_pairs.append((i, j, interaction_matrix[i, j]))

    all_pairs.sort(key=lambda x: x[2], reverse=True)
    top_n_pairs = all_pairs[:100]

    log.info(f"  准备并行分发 {len(top_n_pairs)} 个绘图任务（使用 {os.cpu_count()} 个核心）...")

    # 3. 使用进程池进行并行处理
    # 使用 max_workers=None 默认使用所有 CPU 核心
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        # 准备任务列表
        futures = [
            executor.submit(
                _worker_pdp_2d,
                rank, idx_a, idx_b, strength,
                model, X_all_scaled, feature_cols,
                theme, scaler, sub_dir
            )
            for rank, (idx_a, idx_b, strength) in enumerate(top_n_pairs, start=1)
        ]

        # 监控进度
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            done_count += 1
            if done_count % 20 == 0:
                log.info(f"  已完成 {done_count}/100...")

    log.info(f"  图5并行绘图全部完成，结果保存于: {sub_dir}")

# =====================================================================
#  图6: 特征交互效应复合矩阵图
# =====================================================================
def plot_6_interaction_matrix(shap_interaction, feature_cols, target_name, plot_dir=PLOT_DIR):
    """
    对称矩阵热力图，展示所有变量对之间的 SHAP 交互效应强度。
    """
    log.info(f"绘制图6: 特征交互效应复合矩阵图 — {target_name}")

    theme = get_theme(target_name)

    n_features = len(feature_cols)
    # 计算每对特征的平均绝对交互值
    interaction_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            interaction_matrix[i, j] = np.abs(shap_interaction[:, i, j]).mean()

    # 将对角线（主效应）置零，只看交互效应
    np.fill_diagonal(interaction_matrix, 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.zeros_like(interaction_matrix, dtype=bool)
    # 上三角 mask（因为矩阵对称）
    mask[np.triu_indices_from(mask, k=1)] = True

    sns.heatmap(
        interaction_matrix,
        xticklabels=feature_cols,
        yticklabels=feature_cols,
        cmap=theme['cmap_sequential'],
        annot=False,
        fmt='.4f',
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': '平均 |SHAP 交互值|', 'shrink': 0.8}
    )
    ax.set_title(f'{target_name} — 特征交互效应矩阵', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图6_交互效应矩阵_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")

    # 输出 Top 10 交互对
    pairs_strength = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs_strength.append((feature_cols[i], feature_cols[j], interaction_matrix[i, j]))
    pairs_strength.sort(key=lambda x: x[2], reverse=True)
    log.info(f"  Top 10 交互对:")
    for rank, (f1, f2, val) in enumerate(pairs_strength[:10], 1):
        log.info(f"    {rank}. {f1} × {f2}: {val:.6f}")

    return pairs_strength


# =====================================================================
#  图7: 主效应与交互效应对比图
# =====================================================================
def plot_7_main_vs_interaction(shap_values, shap_interaction, feature_cols, target_name, plot_dir=PLOT_DIR):
    """
    每个变量的主效应 vs 交互效应贡献对比（水平堆叠条形图）。
    """
    log.info(f"绘制图7: 主效应与交互效应对比图 — {target_name}")

    theme = get_theme(target_name)

    n_features = len(feature_cols)

    # 主效应：对角线上的 SHAP 交互值的平均绝对值
    main_effects = np.array([np.abs(shap_interaction[:, i, i]).mean() for i in range(n_features)])

    # 交互效应：非对角线元素的平均绝对值之和
    interaction_effects = np.zeros(n_features)
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                interaction_effects[i] += np.abs(shap_interaction[:, i, j]).mean()

    # 总效应
    total_effects = main_effects + interaction_effects
    # 交互占比
    interaction_ratio = interaction_effects / (total_effects + 1e-10)

    # 按总效应排序
    sort_idx = np.argsort(total_effects)
    sorted_features = [feature_cols[i] for i in sort_idx]
    sorted_main = main_effects[sort_idx]
    sorted_inter = interaction_effects[sort_idx]
    sorted_ratio = interaction_ratio[sort_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})

    y_pos = np.arange(len(sorted_features))
    # 左图：双色对比
    # 主效应：使用 ratio_low
    ax1.barh(y_pos, sorted_main, color=theme['ratio_low'], label='主效应', height=0.7)
    # 交互效应：使用 ratio_high
    ax1.barh(y_pos, sorted_inter, left=sorted_main, color=theme['ratio_high'], label='交互效应', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_features, fontsize=9)
    ax1.set_xlabel('平均 |SHAP 值|', fontsize=12)
    ax1.set_title('主效应 vs 交互效应', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右图：交互占比
    colors = [theme['ratio_high'] if r > 0.3 else theme['ratio_mid'] if r > 0.15 else theme['ratio_low'] for r in sorted_ratio]
    ax2.barh(y_pos, sorted_ratio * 100, color=colors, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel('交互效应占比 (%)', fontsize=12)
    ax2.set_title('交互占比', fontsize=13, fontweight='bold')
    ax2.axvline(x=30, color=theme['ratio_high'], linestyle='--', linewidth=1, alpha=0.5, label='30% 阈值')
    ax2.axvline(x=15, color=theme['ratio_mid'], linestyle='--', linewidth=1, alpha=0.5, label='15% 阈值')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(f'{target_name} — 主效应与交互效应对比', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图7_主效应vs交互效应_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")

    # 输出交互占比 Top 5
    top_inter_idx = np.argsort(interaction_ratio)[::-1][:5]
    log.info(f"  交互效应占比 Top 5:")
    for rank, i in enumerate(top_inter_idx, 1):
        log.info(f"    {rank}. {feature_cols[i]}: 交互占比={interaction_ratio[i]*100:.1f}%")


# =====================================================================
#  图8: SHAP 成对交互散点图
# =====================================================================
def plot_8_interaction_scatter(shap_interaction, X_all_scaled, feature_cols, target_name,
                               top_pairs=None, scaler=None, plot_dir=PLOT_DIR):
    """并行化绘制图8"""
    log.info(f"绘制图8: SHAP成对交互散点图（并行模式 - Top 100）— {target_name}")
    theme = get_theme(target_name)
    sub_dir = os.path.join(plot_dir, '图8_交互散点图')
    os.makedirs(sub_dir, exist_ok=True)

    n_features = len(feature_cols)
    pairs_strength = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = np.abs(shap_interaction[:, i, j]).mean()
            pairs_strength.append((i, j, val))

    pairs_strength.sort(key=lambda x: x[2], reverse=True)
    pairs_strength = pairs_strength[:100]

    X_display = scaler.inverse_transform(X_all_scaled) if scaler is not None else X_all_scaled

    log.info(f"  准备并行绘制前 100 对交互散点图...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(
                _worker_interaction_scatter,
                rank, i, j, strength,
                shap_interaction[:, i, j],
                X_display[:, i],
                X_display[:, j],
                feature_cols[i],
                feature_cols[j],
                theme,
                os.path.join(sub_dir, f'{rank:03d}_{feature_cols[i]}×{feature_cols[j]}.png')
            )
            for rank, (i, j, strength) in enumerate(pairs_strength, start=1)
        ]

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            if i % 20 == 0:
                log.info(f"  图8进度: {i}/100...")

    log.info(f"  图8并行绘制完成，保存于: {sub_dir}")


# =====================================================================
#  图9: SHAP 热力图（Heatmap）
# =====================================================================
def plot_9_shap_heatmap(shap_values, feature_cols, target_name, plot_dir=PLOT_DIR):
    """
    SHAP 热力图：直接在函数调用中设置 cmap，避免索引错误。
    """
    log.info(f"绘制图9: SHAP热力图 — {target_name}")
    theme = get_theme(target_name)

    # 清理当前图形，防止 Axes 冲突
    plt.clf()

    # 直接在调用时传入 cmap
    # 注意：shap.plots.heatmap 内部会处理 Axes 分配
    shap.plots.heatmap(
        shap_values,
        max_display=len(feature_cols),
        show=False,
        cmap=theme['cmap_diverging']
    )

    # 获取当前的 Figure 进行修饰
    fig = plt.gcf()
    fig.set_size_inches(14, 10)

    # 尝试为当前坐标系添加标题
    plt.title(f'{target_name} — SHAP 热力图', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图9_SHAP热力图_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


# =====================================================================
#  图10: 各模型残差图（测试集）
# =====================================================================
def plot_10_residuals(y_test, test_results, target_name, plot_dir=PLOT_DIR):
    """
    为每个模型单独绘制残差图，保存到 plot_dir/图10_残差图/ 子目录中。
    从 experiment.py 迁移而来。
    """
    log.info(f"绘制图10: 各模型残差图（拆分版）— {target_name}")

    theme = get_theme(target_name)

    # 创建子目录
    sub_dir = os.path.join(plot_dir, '图10_残差图')
    os.makedirs(sub_dir, exist_ok=True)

    model_names = list(test_results.keys())

    for idx, name in enumerate(model_names):
        fig, ax = plt.subplots(figsize=(6, 5))

        y_pred = test_results[name]['y_pred']
        residuals = y_test - y_pred
        model_color = MODEL_COLORS.get(name, '#3498db')

        ax.scatter(y_pred, residuals, alpha=0.35, s=14,
                   color=model_color, edgecolors='none')
        ax.axhline(y=0, color=theme['accent'], linestyle='--', linewidth=1)
        ax.set_xlabel('预测值', fontsize=11)
        ax.set_ylabel('残差', fontsize=11)
        ax.set_title(f'{target_name} — {name} 残差图（测试集）', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

        plt.tight_layout()
        save_path = os.path.join(sub_dir, f'{idx + 1:02d}_{name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  [{idx + 1}/{len(model_names)}] 已保存: {save_path}")

    log.info(f"  图10全部完成，共 {len(model_names)} 张图，保存于: {sub_dir}")


# =====================================================================
#  图11: 最佳树模型特征重要性图
# =====================================================================
def plot_11_feature_importance(trained_models, model_name, feature_cols, target_name, plot_dir=PLOT_DIR):
    """
    使用设定的 SHAP 分析模型绘制特征重要性条形图。
    """
    log.info(f"绘制图11: 特征重要性图 — {target_name} (使用设定模型: {model_name})")

    # 直接获取设定的模型
    if model_name not in trained_models:
        log.warning(f"  在已训练模型中未找到 {model_name}，跳过图11")
        return

    best_model = trained_models[model_name]

    # 检查模型是否支持该属性（如 OLS 就不支持）
    if not hasattr(best_model, 'feature_importances_'):
        log.warning(f"  {model_name} 不支持 feature_importances_ 属性，跳过图11")
        return

    importance = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        '特征': feature_cols,
        '重要性': importance
    }).sort_values('重要性', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    theme = get_theme(target_name)

    # 绘图逻辑
    ax.barh(feat_imp['特征'], feat_imp['重要性'], color=theme['ratio_low'], alpha=0.8)
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'{target_name} — {model_name} 特征重要性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图11_特征重要性_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


# =====================================================================
#  图12: SHAP 特征影响-交互网络图
# =====================================================================
def plot_12_shap_network(shap_values, shap_interaction, feature_cols, target_name, plot_dir=PLOT_DIR):
    """
    SHAP 特征影响-交互网络图（圆形布局 + 双 colorbar 样式）。
    - 节点沿圆周均匀分布，大小 ∝ 平均 |SHAP| 主效应（Vimp）
    - 节点颜色：绿色渐变，深浅 ∝ Vimp
    - 边 = 所有特征对之间的交互效应（Vint），粗细 & 紫色深浅 ∝ Vint
    - 右侧两个 colorbar：Vint（紫色渐变）、Vimp（绿色渐变）
    """
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    log.info(f"绘制图12: SHAP特征影响-交互网络图 — {target_name}")

    n_features = len(feature_cols)
    shap_vals_array = shap_values.values  # (n_samples, n_features)

    # ---- 1. 计算节点属性 ----
    mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)  # Vimp

    # ---- 2. 计算所有边的交互强度 ----
    all_edges = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            strength = np.abs(shap_interaction[:, i, j]).mean()
            all_edges.append((i, j, strength))

    # 按交互强度排序，确定阈值
    all_strengths = np.array([s for _, _, s in all_edges])
    max_strength = all_strengths.max() if len(all_strengths) > 0 else 1.0
    # 过滤阈值：低于最大强度 5% 的边直接不画
    filter_threshold = max_strength * 0.05
    # 弱边阈值：低于最大强度 20% 的边画成浅灰色细线
    weak_threshold = max_strength * 0.20

    # 分类：强边（紫色渐变）和弱边（灰色细线）
    strong_edges = [(i, j, s) for i, j, s in all_edges if s >= weak_threshold]
    weak_edges = [(i, j, s) for i, j, s in all_edges if filter_threshold <= s < weak_threshold]
    n_filtered = sum(1 for _, _, s in all_edges if s < filter_threshold)

    log.info(f"  边分类: 强边 {len(strong_edges)} 条, 弱边(灰色) {len(weak_edges)} 条, "
             f"过滤掉 {n_filtered} 条 (< 5% max)")

    # ---- 3. 构建网络图 ----
    G = nx.Graph()
    for i, feat in enumerate(feature_cols):
        G.add_node(feat, importance=mean_abs_shap[i])
    # 只添加未被过滤的边
    for i, j, strength in strong_edges + weak_edges:
        G.add_edge(feature_cols[i], feature_cols[j], weight=strength)

    # ---- 4. 圆形布局 ----
    pos = nx.circular_layout(G)

    # ---- 5. 绘图 ----
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # 根据因变量名称动态设定渐变色列表
    if target_name == TARGET_A:
        # 步行联系强度（%）：边=红色渐变，节点=蓝色渐变
        edge_colors_list = ['#FFF5F0', '#FCBBA1', '#FB6A4A', '#CB181D', '#67000D']
        node_colors_list = ['#F7FBFF', '#DEEBF7', '#6BAED6', '#2171B5', '#08306B']
        vint_label, vimp_label = '交互强度 (Vint)', '特征影响 (Vimp)'
    else:
        # 步行联系收缩度：边=紫色渐变，节点=黄色渐变
        edge_colors_list = ['#F0E6FF', '#C9A0DC', '#9B59B6', '#6C3483', '#2E0854']
        node_colors_list = ['#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#CC4C02']
        vint_label, vimp_label = '交互强度 (Vint)', '特征影响 (Vimp)'

    # 定义 Colormap
    vint_cmap = LinearSegmentedColormap.from_list('vint_map', edge_colors_list)
    vimp_cmap = LinearSegmentedColormap.from_list('vimp_map', node_colors_list)

    # -- 先绘制弱边（灰色细线，画在底层） --
    if weak_edges:
        weak_edge_list = [(feature_cols[i], feature_cols[j]) for i, j, _ in weak_edges]
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=weak_edge_list,
                               width=0.4,
                               edge_color='#D5D5D5',
                               style='solid',
                               alpha=0.6)

    # -- 再绘制强边（动态渐变色，画在上层） --
    if strong_edges:
        strong_edge_list = [(feature_cols[i], feature_cols[j]) for i, j, _ in strong_edges]
        strong_weights = np.array([s for _, _, s in strong_edges])
        # 强边的 colorbar 范围：从 weak_threshold 到 max_strength
        vint_min_strong = weak_threshold
        vint_max_strong = max_strength
        edge_norm = Normalize(vmin=vint_min_strong, vmax=vint_max_strong)

        # 边宽度：映射到 [1.0, 10]
        if vint_max_strong > vint_min_strong:
            strong_widths = 1.0 + (strong_weights - vint_min_strong) / \
                            (vint_max_strong - vint_min_strong) * 9.0
        else:
            strong_widths = np.full_like(strong_weights, 3.0)

        # 边颜色：使用 vint_cmap
        strong_colors = [vint_cmap(edge_norm(w)) for w in strong_weights]

        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=strong_edge_list,
                               width=list(strong_widths),
                               edge_color=strong_colors,
                               style='solid')
    else:
        # 如果没有强边，colorbar 使用全局范围
        edge_norm = Normalize(vmin=0, vmax=max_strength)

    # -- 节点属性 --
    node_importances = np.array([G.nodes[n]['importance'] for n in G.nodes()])
    vimp_min, vimp_max = node_importances.min(), node_importances.max()
    node_norm = Normalize(vmin=vimp_min, vmax=vimp_max)

    # 节点大小：映射到 [300, 2800]
    if vimp_max > vimp_min:
        node_sizes = 300 + (node_importances - vimp_min) / (vimp_max - vimp_min) * 2500
    else:
        node_sizes = np.full_like(node_importances, 1200)

    # 节点颜色：使用 vimp_cmap
    node_colors = [vimp_cmap(node_norm(imp)) for imp in node_importances]

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors='white',
                           linewidths=1.5)

    # -- 标签：放在圆周外侧，字号加大 --
    label_pos = {}
    for node, (x, y) in pos.items():
        # 沿径向向外偏移
        r = np.sqrt(x**2 + y**2)
        if r > 0:
            offset = 0.18
            label_pos[node] = (x + offset * x / r, y + offset * y / r)
        else:
            label_pos[node] = (x, y + 0.18)

    for node, (lx, ly) in label_pos.items():
        # 根据位置决定对齐方式
        cx, cy = pos[node]
        if cx > 0.05:
            ha = 'left'
        elif cx < -0.05:
            ha = 'right'
        else:
            ha = 'center'
        ax.text(lx, ly, node, fontsize=12, fontweight='bold',
                ha=ha, va='center', color='#222222')

    ax.set_title(f'{target_name} — SHAP 特征交互网络图',
                 fontsize=15, fontweight='bold', pad=20)
    ax.axis('off')

    # ---- 6. 双 colorbar (修长垂直版) ----
    # 留出右侧空间给 colorbar
    plt.subplots_adjust(right=0.85)

    cbar_ax1 = fig.add_axes([0.86, 0.55, 0.025, 0.30])
    sm1 = ScalarMappable(cmap=vint_cmap, norm=edge_norm)
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cbar_ax1)
    cb1.set_label(vint_label, fontsize=11, fontweight='bold', labelpad=10)
    cb1.ax.tick_params(labelsize=9)

    # Vimp colorbar (下方，宽度 0.015)
    cbar_ax2 = fig.add_axes([0.86, 0.15, 0.025, 0.30])
    sm2 = ScalarMappable(cmap=vimp_cmap, norm=node_norm)
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cbar_ax2)
    cb2.set_label(vimp_label, fontsize=11, fontweight='bold', labelpad=10)
    cb2.ax.tick_params(labelsize=9)

    save_path = os.path.join(plot_dir, f'图12_SHAP特征交互网络图_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    log.info(f"  已保存: {save_path}")

    # 输出网络统计信息
    log.info(f"  网络统计: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    degree_centrality = nx.degree_centrality(G)
    sorted_dc = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info(f"  度中心性 Top 5:")
    for rank, (feat, dc) in enumerate(sorted_dc, 1):
        log.info(f"    {rank}. {feat}: {dc:.3f}")


# =====================================================================
#  主流程
# =====================================================================
def run_plots_for_target(target_name, y, X, feature_cols, plot_ids=None, max_features=None):
    """
    为指定因变量运行所有图例绘制。
    已修改：SHAP 计算和导出现在基于全量样本（3000+），而非仅训练集。
    """
    log.info(f"\n{'='*70}")
    log.info(f"  开始绘制: {target_name} (全样本分析模式)")
    log.info(f"{'='*70}")

    # 根据因变量名称创建子目录
    target_plot_dir = os.path.join(PLOT_DIR, target_name)
    os.makedirs(target_plot_dir, exist_ok=True)
    log.info(f"  输出子目录: {target_plot_dir}")

    # 加载最优参数
    all_params = load_best_params(target_name)
    if all_params is None:
        return

    # 1. 划分数据集用于模型训练和性能验证 (保持与 experiment.py 一致)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(f"  模型训练划分: 训练集 {X_train.shape[0]}, 测试集 {X_test.shape[0]}")

    # 2. 训练所有模型并获取标准化器
    log.info("  训练所有模型以确定性能...")
    trained_models, test_results, scaler = train_all_models(
        X_train, y_train, X_test, y_test, all_params, feature_cols
    )

    # 3. 准备【全量样本】的标准化数据用于 SHAP 分析
    # 将输入的原始 X 全部进行标准化处理
    X_all_scaled = scaler.transform(X)
    log.info(f"  SHAP 分析将基于全量样本进行，样本总数: {X_all_scaled.shape[0]}")

    # 获取 SHAP 分析所用的模型
    shap_model_name = SHAP_MODEL_CONFIG.get(target_name, 'auto')
    if shap_model_name == 'auto':
        with open(os.path.join(OUTPUT_DIR, 'training_summary.json'), 'r', encoding='utf-8') as f:
            summary = json.load(f)
        shap_model_name = summary['results'][target_name]['best_cv_tree_model']
        if not shap_model_name:
            raise ValueError('缺少本轮交叉验证结果，请先运行调优模式')
    shap_model = trained_models.get(shap_model_name)
    if shap_model is None:
        log.error(f"  未找到 {shap_model_name} 模型，无法进行 SHAP 分析")
        return
    log.info(f"  SHAP 分析模型: {shap_model_name}")

    # ============ 新增：超时确认逻辑 ============
    # 在正式开始高负载计算前停顿，等待用户确认或超时
    if not timed_continue_prompt(timeout=10):
        return
    # ==========================================

    should_plot = lambda n: plot_ids is None or n in plot_ids


    # ===== 图1: 模型性能验证图 (依然基于测试集展示泛化能力) =====
    if should_plot(1):
        train_preds = {}
        X_train_scaled = scaler.transform(X_train)
        for m_name, m_obj in trained_models.items():
            if m_name == 'OLS':
                X_train_const = sm.add_constant(X_train_scaled)
                train_preds[m_name] = m_obj.predict(X_train_const)
            else:
                train_preds[m_name] = m_obj.predict(X_train_scaled)
        plot_1_model_performance(y_test, test_results, target_name,
                                 shap_model_name=shap_model_name,
                                 y_train=y_train, train_preds=train_preds,
                                 plot_dir=target_plot_dir)

    # ===== 图10: 各模型残差图 =====
    if should_plot(10):
        plot_10_residuals(y_test, test_results, target_name, plot_dir=target_plot_dir)

    # ===== 图11: 特征重要性图 =====
    if should_plot(11):
        plot_11_feature_importance(trained_models, shap_model_name, feature_cols, target_name, plot_dir=target_plot_dir)
    # =================================================================
    #  核心修改：计算全量样本的 SHAP 值
    # =================================================================
    shap_values = None
    if any(should_plot(n) for n in [2, 3, 4, 9, 12]):
        # 注意：这里传入的是 X_all_scaled
        shap_values, explainer = compute_shap_values(
            shap_model, X_all_scaled, feature_cols
        )

        # ===== 导出全量样本的 SHAP 值到 CSV =====
        shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
        shap_df.insert(0, '样本序号', range(len(shap_df)))
        # 加入基准值
        shap_df['base_value'] = shap_values.base_values
        csv_path = os.path.join(target_plot_dir, f'SHAP值_逐样本_全量_{target_name}.csv')
        shap_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        log.info(f"  【全量导出】SHAP值CSV已保存: {csv_path} (共 {shap_df.shape[0]} 样本)")

    # ===== 图2: SHAP Bar Plot (全量) =====
    if should_plot(2) and shap_values is not None:
        plot_2_shap_bar(shap_values, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图3: SHAP Beeswarm (全量) =====
    if should_plot(3) and shap_values is not None:
        plot_3_shap_beeswarm(shap_values, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图4: SHAP Dependence (全量) =====
    if should_plot(4) and shap_values is not None:
        plot_4_shap_dependence(shap_values, X_all_scaled, feature_cols, target_name, scaler=scaler, plot_dir=target_plot_dir, max_features=max_features)

    # =================================================================
    #  核心修改：计算全量样本的 SHAP 交互值
    # =================================================================
    shap_interaction = None
    if any(should_plot(n) for n in [5, 6, 7, 8, 12]):
        # 注意：这里传入的是 X_all_scaled
        shap_interaction = compute_shap_interaction_values(
            shap_model, X_all_scaled, feature_cols
        )

    # ===== 图5: 二维 PDP (全量) =====
    if should_plot(5):
        plot_5_pdp_2d(shap_model, X_all_scaled, feature_cols, target_name,
                      shap_interaction=shap_interaction, scaler=scaler, plot_dir=target_plot_dir)

    # ===== 图6: 交互效应矩阵 (全量) =====
    top_pairs = None
    if should_plot(6) and shap_interaction is not None:
        top_pairs = plot_6_interaction_matrix(shap_interaction, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图7: 主效应 vs 交互效应 (全量) =====
    if should_plot(7) and shap_interaction is not None:
        plot_7_main_vs_interaction(shap_values if shap_values is not None else None,
                                   shap_interaction, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图8: 交互散点图 (全量) =====
    if should_plot(8) and shap_interaction is not None:
        plot_8_interaction_scatter(shap_interaction, X_all_scaled, feature_cols,
                                   target_name, top_pairs=top_pairs, scaler=scaler, plot_dir=target_plot_dir)

    # ===== 图9: SHAP 热力图 (全量) =====
    if should_plot(9) and shap_values is not None:
        plot_9_shap_heatmap(shap_values, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图12: SHAP 特征影响-交互网络图 (全量) =====
    if should_plot(12) and shap_values is not None and shap_interaction is not None:
        plot_12_shap_network(shap_values, shap_interaction, feature_cols, target_name, plot_dir=target_plot_dir)

    log.info(f"\n  {target_name} 全量样本分析及图例绘制完成！")
    log.info(f"  图例保存在: {target_plot_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='SHAP 图例绘制脚本',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--plots', type=int, nargs='+', default=None,
        help='要绘制的图例编号（1-12），不指定则绘制全部\n例如: --plots 1 2 3'
    )
    parser.add_argument(
        '--target', type=str, nargs='+', default=None,
        help=(
            '指定因变量（可多选，空格分隔）\n'
            '例如: --target 步行联系强度（%%）\n'
            '      --target 步行联系强度（%%） 步行联系收缩度\n'
            '不指定则自动检测数据中可用的因变量'
        )
    )
    parser.add_argument(
        '--max-features', type=int, default=None,
        help='限制图4输出的前 N 个特征；不指定则输出全部特征'
    )
    parser.add_argument('--output-dir', type=str, required=True,
                        help='本轮训练结果目录（含 best_params 和 training_summary.json）')
    return parser.parse_args()


def main():
    global OUTPUT_DIR, BEST_PARAMS_DIR, PLOT_DIR
    args = parse_args()
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    BEST_PARAMS_DIR = os.path.join(OUTPUT_DIR, 'best_params')
    PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(PLOT_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("  SHAP 图例绘制脚本")
    log.info("=" * 70)
    log.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  输出目录: {PLOT_DIR}")

    if args.plots:
        log.info(f"  指定绘制图例: {args.plots}")
    else:
        log.info("  绘制全部图例 (1-12)")

    # 1. 确定本次运行要处理的因变量（用户指定的或自动检测的）
    if args.target:
        current_run_targets = args.target
    else:
        df_head = pd.read_csv(DATA_PATH, encoding='utf-8-sig', nrows=0)
        current_run_targets = [t for t in ALL_TARGETS if t in df_head.columns]

    # 2. 调用修改后的加载函数
    # 无论 current_run_targets 是多少，load_data 内部都会排除 ALL_TARGETS
    X, y_dict, feature_cols = load_data(current_run_targets)

    for target_name in current_run_targets:
        run_plots_for_target(target_name, y_dict[target_name], X, feature_cols, plot_ids=args.plots, max_features=args.max_features)

    log.info(f"\n{'='*70}")
    log.info(f"  所有图例绘制完成！")
    log.info(f"  输出目录: {PLOT_DIR}")
    log.info(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
