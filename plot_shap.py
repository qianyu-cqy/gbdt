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
  python plot_shap.py --target 步行比例  # 只绘制指定因变量的图例
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

# ============ 中文字体设置 ============
rcParams['font.sans-serif'] = ['STHeiti', 'Heiti TC', 'Songti SC', 'Arial Unicode MS']
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
DATA_PATH = os.path.join(BASE_DIR, 'new_data.csv')
BEST_PARAMS_DIR = os.path.join(BASE_DIR, 'best_params')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

TARGET_A = '步行比例'
TARGET_B = '步行距离'
EXCLUDE_COLS = ['街区编号', TARGET_A, TARGET_B]
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1

# ============ SHAP 分析模型配置 ============
# 为每个因变量指定用于 SHAP 分析的模型（可选: 'XGBOOST' 或 'LightGBM'）
# 根据实验结果选择各因变量表现最优的模型
SHAP_MODEL_CONFIG = {
    TARGET_A: 'LightGBM',   # 步行比例 — LightGBM 表现最优
    TARGET_B: 'XGBOOST',    # 步行距离 — XGBoost 表现最优
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

# 色带1（TARGET_A 步行比例）：蓝色（高值）→ 淡紫灰（中间值）→ 粉红/玫瑰色（低值）
_CMAP_A_DIVERGING = LinearSegmentedColormap.from_list(
    'BluLavenderPink',
    ['#E91E63', '#F8BBD0', '#D7C4D9', '#BBDEFB', '#1565C0']
)
# 色带2（TARGET_B 步行距离）：黄色（高值）→ 淡灰绿（中间值）→ 浅青/墨青色（低值）
_CMAP_B_DIVERGING = LinearSegmentedColormap.from_list(
    'TealSageYellow',
    ['#00695C', '#B2DFDB', '#C8D5C4', '#FFF9C4', '#F9A825']
)

# 色带1 的顺序版本（用于非发散型热力图，如交互矩阵）
_CMAP_A_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'LavenderToPink', ['#F0E6EE', '#F8BBD0', '#E91E63', '#880E4F']
)
# 色带2 的顺序版本
_CMAP_B_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'MintToTeal', ['#E6EDEB', '#B2DFDB', '#00897B', '#004D40']
)

# 每个 TARGET 的完整主题色配置
TARGET_THEME = {
    TARGET_A: {
        # 发散型色带（散点图、PDP、交互散点等）
        'cmap_diverging': _CMAP_A_DIVERGING,
        # 顺序型色带（交互矩阵热力图等）
        'cmap_sequential': _CMAP_A_SEQUENTIAL,
        # 主色调（LOWESS 线、置信区间、y=0 线等）
        'primary': '#1565C0',        # 深蓝
        # 强调色（阈值线、交叉点标注等）
        'accent': '#C2185B',         # 玫瑰红
        # 正向背景色（SHAP > 0 区域）
        'bg_positive': '#E3F2FD',    # 浅蓝
        # 负向背景色（SHAP < 0 区域）
        'bg_negative': '#FCE4EC',    # 浅粉
        # 图7 主效应条形图颜色
        'bar_main': '#1565C0',       # 蓝色
        # 图7 交互效应条形图颜色
        'bar_interact': '#E91E63',   # 粉红
        # 图11 特征重要性条形图颜色
        'bar_importance': '#1976D2', # 蓝色
        # 图12 正向节点 RGB（蓝色系）
        'node_positive': (0.2, 0.4, 0.9),
        # 图12 负向节点 RGB（粉红系）
        'node_negative': (0.91, 0.12, 0.39),
        # 图7 交互占比高阈值颜色
        'ratio_high': '#C2185B',
        # 图7 交互占比中阈值颜色
        'ratio_mid': '#5C6BC0',
        # 图7 交互占比低阈值颜色
        'ratio_low': '#90A4AE',
    },
    TARGET_B: {
        'cmap_diverging': _CMAP_B_DIVERGING,
        'cmap_sequential': _CMAP_B_SEQUENTIAL,
        'primary': '#00695C',        # 墨青
        'accent': '#F57F17',         # 深黄
        'bg_positive': '#E0F2F1',    # 浅青
        'bg_negative': '#FFFDE7',    # 浅黄
        'bar_main': '#00695C',       # 墨青
        'bar_interact': '#F9A825',   # 黄色
        'bar_importance': '#00796B', # 青色
        'node_positive': (0.0, 0.41, 0.36),
        'node_negative': (0.98, 0.66, 0.15),
        'ratio_high': '#E65100',
        'ratio_mid': '#00897B',
        'ratio_low': '#90A4AE',
    },
}


def get_theme(target_name):
    """获取指定 TARGET 的主题色配置，若未找到则返回 TARGET_A 的配置作为默认值"""
    return TARGET_THEME.get(target_name, TARGET_THEME[TARGET_A])

# ============ 日志 ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('plot_shap')


# ============ 数据加载 ============
def load_data():
    """加载数据并返回特征矩阵、因变量、特征名"""
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].values
    y_a = df[TARGET_A].values
    y_b = df[TARGET_B].values
    log.info(f"数据加载完成: {X.shape[0]} 样本, {len(feature_cols)} 特征")
    return X, y_a, y_b, feature_cols


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
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS, **params)
    elif model_name == 'XGBOOST':
        return XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=N_JOBS, **params)
    elif model_name == 'LightGBM':
        return LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=N_JOBS, **params)
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

    # OLS
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
    ax.set_title(f'{target_name} — 多模型 R² 对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(r2_values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
    # 将条形图颜色统一为主题色
    for patch in ax.patches:
        patch.set_facecolor(theme['bar_importance'])
    # 移除条形图上的数字标注
    for txt in ax.texts:
        txt.remove()
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


def _compute_lowess_with_ci(feat_vals, shap_vals, frac=0.3, n_bootstrap=200):
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


def _plot_single_dependence(feat, feat_vals, shap_vals, theme, target_name, save_path):
    """
    为单个特征绘制 SHAP 依赖图（散点 + LOWESS + 内嵌柱状分布）。
    柱状分布以灰色背景直方图形式嵌入散点图底部，无背景分区，无阈值标注。
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # ---- 计算 LOWESS 拟合 + 置信区间 ----
    lowess_x, lowess_y, ci_lower, ci_upper = _compute_lowess_with_ci(
        feat_vals, shap_vals, frac=0.3, n_bootstrap=200
    )

    # ---- 坐标范围 ----
    x_min, x_max = feat_vals.min(), feat_vals.max()
    x_margin = (x_max - x_min) * 0.03
    plot_xmin = x_min - x_margin
    plot_xmax = x_max + x_margin

    # 1. 内嵌柱状分布（灰色，贴在散点图底部，使用副轴）
    ax_hist = ax.twinx()
    n_bins = min(30, max(10, len(feat_vals) // 10))
    ax_hist.hist(feat_vals, bins=n_bins, color='#C0C0C0', alpha=0.35,
                 edgecolor='white', linewidth=0.3, zorder=0)
    # 将柱状图压缩到底部约 25% 的区域
    hist_ymax = ax_hist.get_ylim()[1]
    ax_hist.set_ylim(0, hist_ymax * 4)
    ax_hist.set_yticks([])
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.spines['left'].set_visible(False)
    ax_hist.spines['bottom'].set_visible(False)

    # 2. 散点图（单一颜色）
    ax.scatter(feat_vals, shap_vals, color=theme['primary'],
               alpha=0.35, s=14, edgecolors='none', zorder=2)

    # 3. LOWESS 曲线（使用强调色，与散点区分）
    ax.plot(lowess_x, lowess_y, color=theme['accent'], linewidth=2.5, zorder=4, label='LOWESS')

    # 4. 置信区间
    ax.fill_between(lowess_x, ci_lower, ci_upper, alpha=0.2, color=theme['accent'], zorder=3,
                    label='95% 置信区间')

    # 5. y=0 水平虚线
    ax.axhline(y=0, color=theme['primary'], linestyle='--', linewidth=1, alpha=0.6, zorder=3)

    ax.set_xlabel(f'{feat}', fontsize=11)
    ax.set_ylabel(f'SHAP value for\n{feat}', fontsize=11)
    ax.set_xlim(plot_xmin, plot_xmax)
    ax.grid(True, alpha=0.2, zorder=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_4_shap_dependence(shap_values, X_data, feature_cols, target_name, scaler=None, plot_dir=PLOT_DIR):
    """
    为所有自变量绘制增强版 SHAP 依赖图，每个特征单独一张图。
    每张图包含：上方散点图 + LOWESS 曲线 + Bootstrap 置信区间 + 阈值标注 + 背景分区，
    下方为特征值的柱状分布图。
    所有图保存到 plot_dir/图4_SHAP依赖图/ 子目录中。
    如果提供 scaler，则将横坐标还原为原始值（更直观）。
    """
    log.info(f"绘制图4: SHAP单特征依赖图（LOWESS增强版 + 柱状分布）— {target_name}")

    theme = get_theme(target_name)

    # 创建子目录
    sub_dir = os.path.join(plot_dir, '图4_SHAP依赖图')
    os.makedirs(sub_dir, exist_ok=True)

    # 按平均 |SHAP| 排序（用于文件名编号，方便按重要性排列）
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    # 准备原始值数据（用于横坐标显示）
    if scaler is not None:
        X_display = scaler.inverse_transform(X_data)
    else:
        X_display = X_data

    shap_vals_array = shap_values.values  # (n_samples, n_features)

    for rank, feat_idx in enumerate(sorted_indices, start=1):
        feat = feature_cols[feat_idx]
        feat_vals = X_display[:, feat_idx]
        shap_vals = shap_vals_array[:, feat_idx]

        save_path = os.path.join(sub_dir, f'{rank:02d}_{feat}.png')
        _plot_single_dependence(feat, feat_vals, shap_vals, theme, target_name, save_path)
        log.info(f"  [{rank}/{len(feature_cols)}] 已保存: {save_path}")

    log.info(f"  图4全部完成，共 {len(feature_cols)} 张图，保存于: {sub_dir}")


# =====================================================================
#  图5: 二维 PDP 部分依赖图
# =====================================================================
def plot_5_pdp_2d(model, X_train_scaled, feature_cols, target_name, scaler=None, plot_dir=PLOT_DIR):
    """
    基于 SHAP 交互强度自动选取 Top 变量对，绘制二维 PDP 热力图。
    如果提供 scaler，则将坐标轴还原为原始值。
    """
    log.info(f"绘制图5: 二维PDP部分依赖图 — {target_name}")

    theme = get_theme(target_name)

    # 使用 SHAP 重要性排名前 6 的特征，选取有意义的组合
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(pd.DataFrame(X_train_scaled, columns=feature_cols))
    mean_abs_shap = np.abs(shap_vals.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:6]
    top_features = [feature_cols[i] for i in top_indices]

    # 生成变量对（取前 N_PDP_PAIRS 对）
    pairs = []
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            pairs.append((top_features[i], top_features[j]))
    pairs = pairs[:N_PDP_PAIRS]

    n_cols = 2
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    # 使用 sklearn 的 PartialDependenceDisplay 逐对绘制
    df_train = pd.DataFrame(X_train_scaled, columns=feature_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    for idx, (feat_a, feat_b) in enumerate(pairs):
        ax = axes_flat[idx]
        feat_idx_a = feature_cols.index(feat_a)
        feat_idx_b = feature_cols.index(feat_b)

        # 计算部分依赖
        from sklearn.inspection import partial_dependence
        pd_result = partial_dependence(
            model, X_train_scaled, features=[(feat_idx_a, feat_idx_b)],
            kind='average', grid_resolution=30
        )

        # 提取网格和预测值
        grid_a = pd_result['grid_values'][0]
        grid_b = pd_result['grid_values'][1]

        # 如果提供了 scaler，将网格值反标准化为原始值
        if scaler is not None:
            mean_a, std_a = scaler.mean_[feat_idx_a], scaler.scale_[feat_idx_a]
            mean_b, std_b = scaler.mean_[feat_idx_b], scaler.scale_[feat_idx_b]
            grid_a = grid_a * std_a + mean_a
            grid_b = grid_b * std_b + mean_b

        xx, yy = np.meshgrid(grid_a, grid_b)
        zz = pd_result['average'][0].T  # 转置以匹配 meshgrid

        # 绘制热力图（填充等高线）
        n_levels = 8  # 等高线层数，与参考图一致
        contour_filled = ax.contourf(xx, yy, zz, levels=n_levels, cmap=theme['cmap_diverging'], alpha=0.9)
        # 绘制等高线（线条）
        contour_lines = ax.contour(xx, yy, zz, levels=contour_filled.levels, colors='white',
                                   linewidths=0.8, alpha=0.7)
        # 在等高线上标注数值（大字体、加粗，参考示意图样式）
        clbl = ax.clabel(contour_lines, inline=True, fontsize=13, fmt='%.2f', colors='black')
        for txt in clbl:
            txt.set_fontweight('bold')
        fig.colorbar(contour_filled, ax=ax, shrink=0.8, label='部分依赖值')
        ax.set_xlabel(feat_a, fontsize=11)
        ax.set_ylabel(feat_b, fontsize=11)
        ax.set_title(f'{feat_a} × {feat_b}', fontsize=12, fontweight='bold')

    # 隐藏多余子图
    for idx in range(len(pairs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f'{target_name} — 二维 PDP 部分依赖图', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图5_二维PDP_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


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

    # 左图：堆叠条形图
    y_pos = np.arange(n_features)
    ax1.barh(y_pos, sorted_main, color=theme['bar_main'], label='主效应', height=0.7)
    ax1.barh(y_pos, sorted_inter, left=sorted_main, color=theme['bar_interact'], label='交互效应', height=0.7)
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
def plot_8_interaction_scatter(shap_interaction, X_data, feature_cols, target_name,
                               top_pairs=None, scaler=None, plot_dir=PLOT_DIR):
    """
    为交互最强的 Top 5 变量对绘制 SHAP 交互散点图，每对单独一张图。
    横轴=变量A的值，纵轴=A与B的SHAP交互值，颜色=变量B的值。
    如果提供 scaler，则将横轴和颜色编码还原为原始值。
    所有图保存到 plot_dir/图8_交互散点图/ 子目录中，按交互强度排名编号。
    """
    log.info(f"绘制图8: SHAP成对交互散点图（Top 5）— {target_name}")

    theme = get_theme(target_name)

    # 创建子目录
    sub_dir = os.path.join(plot_dir, '图8_交互散点图')
    os.makedirs(sub_dir, exist_ok=True)

    # 计算所有变量对的交互强度并排序
    n_features = len(feature_cols)
    pairs_strength = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            val = np.abs(shap_interaction[:, i, j]).mean()
            pairs_strength.append((i, j, val))
    pairs_strength.sort(key=lambda x: x[2], reverse=True)
    pairs_strength = pairs_strength[:5]  # 只取 Top 5 交互对

    # 如果提供了 scaler，将特征值反标准化为原始值
    if scaler is not None:
        X_display = scaler.inverse_transform(X_data)
    else:
        X_display = X_data

    total = len(pairs_strength)
    for rank, (i, j, strength) in enumerate(pairs_strength, start=1):
        feat_a = feature_cols[i]
        feat_b = feature_cols[j]

        fig, ax = plt.subplots(figsize=(7, 5.5))

        # SHAP 交互值
        interaction_vals = shap_interaction[:, i, j]
        x_vals = X_display[:, i]
        color_vals = X_display[:, j]

        scatter = ax.scatter(x_vals, interaction_vals, c=color_vals, cmap=theme['cmap_diverging'],
                             alpha=0.5, s=14, edgecolors='none')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(feat_a, fontsize=11)
        ax.set_ylabel(f'SHAP 交互值\n({feat_a} × {feat_b})', fontsize=10)
        ax.set_title(f'{target_name} — {feat_a} × {feat_b}\n(交互强度={strength:.5f})',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(feat_b, fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(sub_dir, f'{rank:03d}_{feat_a}×{feat_b}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if rank <= 5 or rank == total:
            log.info(f"  [{rank}/{total}] 已保存: {save_path}")

    log.info(f"  图8完成，共 {total} 张图（Top 5），保存于: {sub_dir}")


# =====================================================================
#  图9: SHAP 热力图（Heatmap）
# =====================================================================
def plot_9_shap_heatmap(shap_values, feature_cols, target_name, plot_dir=PLOT_DIR):
    """
    SHAP 热力图：横轴为特征，纵轴为样本（按预测值排序），颜色为 SHAP 值。
    """
    log.info(f"绘制图9: SHAP热力图 — {target_name}")

    fig, ax = plt.subplots(figsize=(14, 10))
    shap.plots.heatmap(shap_values, max_display=len(feature_cols), show=False, ax=ax)
    ax.set_title(f'{target_name} — SHAP 热力图', fontsize=14, fontweight='bold')
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        save_path = os.path.join(sub_dir, f'{idx + 1:02d}_{name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"  [{idx + 1}/{len(model_names)}] 已保存: {save_path}")

    log.info(f"  图10全部完成，共 {len(model_names)} 张图，保存于: {sub_dir}")


# =====================================================================
#  图11: 最佳树模型特征重要性图
# =====================================================================
def plot_11_feature_importance(trained_models, test_results, feature_cols, target_name, plot_dir=PLOT_DIR):
    """
    使用测试集 R² 最高的树模型绘制特征重要性条形图。
    从 experiment.py 迁移而来。
    """
    log.info(f"绘制图11: 特征重要性图 — {target_name}")

    # 找到测试集 R² 最高的树模型
    tree_models = ['GBDT', 'RF', 'XGBOOST', 'LightGBM']
    best_tree_name = None
    best_r2 = -np.inf
    for name in tree_models:
        if name in test_results and name in trained_models:
            r2 = test_results[name]['metrics']['R\u00b2']
            if r2 > best_r2:
                best_r2 = r2
                best_tree_name = name

    if best_tree_name is None:
        log.warning("  未找到可用的树模型，跳过特征重要性图")
        return

    best_model = trained_models[best_tree_name]
    if not hasattr(best_model, 'feature_importances_'):
        log.warning(f"  {best_tree_name} 不支持 feature_importances_，跳过")
        return

    log.info(f"  使用最佳树模型: {best_tree_name} (R\u00b2={best_r2:.4f})")

    importance = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        '特征': feature_cols,
        '重要性': importance
    }).sort_values('重要性', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    theme = get_theme(target_name)
    ax.barh(feat_imp['特征'], feat_imp['重要性'], color=theme['bar_importance'], alpha=0.8)
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'{target_name} — {best_tree_name}（调优后）特征重要性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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

    # -- 紫色渐变 colormap（用于强边） --
    purple_cmap = LinearSegmentedColormap.from_list(
        'purple_edge', ['#F0E6FF', '#C9A0DC', '#9B59B6', '#6C3483', '#2E0854'])

    # -- 绿色渐变 colormap（用于节点） --
    green_cmap = LinearSegmentedColormap.from_list(
        'green_node', ['#E8F5E9', '#A5D6A7', '#4CAF50', '#2E7D32', '#1B5E20'])

    # -- 先绘制弱边（灰色细线，画在底层） --
    if weak_edges:
        weak_edge_list = [(feature_cols[i], feature_cols[j]) for i, j, _ in weak_edges]
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=weak_edge_list,
                               width=0.4,
                               edge_color='#D5D5D5',
                               style='solid',
                               alpha=0.6)

    # -- 再绘制强边（紫色渐变，画在上层） --
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

        # 边颜色：紫色渐变
        strong_colors = [purple_cmap(edge_norm(w)) for w in strong_weights]

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

    # 节点颜色：绿色渐变
    node_colors = [green_cmap(node_norm(imp)) for imp in node_importances]

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

    # 留出右侧空间给 colorbar
    plt.subplots_adjust(right=0.82)

    # ---- 6. 双 colorbar ----
    # Vint colorbar（紫色，上方）
    cbar_ax1 = fig.add_axes([0.86, 0.55, 0.025, 0.30])
    sm1 = ScalarMappable(cmap=purple_cmap, norm=edge_norm)
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cbar_ax1)
    cb1.set_label('Vint', fontsize=12, fontweight='bold')
    cb1.ax.tick_params(labelsize=9)

    # Vimp colorbar（绿色，下方）
    cbar_ax2 = fig.add_axes([0.86, 0.15, 0.025, 0.30])
    sm2 = ScalarMappable(cmap=green_cmap, norm=node_norm)
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cbar_ax2)
    cb2.set_label('Vimp', fontsize=12, fontweight='bold')
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
def run_plots_for_target(target_name, y, X, feature_cols, plot_ids=None):
    """
    为指定因变量运行所有图例绘制。
    plot_ids: 要绘制的图例编号列表，None 表示全部绘制。
    """
    log.info(f"\n{'='*70}")
    log.info(f"  开始绘制: {target_name}")
    log.info(f"{'='*70}")

    # 根据因变量名称创建子目录
    target_plot_dir = os.path.join(PLOT_DIR, target_name)
    os.makedirs(target_plot_dir, exist_ok=True)
    log.info(f"  输出子目录: {target_plot_dir}")

    # 加载最优参数
    all_params = load_best_params(target_name)
    if all_params is None:
        return

    # 划分数据集（与 experiment.py 一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(f"  数据划分: 训练集 {X_train.shape[0]}, 测试集 {X_test.shape[0]}")

    # 训练所有模型
    log.info("  训练所有模型...")
    trained_models, test_results, scaler = train_all_models(
        X_train, y_train, X_test, y_test, all_params, feature_cols
    )

    # 标准化后的数据
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 获取 SHAP 分析所用的模型
    shap_model_name = SHAP_MODEL_CONFIG.get(target_name, 'XGBOOST')
    shap_model = trained_models.get(shap_model_name)
    if shap_model is None:
        log.error(f"  未找到 {shap_model_name} 模型，无法进行 SHAP 分析")
        return
    log.info(f"  SHAP 分析模型: {shap_model_name}")

    should_plot = lambda n: plot_ids is None or n in plot_ids

    # ===== 图1: 模型性能验证图 =====
    if should_plot(1):
        # 计算训练集预测值（用于 Joint Plot）
        train_preds = {}
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
        plot_11_feature_importance(trained_models, test_results, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 计算 SHAP 值（图2/3/4/9/12 共用） =====
    shap_values = None
    if any(should_plot(n) for n in [2, 3, 4, 9, 12]):
        shap_values, explainer = compute_shap_values(
            shap_model, X_train_scaled, feature_cols
        )

    # ===== 图2: SHAP Bar Plot =====
    if should_plot(2) and shap_values is not None:
        plot_2_shap_bar(shap_values, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图3: SHAP Beeswarm =====
    if should_plot(3) and shap_values is not None:
        plot_3_shap_beeswarm(shap_values, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图4: SHAP Dependence =====
    if should_plot(4) and shap_values is not None:
        plot_4_shap_dependence(shap_values, X_train_scaled, feature_cols, target_name, scaler=scaler, plot_dir=target_plot_dir)

    # ===== 图5: 二维 PDP =====
    if should_plot(5):
        plot_5_pdp_2d(shap_model, X_train_scaled, feature_cols, target_name, scaler=scaler, plot_dir=target_plot_dir)

    # ===== 计算 SHAP 交互值（图6/7/8/12 共用） =====
    shap_interaction = None
    if any(should_plot(n) for n in [6, 7, 8, 12]):
        shap_interaction = compute_shap_interaction_values(
            shap_model, X_train_scaled, feature_cols
        )

    # ===== 图6: 交互效应矩阵 =====
    top_pairs = None
    if should_plot(6) and shap_interaction is not None:
        top_pairs = plot_6_interaction_matrix(shap_interaction, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图7: 主效应 vs 交互效应 =====
    if should_plot(7) and shap_interaction is not None:
        plot_7_main_vs_interaction(shap_values if shap_values is not None else None,
                                   shap_interaction, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图8: 交互散点图 =====
    if should_plot(8) and shap_interaction is not None:
        plot_8_interaction_scatter(shap_interaction, X_train_scaled, feature_cols,
                                   target_name, top_pairs=top_pairs, scaler=scaler, plot_dir=target_plot_dir)

    # ===== 图9: SHAP 热力图 =====
    if should_plot(9) and shap_values is not None:
        plot_9_shap_heatmap(shap_values, feature_cols, target_name, plot_dir=target_plot_dir)

    # ===== 图12: SHAP 特征影响-交互网络图 =====
    if should_plot(12) and shap_values is not None and shap_interaction is not None:
        plot_12_shap_network(shap_values, shap_interaction, feature_cols, target_name, plot_dir=target_plot_dir)

    log.info(f"\n  {target_name} 所有图例绘制完成！")
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
        '--target', type=str, default=None,
        choices=[TARGET_A, TARGET_B],
        help=f'只绘制指定因变量的图例\n可选: {TARGET_A}, {TARGET_B}'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("  SHAP 图例绘制脚本")
    log.info("=" * 70)
    log.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  输出目录: {PLOT_DIR}")

    if args.plots:
        log.info(f"  指定绘制图例: {args.plots}")
    else:
        log.info("  绘制全部图例 (1-12)")

    # 加载数据
    X, y_a, y_b, feature_cols = load_data()

    targets = {}
    if args.target is None or args.target == TARGET_A:
        targets[TARGET_A] = y_a
    if args.target is None or args.target == TARGET_B:
        targets[TARGET_B] = y_b

    for target_name, y in targets.items():
        run_plots_for_target(target_name, y, X, feature_cols, plot_ids=args.plots)

    log.info(f"\n{'='*70}")
    log.info(f"  所有图例绘制完成！")
    log.info(f"  输出目录: {PLOT_DIR}")
    log.info(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
