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
def plot_1_model_performance(y_test, test_results, target_name, shap_model_name='XGBOOST', plot_dir=PLOT_DIR):
    """
    三联组合图：
    (a) 多模型 R² 对比柱状图
    (b) 最优模型 预测值 vs 实际值散点图
    (c) 最优模型 残差直方图
    """
    log.info(f"绘制图1: 模型性能验证图 — {target_name}（SHAP模型: {shap_model_name}）")

    model_color = MODEL_COLORS.get(shap_model_name, '#e74c3c')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- (a) 多模型 R² 对比柱状图 ---
    ax = axes[0]
    model_names = list(test_results.keys())
    r2_values = [test_results[m]['metrics']['R²'] for m in model_names]
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in model_names]

    bars = ax.bar(model_names, r2_values, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
    # 高亮 SHAP 分析所用的模型
    for i, m in enumerate(model_names):
        if m == shap_model_name:
            bars[i].set_edgecolor('#c0392b')
            bars[i].set_linewidth(2.5)

    # 在柱子上方标注数值
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('(a) 多模型 R² 对比', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(r2_values) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- (b) 最优模型 预测值 vs 实际值散点图 ---
    ax = axes[1]
    y_pred_best = test_results[shap_model_name]['y_pred']
    ax.scatter(y_test, y_pred_best, alpha=0.35, s=12, color=model_color, edgecolors='none')
    # 45° 参考线
    lims = [min(y_test.min(), y_pred_best.min()), max(y_test.max(), y_pred_best.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7, label='理想预测线 (y=x)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('实际值', fontsize=12)
    ax.set_ylabel('预测值', fontsize=12)
    ax.set_title(f'(b) {shap_model_name} 预测值 vs 实际值', fontsize=13, fontweight='bold')
    r2_best = test_results[shap_model_name]['metrics']['R²']
    ax.text(0.05, 0.92, f'R²={r2_best:.4f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color=model_color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # --- (c) 最优模型 残差直方图 ---
    ax = axes[2]
    residuals = y_test - y_pred_best
    ax.hist(residuals, bins=40, color=model_color, alpha=0.7, edgecolor='white', linewidth=0.5,
            density=True)
    # 叠加正态分布曲线
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x_range, norm.pdf(x_range, mu, std), 'k-', linewidth=1.5, label=f'正态拟合 (μ={mu:.3f}, σ={std:.3f})')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('残差 (实际值 - 预测值)', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'(c) {shap_model_name} 残差分布', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle(f'{target_name} — 模型性能验证', fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图1_模型性能验证_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


# =====================================================================
#  图2: SHAP 全局特征贡献图（Bar Plot）
# =====================================================================
def plot_2_shap_bar(shap_values, feature_cols, target_name, plot_dir=PLOT_DIR):
    """SHAP 全局特征重要性条形图"""
    log.info(f"绘制图2: SHAP全局特征贡献图 — {target_name}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=len(feature_cols), show=False, ax=ax)
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

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=len(feature_cols), show=False, plot_size=None, ax=ax)
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


def plot_4_shap_dependence(shap_values, X_data, feature_cols, target_name, scaler=None, plot_dir=PLOT_DIR):
    """
    为 SHAP 排名前 TOP_N_DEPENDENCE 个特征绘制增强版依赖图。
    包含：散点图 + LOWESS 曲线 + Bootstrap 置信区间 + 阈值标注 + 背景分区。
    如果提供 scaler，则将横坐标还原为原始值（更直观）。
    """
    log.info(f"绘制图4: SHAP单特征依赖图（LOWESS增强版）— {target_name}")

    # 按平均 |SHAP| 排序，取 Top N
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_N_DEPENDENCE]
    top_features = [feature_cols[i] for i in top_indices]

    # 准备原始值数据（用于横坐标显示）
    if scaler is not None:
        X_display = scaler.inverse_transform(X_data)
    else:
        X_display = X_data

    # 自动选择交互最强的特征作为颜色编码
    # 使用 SHAP 内置的 approximate_interactions 方法
    shap_vals_array = shap_values.values  # (n_samples, n_features)

    n_cols = 2
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 5.5 * n_rows))
    axes = axes.flatten()

    for idx, feat in enumerate(top_features):
        ax = axes[idx]
        feat_idx = feature_cols.index(feat)

        # 特征值和 SHAP 值
        feat_vals = X_display[:, feat_idx]
        shap_vals = shap_vals_array[:, feat_idx]

        # 自动选择交互最强的特征（排除自身）
        # 使用相关性方法：对每个其他特征，计算其值与当前特征 SHAP 值的相关性
        best_interact_idx = None
        best_interact_corr = -1
        for j in range(len(feature_cols)):
            if j == feat_idx:
                continue
            try:
                corr = np.abs(np.corrcoef(X_display[:, j], shap_vals)[0, 1])
                if not np.isnan(corr) and corr > best_interact_corr:
                    best_interact_corr = corr
                    best_interact_idx = j
            except Exception:
                continue

        if best_interact_idx is not None:
            color_vals = X_display[:, best_interact_idx]
            color_feat_name = feature_cols[best_interact_idx]
        else:
            color_vals = feat_vals
            color_feat_name = feat

        # ---- 计算 LOWESS 拟合 + 置信区间 ----
        lowess_x, lowess_y, ci_lower, ci_upper = _compute_lowess_with_ci(
            feat_vals, shap_vals, frac=0.3, n_bootstrap=200
        )

        # ---- 找到 LOWESS 与 y=0 的交叉点 ----
        crossings = _find_lowess_zero_crossings(lowess_x, lowess_y)

        # ---- 绘图 ----
        # 1. 背景分区（基于交叉点）
        x_min, x_max = feat_vals.min(), feat_vals.max()
        x_margin = (x_max - x_min) * 0.03
        plot_xmin = x_min - x_margin
        plot_xmax = x_max + x_margin

        if len(crossings) > 0:
            # 使用第一个交叉点进行分区
            threshold = crossings[0]
            # 判断交叉点左侧是正还是负（取 LOWESS 曲线在交叉点左侧的值）
            left_mask = lowess_x < threshold
            if np.any(left_mask):
                left_sign = np.sign(lowess_y[left_mask][-1])
            else:
                left_sign = -1  # 默认左侧为负

            if left_sign < 0:
                # 左侧负（浅红），右侧正（浅蓝）
                ax.axvspan(plot_xmin, threshold, color='#FFEBEE', alpha=0.4, zorder=0)
                ax.axvspan(threshold, plot_xmax, color='#E3F2FD', alpha=0.4, zorder=0)
            else:
                # 左侧正（浅蓝），右侧负（浅红）
                ax.axvspan(plot_xmin, threshold, color='#E3F2FD', alpha=0.4, zorder=0)
                ax.axvspan(threshold, plot_xmax, color='#FFEBEE', alpha=0.4, zorder=0)
        else:
            # 无交叉点，根据 LOWESS 均值判断整体正负
            if np.mean(lowess_y) >= 0:
                ax.axvspan(plot_xmin, plot_xmax, color='#E3F2FD', alpha=0.3, zorder=0)
            else:
                ax.axvspan(plot_xmin, plot_xmax, color='#FFEBEE', alpha=0.3, zorder=0)

        # 2. 散点图（颜色映射交互特征）
        scatter = ax.scatter(feat_vals, shap_vals, c=color_vals, cmap='RdBu',
                             alpha=0.5, s=12, edgecolors='none', zorder=2)

        # 3. LOWESS 曲线
        ax.plot(lowess_x, lowess_y, color='#1565C0', linewidth=2.5, zorder=4, label='LOWESS')

        # 4. 置信区间
        ax.fill_between(lowess_x, ci_lower, ci_upper, alpha=0.2, color='#1565C0', zorder=3,
                         label='95% 置信区间')

        # 5. y=0 水平虚线
        ax.axhline(y=0, color='#1565C0', linestyle='--', linewidth=1, alpha=0.6, zorder=3)

        # 6. 阈值标注（交叉点）
        for cross_x in crossings:
            ax.axvline(x=cross_x, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.8, zorder=4)
            # 在交叉点处标注数值
            y_range = shap_vals.max() - shap_vals.min()
            ax.annotate(f'{cross_x:.2f}',
                        xy=(cross_x, 0),
                        xytext=(cross_x, -y_range * 0.08),
                        fontsize=9, fontweight='bold', color='#D32F2F',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#D32F2F'),
                        zorder=5)

        # 7. 颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(color_feat_name, fontsize=9)

        # 8. 标签和标题
        ax.set_xlabel(f'{feat}', fontsize=11)
        ax.set_ylabel('SHAP value', fontsize=11)
        ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
        ax.set_xlim(plot_xmin, plot_xmax)
        ax.grid(True, alpha=0.2, zorder=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 9. 图例（仅第一个子图显示）
        if idx == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.8)

    # 隐藏多余子图
    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{target_name} — SHAP 单特征依赖图 (Top {TOP_N_DEPENDENCE})',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图4_SHAP依赖图_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


# =====================================================================
#  图5: 二维 PDP 部分依赖图
# =====================================================================
def plot_5_pdp_2d(model, X_train_scaled, feature_cols, target_name, scaler=None, plot_dir=PLOT_DIR):
    """
    基于 SHAP 交互强度自动选取 Top 变量对，绘制二维 PDP 热力图。
    如果提供 scaler，则将坐标轴还原为原始值。
    """
    log.info(f"绘制图5: 二维PDP部分依赖图 — {target_name}")

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

        # 绘制热力图
        contour = ax.contourf(xx, yy, zz, levels=20, cmap='RdYlBu_r', alpha=0.9)
        fig.colorbar(contour, ax=ax, shrink=0.8, label='部分依赖值')
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
        cmap='YlOrRd',
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
    ax1.barh(y_pos, sorted_main, color='#3498db', label='主效应', height=0.7)
    ax1.barh(y_pos, sorted_inter, left=sorted_main, color='#e74c3c', label='交互效应', height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_features, fontsize=9)
    ax1.set_xlabel('平均 |SHAP 值|', fontsize=12)
    ax1.set_title('主效应 vs 交互效应', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 右图：交互占比
    colors = ['#e74c3c' if r > 0.3 else '#f39c12' if r > 0.15 else '#95a5a6' for r in sorted_ratio]
    ax2.barh(y_pos, sorted_ratio * 100, color=colors, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel('交互效应占比 (%)', fontsize=12)
    ax2.set_title('交互占比', fontsize=13, fontweight='bold')
    ax2.axvline(x=30, color='red', linestyle='--', linewidth=1, alpha=0.5, label='30% 阈值')
    ax2.axvline(x=15, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='15% 阈值')
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
    为交互最强的 Top 3 对绘制 SHAP 交互散点图。
    横轴=变量A的值，纵轴=A与B的SHAP交互值，颜色=变量B的值。
    如果提供 scaler，则将横轴和颜色编码还原为原始值。
    """
    log.info(f"绘制图8: SHAP成对交互散点图 — {target_name}")

    if top_pairs is None:
        # 自动计算 Top 3 交互对
        n_features = len(feature_cols)
        pairs_strength = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                val = np.abs(shap_interaction[:, i, j]).mean()
                pairs_strength.append((i, j, val))
        pairs_strength.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs_strength[:3]
    else:
        # 将特征名转为索引
        converted = []
        for f1, f2, val in top_pairs[:3]:
            i = feature_cols.index(f1)
            j = feature_cols.index(f2)
            converted.append((i, j, val))
        top_pairs = converted

    # 如果提供了 scaler，将特征值反标准化为原始值
    if scaler is not None:
        X_display = scaler.inverse_transform(X_data)
    else:
        X_display = X_data

    fig, axes = plt.subplots(1, len(top_pairs), figsize=(7 * len(top_pairs), 5.5))
    if len(top_pairs) == 1:
        axes = [axes]

    for idx, (i, j, strength) in enumerate(top_pairs):
        ax = axes[idx]
        feat_a = feature_cols[i]
        feat_b = feature_cols[j]

        # SHAP 交互值
        interaction_vals = shap_interaction[:, i, j]
        x_vals = X_display[:, i]
        color_vals = X_display[:, j]

        scatter = ax.scatter(x_vals, interaction_vals, c=color_vals, cmap='coolwarm',
                             alpha=0.5, s=10, edgecolors='none')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(feat_a, fontsize=11)
        ax.set_ylabel(f'SHAP 交互值\n({feat_a} × {feat_b})', fontsize=10)
        ax.set_title(f'{feat_a} × {feat_b}\n(强度={strength:.5f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(feat_b, fontsize=9)

    fig.suptitle(f'{target_name} — SHAP 成对交互散点图 (Top 3)',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图8_交互散点图_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


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
    绘制所有模型的残差图（子图排列）。
    从 experiment.py 迁移而来。
    """
    log.info(f"绘制图10: 各模型残差图 — {target_name}")

    model_names = list(test_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors_list = ['#7f8c8d', '#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    for idx, name in enumerate(model_names):
        ax = axes[idx]
        y_pred = test_results[name]['y_pred']
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=10,
                   color=colors_list[idx % len(colors_list)], edgecolors='none')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('预测值', fontsize=11)
        if idx == 0:
            ax.set_ylabel('残差', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{target_name} — 各模型残差图（测试集）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f'图10_残差图_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")


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
    ax.barh(feat_imp['特征'], feat_imp['重要性'], color='#2ca02c', alpha=0.8)
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
    SHAP 特征影响-交互网络图。
    - 节点 = 特征，大小 ∝ 平均 |SHAP| 主效应
    - 节点颜色：平均 SHAP > 0 为蓝色（正向促进），< 0 为红色（负向抑制）
    - 边 = 特征对之间的交互效应，粗细 ∝ 平均 |SHAP 交互值|
    - 仅保留交互强度 Top 的边，避免过于密集
    - 力导向布局（spring layout）
    """
    import networkx as nx
    from matplotlib.lines import Line2D

    log.info(f"绘制图12: SHAP特征影响-交互网络图 — {target_name}")

    n_features = len(feature_cols)
    shap_vals_array = shap_values.values  # (n_samples, n_features)

    # ---- 1. 计算节点属性 ----
    # 主效应强度（平均 |SHAP|）
    mean_abs_shap = np.abs(shap_vals_array).mean(axis=0)  # (n_features,)
    # 平均 SHAP 值（判断正/负方向）
    mean_shap = shap_vals_array.mean(axis=0)  # (n_features,)

    # ---- 2. 计算边属性（交互强度） ----
    interaction_strengths = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            strength = np.abs(shap_interaction[:, i, j]).mean()
            interaction_strengths.append((i, j, strength))

    # 按交互强度排序，取 Top N 条边（避免图过于密集）
    interaction_strengths.sort(key=lambda x: x[2], reverse=True)
    # 动态确定保留边数：最多保留 n_features * 1.5 条边，至少 10 条
    max_edges = max(10, int(n_features * 1.5))
    top_edges = interaction_strengths[:max_edges]

    # 过滤掉强度极低的边（低于最大强度的 5%）
    if len(top_edges) > 0:
        max_strength = top_edges[0][2]
        threshold = max_strength * 0.05
        top_edges = [(i, j, s) for i, j, s in top_edges if s > threshold]

    # ---- 3. 构建网络图 ----
    G = nx.Graph()

    # 添加节点
    for i, feat in enumerate(feature_cols):
        G.add_node(feat,
                   importance=mean_abs_shap[i],
                   direction=mean_shap[i])

    # 添加边
    for i, j, strength in top_edges:
        G.add_edge(feature_cols[i], feature_cols[j], weight=strength)

    # ---- 4. 布局 ----
    # 使用 spring layout，重要性高的节点更居中
    # k 参数控制节点间距，iterations 控制迭代次数
    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=RANDOM_STATE,
                           weight='weight')

    # ---- 5. 绘图 ----
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor('#FAFAFA')

    # 节点大小：映射到 [400, 3000] 范围
    node_sizes_raw = np.array([G.nodes[n]['importance'] for n in G.nodes()])
    if node_sizes_raw.max() > node_sizes_raw.min():
        node_sizes = 400 + (node_sizes_raw - node_sizes_raw.min()) / \
                     (node_sizes_raw.max() - node_sizes_raw.min()) * 2600
    else:
        node_sizes = np.full_like(node_sizes_raw, 1500)

    # 节点颜色：正向蓝色，负向红色，颜色深浅 ∝ |平均SHAP|
    node_directions = np.array([G.nodes[n]['direction'] for n in G.nodes()])
    node_colors = []
    for d, imp in zip(node_directions, node_sizes_raw):
        # 归一化强度用于颜色深浅
        if node_sizes_raw.max() > 0:
            intensity = 0.3 + 0.7 * (imp / node_sizes_raw.max())
        else:
            intensity = 0.5
        if d >= 0:
            # 蓝色系（正向促进）
            node_colors.append((0.2 * intensity, 0.4 * intensity, 0.9 * intensity, 0.85))
        else:
            # 红色系（负向抑制）
            node_colors.append((0.9 * intensity, 0.25 * intensity, 0.2 * intensity, 0.85))

    # 边宽度：映射到 [0.5, 8] 范围
    edge_weights = np.array([G.edges[e]['weight'] for e in G.edges()])
    if len(edge_weights) > 0 and edge_weights.max() > edge_weights.min():
        edge_widths = 0.5 + (edge_weights - edge_weights.min()) / \
                      (edge_weights.max() - edge_weights.min()) * 7.5
    elif len(edge_weights) > 0:
        edge_widths = np.full_like(edge_weights, 2.0)
    else:
        edge_widths = []

    # 边颜色：强度越高越深
    edge_colors = []
    for w in edge_weights:
        if edge_weights.max() > 0:
            alpha = 0.15 + 0.7 * (w / edge_weights.max())
        else:
            alpha = 0.3
        edge_colors.append((0.4, 0.4, 0.4, alpha))

    # 绘制边
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=list(edge_widths),
                           edge_color=edge_colors,
                           style='solid')

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors='white',
                           linewidths=1.5)

    # 绘制标签（白色文字，带阴影效果）
    # 先绘制黑色阴影
    label_offset = {k: (v[0], v[1]) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_offset, ax=ax,
                            font_size=9, font_weight='bold',
                            font_color='white')

    # ---- 6. 在边上标注交互强度数值（仅 Top 5） ----
    top_label_edges = top_edges[:5]
    for i, j, strength in top_label_edges:
        feat_a = feature_cols[i]
        feat_b = feature_cols[j]
        if G.has_edge(feat_a, feat_b):
            x = (pos[feat_a][0] + pos[feat_b][0]) / 2
            y = (pos[feat_a][1] + pos[feat_b][1]) / 2
            ax.annotate(f'{strength:.4f}',
                        xy=(x, y), fontsize=7, fontweight='bold',
                        color='#333333', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFCC',
                                  alpha=0.85, edgecolor='#CCCC00', linewidth=0.5))

    # ---- 7. 图例 ----
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.2, 0.4, 0.9, 0.85),
               markersize=14, label='正向促进 (SHAP > 0)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.9, 0.25, 0.2, 0.85),
               markersize=14, label='负向抑制 (SHAP < 0)'),
        Line2D([0], [0], color='gray', linewidth=1, label='弱交互'),
        Line2D([0], [0], color='gray', linewidth=5, label='强交互'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
              framealpha=0.9, edgecolor='#CCCCCC', fancybox=True)

    # ---- 8. 标题和样式 ----
    ax.set_title(f'{target_name} — SHAP 特征影响-交互网络图',
                 fontsize=15, fontweight='bold', pad=20)
    ax.axis('off')

    # 添加说明文字
    fig.text(0.5, 0.02,
             '节点大小 ∝ 特征主效应强度（平均|SHAP|）  |  '
             '连线粗细 ∝ 特征对交互强度  |  '
             '节点颜色：蓝=正向促进，红=负向抑制',
             ha='center', fontsize=9, color='#666666',
             style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_path = os.path.join(plot_dir, f'图12_SHAP特征交互网络图_{target_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"  已保存: {save_path}")

    # 输出网络统计信息
    log.info(f"  网络统计: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    # 输出度中心性 Top 5
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
        plot_1_model_performance(y_test, test_results, target_name,
                                 shap_model_name=shap_model_name, plot_dir=target_plot_dir)

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
