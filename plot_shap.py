#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost-SHAP 图例绘制脚本
=========================
基于 experiment.py 训练好的 XGBoost 模型，生成论文所需的全部图例。

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

使用方式：
  python plot_shap.py                    # 绘制所有图例
  python plot_shap.py --plots 1 2 3      # 只绘制指定编号的图例
python plot_shap.py --target 步行比例  # 只绘制指定因变量的图例
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
def plot_1_model_performance(y_test, test_results, target_name, plot_dir=PLOT_DIR):
    """
    三联组合图：
    (a) 多模型 R² 对比柱状图（10折CV）
    (b) XGBoost 预测值 vs 实际值散点图
    (c) XGBoost 残差直方图
    """
    log.info(f"绘制图1: 模型性能验证图 — {target_name}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- (a) 多模型 R² 对比柱状图 ---
    ax = axes[0]
    model_names = list(test_results.keys())
    r2_values = [test_results[m]['metrics']['R²'] for m in model_names]
    colors = [MODEL_COLORS.get(m, '#95a5a6') for m in model_names]

    bars = ax.bar(model_names, r2_values, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
    # 高亮 XGBoost
    for i, m in enumerate(model_names):
        if m == 'XGBOOST':
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

    # --- (b) XGBoost 预测值 vs 实际值散点图 ---
    ax = axes[1]
    y_pred_xgb = test_results['XGBOOST']['y_pred']
    ax.scatter(y_test, y_pred_xgb, alpha=0.35, s=12, color='#e74c3c', edgecolors='none')
    # 45° 参考线
    lims = [min(y_test.min(), y_pred_xgb.min()), max(y_test.max(), y_pred_xgb.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.7, label='理想预测线 (y=x)')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('实际值', fontsize=12)
    ax.set_ylabel('预测值', fontsize=12)
    ax.set_title('(b) XGBoost 预测值 vs 实际值', fontsize=13, fontweight='bold')
    r2_xgb = test_results['XGBOOST']['metrics']['R²']
    ax.text(0.05, 0.92, f'R²={r2_xgb:.4f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', color='#e74c3c',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # --- (c) XGBoost 残差直方图 ---
    ax = axes[2]
    residuals = y_test - y_pred_xgb
    ax.hist(residuals, bins=40, color='#e74c3c', alpha=0.7, edgecolor='white', linewidth=0.5,
            density=True)
    # 叠加正态分布曲线
    from scipy.stats import norm
    mu, std = norm.fit(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x_range, norm.pdf(x_range, mu, std), 'k-', linewidth=1.5, label=f'正态拟合 (μ={mu:.3f}, σ={std:.3f})')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('残差 (实际值 - 预测值)', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('(c) XGBoost 残差分布', fontsize=13, fontweight='bold')
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
#  图4: SHAP 单特征依赖图（Dependence Plot）
# =====================================================================
def plot_4_shap_dependence(shap_values, X_data, feature_cols, target_name, scaler=None, plot_dir=PLOT_DIR):
    """
    为 SHAP 排名前 TOP_N_DEPENDENCE 个特征绘制依赖图。
    SHAP 会自动选择交互最强的变量作为颜色编码。
    如果提供 scaler，则将横坐标还原为原始值（更直观）。
    """
    log.info(f"绘制图4: SHAP单特征依赖图 — {target_name}")

    # 按平均 |SHAP| 排序，取 Top N
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_N_DEPENDENCE]
    top_features = [feature_cols[i] for i in top_indices]

    # 如果提供了 scaler，将 shap_values 中的 .data 替换为原始值
    # 这样 shap.plots.scatter 的横坐标就会显示原始值
    if scaler is not None:
        import copy
        shap_values_plot = copy.deepcopy(shap_values)
        shap_values_plot.data = scaler.inverse_transform(shap_values_plot.data)
    else:
        shap_values_plot = shap_values

    n_cols = 2
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, feat in enumerate(top_features):
        ax = axes[idx]
        shap.plots.scatter(shap_values_plot[:, feat], color=shap_values_plot, show=False, ax=ax)
        ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

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

    # 获取 XGBoost 模型
    xgb_model = trained_models.get('XGBOOST')
    if xgb_model is None:
        log.error("  未找到 XGBoost 模型，无法进行 SHAP 分析")
        return

    should_plot = lambda n: plot_ids is None or n in plot_ids

    # ===== 图1: 模型性能验证图 =====
    if should_plot(1):
        plot_1_model_performance(y_test, test_results, target_name, plot_dir=target_plot_dir)

    # ===== 计算 SHAP 值（图2/3/4/9 共用） =====
    shap_values = None
    if any(should_plot(n) for n in [2, 3, 4, 9]):
        shap_values, explainer = compute_shap_values(
            xgb_model, X_train_scaled, feature_cols
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
        plot_5_pdp_2d(xgb_model, X_train_scaled, feature_cols, target_name, scaler=scaler, plot_dir=target_plot_dir)

    # ===== 计算 SHAP 交互值（图6/7/8 共用） =====
    shap_interaction = None
    if any(should_plot(n) for n in [6, 7, 8]):
        shap_interaction = compute_shap_interaction_values(
            xgb_model, X_train_scaled, feature_cols
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

    log.info(f"\n  {target_name} 所有图例绘制完成！")


def parse_args():
    parser = argparse.ArgumentParser(
        description='XGBoost-SHAP 图例绘制脚本',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--plots', type=int, nargs='+', default=None,
        help='要绘制的图例编号（1-9），不指定则绘制全部\n例如: --plots 1 2 3'
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
    log.info("  XGBoost-SHAP 图例绘制脚本")
    log.info("=" * 70)
    log.info(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  输出目录: {PLOT_DIR}")

    if args.plots:
        log.info(f"  指定绘制图例: {args.plots}")
    else:
        log.info("  绘制全部图例 (1-9)")

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
