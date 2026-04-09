#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步行活力与步行范围的多模型对比实验（改进版）
=============================================
模型A：Y = 短距步行比例 ~ f(X₁, X₂, ..., X₂₃)  → 解释"步行活力的形成"
模型B：Y = 步行距离     ~ f(X₁, X₂, ..., X₂₃)  → 解释"步行范围的限制"

实验流程：
  1. 数据 80/20 划分为训练集和测试集（测试集完全隔离）
  2. 在训练集上使用 RandomizedSearchCV（内层5折交叉验证）进行超参数调优
  3. 使用最优参数在完整训练集上重新训练
  4. 在 hold-out 测试集上评估最终性能（MAE、MSE、RMSE、R²）
  5. 额外进行10折交叉验证报告泛化性能

数据规模：~3028 样本，23 个自变量（中小规模数据集）
对比模型：OLS、GBDT、RF、XGBOOST、LightGBM
"""

import os
import sys
import json
import time
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============ 中文字体设置 ============
rcParams['font.sans-serif'] = ['STHeiti', 'Heiti TC', 'Songti SC', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import (
    train_test_split, KFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import statsmodels.api as sm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'base_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TARGET_A = '短距步行比例'
TARGET_B = '步行距离'
EXCLUDE_COLS = ['街区编号', TARGET_A, TARGET_B]

TEST_SIZE = 0.2          # 测试集比例
INNER_CV_FOLDS = 5       # 调参内层交叉验证折数
OUTER_CV_FOLDS = 10      # 外层交叉验证折数（泛化性能评估）
N_ITER_SEARCH = 30       # RandomizedSearchCV 最大搜索次数
RANDOM_STATE = 42
N_JOBS = -1              # 并行核数


# ============ 日志系统 ============
def setup_logger():
    """
    配置日志系统：所有输出同时写入控制台和带时间戳后缀的日志文件
    日志文件路径: logs/experiment_YYYYMMDD_HHMMSS.log
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'experiment_{timestamp}.log'
    log_path = os.path.join(LOG_DIR, log_filename)

    # 创建 logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 防止重复添加

    # 格式：时间 + 级别 + 消息
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件 handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_path}")
    return logger, log_path


# 全局 logger
logger = None


def log(msg=''):
    """统一日志输出函数，同时写入控制台和日志文件"""
    if logger:
        logger.info(msg)
    else:
        print(msg)


# ============ 数据加载与预处理 ============
def load_data():
    """加载并预处理数据"""
    df = pd.read_csv(DATA_PATH, encoding='gbk')
    log(f"数据维度: {df.shape}")
    log(f"列名: {df.columns.tolist()}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        log(f"\n缺失值统计:\n{missing[missing > 0]}")
    else:
        log("无缺失值")

    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].values
    y_a = df[TARGET_A].values
    y_b = df[TARGET_B].values

    n_samples = X.shape[0]
    n_features = len(feature_cols)
    log(f"\n自变量数量: {n_features}")
    log(f"自变量列表: {feature_cols}")
    log(f"样本数量: {n_samples}")
    log(f"样本/特征比: {n_samples / n_features:.1f}:1")
    log(f"训练集预计: {int(n_samples * (1 - TEST_SIZE))} 样本")

    return X, y_a, y_b, feature_cols


# ============ 超参数搜索空间定义 ============
def get_param_grids():
    """
    为每个模型定义超参数搜索空间

    设计原则（针对 ~3028 样本、23 特征的中小数据集）：
    ─────────────────────────────────────────────────────
    1. 树深度控制在较浅范围（3~6），防止在小数据上过拟合
    2. n_estimators 适中（100~300），小数据集不需要太多树
    3. learning_rate 偏保守（0.01~0.1），配合适中的树数量
    4. min_samples_leaf / min_samples_split 适当增大，
       确保叶节点有足够样本（训练集约2422样本）
    5. subsample < 1.0 引入随机性，增强泛化
    6. 正则化参数适当增大，抑制过拟合
    """
    param_grids = {
        # GBDT: 不支持 n_jobs 并行，搜索空间需精简
        'GBDT': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],           # 小数据集浅树即可
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],     # 引入随机性防过拟合
            'min_samples_leaf': [5, 10, 20],   # 叶节点至少5~20个样本
            'min_samples_split': [10, 20],     # 分裂至少需要10~20个样本
        },
        'RF': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 8, 12, None],     # None允许完全生长，但配合min_samples_leaf控制
            'min_samples_split': [5, 10, 20],  # 小数据集适当增大
            'min_samples_leaf': [3, 5, 10],    # 叶节点至少3~10个样本
            'max_features': ['sqrt', 0.5, 0.7],  # 特征采样比例
        },
        'XGBOOST': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],         # 控制树深度
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'reg_alpha': [0, 0.1, 1.0],        # L1正则化
            'reg_lambda': [1.0, 3.0, 5.0],     # L2正则化，适当增大
            'min_child_weight': [3, 5, 10],     # 类似min_samples_leaf
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],             # 限制深度
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [0, 1.0, 5.0],
            'num_leaves': [15, 31, 50],         # 小数据集不宜太多叶子
            'min_child_samples': [10, 20, 30],  # 叶节点最少样本数
        },
    }
    return param_grids


def get_base_models():
    """返回各模型的基础实例（用于 RandomizedSearchCV）"""
    models = {
        'GBDT': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'RF': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'XGBOOST': XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=N_JOBS),
        'LightGBM': LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=N_JOBS),
    }
    return models


# ============ 计算评估指标 ============
def compute_metrics(y_true, y_pred):
    """计算所有评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2}


# ============ OLS 回归 ============
def ols_fit_predict(X_train, y_train, X_test):
    """使用 statsmodels 进行 OLS 回归"""
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train_const).fit()
    y_pred = model.predict(X_test_const)
    return y_pred, model


# ============ 超参数调优（核心） ============
def tune_hyperparameters(model_name, base_model, param_grid, X_train, y_train):
    """
    使用带进度条的超参数搜索在训练集上进行调优
    - 手动实现随机搜索 + tqdm 进度条
    - 内层5折交叉验证
    - 评分指标: MSE（越小越好）
    """
    from sklearn.model_selection import ParameterSampler, cross_val_score

    total_combos = int(np.prod([len(v) for v in param_grid.values()]))
    n_iter = min(N_ITER_SEARCH, total_combos)
    log(f"    超参数搜索空间: {total_combos} 种组合, 随机采样 {n_iter} 种")

    inner_cv = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 生成随机参数组合
    param_list = list(ParameterSampler(
        param_grid, n_iter=n_iter, random_state=RANDOM_STATE
    ))

    best_score = -np.inf
    best_params = None
    best_estimator = None

    # 带进度条的搜索
    pbar = tqdm(
        param_list,
        desc=f"    {model_name} 调参",
        unit="组合",
        ncols=90,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for params in pbar:
        # 克隆模型并设置参数
        model_clone = base_model.__class__(**{**base_model.get_params(), **params})

        # 内层交叉验证评分
        scores = cross_val_score(
            model_clone, X_train, y_train,
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=N_JOBS
        )
        mean_score = scores.mean()

        # 更新最优（neg_mse 越大越好，即 MSE 越小越好）
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        # 进度条显示当前最优 MSE
        pbar.set_postfix({'最优MSE': f'{-best_score:.4f}'})

    pbar.close()

    # 用最优参数在全部训练集上重新训练
    best_estimator = base_model.__class__(**{**base_model.get_params(), **best_params})
    best_estimator.fit(X_train, y_train)

    best_cv_mse = -best_score
    log(f"    最优内层CV MSE: {best_cv_mse:.4f}")
    log(f"    最优参数: {best_params}")

    return best_estimator, best_params, best_cv_mse


# ============ 外层10折交叉验证（泛化性能） ============
def outer_cross_validate(model_name, best_params, X, y):
    """
    使用最优参数进行外层10折交叉验证，评估泛化性能（带进度条）
    """
    outer_cv = KFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scaler = StandardScaler()

    all_y_true = []
    all_y_pred = []

    folds = list(outer_cv.split(X))
    pbar = tqdm(
        folds,
        desc=f"    {model_name} 10折CV",
        unit="折",
        ncols=90,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    for train_idx, test_idx in pbar:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        if model_name == 'OLS':
            y_pred, _ = ols_fit_predict(X_tr_s, y_tr, X_te_s)
        else:
            model = create_model_with_params(model_name, best_params)
            model.fit(X_tr_s, y_tr)
            y_pred = model.predict(X_te_s)

        all_y_true.extend(y_te)
        all_y_pred.extend(y_pred)

    pbar.close()
    return np.array(all_y_true), np.array(all_y_pred)


# ============ 残差图绘制 ============
def plot_residuals(results_dict, target_name, output_path):
    """绘制所有模型的残差图（子图排列）"""
    model_names = list(results_dict.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, (name, (y_true, y_pred)) in enumerate(results_dict.items()):
        ax = axes[idx]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=colors[idx % len(colors)],
                   edgecolors='none')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('预测值', fontsize=11)
        if idx == 0:
            ax.set_ylabel('残差', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{target_name} — 各模型残差图（测试集）', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  残差图已保存: {output_path}")


# ============ 结果表格输出 ============
def format_results_table(all_metrics, target_name):
    """将所有模型的指标整理为论文格式的表格"""
    metric_names = ['MAE', 'MSE', 'RMSE', 'R²']
    model_names = list(all_metrics.keys())

    table_data = []
    for metric in metric_names:
        row = [metric]
        values = [all_metrics[m][metric] for m in model_names]

        best_idx = np.argmax(values) if metric == 'R²' else np.argmin(values)

        for i, val in enumerate(values):
            formatted = f"{val:.4f}"
            if i == best_idx:
                formatted = f"**{formatted}**"
            row.append(formatted)
        table_data.append(row)

    headers = ['指标'] + model_names
    return headers, table_data


def print_and_save_table(headers, table_data, target_name, output_path):
    """打印并保存表格"""
    log(f"\n{'='*80}")
    log(f"  {target_name}")
    log(f"{'='*80}")
    table_str = tabulate(table_data, headers=headers, tablefmt='grid', stralign='center')
    log(table_str)

    clean_data = []
    for row in table_data:
        clean_row = [cell.replace('**', '') if isinstance(cell, str) else cell for cell in row]
        clean_data.append(clean_row)

    df_result = pd.DataFrame(clean_data, columns=headers)
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    log(f"  结果表格已保存: {output_path}")
    return df_result


# ============ 特征重要性分析 ============
def analyze_feature_importance(best_model, model_name, feature_cols, target_name, X_train, y_train):
    """使用调优后的最佳模型分析特征重要性"""
    if not hasattr(best_model, 'feature_importances_'):
        log(f"  {model_name} 不支持特征重要性分析，跳过")
        return None

    importance = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        '特征': feature_cols,
        '重要性': importance
    }).sort_values('重要性', ascending=False).reset_index(drop=True)

    log(f"\n{target_name} — {model_name}（调优后）特征重要性排名 (Top 10):")
    table_str = tabulate(feat_imp.head(10).values.tolist(),
                         headers=['特征', '重要性'], tablefmt='grid', floatfmt='.4f')
    log(table_str)

    # 绘制特征重要性图
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp_sorted = feat_imp.sort_values('重要性', ascending=True)
    ax.barh(feat_imp_sorted['特征'], feat_imp_sorted['重要性'], color='#2ca02c', alpha=0.8)
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'{target_name} — {model_name}（调优后）特征重要性', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'feature_importance_{target_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log(f"  特征重要性图已保存: {output_path}")

    csv_path = os.path.join(OUTPUT_DIR, f'feature_importance_{target_name}.csv')
    feat_imp.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return feat_imp


# ============ OLS 回归摘要 ============
def ols_summary(X_train, y_train, feature_cols, target_name):
    """输出 OLS 回归的详细统计摘要（基于训练集）"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_const = sm.add_constant(X_scaled)

    model = sm.OLS(y_train, X_const).fit()

    summary_path = os.path.join(OUTPUT_DIR, f'ols_summary_{target_name}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"OLS 回归摘要 — {target_name}（基于训练集）\n")
        f.write("=" * 80 + "\n")
        f.write(str(model.summary()) + "\n\n")
        f.write("特征名称对照:\n")
        for i, col in enumerate(feature_cols):
            f.write(f"  x{i+1} -> {col}\n")

    log(f"\n  OLS 回归摘要已保存: {summary_path}")
    log(f"  训练集 R² = {model.rsquared:.4f}, Adj. R² = {model.rsquared_adj:.4f}")
    return model


# ============ 最优参数的保存与加载 ============
BEST_PARAMS_DIR = os.path.join(OUTPUT_DIR, 'best_params')
os.makedirs(BEST_PARAMS_DIR, exist_ok=True)


def save_best_params(all_best_params, target_name):
    """
    保存所有模型的最优超参数
    - CSV 格式：方便人工查看
    - JSON 格式：方便程序加载（推理模式使用）
    """
    # 1. 保存为 CSV（人工可读）
    rows = []
    for model_name, params in all_best_params.items():
        if params is None:
            rows.append({'模型': model_name, '参数名': 'N/A', '最优值': '无超参数'})
        else:
            for k, v in params.items():
                rows.append({'模型': model_name, '参数名': k, '最优值': v})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, f'best_params_{target_name}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log(f"  最优参数(CSV)已保存: {csv_path}")

    # 2. 保存为 JSON（程序可加载，推理模式使用）
    # 将 numpy 类型转为 Python 原生类型，确保 JSON 可序列化
    serializable_params = {}
    for model_name, params in all_best_params.items():
        if params is None:
            serializable_params[model_name] = None
        else:
            serializable_params[model_name] = {
                k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else
                    v)
                for k, v in params.items()
            }

    json_path = os.path.join(BEST_PARAMS_DIR, f'best_params_{target_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_params, f, ensure_ascii=False, indent=2)
    log(f"  最优参数(JSON)已保存: {json_path}")


def load_best_params(target_name):
    """
    从 JSON 文件加载已保存的最优超参数（推理模式使用）
    返回: {model_name: {param_name: value}} 或 None（文件不存在时）
    """
    json_path = os.path.join(BEST_PARAMS_DIR, f'best_params_{target_name}.json')
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    log(f"  已加载最优参数: {json_path}")
    return params


# ============ 根据最优参数创建模型实例 ============
def create_model_with_params(model_name, params):
    """根据模型名称和参数字典创建模型实例"""
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


# ============ 命令行参数解析 ============
def parse_args():
    parser = argparse.ArgumentParser(
        description='步行活力与步行范围的多模型对比实验',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--mode', type=str, default='tune',
        choices=['tune', 'infer'],
        help=(
            '运行模式:\n'
            '  tune  — 完整调优模式（默认）：超参数搜索 + 训练 + 评估\n'
            '  infer — 推理模式：加载已保存的最优参数，跳过调参直接训练评估'
        )
    )
    return parser.parse_args()


# ============ 主函数 ============
def main():
    global logger

    args = parse_args()
    infer_mode = (args.mode == 'infer')

    logger, log_path = setup_logger()

    total_start = time.time()

    log("=" * 80)
    if infer_mode:
        log("  步行活力与步行范围的多模型对比实验【推理模式】")
        log("  流程: 加载最优参数 → 训练 → 测试集评估 → 外层10折CV")
    else:
        log("  步行活力与步行范围的多模型对比实验【调优模式】")
        log("  流程: 数据划分 → 超参数调优(内层5折CV) → 测试集评估 → 外层10折CV")
    log("=" * 80)
    log(f"  运行模式: {'推理模式 (--mode infer)' if infer_mode else '调优模式 (--mode tune)'}")
    log(f"  实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  日志文件: {log_path}")

    # 1. 加载数据
    X, y_a, y_b, feature_cols = load_data()

    # 2. 调优模式：获取模型和参数搜索空间
    if not infer_mode:
        base_models = get_base_models()
        param_grids = get_param_grids()

        log("\n超参数搜索空间摘要:")
        for name, grid in param_grids.items():
            total = int(np.prod([len(v) for v in grid.values()]))
            actual = min(N_ITER_SEARCH, total)
            log(f"  {name}: 总共 {total} 种组合, 将随机采样 {actual} 种")

    # 3. 对两个因变量分别进行实验
    targets = {
        TARGET_A: (y_a, '模型A：短距步行比例（步行活力的形成）'),
        TARGET_B: (y_b, '模型B：步行距离（步行范围的限制）')
    }

    for target_col, (y, description) in targets.items():
        log(f"\n\n{'#'*80}")
        log(f"  {description}")
        log(f"{'#'*80}")

        # ========== 推理模式：加载已保存的最优参数 ==========
        loaded_params = None
        if infer_mode:
            loaded_params = load_best_params(target_col)
            if loaded_params is None:
                log(f"  ❌ 错误: 未找到 {target_col} 的最优参数文件！")
                log(f"     请先运行调优模式: python experiment.py --mode tune")
                log(f"     参数文件应位于: {BEST_PARAMS_DIR}/best_params_{target_col}.json")
                continue
            log("  已加载的最优参数:")
            for m_name, m_params in loaded_params.items():
                if m_params is None:
                    log(f"    {m_name}: 无超参数")
                else:
                    log(f"    {m_name}: {m_params}")

        # ========== 步骤1: 划分训练集和测试集 ==========
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        log(f"\n  数据划分: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
        log(f"  因变量统计 — 训练集: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        log(f"  因变量统计 — 测试集: mean={y_test.mean():.4f}, std={y_test.std():.4f}")

        # 标准化（基于训练集拟合）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 存储结果
        test_metrics = {}       # 测试集指标
        cv_metrics = {}         # 10折CV指标
        test_predictions = {}   # 测试集预测值（用于残差图）
        cv_predictions = {}     # CV预测值（用于残差图）
        all_best_params = {}    # 最优参数
        best_trained_models = {}  # 训练好的最佳模型

        # ========== 步骤2: OLS（无超参数调优） ==========
        log(f"\n  {'─'*60}")
        log(f"  [1/5] OLS 回归（无超参数，直接训练）")
        log(f"  {'─'*60}")

        ols_start = time.time()
        y_pred_test_ols, ols_model = ols_fit_predict(X_train_scaled, y_train, X_test_scaled)
        test_metrics['OLS'] = compute_metrics(y_test, y_pred_test_ols)
        test_predictions['OLS'] = (y_test, y_pred_test_ols)
        all_best_params['OLS'] = None

        # OLS 外层10折CV
        y_true_cv, y_pred_cv = outer_cross_validate('OLS', {}, X, y)
        cv_metrics['OLS'] = compute_metrics(y_true_cv, y_pred_cv)
        cv_predictions['OLS'] = (y_true_cv, y_pred_cv)

        ols_elapsed = time.time() - ols_start
        m = test_metrics['OLS']
        log(f"    测试集: MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R²']:.4f}")
        m_cv = cv_metrics['OLS']
        log(f"    10折CV: MAE={m_cv['MAE']:.4f}  MSE={m_cv['MSE']:.4f}  RMSE={m_cv['RMSE']:.4f}  R²={m_cv['R²']:.4f}")
        log(f"    耗时: {ols_elapsed:.1f}秒")

        # ========== 步骤3: 树模型 ==========
        model_list = ['GBDT', 'RF', 'XGBOOST', 'LightGBM']
        for idx, model_name in enumerate(model_list):
            log(f"\n  {'─'*60}")

            if infer_mode:
                # ===== 推理模式：直接用已保存的最优参数训练 =====
                log(f"  [{idx+2}/5] {model_name} — 推理模式（加载最优参数直接训练）")
                log(f"  {'─'*60}")

                best_params = loaded_params.get(model_name)
                if best_params is None:
                    log(f"    ⚠️ 未找到 {model_name} 的最优参数，跳过")
                    continue

                log(f"    使用参数: {best_params}")
                start_time = time.time()

                # 用最优参数创建模型并训练
                best_model = create_model_with_params(model_name, best_params)
                best_model.fit(X_train_scaled, y_train)

                elapsed = time.time() - start_time
                log(f"    训练耗时: {elapsed:.1f}秒")

            else:
                # ===== 调优模式：超参数搜索 =====
                log(f"  [{idx+2}/5] {model_name} — 超参数调优中...")
                log(f"  {'─'*60}")

                start_time = time.time()

                best_model, best_params, best_cv_mse = tune_hyperparameters(
                    model_name, base_models[model_name],
                    param_grids[model_name],
                    X_train_scaled, y_train
                )

                elapsed = time.time() - start_time
                log(f"    调参耗时: {elapsed:.1f}秒")

            # 在测试集上评估
            y_pred_test = best_model.predict(X_test_scaled)
            test_metrics[model_name] = compute_metrics(y_test, y_pred_test)
            test_predictions[model_name] = (y_test, y_pred_test)
            all_best_params[model_name] = best_params
            best_trained_models[model_name] = best_model

            # 外层10折CV（使用最优参数）
            y_true_cv, y_pred_cv = outer_cross_validate(model_name, best_params, X, y)
            cv_metrics[model_name] = compute_metrics(y_true_cv, y_pred_cv)
            cv_predictions[model_name] = (y_true_cv, y_pred_cv)

            m = test_metrics[model_name]
            log(f"    测试集: MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R²']:.4f}")
            m_cv = cv_metrics[model_name]
            log(f"    10折CV: MAE={m_cv['MAE']:.4f}  MSE={m_cv['MSE']:.4f}  RMSE={m_cv['RMSE']:.4f}  R²={m_cv['R²']:.4f}")

        # ========== 步骤4: 输出结果 ==========
        # 4.1 测试集性能对比表格
        headers, table_data = format_results_table(test_metrics, target_col)
        csv_path = os.path.join(OUTPUT_DIR, f'model_comparison_test_{target_col}.csv')
        print_and_save_table(headers, table_data, f'{description} — 测试集性能对比', csv_path)

        # 4.2 10折CV性能对比表格
        headers_cv, table_data_cv = format_results_table(cv_metrics, target_col)
        csv_path_cv = os.path.join(OUTPUT_DIR, f'model_comparison_cv_{target_col}.csv')
        print_and_save_table(headers_cv, table_data_cv, f'{description} — 10折交叉验证性能对比', csv_path_cv)

        # 4.3 残差图（测试集）
        residual_path = os.path.join(OUTPUT_DIR, f'residual_plot_test_{target_col}.png')
        plot_residuals(test_predictions, f'{description}（测试集）', residual_path)

        # 4.4 残差图（10折CV）
        residual_path_cv = os.path.join(OUTPUT_DIR, f'residual_plot_cv_{target_col}.png')
        plot_residuals(cv_predictions, f'{description}（10折CV）', residual_path_cv)

        # 4.5 保存最优参数（调优模式和推理模式都保存，确保一致性）
        save_best_params(all_best_params, target_col)

        # 4.6 特征重要性分析（使用测试集表现最好的树模型）
        if best_trained_models:
            best_tree_name = max(
                [n for n in model_list if n in test_metrics],
                key=lambda n: test_metrics[n]['R²']
            )
            log(f"\n  测试集最佳树模型: {best_tree_name} (R²={test_metrics[best_tree_name]['R²']:.4f})")
            analyze_feature_importance(
                best_trained_models[best_tree_name], best_tree_name,
                feature_cols, target_col, X_train_scaled, y_train
            )

        # 4.7 OLS 回归摘要
        ols_summary(X_train, y_train, feature_cols, target_col)

    total_elapsed = time.time() - total_start
    log(f"\n\n{'='*80}")
    log(f"  所有实验完成！")
    log(f"  运行模式: {'推理模式' if infer_mode else '调优模式'}")
    log(f"  总耗时: {total_elapsed:.1f}秒 ({total_elapsed/60:.1f}分钟)")
    log(f"  结果保存在: {OUTPUT_DIR}")
    log(f"  日志文件: {log_path}")
    log(f"  实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not infer_mode:
        log(f"  💡 下次可用推理模式快速运行: python experiment.py --mode infer")
    log(f"{'='*80}")


if __name__ == '__main__':
    main()
