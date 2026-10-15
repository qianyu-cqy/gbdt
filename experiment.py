#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步行联系强度（%）与步行联系收缩度的多模型对比实验（改进版 v2）
=============================================
支持动态选择因变量：
  --targets 步行联系强度（%）              → 仅运行步行联系强度（%）模型
  --targets 步行联系强度（%） 步行联系收缩度    → 同时运行两个模型
  --targets 步行联系收缩度              → 仅运行步行联系收缩度模型

实验流程：
  1. 数据 80/20 划分为训练集和测试集（测试集完全隔离）
  2. 在训练集上使用 随机参数采样（内层5折交叉验证）进行超参数调优
  3. 使用最优参数在完整训练集上重新训练
  4. 在 hold-out 测试集上评估最终性能（MAE、MSE、RMSE、R²）

对比模型：OLS、GBDT、RF、XGBOOST、LightGBM
"""

import os
import sys
import json
import time
import logging
import argparse
import hashlib
from pathlib import Path
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
    train_test_split, KFold
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
DATA_PATH = os.path.join(BASE_DIR, '建成环境数据.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output-built-environment')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# ============ 不再硬编码因变量，改为动态获取 ============
ALWAYS_EXCLUDE = ['FID']  # 始终排除的列（非特征列）

TEST_SIZE = 0.2          # 测试集比例
INNER_CV_FOLDS = 5       # 调参内层交叉验证折数
N_ITER_SEARCH = 60       # RandomizedSearchCV 最大搜索次数
RANDOM_STATE = 42
N_JOBS = min(5, os.cpu_count() or 1)  # 仅并行CV，模型内部单线程

# ============ 因变量描述映射（可扩展） ============
TARGET_DESCRIPTIONS = {
    '步行联系强度（%）': '模型A：步行联系强度（%）',
    '步行联系收缩度': '模型B：步行联系收缩度',
}

# 定义所有潜在的因变量，用于在特征中互相排除
ALL_TARGET_COLS = ['步行联系强度（%）', '步行联系收缩度']

# ============ 日志系统 ============
def setup_logger():
    """配置日志系统"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'experiment_{timestamp}.log'
    log_path = os.path.join(LOG_DIR, log_filename)

    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_path}")
    return logger, log_path

# 全局 logger
logger = None

def log(msg=''):
    """统一日志输出函数"""
    if logger:
        logger.info(msg)
    else:
        print(msg)

# ============ 数据加载与预处理 ============
def load_data(target_cols):
    """
    加载并预处理数据
    """
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    log(f"数据维度: {df.shape}")

    # 检查当前请求的因变量是否存在
    missing_targets = [t for t in target_cols if t not in df.columns]
    if missing_targets:
        log(f"\n❌ 错误: 以下因变量在数据中不存在: {missing_targets}")
        sys.exit(1)

    df = df.dropna().reset_index(drop=True)

    # 【关键修改】：排除列 = 始终排除的列 + 所有已知的潜在因变量
    # 这样当跑其中一个目标时，另一个目标也会被排除，避免目标泄漏。
    exclude_cols = ALWAYS_EXCLUDE + ALL_TARGET_COLS + list(target_cols)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not feature_cols:
        raise ValueError('没有可用的自变量')
    numeric_cols = feature_cols + list(target_cols)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df[numeric_cols].to_numpy()).all():
        raise ValueError('特征或目标中含有无穷值')
    n_samples = df.shape[0]
    n_features = len(feature_cols)
    log(f"\n因变量: {target_cols}")
    log(f"自变量数量: {n_features}")
    log(f"自变量列表: {feature_cols}")
    log(f"样本数量: {n_samples}")
    log(f"样本/特征比: {n_samples / n_features:.1f}:1")
    log(f"训练集预计: {int(n_samples * (1 - TEST_SIZE))} 样本")

    return df, feature_cols

# ============ 自动检测数据中可用的因变量 ============
def detect_available_targets():
    """
    扫描数据文件，检测哪些已知因变量存在
    """
    df_head = pd.read_csv(DATA_PATH, encoding='utf-8-sig', nrows=0)
    columns = df_head.columns.tolist()

    # 修改此处匹配新的变量名
    known_targets = ALL_TARGET_COLS
    available = [t for t in known_targets if t in columns]
    return available, columns

# ============ 交互式选择因变量 ============
def interactive_select_targets():
    """
    当用户未通过命令行指定因变量时，进入交互式选择
    """
    available, all_columns = detect_available_targets()

    print("\n" + "=" * 60)
    print("  因变量选择")
    print("=" * 60)
    print(f"\n数据文件: {DATA_PATH}")
    print(f"数据列名: {all_columns}")
    print(f"\n已识别的可用因变量:")

    if not available:
        print("  ⚠️ 未检测到已知因变量（步行联系强度（%）/步行联系收缩度）")
        print("  请手动输入因变量名称（用逗号分隔）:")
        user_input = input("  > ").strip()
        if not user_input:
            print("  未输入任何因变量，退出。")
            sys.exit(0)
        targets = [t.strip() for t in user_input.split(',')]
        return targets

    for i, t in enumerate(available, 1):
        desc = TARGET_DESCRIPTIONS.get(t, t)
        print(f"  [{i}] {t} — {desc}")

    print(f"\n请选择（输入编号，多个用逗号分隔，直接回车=全选）:")
    user_input = input("  > ").strip()

    if not user_input:
        # 直接回车 = 全选
        targets = available
        print(f"  已选择全部: {targets}")
    else:
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            targets = [available[i - 1] for i in indices if 1 <= i <= len(available)]
            if not targets:
                print("  选择无效，退出。")
                sys.exit(0)
            print(f"  已选择: {targets}")
        except (ValueError, IndexError):
            print("  输入无效，退出。")
            sys.exit(0)

    return targets

# ============ 超参数搜索空间定义 ============
def get_param_grids():
    """为每个模型定义超参数搜索空间（与原版完全一致）"""
    param_grids = {
        'GBDT': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_leaf': [5, 10, 20],
            'min_samples_split': [10, 20, 30],
        },
        'RF': {
            'n_estimators': [200, 300, 500],
            'max_depth': [8, 12, 16, 20, None],
            'min_samples_split': [5, 10, 15, 20],
            'min_samples_leaf': [5, 10, 15, 20],
            'max_features': ['sqrt', 0.3, 0.5, 0.7],
        },
        'XGBOOST': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 3.0],
            'reg_lambda': [0.1, 0.5, 1.0, 3.0, 5.0, 8.0, 10.0],
            'min_child_weight': [3, 5, 10, 15],
        },
        'LightGBM': {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [4, 6, 8, -1],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
            'reg_alpha': [0, 0.01, 0.1, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 3.0],
            'num_leaves': [15, 20, 31, 40],
            'min_child_samples': [10, 20, 30],
        },
    }
    return param_grids

def get_base_models():
    """返回各模型的基础实例"""
    models = {
        'GBDT': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'RF': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        'XGBOOST': XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1),
        'LightGBM': LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
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

# ============ 超参数调优 ============
def tune_hyperparameters(model_name, base_model, param_grid, X_train, y_train, target_name):
    """使用带进度条的超参数搜索"""
    from sklearn.model_selection import ParameterSampler, cross_val_score

    total_combos = int(np.prod([len(v) for v in param_grid.values()]))
    n_iter = min(N_ITER_SEARCH, total_combos)
    log(f"    超参数搜索空间: {total_combos} 种组合, 随机采样 {n_iter} 种")

    inner_cv = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    param_list = list(ParameterSampler(
        param_grid, n_iter=n_iter, random_state=RANDOM_STATE
    ))

    best_score = -np.inf
    best_params = None
    search_rows = []
    search_dir = os.path.join(OUTPUT_DIR, 'cv_results')
    os.makedirs(search_dir, exist_ok=True)
    search_path = os.path.join(search_dir, f'{target_name}_{model_name}.csv')

    sys.stdout.flush()
    pbar = tqdm(
        param_list,
        desc=f"    {model_name} 调参",
        unit="组合",
        ncols=90,
        delay=0.5,
        disable=not sys.stdout.isatty(),
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for params in pbar:
        model_clone = base_model.__class__(**{**base_model.get_params(), **params})
        scores = cross_val_score(
            model_clone, X_train, y_train,
            cv=inner_cv,
            scoring='neg_mean_squared_error',
            n_jobs=N_JOBS, error_score='raise'
        )
        mean_score = scores.mean()
        if not np.isfinite(scores).all():
            raise ValueError(f'{model_name} 交叉验证产生非有限指标')
        search_rows.append({
            'candidate': len(search_rows) + 1,
            'params': json.dumps(params, ensure_ascii=False),
            'mean_cv_mse': float(-mean_score), 'std_cv_mse': float(scores.std()),
            **{f'fold_{i + 1}_mse': float(-score) for i, score in enumerate(scores)},
        })
        pd.DataFrame(search_rows).to_csv(search_path, index=False, encoding='utf-8-sig')
        if len(search_rows) % 10 == 0:
            log(f'    {model_name} 调参进度: {len(search_rows)}/{n_iter}')

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

        pbar.set_postfix({'最优MSE': f'{-best_score:.4f}'})

    pbar.close()

    best_estimator = base_model.__class__(**{**base_model.get_params(), **best_params})
    best_estimator.fit(X_train, y_train)

    best_cv_mse = -best_score
    log(f"    最优内层CV MSE: {best_cv_mse:.4f}")
    log(f"    最优参数: {best_params}")

    return best_estimator, best_params, best_cv_mse

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
    log(f"\n{'=' * 80}")
    log(f"  {target_name}")
    log(f"{'=' * 80}")
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
    """特征重要性分析"""
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

    csv_path = os.path.join(OUTPUT_DIR, f'feature_importance_{target_name}.csv')
    feat_imp.to_csv(csv_path, index=False, encoding='utf-8-sig')
    log(f"  特征重要性CSV已保存: {csv_path}")

    return feat_imp

# ============ OLS 回归摘要 ============
def ols_summary(X_train, y_train, feature_cols, target_name):
    """输出 OLS 回归的详细统计摘要"""
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
            f.write(f"  x{i + 1} -> {col}\n")

    log(f"\n  OLS 回归摘要已保存: {summary_path}")
    log(f"  训练集 R² = {model.rsquared:.4f}, Adj. R² = {model.rsquared_adj:.4f}")
    return model

# ============ 最优参数的保存与加载 ============
BEST_PARAMS_DIR = os.path.join(OUTPUT_DIR, 'best_params')

def save_best_params(all_best_params, target_name):
    """保存最优超参数（CSV + JSON）"""
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
    """从 JSON 文件加载已保存的最优超参数"""
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
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    elif model_name == 'XGBOOST':
        return XGBRegressor(random_state=RANDOM_STATE, verbosity=0, n_jobs=1, **params)
    elif model_name == 'LightGBM':
        return LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, n_jobs=1, **params)
    else:
        raise ValueError(f"未知模型: {model_name}")

# ============ 命令行参数解析 ============
def parse_args():
    parser = argparse.ArgumentParser(
        description='步行联系强度（%）与步行联系收缩度的多模型对比实验',
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
    parser.add_argument(
        '--targets', type=str, nargs='*', default=None,
        help=(
            '指定因变量（可多选，空格分隔）:\n'
            '  --targets 步行联系强度（%%）              → 仅运行步行联系强度（%%）\n'
            '  --targets 步行联系强度（%%） 步行联系收缩度    → 同时运行两个\n'
            '  不指定则进入交互式选择模式'
        )
    )
    parser.add_argument('--output-dir', type=str, default=None,
                        help='结果目录；调优默认新建时间戳目录，推理须指定已有目录')
    parser.add_argument('--jobs', type=int, default=N_JOBS,
                        help='交叉验证并行进程数（默认最多5个，模型内部单线程）')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs 必须大于等于1')
    if args.mode == 'infer' and not args.output_dir:
        parser.error('推理模式须通过 --output-dir 指定已有结果目录')
    return args

# ============ 主函数 ============
def main():
    global logger, OUTPUT_DIR, LOG_DIR, BEST_PARAMS_DIR, N_JOBS

    args = parse_args()
    infer_mode = (args.mode == 'infer')
    N_JOBS = args.jobs
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)
    else:
        OUTPUT_DIR = os.path.join(OUTPUT_DIR, datetime.now().strftime('run_%Y%m%d_%H%M%S_%f'))
    if not infer_mode and os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        raise FileExistsError(f'调优结果目录非空，拒绝覆盖: {OUTPUT_DIR}')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    BEST_PARAMS_DIR = os.path.join(OUTPUT_DIR, 'best_params')
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_PARAMS_DIR, exist_ok=True)

    logger, log_path = setup_logger()

    total_start = time.time()

    # ========== 确定因变量 ==========
    if args.targets:
        # 命令行指定
        target_cols = args.targets
        log(f"命令行指定因变量: {target_cols}")
    else:
        # 交互式选择
        target_cols = interactive_select_targets()

    log("=" * 80)
    if infer_mode:
        log("  步行联系强度（%）与步行联系收缩度的多模型对比实验【推理模式】")
        log("  流程: 加载最优参数 → 训练 → 测试集评估")
    else:
        log("  步行联系强度（%）与步行联系收缩度的多模型对比实验【调优模式】")
        log("  流程: 数据划分 → 超参数调优(内层5折CV) → 测试集评估")
    log("=" * 80)
    log(f"  运行模式: {'推理模式 (--mode infer)' if infer_mode else '调优模式 (--mode tune)'}")
    log(f"  因变量: {target_cols}")
    log(f"  实验开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  日志文件: {log_path}")

    # 1. 加载数据（传入因变量列表，用于确定自变量）
    df, feature_cols = load_data(target_cols)
    X = df[feature_cols].values
    train_indices, test_indices = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    split = np.full(len(df), 'train', dtype=object)
    split[test_indices] = 'test'
    pd.DataFrame({'FID': df['FID'], 'split': split}).to_csv(
        os.path.join(OUTPUT_DIR, 'data_split.csv'), index=False, encoding='utf-8-sig'
    )
    summary_path = os.path.join(OUTPUT_DIR, 'training_summary.json')
    previous_results = {}
    if infer_mode and os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            previous_results = json.load(f).get('results', {})
    run_summary = {
        'status': 'running', 'mode': args.mode, 'data_file': DATA_PATH,
        'data_sha256': hashlib.sha256(Path(DATA_PATH).read_bytes()).hexdigest(),
        'started_at': datetime.now().isoformat(),
        'n_samples': len(df), 'features': feature_cols, 'targets': target_cols,
        'test_size': TEST_SIZE, 'random_state': RANDOM_STATE,
        'cv_folds': INNER_CV_FOLDS, 'search_iterations': N_ITER_SEARCH,
        'cv_jobs': N_JOBS, 'results': previous_results,
    }
    def save_summary():
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
    save_summary()
    log(f'  本轮结果目录: {OUTPUT_DIR}')
    log(f'  交叉验证并行进程数: {N_JOBS}；模型内部单线程')

    # 2. 调优模式：获取模型和参数搜索空间
    if not infer_mode:
        base_models = get_base_models()
        param_grids = get_param_grids()

        log("\n超参数搜索空间摘要:")
        for name, grid in param_grids.items():
            total = int(np.prod([len(v) for v in grid.values()]))
            actual = min(N_ITER_SEARCH, total)
            log(f"  {name}: 总共 {total} 种组合, 将随机采样 {actual} 种")

    # 3. 对每个选择的因变量分别进行实验
    for target_col in target_cols:
        y = df[target_col].values
        description = TARGET_DESCRIPTIONS.get(target_col, f'模型：{target_col}')

        log(f"\n\n{'#' * 80}")
        log(f"  {description}")
        log(f"{'#' * 80}")

        # ========== 推理模式：加载已保存的最优参数 ==========
        loaded_params = None
        if infer_mode:
            loaded_params = load_best_params(target_col)
            if loaded_params is None:
                log(f"  ❌ 错误: 未找到 {target_col} 的最优参数文件！")
                log(f"     请先运行调优模式: python experiment.py --mode tune --targets {target_col}")
                log(f"     参数文件应位于: {BEST_PARAMS_DIR}/best_params_{target_col}.json")
                continue
            log("  已加载的最优参数:")
            for m_name, m_params in loaded_params.items():
                if m_params is None:
                    log(f"    {m_name}: 无超参数")
                else:
                    log(f"    {m_name}: {m_params}")

        # ========== 步骤1: 划分训练集和测试集 ==========
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        log(f"\n  数据划分: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
        log(f"  因变量统计 — 训练集: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        log(f"  因变量统计 — 测试集: mean={y_test.mean():.4f}, std={y_test.std():.4f}")

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 存储结果
        test_metrics = {}
        test_predictions = {}
        all_best_params = {}
        best_trained_models = {}
        cv_mse = run_summary['results'].get(target_col, {}).get('best_cv_mse', {}) if infer_mode else {}

        # ========== 步骤2: OLS ==========
        log(f"\n  {'─' * 60}")
        log(f"  [1/5] OLS 回归（无超参数，直接训练）")
        log(f"  {'─' * 60}")

        ols_start = time.time()
        y_pred_test_ols, ols_model = ols_fit_predict(X_train_scaled, y_train, X_test_scaled)
        test_metrics['OLS'] = compute_metrics(y_test, y_pred_test_ols)
        test_predictions['OLS'] = (y_test, y_pred_test_ols)
        all_best_params['OLS'] = None

        ols_elapsed = time.time() - ols_start
        m = test_metrics['OLS']
        log(f"    测试集: MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R²']:.4f}")
        log(f"    耗时: {ols_elapsed:.1f}秒")

        # ========== 步骤3: 树模型 ==========
        model_list = ['GBDT', 'RF', 'XGBOOST', 'LightGBM']
        for idx, model_name in enumerate(model_list):
            log(f"\n  {'─' * 60}")

            if infer_mode:
                log(f"  [{idx + 2}/5] {model_name} — 推理模式（加载最优参数直接训练）")
                log(f"  {'─' * 60}")

                best_params = loaded_params.get(model_name)
                if best_params is None:
                    log(f"    ⚠️ 未找到 {model_name} 的最优参数，跳过")
                    continue

                log(f"    使用参数: {best_params}")
                start_time = time.time()

                best_model = create_model_with_params(model_name, best_params)
                best_model.fit(X_train_scaled, y_train)

                elapsed = time.time() - start_time
                log(f"    训练耗时: {elapsed:.1f}秒")

            else:
                log(f"  [{idx + 2}/5] {model_name} — 超参数调优中...")
                log(f"  {'─' * 60}")

                start_time = time.time()

                best_model, best_params, best_cv_mse = tune_hyperparameters(
                    model_name, base_models[model_name],
                    param_grids[model_name],
                    X_train_scaled, y_train, target_col
                )

                elapsed = time.time() - start_time
                log(f"    调参耗时: {elapsed:.1f}秒")
                cv_mse[model_name] = best_cv_mse

            # 在测试集上评估
            y_pred_test = best_model.predict(X_test_scaled)
            test_metrics[model_name] = compute_metrics(y_test, y_pred_test)
            test_predictions[model_name] = (y_test, y_pred_test)
            all_best_params[model_name] = best_params
            best_trained_models[model_name] = best_model

            m = test_metrics[model_name]
            log(f"    测试集: MAE={m['MAE']:.4f}  MSE={m['MSE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R²']:.4f}")

        # ========== 步骤4: 输出结果 ==========
        headers, table_data = format_results_table(test_metrics, target_col)
        csv_path = os.path.join(OUTPUT_DIR, f'model_comparison_test_{target_col}.csv')
        print_and_save_table(headers, table_data, f'{description} — 测试集性能对比', csv_path)

        save_best_params(all_best_params, target_col)
        prediction_frame = pd.DataFrame({
            'FID': df.iloc[test_indices]['FID'].to_numpy(), 'y_true': y_test,
            **{name: pred[1] for name, pred in test_predictions.items()},
        })
        prediction_frame.to_csv(
            os.path.join(OUTPUT_DIR, f'test_predictions_{target_col}.csv'),
            index=False, encoding='utf-8-sig'
        )
        run_summary['results'][target_col] = {
            'test_metrics': test_metrics, 'best_cv_mse': cv_mse,
            'best_cv_tree_model': min(cv_mse, key=cv_mse.get) if cv_mse else None,
            'best_test_tree_model': max(best_trained_models, key=lambda name: test_metrics[name]['R²']),
        }
        save_summary()

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

        ols_summary(X_train, y_train, feature_cols, target_col)

    total_elapsed = time.time() - total_start
    run_summary['status'] = 'completed'
    run_summary['elapsed_seconds'] = total_elapsed
    run_summary['finished_at'] = datetime.now().isoformat()
    save_summary()
    log(f"\n\n{'=' * 80}")
    log(f"  所有实验完成！")
    log(f"  运行模式: {'推理模式' if infer_mode else '调优模式'}")
    log(f"  因变量: {target_cols}")
    log(f"  总耗时: {total_elapsed:.1f}秒 ({total_elapsed / 60:.1f}分钟)")
    log(f"  结果保存在: {OUTPUT_DIR}")
    log(f"  日志文件: {log_path}")
    log(f"  实验结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not infer_mode:
        log(f"  💡 下次可用推理模式快速运行: python experiment.py --mode infer --output-dir \"{OUTPUT_DIR}\"")
    log(f'  📊 绘制 SHAP 图例: python plot_shap.py --output-dir "{OUTPUT_DIR}"')
    log(f"{'=' * 80}")

if __name__ == '__main__':
    main()