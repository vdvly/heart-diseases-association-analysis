#  Heart Disease — Association Rule Mining
#  Algorithms: Apriori · FP-Growth · ECLAT (custom)

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
from matplotlib import patches 
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ── Parameters (change these if needed) ──────────────────────
DATA_PATH   = r"C:\Users\dvly\Downloads\heart+disease\processed.cleveland.data"
MIN_SUPPORT = 0.30
MIN_CONF    = 0.70


# 1. LOAD DATA
print("\n" + "="*60)
print("  1. LOAD DATA")
print("="*60)

df = pd.read_csv(DATA_PATH, header=None, na_values="?")
df.columns = ["age","sex","cp","trestbps","chol","fbs",
              "restecg","thalach","exang","oldpeak",
              "slope","ca","thal","target"]

df.dropna(inplace=True)
df['target'] = (df['target'] > 0).astype(int)   # 0=no disease, 1=disease

print(f"  Rows: {len(df)}  |  Columns: {df.shape[1]}")
print(df.head(3).to_string(index=False))


# 2. DATA PREPARATION  (discretise + one-hot encode)
print("\n" + "="*60)
print("  2. DATA PREPARATION")
print("="*60)

data = df.copy()

# ── Discretise continuous variables ──────────────────────────
data['age']      = pd.cut(data['age'],      bins=[0,40,55,100],
                          labels=['age_young','age_middle','age_old'])
data['trestbps'] = pd.cut(data['trestbps'], bins=[0,120,140,300],
                          labels=['bp_normal','bp_elevated','bp_high'])
data['chol']     = pd.cut(data['chol'],     bins=[0,200,240,600],
                          labels=['chol_normal','chol_borderline','chol_high'])
data['thalach']  = pd.cut(data['thalach'],  bins=[0,100,150,250],
                          labels=['hr_low','hr_medium','hr_high'])
data['oldpeak']  = pd.cut(data['oldpeak'],  bins=[-1,1,2.5,10],
                          labels=['st_normal','st_moderate','st_severe'])

# ── Label categorical columns ─────────────────────────────────
cat_map = {
    'sex':     {0:'sex_female',   1:'sex_male'},
    'cp':      {0:'cp_typical',   1:'cp_atypical', 2:'cp_nonanginal', 3:'cp_asymptomatic'},
    'fbs':     {0:'fbs_normal',   1:'fbs_high'},
    'restecg': {0:'ecg_normal',   1:'ecg_ST',      2:'ecg_LVH'},
    'exang':   {0:'exang_no',     1:'exang_yes'},
    'slope':   {1:'slope_up',     2:'slope_flat',  3:'slope_down'},
    'thal':    {3:'thal_normal',  6:'thal_fixed',  7:'thal_reversal'},
    'target':  {0:'target_no',    1:'target_yes'},
}
for col, mapping in cat_map.items():
    data[col] = data[col].map(mapping)

data['ca'] = 'ca_' + data['ca'].astype(int).astype(str)

# ── One-hot encode → boolean DataFrame ───────────────────────
binary_df = pd.get_dummies(data, prefix='', prefix_sep='').astype(bool)

print(f"  Encoded shape: {binary_df.shape}")
print(f"  Features: {list(binary_df.columns)}")


# 3. APRIORI — FREQUENT ITEMSETS
print("\n" + "="*60)
print(f"  3. APRIORI  (min_support={MIN_SUPPORT})")
print("="*60)

t0 = time.time()
freq_apriori = apriori(binary_df, min_support=MIN_SUPPORT,
                       use_colnames=True, max_len=3)
apriori_time = time.time() - t0
freq_apriori['length'] = freq_apriori['itemsets'].apply(len)

for k in [1, 2, 3]:
    subset = freq_apriori[freq_apriori['length'] == k] \
                 .sort_values('support', ascending=False)
    print(f"\n  Frequent {k}-itemsets  ({len(subset)} found):")
    for _, row in subset.iterrows():
        items = ', '.join(sorted(row['itemsets']))
        print(f"    {{ {items} }}   sup={row['support']:.3f}")

freq_apriori[freq_apriori['length']==1].to_csv("frequent_1_itemsets.csv", index=False)
freq_apriori[freq_apriori['length']==2].to_csv("frequent_2_itemsets.csv", index=False)
freq_apriori[freq_apriori['length']==3].to_csv("frequent_3_itemsets.csv", index=False)


# 4. ASSOCIATION RULES  (min_confidence = 0.7)
print("\n" + "="*60)
print(f"  4. ASSOCIATION RULES  (min_confidence={MIN_CONF})")
print("="*60)

rules = association_rules(freq_apriori, metric="confidence",
                          min_threshold=MIN_CONF)
rules = rules[['antecedents','consequents','support',
               'confidence','lift','leverage','conviction']] \
        .sort_values('confidence', ascending=False)

def show_rules(df_rules, title):
    print(f"\n  ── {title}  ({len(df_rules)} rules) ──")
    for _, r in df_rules.head(10).iterrows():
        lhs = ', '.join(sorted(r['antecedents']))
        rhs = ', '.join(sorted(r['consequents']))
        print(f"    IF {{ {lhs} }}  ->  {{ {rhs} }}")
        print(f"       sup={r['support']:.3f}  conf={r['confidence']:.3f}"
              f"  lift={r['lift']:.3f}  lev={r['leverage']:.4f}"
              f"  conv={r['conviction']:.3f}")

rules_AB = rules[(rules['antecedents'].apply(len) == 1) &
                 (rules['consequents'].apply(len) == 1)]
show_rules(rules_AB, "IF A -> B")

rules_ABC = rules[(rules['antecedents'].apply(len) == 2) &
                  (rules['consequents'].apply(len) == 1)]
show_rules(rules_ABC, "IF A AND B -> C")

rules_target = rules[rules['consequents'].apply(
    lambda x: bool(x & {'target_yes', 'target_no'}))]
show_rules(rules_target, "IF A -> target")

rules.to_csv("all_rules.csv",             index=False)
rules_AB.to_csv("rules_A_then_B.csv",    index=False)
rules_ABC.to_csv("rules_AB_then_C.csv",  index=False)
rules_target.to_csv("rules_A_then_target.csv", index=False)


# 5. FP-GROWTH
print("\n" + "="*60)
print(f"  5. FP-GROWTH  (min_support={MIN_SUPPORT})")
print("="*60)

t0 = time.time()
freq_fp = fpgrowth(binary_df, min_support=MIN_SUPPORT, use_colnames=True)
fp_time = time.time() - t0
freq_fp['length'] = freq_fp['itemsets'].apply(len)

print(f"  Found {len(freq_fp)} itemsets in {fp_time:.4f}s")
freq_fp.to_csv("frequent_itemsets_fpgrowth.csv", index=False)


# 6. ECLAT  (custom implementation)
print("\n" + "="*60)
print(f"  6. ECLAT  (custom, min_support={MIN_SUPPORT})")
print("="*60)

"""
ECLAT uses a vertical data format: each item maps to its
TIDset (set of transaction IDs). k-itemsets are found by
intersecting TIDsets of (k-1)-itemsets — no candidate
generation needed. Depth-first search traversal.
"""

def build_tidsets(binary_dataframe):
    tidsets = defaultdict(set)
    for tid, row in binary_dataframe.iterrows():
        for item in binary_dataframe.columns[row.values]:
            tidsets[item].add(tid)
    return tidsets

def eclat_recursive(prefix, items, min_count, results):
    while items:
        item, tids = items.pop()
        if len(tids) >= min_count:
            key = frozenset(prefix + [item])
            results[key] = len(tids)
            extensions = [
                (other, tids & other_tids)
                for other, other_tids in items
                if len(tids & other_tids) >= min_count
            ]
            eclat_recursive(prefix + [item], extensions, min_count, results)

t0 = time.time()
tidsets      = build_tidsets(binary_df)
min_count    = int(MIN_SUPPORT * len(binary_df))
eclat_raw    = {}
eclat_recursive([], list(tidsets.items()), min_count, eclat_raw)
eclat_time   = time.time() - t0

eclat_df = pd.DataFrame([
    {'itemsets': fs, 'support': cnt/len(binary_df), 'length': len(fs)}
    for fs, cnt in eclat_raw.items()
]).sort_values('support', ascending=False)

print(f"  Found {len(eclat_df)} itemsets in {eclat_time:.4f}s")
eclat_df.to_csv("frequent_itemsets_eclat.csv", index=False)


# 7. ALGORITHM COMPARISON
print("\n" + "="*60)
print("  7. ALGORITHM COMPARISON")
print("="*60)

comp = pd.DataFrame({
    'Algorithm':  ['Apriori',      'FP-Growth',       'ECLAT'],
    'Paradigm':   ['BFS + prune',  'FP-Tree divide',  'DFS + tidsets'],
    'DB Scans':   ['k + 1',        '2',               '1'],
    'Candidates': ['Yes',          'No',              'No'],
    'Time (s)':   [f'{apriori_time:.4f}', f'{fp_time:.4f}', f'{eclat_time:.4f}'],
    'Itemsets':   [len(freq_apriori), len(freq_fp),   len(eclat_df)],
})
print(comp.to_string(index=False))
comp.to_csv("algorithm_comparison.csv", index=False)

print("""
  Concept Summary:
  Apriori   : Anti-monotone pruning, BFS, k+1 DB scans per level
  FP-Growth : Builds a compact FP-Tree, mines conditional pattern bases
  ECLAT     : Vertical TIDsets, k-way intersection, depth-first DFS
""")


# 8. VISUALISATION
print("="*60)
print("  8. VISUALISATION")
print("="*60)

# Fig 1 — Top 15 frequent 1-itemsets
top_items = (freq_apriori[freq_apriori['length'] == 1]
             .sort_values('support', ascending=False)
             .head(15).copy())
top_items['label'] = top_items['itemsets'].apply(lambda x: list(x)[0])

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e94560' if 'target' in lbl else '#3b82f6'
          for lbl in top_items['label']]
bars = ax.barh(top_items['label'], top_items['support'],
               color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, top_items['support']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.axvline(MIN_SUPPORT, color='orange', linestyle='--',
           label=f'min_sup={MIN_SUPPORT}')
ax.set_xlabel('Support')
ax.set_title('Top 15 Frequent 1-Itemsets (Apriori)')
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.savefig('fig1_top_itemsets.png', dpi=150)
plt.close()

# Fig 2 — Support vs Confidence scatter
if not rules.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(rules['support'], rules['confidence'],
                    c=rules['lift'], cmap='plasma',
                    s=60, alpha=0.8, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Lift')
    ax.axhline(MIN_CONF, color='orange', linestyle='--',
               label=f'min_conf={MIN_CONF}')
    ax.set_xlabel('Support')
    ax.set_ylabel('Confidence')
    ax.set_title('Association Rules — Support vs Confidence')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig2_rules_scatter.png', dpi=150)
    plt.close()

# Fig 3 — Correlation heatmap of rule metrics
if not rules.empty:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        rules[['support','confidence','lift','leverage','conviction']].corr(),
        annot=True, fmt='.2f', cmap='coolwarm', ax=ax
    )
    ax.set_title('Correlation of Rule Metrics')
    plt.tight_layout()
    plt.savefig('fig3_metrics_heatmap.png', dpi=150)
    plt.close()

# Fig 4 — Algorithm comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
algo_names  = ['Apriori', 'FP-Growth', 'ECLAT']
algo_colors = ['#e94560', '#3b82f6', '#10b981']
times  = [apriori_time, fp_time, eclat_time]
counts = [len(freq_apriori), len(freq_fp), len(eclat_df)]

for ax, vals, title, ylabel in zip(
        axes,
        [times, counts],
        ['Execution Time (s)', 'Frequent Itemsets Found'],
        ['Time (s)', 'Count']):
    b = ax.bar(algo_names, vals, color=algo_colors, width=0.5,
               edgecolor='white', linewidth=0.8)
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.02,
                f'{v:.4f}' if isinstance(v, float) else str(v),
                ha='center', fontweight='bold', fontsize=11)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

plt.suptitle('Algorithm Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_algorithm_comparison.png', dpi=150)
plt.close()

print("\n  Saved: fig1_top_itemsets.png")
print("  Saved: fig2_rules_scatter.png")
print("  Saved: fig3_metrics_heatmap.png")
print("  Saved: fig4_algorithm_comparison.png")

print("\n" + "="*60)
print("  DONE — CSV files + 4 figures saved.")
print("="*60)
# ── Fig 5: Frequent Itemset Count by Length (Bar) ────────────
fig, ax = plt.subplots(figsize=(7, 5))
lengths = freq_apriori['length'].value_counts().sort_index()
bars = ax.bar([f'{k}-itemsets' for k in lengths.index], lengths.values,
              color=['#3b82f6','#e94560','#10b981'], edgecolor='white', width=0.5)
for bar, val in zip(bars, lengths.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha='center', fontweight='bold', fontsize=12)
ax.set_title('Number of Frequent Itemsets by Length (Apriori)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('fig5_itemset_counts.png', dpi=150)
plt.close()

# ── Fig 6: Metrics Distribution Boxplot ─────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
metrics = ['support', 'confidence', 'lift', 'conviction']
colors  = ['#3b82f6', '#10b981', '#e94560', '#f59e0b']
for ax, metric, color in zip(axes, metrics, colors):
    ax.boxplot(rules[metric].dropna(), patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7),
               medianprops=dict(color='white', linewidth=2))
    ax.set_title(metric.capitalize(), fontsize=12)
    ax.set_xticks([])
plt.suptitle('Distribution of Rule Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig6_metrics_boxplot.png', dpi=150)
plt.close()