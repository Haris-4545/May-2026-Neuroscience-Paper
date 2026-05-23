# Smith et al. (2024) Demographic Analysis
#raw fMRI data could not be used, so using participants.tsv only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

df_smith = pd.read_csv('participants.tsv', sep='\t')

string_cols = ['sex', 'race', 'ethnicity', 'handedness', 'group']
for col in string_cols:
    df_smith[col] = df_smith[col].str.strip()
print("=== SMITH DATASET")
print(f"Total participants: {len(df_smith)}")
print(f"Columns: {df_smith.columns.tolist()}")
print(df_smith.head())

print("\n=== AGE DESCRIPTIVES ===")
print(df_smith['age'].describe().round(2))
print(f"Age range: {df_smith['age'].min()} to {df_smith['age'].max()} years")

df_smith['age_group'] = pd.cut(
    df_smith['age'],
    bins   = [20, 35, 49, 80],
    labels = ['young_adult', 'middle_adult', 'older_adult']
)

print("\n=== AGE GROUP BREAKDOWN ===")
age_group_counts = df_smith['age_group'].value_counts().sort_index()
print(age_group_counts)
print("\nAs percentages:")
print((age_group_counts / len(df_smith) * 100).round(1))

print("\n=== SEX DISTRIBUTION ===")
sex_counts = df_smith['sex'].value_counts()
print(sex_counts)
print("\nAs percentages:")
print((sex_counts / len(df_smith) * 100).round(1))

print("\nSex by age group:")
print(pd.crosstab(df_smith['age_group'], df_smith['sex']))

print("\n=== RACE BREAKDOWN ===")
race_counts = df_smith['race'].value_counts()
print(race_counts)
print("\nAs percentages:")
print((race_counts / len(df_smith) * 100).round(1))

print("\n=== ETHNICITY BREAKDOWN ===")
eth_counts = df_smith['ethnicity'].value_counts()
print(eth_counts)

# Proportion of non-White participants
non_white = df_smith[df_smith['race'] != 'White']
print(f"\nNon-White participants: {len(non_white)} ({len(non_white)/len(df_smith)*100:.1f}%)")
print(f"White participants: {len(df_smith) - len(non_white)} ({(len(df_smith)-len(non_white))/len(df_smith)*100:.1f}%)")

print("\n=== BMI BY AGE GROUP ===")
print(df_smith.groupby('age_group')['BMI'].describe().round(2))

# Vaidya sample characteristics (from published paper)
vaidya_data = {
    'participant_id' : [f'adol_{i}' for i in range(18)] + [f'adult_{i}' for i in range(18)],
    'age_group'      : ['adolescent'] * 18 + ['young_adult'] * 18,
    'mean_age'       : [13.39] * 18 + [27.72] * 18,
    'sex'            : (['F'] * 9 + ['M'] * 9) + (['F'] * 9 + ['M'] * 9)
}
df_vaidya_demo = pd.DataFrame(vaidya_data)

print("\n=== VAIDYA SAMPLE SUMMARY ===")
print(f"Adolescents: N=18, mean age=13.39, SD=0.92, range 12-15")
print(f"Young adults: N=18, mean age=27.72, SD=1.36, range 26-30")
print(f"Sex: 9 female, 9 male per group")
print(f"Race: Not reported in Vaidya et al. (2013)")

print("\n=== COMBINED LIFESPAN COVERAGE ===")
print("Source         | Age Range | N   | Groups")
print("---------------|-----------|-----|---------------------------")
print("Vaidya (2013)  | 12-30     | 36  | Adolescents, Young Adults")
print("Smith (2024)   | 21-80     | 114 | Young, Middle, Older Adults")
print("Combined       | 12-80     | 150 | Full lifespan coverage")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Smith et al. (2024) — Sample Demographic Analysis', fontsize=14)

# Plot 1 — Age distribution with age group bands
ax1 = axes[0, 0]
sns.histplot(df_smith['age'], bins=20, kde=True, ax=ax1, color='steelblue')
ax1.axvspan(21, 35, alpha=0.1, color='green',  label='Young adult')
ax1.axvspan(35, 49, alpha=0.1, color='orange', label='Middle adult')
ax1.axvspan(49, 80, alpha=0.1, color='red',    label='Older adult')
ax1.set_title('Age Distribution')
ax1.set_xlabel('Age (years)')
ax1.set_ylabel('Count')
ax1.legend(fontsize=8)

# Plot 2 — Age group counts
ax2 = axes[0, 1]
age_group_counts.plot(kind='bar', ax=ax2, color=['green', 'orange', 'red'], alpha=0.7)
ax2.set_title('Participants per Age Group')
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(age_group_counts):
    ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# Plot 3 — Race breakdown
ax3 = axes[1, 0]
race_counts.plot(kind='bar', ax=ax3, color='steelblue', alpha=0.7)
ax3.set_title('Race Distribution (Smith 2024)')
ax3.set_xlabel('Race')
ax3.set_ylabel('Count')
ax3.tick_params(axis='x', rotation=45)

# Plot 4 — Combined lifespan coverage
ax4 = axes[1, 1]
# Vaidya age ranges
ax4.barh('Vaidya\nAdolescents', 3,  left=12, color='navy',      alpha=0.8, height=0.4)
ax4.barh('Vaidya\nYoung Adults', 4, left=26, color='steelblue', alpha=0.8, height=0.4)
# Smith age groups
ax4.barh('Smith\nYoung Adult',   14, left=21, color='green',     alpha=0.7, height=0.4)
ax4.barh('Smith\nMiddle Adult',  14, left=35, color='orange',    alpha=0.7, height=0.4)
ax4.barh('Smith\nOlder Adult',   30, left=50, color='red',       alpha=0.7, height=0.4)
ax4.set_title('Combined Lifespan Coverage Across Both Sources')
ax4.set_xlabel('Age (years)')
ax4.axvline(x=18, color='black', linestyle='--', alpha=0.5, label='Age 18')
ax4.legend(fontsize=8)

plt.tight_layout()
plt.savefig('smith_demographic_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: smith_demographic_analysis.png")


print(f"\nSmith (2024) sample:")
print(f"  N = {len(df_smith)}")
print(f"  Age: M = {df_smith['age'].mean():.2f}, SD = {df_smith['age'].std():.2f}, range {df_smith['age'].min()}-{df_smith['age'].max()}")
print(f"  Sex: {sex_counts.get('F', 0)} female, {sex_counts.get('M', 0)} male, {sex_counts.get('O', 0)} other")
print(f"  Young adults (21-35):   n = {age_group_counts.get('young_adult', 0)}")
print(f"  Middle adults (36-49):  n = {age_group_counts.get('middle_adult', 0)}")
print(f"  Older adults (50-80):   n = {age_group_counts.get('older_adult', 0)}")
print(f"  White: {(df_smith['race']=='White').sum()} ({(df_smith['race']=='White').sum()/len(df_smith)*100:.1f}%)")
print(f"  Non-White: {(df_smith['race']!='White').sum()} ({(df_smith['race']!='White').sum()/len(df_smith)*100:.1f}%)")
print(f"\nCombined coverage:")
print(f"  Total participants across both sources: {len(df_smith) + 36}")
print(f"  Age range: 12-80 years")
print(f"  Developmental groups: adolescents, young adults, middle adults, older adults")