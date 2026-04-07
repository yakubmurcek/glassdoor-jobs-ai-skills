import pandas as pd

de = pd.read_csv('data/outputs/de/de_relevant_ai_stata.csv', sep=';', encoding='utf-8-sig', low_memory=False)
sal = pd.to_numeric(de['salary_mid'], errors='coerce')

# Pay period breakdown for rows that HAVE salary
has_sal = sal.notna()
print('--- pay_period for rows WITH salary_mid ---')
print(de.loc[has_sal, 'pay_period'].value_counts(dropna=False).to_string())

# MONTHLY rows: raw values (will become x12 in Stata)
monthly_mask = de['pay_period'] == 'MONTHLY'
monthly_sal = sal[monthly_mask]
print()
print('--- MONTHLY salary_mid (raw, pre x12 in Stata) ---')
print('Count:', monthly_sal.notna().sum())
print(monthly_sal.describe())
print('After x12: min', monthly_sal.min()*12, 'median', monthly_sal.median()*12, 'max', monthly_sal.max()*12)

# HOURLY rows: what are the hourly rates?
hourly_mask = de['pay_period'] == 'HOURLY'
hourly_sal = sal[hourly_mask]
print()
print('--- HOURLY salary_mid (raw, pre x1607 in Stata) ---')
print('Count:', hourly_sal.notna().sum())
print(hourly_sal.describe())
print('After x1607: min', round(hourly_sal.min()*1607), 'median', round(hourly_sal.median()*1607))

# Values under 3000 that get nulled in Stata
print()
print('--- Rows with salary_mid < 3000 (dropped in Stata) ---')
low = de[sal < 3000]
print('Count:', len(low))
print(low[['salary_mid', 'pay_period', 'pay_currency']].to_string())

# ANNUAL rows summary
annual_sal = sal[de['pay_period'] == 'ANNUAL']
print()
print('--- ANNUAL salary_mid ---')
print('Count:', annual_sal.notna().sum())
print(annual_sal.describe())
