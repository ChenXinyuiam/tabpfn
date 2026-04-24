import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("d:/2026/others/tabpfn/data/processed/2026-04-22")
xlsx_files = sorted(data_dir.glob("*.xlsx"))

print(f"Found {len(xlsx_files)} Excel files\n")
print("=" * 100)

for file_path in xlsx_files:
    print(f"\n\nFILE: {file_path.name}")
    print("-" * 100)
    
    try:
        df = pd.read_excel(file_path)
        
        # 1. Column names
        print(f"\n1. COLUMN NAMES ({len(df.columns)} columns):")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # 2. Shape and dtypes
        print(f"\n2. SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n   DTYPES:")
        for col, dtype in df.dtypes.items():
            print(f"   {col}: {dtype}")
        
        # 3. First 10 rows
        print(f"\n3. FIRST 10 ROWS:")
        print(df.head(10).to_string())
        
        # 4. Descriptive statistics
        print(f"\n4. DESCRIPTIVE STATISTICS (df.describe()):")
        print(df.describe().to_string())
        
        # 5. Check for bad/wrong values
        print(f"\n5. DATA QUALITY CHECKS:")
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\n   NaN values found:")
            for col in df.columns:
                if nan_counts[col] > 0:
                    print(f"   {col}: {nan_counts[col]} NaN values")
        else:
            print(f"   No NaN values found")
        
        # Check for negative values in numeric columns
        print(f"\n   Negative values in numeric columns:")
        has_negative = False
        for col in df.select_dtypes(include=[np.number]).columns:
            if (df[col] < 0).any():
                neg_count = (df[col] < 0).sum()
                print(f"   {col}: {neg_count} negative values (min: {df[col].min()})")
                has_negative = True
        if not has_negative:
            print(f"   No negative values found")
        
        # Check for geometry_valid column
        if 'geometry_valid' in df.columns:
            print(f"\n   geometry_valid column analysis:")
            print(f"   Value counts: {df['geometry_valid'].value_counts().to_dict()}")
            false_count = (df['geometry_valid'] == False).sum()
            if false_count > 0:
                print(f"   WARNING: {false_count} rows with geometry_valid==False")
        
        # Check for outliers using IQR method for numeric columns
        print(f"\n   Potential outliers (using IQR method):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_found = False
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"   {col}: {outliers} potential outliers")
                outlier_found = True
        if not outlier_found:
            print(f"   No significant outliers detected")
        
        # Check for zero or near-zero values in columns where they shouldn't be
        print(f"\n   Zero/Near-zero value analysis:")
        for col in numeric_cols:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"   {col}: {zero_count} zero values")
        
        # Check for duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            print(f"\n   Duplicate rows: {dup_count} duplicates found")
        else:
            print(f"\n   No duplicate rows found")
        
    except Exception as e:
        print(f"ERROR reading file: {e}")

print("\n" + "=" * 100)
print("\nEXPLORATION COMPLETE")
