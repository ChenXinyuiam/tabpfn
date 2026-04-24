import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

data_dir = Path("d:/2026/others/tabpfn/data/processed/2026-04-22")
xlsx_files = sorted(data_dir.glob("*.xlsx"))

for file_path in xlsx_files:
    print(f"\n{'='*100}")
    print(f"FILE: {file_path.name}")
    print(f"{'='*100}")
    
    try:
        df = pd.read_excel(file_path)
        
        print(f"\nCOLUMN NAMES ({len(df.columns)} columns):")
        print(", ".join(df.columns.tolist()))
        
        print(f"\nSHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print(f"\nDTYPES:")
        dtype_summary = df.dtypes.astype(str).value_counts()
        for dtype, count in dtype_summary.items():
            print(f"  {dtype}: {count} columns")
        
        print(f"\nFIRST 10 ROWS:")
        print(df.head(10).to_string())
        
        print(f"\nDESCRIPTIVE STATISTICS:")
        print(df.describe().to_string())
        
        print(f"\n*** DATA QUALITY ISSUES ***")
        
        # NaN values
        nan_counts = df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\nNaN VALUES:")
            for col in df.columns:
                if nan_counts[col] > 0:
                    pct = 100 * nan_counts[col] / len(df)
                    print(f"  {col}: {nan_counts[col]} ({pct:.1f}%)")
        else:
            print(f"\nNo NaN values")
        
        # Negative values
        print(f"\nNEGATIVE VALUES:")
        has_neg = False
        for col in df.select_dtypes(include=[np.number]).columns:
            if (df[col] < 0).any():
                neg_count = (df[col] < 0).sum()
                print(f"  {col}: {neg_count} negative values (min: {df[col].min()})")
                has_neg = True
        if not has_neg:
            print(f"  None detected")
        
        # geometry_valid
        if 'geometry_valid' in df.columns:
            false_count = (df['geometry_valid'] == False).sum()
            print(f"\ngeometry_valid: {false_count} False values (out of {len(df)})")
        
        # Outliers
        print(f"\nOUTLIERS (IQR method):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_found = False
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                if outliers > 0:
                    print(f"  {col}: {outliers} outliers")
                    outlier_found = True
        if not outlier_found:
            print(f"  None detected")
        
        # Zero values
        print(f"\nZERO VALUES:")
        zero_found = False
        for col in numeric_cols:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"  {col}: {zero_count} zeros")
                zero_found = True
        if not zero_found:
            print(f"  None detected")
        
        # Duplicates
        dup_count = df.duplicated().sum()
        print(f"\nDUPLICATE ROWS: {dup_count}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*100}")
print("COMPLETE")
