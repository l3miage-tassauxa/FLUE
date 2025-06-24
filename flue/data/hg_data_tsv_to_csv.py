import os
import shutil
import pandas as pd

def convert_folder_to_csv(src_folder, add_headers=None):
    src_folder = os.path.abspath(src_folder)
    parent_dir = os.path.dirname(src_folder)
    folder_name = os.path.basename(src_folder)
    dst_folder = os.path.join(parent_dir, f"{folder_name}-csv")
    os.makedirs(dst_folder, exist_ok=True)

    # Auto-detect if we need headers based on folder path
    if add_headers is None:
        add_headers = 'xnli' in src_folder.lower()

    for fname in os.listdir(src_folder):
        src_path = os.path.join(src_folder, fname)
        if os.path.isfile(src_path):
            if fname.lower().endswith('.tsv'):
                # Read TSV without headers (since TSV files don't have headers)
                df = pd.read_csv(src_path, sep='\t', engine='python', dtype=str, quoting=3, header=None)
                
                if add_headers:
                    # Add proper column names based on the number of columns
                    if df.shape[1] == 2:
                        df.columns = ['text', 'label']
                    elif df.shape[1] == 3:
                        df.columns = ['sentence1', 'sentence2', 'label']
                    else:
                        # Generic column names if we have a different number of columns
                        df.columns = [f'column_{i}' for i in range(df.shape[1])]
                    
                    # Remove quotes from all string values (optional)
                    df = df.map(lambda x: x.strip('"') if isinstance(x, str) else x)
                    new_name = fname[:-4] + '.csv'
                    dst_path = os.path.join(dst_folder, new_name)
                    df.to_csv(dst_path, index=False)
                    print(f"Converted {fname} -> {new_name} (with headers)")
                else:
                    # Original behavior for PAWSX - no headers, just convert format
                    new_name = fname[:-4] + '.csv'
                    dst_path = os.path.join(dst_folder, new_name)
                    df.to_csv(dst_path, index=False, header=False)
                    print(f"Converted {fname} -> {new_name} (no headers)")
            else:
                # Copy non-TSV files as-is
                dst_path = os.path.join(dst_folder, fname)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {fname}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python flue/data/hg_data_tsv_to_csv.py <source_folder> [add_headers]")
        print("  source_folder: Path to the folder containing TSV files")
        print("  add_headers: 'true'/'false' to force header behavior, or auto-detect based on path")
        sys.exit(1)

    source_folder = sys.argv[1]
    
    # Parse optional add_headers parameter
    add_headers = None
    if len(sys.argv) > 2:
        if sys.argv[2].lower() == 'true':
            add_headers = True
        elif sys.argv[2].lower() == 'false':
            add_headers = False
        # If not 'true' or 'false', leave as None for auto-detection
    
    convert_folder_to_csv(source_folder, add_headers)