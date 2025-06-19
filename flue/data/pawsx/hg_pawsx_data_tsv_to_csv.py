import os
import shutil
import pandas as pd

def convert_folder_to_csv(src_folder):
    src_folder = os.path.abspath(src_folder)
    parent_dir = os.path.dirname(src_folder)
    folder_name = os.path.basename(src_folder)
    dst_folder = os.path.join(parent_dir, f"{folder_name}-csv")
    os.makedirs(dst_folder, exist_ok=True)

    for fname in os.listdir(src_folder):
        src_path = os.path.join(src_folder, fname)
        if os.path.isfile(src_path):
            if fname.lower().endswith('.tsv'):
                # Read TSV
                df = pd.read_csv(src_path, sep='\t', engine='python', dtype=str, quoting=3)
                # Keep only columns with a real name (not Unnamed or empty)
                keep_cols = [col for col in df.columns if col and not col.startswith("Unnamed")]
                df = df[keep_cols]
                # Remove quotes from all string values (optional)
                df = df.map(lambda x: x.strip('"') if isinstance(x, str) else x)
                new_name = fname[:-4] + '.csv'
                dst_path = os.path.join(dst_folder, new_name)
                df.to_csv(dst_path, index=False)
                print(f"Converted {fname} -> {new_name}")
            else:
                # Copy non-TSV files as-is
                dst_path = os.path.join(dst_folder, fname)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {fname}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python flue/data/pawsx/hg_pawsx_data_tsv_to_csv.py flue/data/pawsx/processed")
        sys.exit(1)

    source_folder = sys.argv[1]
    convert_folder_to_csv(source_folder)