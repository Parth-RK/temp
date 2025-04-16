# --- dataman.py ---
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import sys

# Dynamically add project root to path if needed (adjust relative path)
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

try:
    import config # Use config for default paths/columns if available
except ImportError:
    print("Warning: config.py not found. Using hardcoded defaults in dataman.")
    # Define minimal defaults if config cannot be imported
    class ConfigFallback:
        DATA_DIR = "data"
        TEXT_COLUMN_INDEX = 1
        LABEL_COLUMN_INDEX = 0
        COLUMN_NAMES = ['label', 'text']
        HAS_HEADER = True
        SEED = 42
    config = ConfigFallback()

def _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format):
    """Helper to load data based on format."""
    print(f"Loading data from: {input_path}")
    try:
        if file_format == "csv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(input_path, header=header, names=names)
        elif file_format == "tsv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(input_path, sep='\t', header=header, names=names)
        elif file_format == "jsonl":
            df = pd.read_json(input_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Select and rename columns based on indices provided
        text_col_name = df.columns[text_col_idx]
        label_col_name = df.columns[label_col_idx]
        df = df[[label_col_name, text_col_name]]
        df.columns = ['label', 'text'] # Standardize column names

        print(f"Loaded {len(df)} rows.")
        print(f"Using columns: label='{label_col_name}' (idx {label_col_idx}), text='{text_col_name}' (idx {text_col_idx})")
        return df

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except IndexError:
         print(f"Error: Column index out of bounds. Check TEXT_COLUMN_INDEX ({text_col_idx}) and LABEL_COLUMN_INDEX ({label_col_idx}) for file {input_path}")
         return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_balanced_subset(input_path, output_path, n_samples_per_class,
                           text_col_idx=config.TEXT_COLUMN_INDEX,
                           label_col_idx=config.LABEL_COLUMN_INDEX,
                           col_names=config.COLUMN_NAMES,
                           has_header=config.HAS_HEADER,
                           file_format="csv"):
    """
    Creates a balanced dataset subset with n samples from each label category.

    Args:
        input_path (str): Path to the input data file.
        output_path (str): Path to save the balanced dataset.
        n_samples_per_class (int): Number of samples per label.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header.
        has_header (bool): If the file has a header.
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    df = _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format)
    if df is None:
        return

    print(f"\nOriginal dataset shape: {df.shape}")
    df = df.dropna(subset=['label', 'text']).reset_index(drop=True)
    print(f"Shape after dropping NaNs: {df.shape}")

    print("\nOriginal Label Distribution:")
    print(df['label'].value_counts())

    balanced_dfs = []
    unique_labels = df['label'].unique()

    print(f"\nCreating balanced subset with {n_samples_per_class} samples per class...")
    for label in unique_labels:
        label_df = df[df['label'] == label]
        available_samples = len(label_df)
        sample_size = min(n_samples_per_class, available_samples)

        if sample_size < n_samples_per_class:
            print(f"  Warning: Label '{label}' has only {sample_size} samples (requested {n_samples_per_class}). Taking all available.")

        sampled_df = label_df.sample(n=sample_size, random_state=config.SEED, replace=False)
        balanced_dfs.append(sampled_df)

    if not balanced_dfs:
        print("Error: No data collected for balancing. Check input data and parameters.")
        return

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True) # Shuffle

    print(f"\nBalanced subset shape: {balanced_df.shape}")
    print("Balanced Subset Label Distribution:")
    print(balanced_df['label'].value_counts())

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Save with header using standard column names
        balanced_df.to_csv(output_path, index=False)
        print(f"\nBalanced subset saved to {output_path}")
    except Exception as e:
        print(f"Error saving balanced subset: {e}")

def split_data(input_path, train_path, val_path, test_path,
               val_size=0.15, test_size=0.15, stratify=True,
               text_col_idx=config.TEXT_COLUMN_INDEX,
               label_col_idx=config.LABEL_COLUMN_INDEX,
               col_names=config.COLUMN_NAMES,
               has_header=config.HAS_HEADER,
               file_format="csv"):
    """
    Splits the data into train, validation, and test sets.

    Args:
        input_path (str): Path to the input data file.
        train_path (str): Path to save the training set.
        val_path (str): Path to save the validation set.
        test_path (str): Path to save the test set.
        val_size (float): Proportion for validation set.
        test_size (float): Proportion for test set (from the original data).
        stratify (bool): Whether to stratify based on labels.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header.
        has_header (bool): If the file has a header.
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    df = _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format)
    if df is None:
        return

    df = df.dropna(subset=['label', 'text']).reset_index(drop=True)
    print(f"Total data for splitting: {len(df)} rows")

    if len(df) < 3:
        print("Error: Not enough data to perform train/val/test split.")
        return

    stratify_col = df['label'] if stratify else None

    # Calculate test size relative to original, val size relative to remaining
    if (val_size + test_size) >= 1.0:
        print("Error: Sum of validation and test sizes must be less than 1.0")
        return

    # Split off test set first
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=config.SEED,
        stratify=stratify_col
    )

    # Split remaining into train and validation
    # Adjust val_size relative to the remaining data after test split
    relative_val_size = val_size / (1.0 - test_size)
    stratify_col_train_val = train_val_df['label'] if stratify else None

    if len(train_val_df) < 2:
         print("Warning: Very few samples remaining after test split. Validation split might be empty.")
         train_df = train_val_df
         val_df = pd.DataFrame(columns=df.columns) # Empty df
    else:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=config.SEED,
            stratify=stratify_col_train_val
        )

    print(f"\nSplit complete:")
    print(f"  Train set size: {len(train_df)}")
    print(f"  Validation set size: {len(val_df)}")
    print(f"  Test set size: {len(test_df)}")

    try:
        for pth, dframe in [(train_path, train_df), (val_path, val_df), (test_path, test_df)]:
            out_dir = os.path.dirname(pth)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            # Save with header using standard column names
            dframe.to_csv(pth, index=False)
            print(f"  Saved {os.path.basename(pth)} ({len(dframe)} rows)")
        print("\nData splitting and saving finished.")
    except Exception as e:
        print(f"Error saving split files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Manipulation Utility (Manual Use)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Balance Subcommand ---
    parser_balance = subparsers.add_parser("balance", help="Create a balanced subset of the data.")
    parser_balance.add_argument("-i", "--input", type=str, default=config.INPUT_FILE_PATH, help="Path to the input data file.")
    parser_balance.add_argument("-o", "--output", type=str, required=True, help="Path to save the balanced output file.")
    parser_balance.add_argument("-n", "--num_samples", type=int, required=True, help="Number of samples per class.")
    parser_balance.add_argument("--format", type=str, default="csv", choices=["csv", "tsv", "jsonl"], help="Input file format.")
    parser_balance.add_argument("--text_col", type=int, default=config.TEXT_COLUMN_INDEX, help="Index of the text column.")
    parser_balance.add_argument("--label_col", type=int, default=config.LABEL_COLUMN_INDEX, help="Index of the label column.")
    parser_balance.add_argument("--no_header", action="store_true", help="Specify if input file has no header.")

    # --- Split Subcommand ---
    parser_split = subparsers.add_parser("split", help="Split data into train, validation, and test sets.")
    parser_split.add_argument("-i", "--input", type=str, default=config.INPUT_FILE_PATH, help="Path to the input data file.")
    parser_split.add_argument("--train_out", type=str, required=True, help="Path to save the training set.")
    parser_split.add_argument("--val_out", type=str, required=True, help="Path to save the validation set.")
    parser_split.add_argument("--test_out", type=str, required=True, help="Path to save the test set.")
    parser_split.add_argument("--val_size", type=float, default=0.15, help="Validation set proportion.")
    parser_split.add_argument("--test_size", type=float, default=0.15, help="Test set proportion.")
    parser_split.add_argument("--no_stratify", action="store_true", help="Disable stratification during split.")
    parser_split.add_argument("--format", type=str, default="csv", choices=["csv", "tsv", "jsonl"], help="Input file format.")
    parser_split.add_argument("--text_col", type=int, default=config.TEXT_COLUMN_INDEX, help="Index of the text column.")
    parser_split.add_argument("--label_col", type=int, default=config.LABEL_COLUMN_INDEX, help="Index of the label column.")
    parser_split.add_argument("--no_header", action="store_true", help="Specify if input file has no header.")


    args = parser.parse_args()

    if args.command == "balance":
        print("--- Running Balance Data ---")
        create_balanced_subset(
            input_path=args.input,
            output_path=args.output,
            n_samples_per_class=args.num_samples,
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=not args.no_header,
            file_format=args.format
        )
    elif args.command == "split":
        print("--- Running Split Data ---")
        split_data(
            input_path=args.input,
            train_path=args.train_out,
            val_path=args.val_out,
            test_path=args.test_out,
            val_size=args.val_size,
            test_size=args.test_size,
            stratify=not args.no_stratify,
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=not args.no_header,
            file_format=args.format
        )
    else:
        parser.print_help()