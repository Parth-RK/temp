# --- dataman.py ---
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import sys

# Attempt to import config for defaults, provide fallbacks if it fails
try:
    import config
except ImportError:
    print("Warning: config.py not found. Using hardcoded defaults in dataman.")
    # Define minimal defaults if config cannot be imported
    class ConfigFallback:
        DATA_DIR = "." # Use current directory as default data dir
        # Assume standard 'text', 'label' columns if config is missing
        TEXT_COLUMN_INDEX = 0 # Default if no header/names from config
        LABEL_COLUMN_INDEX = 1 # Default if no header/names from config
        COLUMN_NAMES = ['text', 'label'] # Default if no header and config missing
        HAS_HEADER = True # Default assumption
        SEED = 42
        # Define INPUT_FILE_PATH based on common names if config unavailable
        if os.path.exists("training.csv"):
             INPUT_FILE_PATH = "training.csv"
        elif os.path.exists("data.csv"):
             INPUT_FILE_PATH = "data.csv"
        else:
             INPUT_FILE_PATH = None # Cannot determine default input
    config = ConfigFallback()
except Exception as e:
     print(f"Warning: Error importing config.py: {e}. Using hardcoded defaults in dataman.")
     config = ConfigFallback() # Use fallback


def _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format):
    """Helper to load data based on format and standardize columns."""
    print(f"Loading data from: {input_path} (Format: {file_format})")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return None

    try:
        read_opts = {'on_bad_lines': 'warn', 'low_memory': False}
        if file_format == "csv":
            header = 0 if has_header else None
            # Provide default names only if no header AND col_names is None (use config default/fallback)
            names = None if has_header else (col_names if col_names else ConfigFallback.COLUMN_NAMES)
            df = pd.read_csv(input_path, header=header, names=names, **read_opts)
        elif file_format == "tsv":
            header = 0 if has_header else None
            names = None if has_header else (col_names if col_names else ConfigFallback.COLUMN_NAMES)
            df = pd.read_csv(input_path, sep='\t', header=header, names=names, **read_opts)
        elif file_format == "jsonl":
            df = pd.read_json(input_path, lines=True)
            # JSONL often lacks headers, column order might vary. Indices are crucial.
            has_header = False # Assume no standard header row for selection logic
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Select and rename columns based on indices provided
        num_cols = len(df.columns)
        if text_col_idx >= num_cols or label_col_idx >= num_cols:
             raise IndexError(f"Column index out of bounds (Text: {text_col_idx}, Label: {label_col_idx}). File '{os.path.basename(input_path)}' has {num_cols} columns: {list(df.columns)}")

        text_col_name = df.columns[text_col_idx]
        label_col_name = df.columns[label_col_idx]

        # Create new DataFrame with standardized names
        df_std = pd.DataFrame({
            'label': df[label_col_name],
            'text': df[text_col_name]
        })
        df_std = df_std.dropna(subset=['label', 'text']).reset_index(drop=True)
        df_std['text'] = df_std['text'].astype(str) # Ensure text is string

        print(f"Loaded {len(df_std)} rows (after dropping NaNs).")
        print(f"Using columns: label='{label_col_name}' (idx {label_col_idx}), text='{text_col_name}' (idx {text_col_idx})")
        return df_std

    except FileNotFoundError:
        # Should be caught above, but for safety
        print(f"Error: Input file not found at {input_path}")
        return None
    except IndexError as e:
         print(f"Error: Column index out of bounds. Check --text_col ({text_col_idx}) and --label_col ({label_col_idx}) for file {input_path}. Details: {e}")
         return None
    except Exception as e:
        print(f"Error loading data from {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_balanced_subset(input_path, output_path, n_samples_per_class,
                           text_col_idx=config.TEXT_COLUMN_INDEX,
                           label_col_idx=config.LABEL_COLUMN_INDEX,
                           col_names=config.COLUMN_NAMES, # Use config default/fallback
                           has_header=config.HAS_HEADER, # Use config default/fallback
                           file_format="csv"):
    """
    Creates a balanced dataset subset with n samples from each label category.

    Args:
        input_path (str): Path to the input data file.
        output_path (str): Path to save the balanced dataset.
        n_samples_per_class (int): Number of samples per label.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header (overrides config).
        has_header (bool): If the file has a header (overrides config).
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    # Use provided args if they differ from defaults, otherwise use config/fallback values
    current_text_col = text_col_idx
    current_label_col = label_col_idx
    current_has_header = has_header
    current_col_names = col_names # May be None

    df = _load_data(input_path, current_text_col, current_label_col, current_col_names, current_has_header, file_format)
    if df is None:
        return

    print(f"\nOriginal dataset shape (after load & NaN drop): {df.shape}")

    # Ensure label column is treated as string for value counts and grouping
    df['label'] = df['label'].astype(str)

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
            print(f"  Warning: Label '{label}' has only {available_samples} samples (requested {n_samples_per_class}). Taking all {sample_size} available.")
        elif sample_size == 0:
             print(f"  Warning: Label '{label}' has 0 samples. Skipping.")
             continue # Skip labels with no samples

        # Use replace=False for sampling without replacement
        sampled_df = label_df.sample(n=sample_size, random_state=config.SEED, replace=False)
        balanced_dfs.append(sampled_df)

    if not balanced_dfs:
        print("Error: No data collected for balancing. Check input data and parameters.")
        return

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True) # Shuffle rows

    print(f"\nBalanced subset shape: {balanced_df.shape}")
    print("Balanced Subset Label Distribution:")
    print(balanced_df['label'].value_counts())

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
        # Save with header using standard column names 'label', 'text'
        balanced_df.to_csv(output_path, index=False, header=True)
        print(f"\nBalanced subset saved to {output_path}")
    except Exception as e:
        print(f"Error saving balanced subset: {e}")

def split_data(input_path, train_path, val_path, test_path,
               val_size=0.15, test_size=0.15, stratify=True,
               text_col_idx=config.TEXT_COLUMN_INDEX,
               label_col_idx=config.LABEL_COLUMN_INDEX,
               col_names=config.COLUMN_NAMES, # Use config default/fallback
               has_header=config.HAS_HEADER, # Use config default/fallback
               file_format="csv"):
    """
    Splits the data into train, validation, and test sets.

    Args:
        input_path (str): Path to the input data file.
        train_path (str): Path to save the training set.
        val_path (str): Path to save the validation set.
        test_path (str): Path to save the test set.
        val_size (float): Proportion for validation set (from original).
        test_size (float): Proportion for test set (from original).
        stratify (bool): Whether to stratify based on labels.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header (overrides config).
        has_header (bool): If the file has a header (overrides config).
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    # Use provided args if they differ from defaults, otherwise use config/fallback values
    current_text_col = text_col_idx
    current_label_col = label_col_idx
    current_has_header = has_header
    current_col_names = col_names # May be None

    df = _load_data(input_path, current_text_col, current_label_col, current_col_names, current_has_header, file_format)
    if df is None:
        return

    print(f"Total data for splitting (after load & NaN drop): {len(df)} rows")

    if len(df) < 3:
        print("Error: Not enough data (less than 3 rows) to perform train/val/test split.")
        return

    if (val_size + test_size) >= 1.0:
        print(f"Error: Sum of validation ({val_size}) and test ({test_size}) sizes must be less than 1.0")
        return
    if val_size < 0 or test_size < 0:
         print(f"Error: Validation ({val_size}) and test ({test_size}) sizes must be non-negative.")
         return


    # Ensure label column is suitable for stratification if requested
    stratify_col = None
    if stratify:
        # Convert label to string for robust stratification, handle potential errors
        try:
            df['label_str'] = df['label'].astype(str)
            # Check if stratification is possible (at least 2 samples per class, or at least 1 sample if n_splits=1)
            label_counts = df['label_str'].value_counts()
            if any(count < 2 for count in label_counts):
                 print("Warning: Stratification may not be possible due to classes with fewer than 2 samples. Attempting anyway, but sklearn might raise an error or fallback.")
                 # Sklearn handles this internally in recent versions, often with a warning.
            stratify_col = df['label_str']
        except Exception as e:
             print(f"Warning: Could not prepare label column for stratification ({e}). Stratification disabled.")
             stratify = False # Disable stratification


    # --- Splitting Logic ---
    # 1. Split off Test set first
    train_val_df = df
    test_df = pd.DataFrame(columns=df.columns) # Initialize empty
    if test_size > 0:
        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=config.SEED,
                stratify=stratify_col
            )
        except ValueError as e:
             # This might happen if stratification fails (e.g., single sample classes)
             print(f"Warning: Stratified split for test set failed ({e}). Performing non-stratified split.")
             train_val_df, test_df = train_test_split(
                 df, test_size=test_size, random_state=config.SEED, stratify=None)
    else:
        print("Test size is 0. Test set will be empty.")
        # train_val_df remains the full dataset (minus label_str if added)


    # 2. Split remaining into Train and Validation
    train_df = train_val_df
    val_df = pd.DataFrame(columns=df.columns) # Initialize empty
    if val_size > 0 and len(train_val_df) > 0:
        # Adjust val_size relative to the *remaining* data after test split
        # Avoid division by zero if original df had size 0 or test_size was 1
        denominator = (1.0 - test_size)
        if denominator > 0 and len(train_val_df) >= 2: # Need at least 2 samples to split further
            relative_val_size = val_size / denominator
            # Ensure relative size is valid (e.g., not > 1)
            relative_val_size = min(max(0.0, relative_val_size), 1.0 - (1 / len(train_val_df)))

            if relative_val_size > 0:
                 # Prepare stratification column for the train_val split
                 stratify_col_train_val = None
                 if stratify and 'label_str' in train_val_df:
                     # Check stratification possibility for the remaining data
                      label_counts_tv = train_val_df['label_str'].value_counts()
                      if any(count < 2 for count in label_counts_tv):
                           print("Warning: Stratification for validation split might fail (classes < 2 samples). Attempting anyway.")
                      stratify_col_train_val = train_val_df['label_str']

                 try:
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=relative_val_size,
                        random_state=config.SEED,
                        stratify=stratify_col_train_val
                    )
                 except ValueError as e:
                    print(f"Warning: Stratified split for validation set failed ({e}). Performing non-stratified split.")
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=relative_val_size, random_state=config.SEED, stratify=None)
            else:
                print("Calculated relative validation size is 0. Validation set will be empty.")
                train_df = train_val_df # All remaining is train
        elif len(train_val_df) < 2:
             print("Not enough data remaining after test split to create validation split. Validation set will be empty.")
             train_df = train_val_df
        else:
            print("Validation size is 0 or cannot split further. Validation set will be empty.")
            train_df = train_val_df # All remaining is train
    else:
         print("Validation size is 0. Validation set will be empty.")
         train_df = train_val_df # All remaining is train


    # Remove temporary stratification column if it exists
    if 'label_str' in train_df.columns: train_df = train_df.drop(columns=['label_str'])
    if 'label_str' in val_df.columns: val_df = val_df.drop(columns=['label_str'])
    if 'label_str' in test_df.columns: test_df = test_df.drop(columns=['label_str'])


    print(f"\nSplit complete:")
    print(f"  Train set size:      {len(train_df)}")
    print(f"  Validation set size: {len(val_df)}")
    print(f"  Test set size:       {len(test_df)}")
    print(f"  (Total rows assigned: {len(train_df) + len(val_df) + len(test_df)} / Original: {len(df)})")


    # --- Saving Logic ---
    try:
        # Save only the 'label' and 'text' columns with standard headers
        cols_to_save = ['label', 'text']
        for pth, dframe in [(train_path, train_df), (val_path, val_df), (test_path, test_df)]:
            out_dir = os.path.dirname(pth)
            if out_dir: # Ensure output directory exists
                os.makedirs(out_dir, exist_ok=True)

            if dframe is not None and not dframe.empty:
                 # Select standard columns, ensure they exist
                 if all(col in dframe.columns for col in cols_to_save):
                     dframe_to_save = dframe[cols_to_save]
                     dframe_to_save.to_csv(pth, index=False, header=True)
                     print(f"  Saved {os.path.basename(pth)} ({len(dframe_to_save)} rows)")
                 else:
                      print(f"  Warning: Could not save {os.path.basename(pth)}. Missing required columns 'label' or 'text'.")
            else:
                 # Optionally create an empty file with header for consistency?
                 print(f"  Skipping save for empty dataset: {os.path.basename(pth)}")
                 # pd.DataFrame(columns=cols_to_save).to_csv(pth, index=False, header=True) # Uncomment to save empty file

        print("\nData splitting and saving finished.")
    except Exception as e:
        print(f"Error saving split files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Manipulation Utility (Manual Use)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Common Arguments ---
    common_parser = argparse.ArgumentParser(add_help=False)
    # Input File (optional, try to use config default)
    default_input = config.INPUT_FILE_PATH if hasattr(config, 'INPUT_FILE_PATH') and config.INPUT_FILE_PATH else None
    input_help = "Path to the input data file." + (f" (Default from config: {default_input})" if default_input else " (Default: try 'training.csv' or 'data.csv')")
    common_parser.add_argument("-i", "--input", type=str, default=default_input, help=input_help)
    common_parser.add_argument("--format", type=str, default="csv", choices=["csv", "tsv", "jsonl"], help="Input file format (Default: csv).")
    # Use config defaults for column indices/header, allow override
    common_parser.add_argument("--text_col", type=int, default=config.TEXT_COLUMN_INDEX, help=f"Index of the text column (Default: {config.TEXT_COLUMN_INDEX}).")
    common_parser.add_argument("--label_col", type=int, default=config.LABEL_COLUMN_INDEX, help=f"Index of the label column (Default: {config.LABEL_COLUMN_INDEX}).")
    # HAS_HEADER from config controls default behavior; --no_header overrides it
    header_action = 'store_false' if config.HAS_HEADER else 'store_true'
    header_default = config.HAS_HEADER
    common_parser.add_argument("--header", action=header_action, default=header_default, help=f"Specify if input file has a header row (Default: {config.HAS_HEADER}). Use --no-header to disable if default is True, or --header to enable if default is False.")


    # --- Balance Subcommand ---
    parser_balance = subparsers.add_parser("balance", help="Create a balanced subset of the data.", parents=[common_parser])
    parser_balance.add_argument("-o", "--output", type=str, required=True, help="Path to save the balanced output file (CSV format).")
    parser_balance.add_argument("-n", "--num_samples", type=int, required=True, help="Number of samples per class.")

    # --- Split Subcommand ---
    parser_split = subparsers.add_parser("split", help="Split data into train, validation, and test sets.", parents=[common_parser])
    parser_split.add_argument("--train_out", type=str, required=True, help="Path to save the training set (CSV format).")
    parser_split.add_argument("--val_out", type=str, required=True, help="Path to save the validation set (CSV format).")
    parser_split.add_argument("--test_out", type=str, required=True, help="Path to save the test set (CSV format).")
    parser_split.add_argument("--val_size", type=float, default=0.15, help="Validation set proportion (from original data) (Default: 0.15).")
    parser_split.add_argument("--test_size", type=float, default=0.15, help="Test set proportion (from original data) (Default: 0.15).")
    parser_split.add_argument("--no_stratify", action="store_true", default=False, help="Disable stratification during split (Default: Stratify).")


    args = parser.parse_args()

    # Ensure input file is specified if default couldn't be found
    if not args.input:
         parser.error("Input file path (-i/--input) is required as no default could be determined.")

    # Handle header logic based on action ('store_false' means header=False if flag is present)
    has_header = args.header # This now correctly reflects the presence/absence of the flag relative to the default

    if args.command == "balance":
        print("--- Running Balance Data ---")
        create_balanced_subset(
            input_path=args.input,
            output_path=args.output,
            n_samples_per_class=args.num_samples,
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=has_header, # Use processed value
            file_format=args.format,
            col_names=config.COLUMN_NAMES # Pass default names from config/fallback
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
            stratify=not args.no_stratify, # Stratify unless --no_stratify is given
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=has_header, # Use processed value
            file_format=args.format,
            col_names=config.COLUMN_NAMES # Pass default names from config/fallback
        )
    else:
        parser.print_help()
