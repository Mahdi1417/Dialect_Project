import os

def merge_folders(folder_a, folder_b, output_folder, separator = "================================================================================="):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = sorted([f for f in os.listdir(folder_a) if f.endswith(".txt")],
                   key=lambda x: int(x.replace(".txt","")))

    for fname in files:
        path_a = os.path.join(folder_a, fname)
        path_b = os.path.join(folder_b, fname)

        if not os.path.exists(path_b):
            print(f"WARNING: {path_b} missing, skipping")
            continue

        # -------------------------
        # Read files
        # -------------------------
        text_a = open(path_a, "r", encoding="utf-8").read()
        text_b = open(path_b, "r", encoding="utf-8").read()

        # -------------------------
        # Split by separator
        # -------------------------
        parts_a = text_a.split(separator)
        parts_b = text_b.split(separator)

        if len(parts_a) != 3:
            print(f"ERROR: {path_a} does not contain exactly 2 separators")
            continue

        if len(parts_b) != 3:
            print(f"ERROR: {path_b} does not contain exactly 2 separators")
            continue

        A1, A2, A3 = parts_a
        B1, B2, B3 = parts_b  # we only need B2

        # -------------------------
        # Create merged file
        # -------------------------
        merged = A1.strip() + "\n" + separator + "\n" \
               + B2.strip() + "\n" + separator + "\n" \
               + A3.strip()

        # Save to output
        out_path = os.path.join(output_folder, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(merged)

        print(f"Created {out_path}")



# Example usage:
merge_folders("zs_preds/Q8B_irab_ar", "3s_preds/Q8B_irab_ar", "zs_preds2/Q8B_irab_ar", "=================================================================================")
merge_folders("zs_preds/Q14B_irab_ar", "3s_preds/Q14B_irab_ar", "zs_preds2/Q14B_irab_ar", "=================================================================================")