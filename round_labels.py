import os
from pathlib import Path

def round_labels(labels_dir):
    """
    Round all label values to 4 decimal places.
    Keeps the first value (class ID) as-is and rounds the next 4 values.
    """
    labels_path = Path(labels_dir)
    
    # Process all .txt files in the labels directory
    for label_file in labels_path.glob("*.txt"):
        print(f"Processing {label_file.name}...")
        
        lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                
                if len(parts) >= 5:
                    # Keep class ID (first value) as-is
                    class_id = parts[0]
                    
                    # Round the next 4 values to 4 decimal places
                    rounded_values = [f"{float(val):.4f}" for val in parts[1:5]]
                    
                    # Reconstruct the line
                    new_line = f"{class_id} {' '.join(rounded_values)}"
                    lines.append(new_line)
                else:
                    # If line has fewer values, keep it as-is
                    lines.append(line.strip())
        
        # Write back to the file
        with open(label_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
        
        print(f"  ✓ {label_file.name} updated")
    
    print(f"\nDone! Processed {len(list(labels_path.glob('*.txt')))} files.")

if __name__ == "__main__":
    labels_dir = r"c:\Users\storo\IdeaProjects\AlbumentationsX\outputs\run_1\labels"
    round_labels(labels_dir)
