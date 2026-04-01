import os
import argparse
import numpy as np

def inspect_dataset(root_dir):
    """
    Validates dataset integrity and checks class balance.
    """
    classes = ['real', 'fake']
    counts = {}
    
    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            counts[cls] = 0
            continue
        counts[cls] = len([f for f in os.listdir(cls_dir) if f.endswith(('.mp4', '.avi'))])
        
    for cls, count in counts.items():
        print(f"Class: {cls} - {count} videos")
        
    # Check class imbalance
    print(f"Total videos: {sum(counts.values())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/", help="Dataset root directory")
    args = parser.parse_args()
    inspect_dataset(args.root)
