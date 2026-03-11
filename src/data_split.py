import splitfolders

# Input raw data folder (where all disease class folders are present)
input_folder = "../data/raw"

# Output folder (train/val/test will be created here)
output_folder = "../data"

# Split ratio (70% train, 15% val, 15% test)
splitfolders.ratio(input_folder, 
                   output=output_folder, 
                   seed=42, 
                   ratio=(0.7, 0.15, 0.15))
