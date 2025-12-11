import splitfolders # type: ignore

# Input path: where you extracted the Kaggle dataset
# Output path: where the split data will be saved
input_folder = "dataset" 
output_folder = "split_dataset"

# Split with a ratio. To only split into training and validation, set test_probs to 0.
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .1, .1), group_prefix=None)
