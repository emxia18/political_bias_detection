import kagglehub

download_location = "political_bias_detection/data"
path = kagglehub.dataset_download("surajkarakulath/labelled-corpus-political-bias-hugging-face", path=download_location)

print("Path to dataset files:", path)