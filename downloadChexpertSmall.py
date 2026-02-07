import kagglehub

# Download latest version
path = kagglehub.dataset_download("ashery/chexpert")

print("Path to dataset files:", '~/med_img_xai')
#where it is downloaded:
'''ls ~/.cache/kagglehub/datasets/ashery/chexpert
'''