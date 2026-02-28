from infra_gis_detect import run_full_pipeline
import os

# Change this to your processed image
image_path = 'src/processed/PoleTag_25_processed.jpg'
# Name the output CSV based on the image name
base = os.path.splitext(os.path.basename(image_path))[0]
output_csv = f'{base}_gis.csv'

run_full_pipeline(image_path, output_csv)