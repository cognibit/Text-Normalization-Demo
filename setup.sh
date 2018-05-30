#!/bin/sh
echo "Downloading and extracting required files"
wget https://storage.googleapis.com/ainstein_text_normalization/test_data.zip
wget https://storage.googleapis.com/ainstein_text_normalization/dnc_model.zip
rm -rf data
rm -rf models
unzip test_data.zip
unzip dnc_model.zip
rm test_data.zip 
rm dnc_model.zip
echo "Finished"
    