# script adapted from MIST repo: https://github.com/samgoldman97/mist/blob/main_v2/data_processing/canopus_train/00_download_canopus_data.sh
# This script downloads preprocessed data from the MassSpecGym project
# Original MassSpecGym code/data: https://github.com/pluskal-lab/MassSpecGym

export_link="https://zenodo.org/records/15008938/files/msg_preprocessed.tar.gz"

# mkdir data/
# cd data/

DATA_DIR="/bigdat2/user/MBGen/data"
mkdir -p $DATA_DIR
cd $DATA_DIR

wget $export_link

tar -xvzf msg_preprocessed.tar.gz

rm -f msg_preprocessed.tar.gz