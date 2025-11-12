for dataset in hmdb dss coconut moses canopus msg combined
do
    mkdir /bigdat2/user/MBGen/data/fp2mol/$dataset/
    mkdir /bigdat2/user/MBGen/data/fp2mol/$dataset/preprocessed/
    mkdir /bigdat2/user/MBGen/data/fp2mol/$dataset/processed/
    mkdir /bigdat2/user/MBGen/data/fp2mol/$dataset/stats/
done

cd data_processing/
python build_fp2mol_datasets.py