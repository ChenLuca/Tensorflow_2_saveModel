Make folder :
$ mkdir Save_model

Download dataset :
$ wget --no-check-certificate #     https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \ #     -O ./cats_and_dogs_filtered.zip

Then run :
$ python3 Build_Classificiation.py --Dataset ./DataSets/Cats_and_dogs/ --Num_Class 2 --Batch_size 128 --Pre_Train_Epochs 5 --Fine_Tuning_Epochs 10
