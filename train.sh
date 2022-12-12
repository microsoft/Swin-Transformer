horovodrun -np 4 python main_hvd.py --cfg configs/swinv2/swinv2_base_patch4_window8_128_hvd.yaml --data-path ../../datasets/tiny-imagenet-200 --batch-size 64 --disable_amp --local_rank  0
