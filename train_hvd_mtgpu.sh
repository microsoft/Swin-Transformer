horovodrun -np 1 python main_hvd_mtgpu.py --cfg configs/swinv2/swinv2_base_patch4_window8_128_hvd.yaml --data-path ../tiny-imagenet-200 --batch-size 4 --disable_amp --local_rank 0 --device mtgpu
