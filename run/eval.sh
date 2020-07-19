data_path=/tudelft.net/staff-bulk/ewi/insy/MMC/xinsheng/data/coco/
save_root=outputs/L5
lr=0.001
wd=1e-4
batch_size=64
n_epochs=50
start_epoch=50
lr_decay=50
bce_weight=5
BK_train=0
BK=2
LAMDA=5
evaluation=True

python WordDisc.py --data_path $data_path \
--lr $lr \
--save_root $save_root \
--weight-decay $wd \
--batch_size $batch_size \
--epoch $n_epochs \
--lr_decay $lr_decay \
--LAMDA $LAMDA \
--start_epoch $start_epoch \
--evaluation $evaluation

                        
                               
