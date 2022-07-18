python -W ignore test.py \
  --dataset pascal --data-root /mnt/bd/det-qi/data/VOC2012 \
  --backbone resnet101 --fold 0 --shot 5 --refine
python -W ignore test.py \
  --dataset pascal --data-root /mnt/bd/det-qi/data/VOC2012 \
  --backbone resnet101 --fold 1 --shot 5 --refine
python -W ignore test.py \
  --dataset pascal --data-root /mnt/bd/det-qi/data/VOC2012 \
  --backbone resnet101 --fold 2 --shot 5 --refine
python -W ignore test.py \
  --dataset pascal --data-root /mnt/bd/det-qi/data/VOC2012 \
  --backbone resnet101 --fold 3 --shot 5 --refine

