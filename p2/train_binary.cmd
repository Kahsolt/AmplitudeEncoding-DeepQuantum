@ECHO OFF

:AB

python train_binary.py -T 0,1
python train_binary.py -T 0,2
python train_binary.py -T 0,3
python train_binary.py -T 0,4

python train_binary.py -T 1,2
python train_binary.py -T 1,3
python train_binary.py -T 1,4

python train_binary.py -T 2,3
python train_binary.py -T 2,4

python train_binary.py -T 3,4


:BA

python train_binary.py -T 1,0

python train_binary.py -T 2,0
python train_binary.py -T 2,1

python train_binary.py -T 3,0
python train_binary.py -T 3,1
python train_binary.py -T 3,2

python train_binary.py -T 4,0
python train_binary.py -T 4,1
python train_binary.py -T 4,2
python train_binary.py -T 4,3
