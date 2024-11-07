@ECHO OFF

SET METHOD=qcnn

python run_qcnn.py -M %METHOD% -T 0,1
python run_qcnn.py -M %METHOD% -T 0,2
python run_qcnn.py -M %METHOD% -T 0,3
python run_qcnn.py -M %METHOD% -T 0,4

python run_qcnn.py -M %METHOD% -T 1,0
python run_qcnn.py -M %METHOD% -T 1,2
python run_qcnn.py -M %METHOD% -T 1,3
python run_qcnn.py -M %METHOD% -T 1,4

python run_qcnn.py -M %METHOD% -T 2,0
python run_qcnn.py -M %METHOD% -T 2,1
python run_qcnn.py -M %METHOD% -T 2,3
python run_qcnn.py -M %METHOD% -T 2,4

python run_qcnn.py -M %METHOD% -T 3,0
python run_qcnn.py -M %METHOD% -T 3,1
python run_qcnn.py -M %METHOD% -T 3,2
python run_qcnn.py -M %METHOD% -T 3,4

python run_qcnn.py -M %METHOD% -T 4,0
python run_qcnn.py -M %METHOD% -T 4,1
python run_qcnn.py -M %METHOD% -T 4,2
python run_qcnn.py -M %METHOD% -T 4,3
