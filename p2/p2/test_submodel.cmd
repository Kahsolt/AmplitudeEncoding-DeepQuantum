@ECHO OFF

python test_submodel.py -T 0,1
python test_submodel.py -T 0,2
python test_submodel.py -T 0,3
python test_submodel.py -T 0,4

python test_submodel.py -T 1,2
python test_submodel.py -T 1,3
python test_submodel.py -T 1,4

python test_submodel.py -T 2,3
python test_submodel.py -T 2,4

python test_submodel.py -T 3,4
