@ECHO OFF

:AB

python test_binary.py -T 0,1 > output\bin_0-1\test.log
python test_binary.py -T 0,2 > output\bin_0-2\test.log
python test_binary.py -T 0,3 > output\bin_0-3\test.log
python test_binary.py -T 0,4 > output\bin_0-4\test.log

python test_binary.py -T 1,2 > output\bin_1-2\test.log
python test_binary.py -T 1,3 > output\bin_1-3\test.log
python test_binary.py -T 1,4 > output\bin_1-4\test.log

python test_binary.py -T 2,3 > output\bin_2-3\test.log
python test_binary.py -T 2,4 > output\bin_2-4\test.log

python test_binary.py -T 3,4 > output\bin_3-4\test.log


:BA

python test_binary.py -T 1,0 > output\bin_1-0\test.log

python test_binary.py -T 2,0 > output\bin_2-0\test.log
python test_binary.py -T 2,1 > output\bin_2-1\test.log

python test_binary.py -T 3,0 > output\bin_3-0\test.log
python test_binary.py -T 3,1 > output\bin_3-1\test.log
python test_binary.py -T 3,2 > output\bin_3-2\test.log

python test_binary.py -T 4,0 > output\bin_4-0\test.log
python test_binary.py -T 4,1 > output\bin_4-1\test.log
python test_binary.py -T 4,2 > output\bin_4-2\test.log
python test_binary.py -T 4,3 > output\bin_4-3\test.log
