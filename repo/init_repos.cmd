@ECHO OFF

PUSHD %~dp0

REM QCNN official
git clone https://github.com/takh04/QCNN/tree/main/QCNN

REM low-rank approx for amp_enc
REM git clone https://github.com/qclib/qclib

POPD

ECHO Done!
ECHO.

PAUSE
