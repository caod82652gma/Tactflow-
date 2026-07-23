@echo off
set folder=C:\Users\FangYuxuan\Desktop\Vet6USB_demo_software_v1.14\test5
echo Deleting CSV files in test5...

for /l %%i in (0,1,4) do (
    if exist "%folder%\%%i\*.csv" (
        del "%folder%\%%i\*.csv"
    )
)

echo Done.
pause