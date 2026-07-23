@echo off
set folder=C:\Users\FangYuxuan\Desktop\Vet6USB_demo_software_v1.14\test3
echo Deleting CSV files in test3...

for /l %%i in (0,1,10) do (
    if exist "%folder%\%%i\*.csv" (
        del "%folder%\%%i\*.csv"
    )
)

echo Done.
pause