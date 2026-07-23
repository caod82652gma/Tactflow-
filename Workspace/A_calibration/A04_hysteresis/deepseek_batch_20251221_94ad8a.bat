@echo off
set folder=C:\Users\FangYuxuan\Desktop\Vet6USB_demo_software_v1.14\test4
echo Deleting CSV files in test4...

for /l %%i in (0,1,10) do (
    if exist "%folder%\%%i\*.csv" (
        del "%folder%\%%i\*.csv"
    )
)

echo Done.
pause