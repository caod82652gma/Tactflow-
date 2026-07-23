@echo off
set folder=C:\Users\FangYuxuan\Desktop\Vet6USB_demo_software_v1.14\test2
echo Deleting CSV files...

for /l %%i in (0,1,10) do (
    if exist "%folder%\%%iN\*.csv" (
        del "%folder%\%%iN\*.csv"
    )
)

echo Done.
pause