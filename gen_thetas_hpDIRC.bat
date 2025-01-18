@echo off

REM Initialize theta value
set theta=30
set momentum=6
set model_type="CNF"

REM Loop over theta values from 30 to 150 in steps of 5
:loop
if %theta% LEQ 150 (
    REM Run the Python script for "Pion"
    python generate_fixedpoint_hpDIRC.py ^
        --config config/hpDIRC_config_clean.json ^
        --momentum %momentum% ^
        --theta %theta% ^
        --method "Pion" ^
        --model_type %model_type%

    REM Run the Python script for "Kaon"
    python generate_fixedpoint_hpDIRC.py ^
        --config config/hpDIRC_config_clean.json  ^
        --momentum %momentum% ^
        --theta %theta% ^
        --method "Kaon" ^
        --model_type %model_type%

    REM Increment theta by 5
    set /a theta=theta+5

    REM Go back to the start of the loop
    goto loop
)

REM # Make plots.
python make_plots.py --config config/hpDIRC_config_clean.json --momentum %momentum%