@echo off
echo Federated Learning Visualization Pipeline
echo =======================================
echo.

echo Step 1: Installing dependencies...
python install_viz_deps.py
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies!
    exit /b %ERRORLEVEL%
)
echo.

echo Step 2: Generating visualizations...
python visualize_results.py
if %ERRORLEVEL% NEQ 0 (
    echo Error generating visualizations!
    exit /b %ERRORLEVEL%
)
echo.

echo All done! Visualizations are available in the 'plots' directory.
echo Open the files to view your results.

pause 