@echo off
TITLE LUMINARK ANTIKYTHERA - TEST SUITE

echo ========================================================
echo   LUMINARK API SMOKE TEST
echo   Target: http://localhost:8000
echo ========================================================
echo.

echo [1/3] Testing Health Endpoint...
curl -s http://localhost:8000/
echo.
echo.

echo [2/3] Testing Deep Analysis (Quantum Brain)...
curl -s -X POST "http://localhost:8000/analyze" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"test-1\",\"spat_vectors\":{\"complexity\":8.0,\"stability\":2.0,\"tension\":9.0,\"adaptability\":7.0,\"coherence\":5.0},\"life_vector\":\"career\",\"temporal_sentiment\":{\"future\":\"anxious\",\"past\":\"heavy\"}}"
echo.
echo.

echo [3/3] Check completed. If JSON output appears above, system is GREEN.
echo.
pause
