#!/bin/bash

# Define the executable
EXEC="./exponentialIntegral.out"

# Define the test sizes
sizes=(
  "5000"
  "8192"
  "16384"
  "20000"
)

# Output file
OUTFILE="test_results.log"
> "$OUTFILE"

# Loop over sizes and run tests
for size in "${sizes[@]}"; do
  echo "=============================================" | tee -a "$OUTFILE"
  echo "Running test with -n $size -m $size" | tee -a "$OUTFILE"
  echo "=============================================" | tee -a "$OUTFILE"
  
  $EXEC -n "$size" -m "$size" -t 2>&1 | tee -a "$OUTFILE"
  
  echo -e "\n\n" | tee -a "$OUTFILE"
done

echo "All tests completed. Results saved in $OUTFILE"
