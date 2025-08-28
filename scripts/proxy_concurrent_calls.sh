#!/bin/bash
echo "=== Testing Proxy Concurrent Calls ==="

for i in {1..30}; do
  curl -s -X POST http://localhost:3000/embed \
    -H "Content-Type: application/json" \
     -d '{"inputs": ["Request '"$i"' - What is Vector search?"]}' > /dev/null &
done

# Wait for all background jobs to complete
wait

echo "All requests completed! (Check proxy logs to verify these are indeed handled by proxy)"

