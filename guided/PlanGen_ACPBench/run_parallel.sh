#!/bin/bash
# Run re-search for domains in parallel
echo "Starting parallel search..."

python3 rerun_search.py hanoi --timeout 60 --strategy greedy_bfs &
PID1=$!
echo "Started Hanoi search (PID $PID1)"

python3 rerun_search.py logistics --timeout 60 --strategy greedy_bfs &
PID2=$!
echo "Started Logistics search (PID $PID2)"

python3 rerun_search.py alfworld --timeout 60 --strategy greedy_bfs &
PID3=$!
echo "Started Alfworld search (PID $PID3)"

wait $PID1
echo "Hanoi done."
wait $PID2
echo "Logistics done."
wait $PID3
echo "Alfworld done."

echo "All searches done. Updating summaries..."
python3 src/results.py --split test --data_root data/test_baseline --out search_result_test_domain.json --summary_out accuracy_summary_test_domain.json --dom_summary_out search_accuracy_test_domain.json
echo "Done."
