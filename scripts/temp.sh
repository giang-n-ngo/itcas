# cancel all running jobs within an ID range
# Usage: ./temp.sh <start_id> <end_id>
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_id> <end_id>"
    exit 1
fi

start_id=$1
end_id=$2

for job_id in $(seq "$start_id" "$end_id"); do
    scancel "$job_id"
done