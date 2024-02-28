# Call wit batch size as 1st argument, input_file WITHOUT extension as 2nd.

IN="$2.csv"  #chatgpt_text2shape_textembeds.csv
OUT="$2_batched.csv"   #chatgpt_text2shape_te_batched.csv

rm $OUT >/dev/null 
gcc -Wall make_batches.c -o make_batches
python3 sort.py $IN "$(echo $IN)_tmp"
./make_batches "$(echo $IN)_tmp" $OUT $(cat $IN | wc --lines) $1
python3 check.py $OUT $1
rm "$(echo $IN)_tmp"

