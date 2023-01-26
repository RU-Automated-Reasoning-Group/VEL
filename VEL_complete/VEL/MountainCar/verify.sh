iters=$1
python3="python3.7"
for ((i=1;i<=iters;i++));
do
    echo "+++++++++++++++ run $i begin +++++++++++++"
    time $python3 verify_mc.py output.model
    echo "+++++++++++++++ run $i end +++++++++++++"
done