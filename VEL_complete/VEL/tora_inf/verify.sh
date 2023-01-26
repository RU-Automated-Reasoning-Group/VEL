python3="python3.7"
iters=$1
for ((i=1;i<=iters;i++));
do
    echo "+++++++++++++++ run $i begin +++++++++++++"
    time $python3 improve_toraeq.py --eval output.model
    echo "+++++++++++++++ run $i end +++++++++++++"
done
