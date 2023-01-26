iters=$1
for ((i=1;i<=iters;i++));
do
    echo "+++++++++++++++ run $i begin +++++++++++++"
    time ./pendulum $(cat output.model)
    echo "+++++++++++++++ run $i end +++++++++++++"
done
