iters=$1
for ((i=1;i<=iters;i++));
do
    echo "+++++++++++++++ run $i begin +++++++++++++"
    time ./B2 $(cat output.model)
    echo "+++++++++++++++ run $i end +++++++++++++"
done
