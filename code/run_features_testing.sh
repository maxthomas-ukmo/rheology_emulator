# loop over 1 to 7 and run main for each nn_it_x
for i in {1..5}
do
  python -m main --train --training_cfg nn_ft_$i
done
