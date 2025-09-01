# loop over 1 to 7 and run main for each nn_it_x
for i in {1..7}
do
  python -m main --train --training_cfg initial_testing/nn-it-$i
done