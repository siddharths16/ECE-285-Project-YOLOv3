#/bin/sh

startvalue=224

for((i=$startvalue;i<=610;i++)) do
	#python test.py ctdet --exp_id dla_test_$i --dataset pascal --load_model ../exp/ctdet/default_dla_384/model_best.pth --input_res $i --flip_test  > dla_test_$i
	python test.py ctdet --exp_id  pascal_resdcn101_no_dcn_Dec_7  --arch resdcn_101 --input_res $i --resume --dataset pascal --flip_test > resdcn101_no_dcn_Dec7_$i
 	i=($i+31)
done
