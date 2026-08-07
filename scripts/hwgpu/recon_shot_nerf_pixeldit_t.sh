for ((i = 100; i <=2000; i += 100)); do
	j=$(printf '%05d' "$i")
	echo "processing epoch = $j"
	/hwdata/24ydz3d/deeplearning/environments/py313/bin/python3 /hwdata/24ydz3d/deeplearning/SeisFlow-main/recon_shot_dataset2.py \
		--input_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset128/valid_pixeldit_nerf_t_i128_epoch__$j/0/ \
		--input_aux_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset128/valid_aux/ \
		--output_dir /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset128/valid_recon_shot_nerf_piexldit_t_epoch_$j/

	/hwdata/24ydz3d/deeplearning/environments/py313/bin/python3 /hwdata/24ydz3d/deeplearning/SeisFlow-main/diff_shot.py \
		--input1 /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset128/shot/ \
		--input2 /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset128/valid_recon_shot_nerf_piexldit_t_epoch_$j/ \
		--output /hwdata/24ydz3d/deeplearning/SeisFlow-main/shot_dataset128/diff_recon_shot_nerf_piexldit_t_epoch_$j/
done
