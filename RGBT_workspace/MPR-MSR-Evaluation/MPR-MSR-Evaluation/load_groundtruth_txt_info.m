 function [ ground_truth] = load_groundtruth_txt_info(gt_file_path)
%LOAD_TXT_INFO

	%try to load ground truth from text file (Benchmark's format)
	filename = gt_file_path;
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]
	try
		ground_truth = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  %#ok, try different format (no commas)
		frewind(f);
		ground_truth = textscan(f, '%f %f %f %f');  
	end
% 	ground_truth = cat(2, ground_truth{:});
    ground_truth{1,3} = ground_truth{1,1}+ground_truth{1,3};
    ground_truth{1,4} = ground_truth{1,2}+ground_truth{1,4};
    ground_truth = cat(2, ground_truth{:});
    
	fclose(f);
	
end

