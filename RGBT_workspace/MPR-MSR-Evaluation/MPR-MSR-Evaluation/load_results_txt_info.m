 function [ track_result ] = load_results_txt_info(video_path)
%LOAD_TXT_INFO




	%try to load ground truth from text file (Benchmark's format)
%     filename = [base_path video suffix '_OurScale.txt'];
    filename = ['./',video_path];
	
	f = fopen(filename);
	assert(f ~= -1, ['No initial position or ground truth to load ("' filename '").'])
	
	%the format is [x, y, width, height]
	try
		track_result = textscan(f, '%f,%f,%f,%f', 'ReturnOnError',false);  
	catch  %#ok, try different format (no commas)
		frewind(f);
        track_result = textscan(f, '%f %f %f %f'); 
%         str = fgetl(f); 
%         track_result = textscan(str,'%f');
%         str = track_result{1}';
	end
	track_result = cat(2, track_result{:});
	fclose(f);
	
end

