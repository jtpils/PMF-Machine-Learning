function I = sampleData(data_path, usr_num, movie_num, list)

fid = fopen(data_path, 'r');
formatSpec = '%u %u %u %u';
sizeA = [4, inf];
A = fscanf(fid, formatSpec, sizeA);
I = zeros(usr_num, movie_num);

for n = 1:size(list, 2)
    sample = list(1, n);
    line = A(:, sample);
    usr_id = uint32(line(1));
    movie_id = uint32(line(2));
    I(usr_id, movie_id) = 1;
end

end