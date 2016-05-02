function rating_mat = readData(data_path, usr_num, movie_num)

fid = fopen(data_path, 'r');
formatSpec = '%u %u %u %u';
sizeA = [4, inf];
A = fscanf(fid, formatSpec, sizeA);

rating_mat = zeros(usr_num, movie_num);
for n = 1:size(A, 2)
    line = A(:, n);
    usr_id = uint32(line(1));
    movie_id = uint32(line(2));
    rating = line(3);
    rating_mat(usr_id, movie_id) = rating;
end




end