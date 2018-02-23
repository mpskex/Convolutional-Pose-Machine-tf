function cache_annotations(annolist, rectidxs, root)
    ksz = 16;
    f_train_list = fopen('train_list.txt', 'w+');
    %   caching
    for imgidx = 1:length(annolist)
        %   judge whether it is empty in rect you have just extract
        if ~isempty(rectidxs(imgidx))
            string2file = '';
            %   avaliable rect id
            ridx_list = rectidxs(imgidx);
            %   every rect in single annotation
            annotation = annolist(imgidx).annorect;
            if isempty(cell2mat(ridx_list))
                continue;
            else
                for idx = 1:length(ridx_list)
                    ridx = cell2mat(ridx_list(idx));
                    if (isfield(annotation(ridx(idx)),'annopoints') ...
                        && ~isempty(annotation(ridx(idx)).annopoints) ...
                        && isfield(annotation(ridx(idx)).annopoints,'point') ...
                        && ~isempty(annotation(ridx(idx)).annopoints.point) ...
                        )
                        %   get points   
                        points = annotation(ridx(idx)).annopoints.point;
                        for kidx = 0:ksz-1
                            p = util_get_annopoint_by_id(points,kidx);
                            if (~isempty(p))
                                string2file = strcat(string2file, sprintf('%.3f\t%.3f', p.x, p.y), '\t');
                            else
                                %fprintf('warning: empty Point!\n');
                                %   fill with nan
                                string2file = strcat(string2file, sprintf('nan\tnan'), '\t');
                            end
                        end
                        string2file = strcat(string2file, '\r\n');
                    end
                end
                file_path = strcat(root, annolist(imgidx).image.name, '.txt');
                fid = fopen(file_path, 'w+');
                fprintf(fid, string2file);
                fclose(fid);
                fprintf(f_train_list, '%s\r\n', annolist(imgidx).image.name);
            end
        end
    end
    fclose(f_train_list);
end
