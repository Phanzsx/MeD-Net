clear
modefile = dir('E:\work\meli\origen_data\data_mat\');
fi = 1;
for ii = 3:length(modefile)
    idfile = dir([modefile(1).folder,'\',modefile(ii).name]);
    for i = 3:length(idfile)
        timefile = dir([idfile(1).folder,'\',idfile(i).name]);
        for j = 3:length(timefile)
            try
            load([timefile(1).folder,'\',timefile(j).name,'\data.mat']);
            img_o = img;
            [x,y,z] = size(img_o);
            %lesion
            lesionmsk = zeros(x,y,z);
            lesionmsk(lesionprob>=0.5) = 1;
            for k = 1:z
                lesionmsk_single = lesionmsk(:,:,k);
                lesionmsk(:,:,k) = bwareaopen(lesionmsk_single,9);
            end
            %lung
            lungmsk = lungmsk+lesionmsk;
            lungmsk(lungmsk>1) = 1;
            L = bwconncomp(lungmsk,26);
            pixelidxlist = L.PixelIdxList;
            num_conn = [];
            for l = 1:length(pixelidxlist)
                num_conn = [num_conn;length(pixelidxlist{1,l})];
            end
            if isempty(num_conn)
                continue;
            end
            [~,index] = sort(num_conn,'descend');
            segx = []; segy = []; segz = [];
            %choose one/two maximum connected area
            for l = 1:min(length(index),2)
                pixelidx = pixelidxlist{1,index(l)};
                [segx0,segy0,segz0] = ind2sub([512,512,z],pixelidx);
                segx = [segx;segx0]; segy = [segy;segy0]; segz = [segz;segz0];
            end
            segx1 = min(segx); segx2 = max(segx);
            segy1 = min(segy); segy2 = max(segy);
            segz1 = min(segz); segz2 = max(segz);
            %make data
            img = img_o(segx1:segx2,segy1:segy2,segz1:segz2);
            lesionmsk = lesionmsk(segx1:segx2,segy1:segy2,segz1:segz2);
            [img,position] = mgrid(img,lesionmsk,16,48);
            path = ['E:\work\meli\data_48patch_16stride_n\',modefile(ii).name,...
                '\',idfile(i).name,'\',timefile(j).name,'\'];
            if ~isdir(path) mkdir(path); end
            save([path,'data.mat'],'img','position');
            %make single
            path = ['E:\work\meli\data_single_n\',modefile(ii).name,...
                '\',idfile(i).name,'\',timefile(j).name,'\'];
            if ~isdir(path) mkdir(path); end
            area = squeeze(sum(lesionprob,[1,2]));
            [~,index] = sort(area,'descend');
            save([path,'index.mat'],'index');
            for l = 1:9
                img_s = imresize(img_o(:,:,index(l)),[256,256]);
                save([path,mat2str(index(l))],'img_s');
            end
            disp([modefile(ii).name,'\',idfile(i).name,'\',timefile(j).name]);
            catch exception
                disp(['Fault:',modefile(ii).name,'\',...
                    idfile(i).name,'\',timefile(j).name]);
                faultmess{fi,1} = exception;
                fi = fi+1;
                continue
            end
        end
    end
    
end

