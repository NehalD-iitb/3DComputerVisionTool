function [occlusion_map] = getOcclusionMap(imgsInfo, imgs, projMatList, camPosList, camViewDirList, P, key_points_list, avg_d)
% calculates occlusion map for all the images
% returns a boolean array, with no. images X no. pts.
% 0 means not visible, 1 means visible

    voxel_factor = 3;

    P_HOM = [P ones(size(P,1),1)]';
    P_range = 1:size(P, 1);
    occlusion_map = false(length(imgsInfo), size(P,1));
       
    d = avg_d * voxel_factor;
   
    k_list = keys(imgsInfo);
    
    for i = 1:length(imgsInfo)
        imgInfo = imgsInfo(k_list{i});
        imgId = imgInfo.image_id;
        Img = imgs{imgId};
        
        fprintf('Processing Img %d\n', imgId);
        
        M = projMatList{imgId};
        C = camPosList{imgId};
        C_DIR = camViewDirList{imgId};
        
        %% direction from camera
        P_DIR = [(P(:,1) - C(1,1)) , (P(:,2)- C(2,1)), (P(:,3)-C(3,1))];
        P_DIST = sqrt(sum( P_DIR(:,1).^2 + P_DIR(:,2).^2 +  P_DIR(:,3).^2,2));
    
        %% converting into unit vector
        P_DIR = [P_DIR(:,1)./P_DIST, P_DIR(:,2)./P_DIST, P_DIR(:,3)./P_DIST ];
    
        %% Selecting points in front of camera
        % calculating cos theta between cam view and line between pt and
        % cam
        
        cos_theta = ( P_DIR(:, 1) .* C_DIR(1,1) ) + ( P_DIR(:, 2) .* C_DIR(2,1) ) + ( P_DIR(:, 3) .* C_DIR(3,1) );
        mask_same_dir = cos_theta > 0;
        
        %P = P(mask_same_dir, :);
        %P_HOM = P_HOM(:, mask_same_dir);
        %P_DIR = P_DIR(mask_same_dir, :);
        %P_range = P_range(1, mask_same_dir);
        
        
        %% selecting pts within image
        max_Y = size(Img,1);
        max_X = size(Img,2);
        
        p = M * P_HOM;
        p = p ./ repmat(p(3,:), 3, 1);
        
        mask_inside = p(1, :) > 0 & p(2, :) > 0 & p(1, :) <= max_X & p(2, :) <= max_Y;
                
        mask_visible = logical( mask_same_dir .* mask_inside' );
        
        p_v = p(:, mask_visible);
        P_v = P(mask_visible, :);
        P_DIR_v = P_DIR(mask_visible, :);
        P_DIST_v = P_DIST(mask_visible, :);
        P_range_v = P_range(1, mask_visible);
        
        % Refreshing mask
        mask = false(size(P_HOM, 2), 1);
        
        %% Taking neighborhood of key-points
        keypoints = key_points_list{imgId};
        [idx, ~] = rangesearch(single(p_v(1:2,:)'), single(keypoints(1:2,:)'), 0.75, 'NSMethod', 'exhaustive');
        
        for j = 1:size(keypoints, 2)
            if isempty(idx{j}) == 1
                [idx_temp, ~] = rangesearch(single(p_v(1:2,:)'), single(keypoints(1:2,j)'), 1.5, 'NSMethod', 'exhaustive');
                idx{j} = idx_temp{1};
            end
            
            max_neigh = min(size(idx{j}, 2), 25);
            if max_neigh > 0
                idx{j} = idx{j}(1, 1:max_neigh);
            end
        end
        
        near_keys = false(size(p_v, 2), 1);
        
        for j = 1:size(keypoints, 2)
            near_keys(idx{j}, 1) = true;
            mask( P_range_v(1, idx{j}), 1 ) = true;
        end
        
        
        %% Looping over valid points for occlusion
        P_v_init = P_v;
        P_DIST_v_init = P_DIST_v;
        P_DIR_v_init = P_DIR_v;
        %P_range_v_init = P_range_v;
        %P_range_v_near = 1:size(P_v_init, 1);
        %P_range_v_near = P_range_v_near(1, near_keys(:,1));
        
        ptCount = 0;
        
        for k = 1:size(keypoints, 2)
            
            for j = idx{k} %1:size(P_v_init, 1)
                %if near_keys(j) == true %&& mask(P_range_v_init(j)) == true

                    dist_j = P_DIST_v_init(j); % dist of current point
                    dir_j = P_DIR_v_init(j,:); % dir current point
                    P_j = P_v_init(j, :);
                
                    V_pts = [ P_j(1)+d P_j(2)+d P_j(3)+d ;
                          P_j(1)+d P_j(2)+d P_j(3)-d ;
                          P_j(1)+d P_j(2)-d P_j(3)+d ;
                          P_j(1)+d P_j(2)-d P_j(3)-d ;
                          P_j(1)-d P_j(2)+d P_j(3)+d ;
                          P_j(1)-d P_j(2)+d P_j(3)-d ;
                          P_j(1)-d P_j(2)-d P_j(3)+d ;
                          P_j(1)-d P_j(2)-d P_j(3)-d ;
                        ];

                    % cal min cos angle between current direction and voxel end
                    % points
                    V_pts_dir = V_pts - repmat(C(1:3, :)', 8, 1);
                    denom = sqrt(sum(V_pts_dir .* V_pts_dir, 2));
                    V_pts_dir = V_pts_dir ./ repmat(denom, 1, 3); 
                    min_cos_angle = min(dir_j * V_pts_dir');
                
                    cos_angle = dir_j * P_DIR_v';
                    in = cos_angle > min_cos_angle;
                
                    if sum(in) > 0
                        in = logical(in);
                        P_cone = P_v(in, :);
                        P_range_cone = P_range_v(:,in);
                        P_DIR_cone = P_DIR_v(in, :);
                        P_DIST_cone = P_DIST_v(in, :);
                    
                        assert(size(P_DIST_cone, 1) > 0)
                        % distance of all points in cone along direction of current point
                        cos_theta = (dir_j * P_DIR_cone')';
                        P_DIST_PROJ = cos_theta .* P_DIST_cone;
                
                        %calculating th for occlusion
                        mask_th = (P_cone(:,1) >= P_j(1) - d) & (P_cone(:,1) <= P_j(1) + d) ...
                              & (P_cone(:,2) >= P_j(2) - d) & (P_cone(:,2) <= P_j(2) + d) ...
                              & (P_cone(:,3) >= P_j(3) - d) & (P_cone(:,3) <= P_j(3) + d);
                        mask_th = logical(mask_th);
                        dist_th = 5 * (max(P_DIST_PROJ(mask_th,:)) - dist_j) + dist_j;
                
                        if size(dist_th, 1) > 0
                            mask_occ = logical(P_DIST_PROJ(:,1) > dist_th);
                        else
                            mask_occ = false(size(P_DIST_PROJ, 1), 1);
                        end
                    
                        if sum(mask_occ) > 0
                        
                            mask( P_range_cone(:, mask_occ) ) = false;
                    
                            %p_v = p(:, mask);
                            %P_v = P(mask, :);
                            %P_DIR_v = P_DIR(mask, :);
                            %P_DIST_v = P_DIST(mask, :);
                            %P_range_v = P_range(1, mask);

                        end
                
                    end
 
            end
            
            ptCount = ptCount + 1;
            if mod(ptCount, 50) == 0
                fprintf('%d done ', ptCount);
            end
        end
        fprintf('\n');
                
        total_visible_pts = 0;
        visible_key_points = 0;
        
        for k = 1:size(keypoints, 2)
            
            if size(idx{k}, 1) > 0
                visible_key_points = visible_key_points + 1;
                
                P_local = P_v(idx{k}, :);
                [idx_ngh, ~] = rangesearch(P_v, P_local, 15 * avg_d, 'NSMethod', 'exhaustive');
                if isempty(idx_ngh) == 0
                    for j = 1:size(idx{k})
                        if mask( P_range_v(1, idx{k}(j)), 1) == true && size(idx_ngh{j}, 2) > 0
                            mask(idx_ngh{j}, 1) = true;
                        
                            total_visible_pts = total_visible_pts + length(idx_ngh{j});
                        end
                    end
                    
                end
                
            end
        end
        
        
        
        occlusion_map(imgId, :) = mask;
        
        %visualize cloud
        %visualizeScene(P(mask,:), C, C_DIR);
       
    end
    
end