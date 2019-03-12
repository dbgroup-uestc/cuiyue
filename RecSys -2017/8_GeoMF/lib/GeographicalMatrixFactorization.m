function [] = GeographicalMatrixFactorization(size_file, poi_file, train_file)
   data_size = dlmread(size_file);
   poi_coos=dlmread(poi_file);
   train_data=dlmread(train_file);
   
   K = 100;
   delta = 50;
   gamma = 0.01;
   epsilon = 10;
   lambda = 10;
   max_iters = 7;
   lat_grid_num = 50;
   lng_grid_num = 50;
   
   M = data_size(1); N = data_size(2);
   C = sparse(train_data(:,1), train_data(:,2), train_data(:,3), M, N);
   
   W = C;
   [row_id, col_id, ] = find(C);
   for sp_id = 1: length(row_id)
       i = row_id(sp_id);
       j = col_id(sp_id);
       W(i, j) = log(1 + C(i, j) * (10 ^ epsilon));
   end
   C = C > 0;
   
   if cont == 0
       P = rand(M, K);
       Q = rand(N, K);
       X = rand(M, lat_grid_num * lng_grid_num);
       Y = zeros(N, lat_grid_num * lng_grid_num);
   
       max_lat = max(poi_coos(:,1));
       min_lat = min(poi_coos(:,1));
       max_lng = max(poi_coos(:,2));
       min_lng = max(poi_coos(:,2));

       % Initialize Y.
       lat_step = (max_lat - min_lat) / M;
       lng_step = (max_lng - min_lng) / N;

       for lid = 1: N
           poi_coo = poi_coos(lid, :);
           for lat_grid_idx = 1: lat_grid_num
               for lng_grid_idx = 1: lng_grid_num
                   grid_center(1) = (lat_grid_idx - 0.5) * lat_step;
                   grid_center(2) = (lng_grid_idx - 0.5) * lng_step;
                   dist = getEarthDist(poi_coo(1), poi_coo(2), grid_center(1), grid_center(2));

                   if(dist < delta)
                       Y(lid,(lat_grid_idx - 1) * lat_grid_num + lng_grid_idx) = exp(-dist / 1.5);
                   end
               end
           end
       end
   else
       disp('Loading result and continue training...');
       fout=sprintf('%s/newGeoMF.mat', filename);
       load(fout,'P','Q','X','Y');
       test_performance_2new(filename, 10);
   end
   
   disp('Initializing weight matrix...');
   tic;
   W_u = cell(M, 1); 
   parfor u = 1: M
       W_u{u} = spdiags(full(W(u,:)'), 0, N, N);
   end
   W_i = cell(N, 1); 
   parfor i = 1: N
       W_i{i} = spdiags(full(W(:,i)), 0, M, M);
   end
   toc;
   
   Ct = C.';
   Wt = W.';
   Yt = Y.';
   YtY = Yt * Y;
   
   disp('Start training...');
   for iters = 1: max_iters
       fprintf('iteration=%d\n', iters);
       
       disp('Updating P...');
       tic;
       QQ = Q' * Q; 
       parfor u = 1: M
           P(u, :) = (Q'*W_u{u}*Q + QQ + gamma*eye(K)) \ ((Q'*W_u{u} + Q') * (C(u,:)' - Y * X(u,:)'));
       end
       toc;
       
       disp('Updating Q...');
       tic;
       PP = P' * P;
       parfor i = 1: N
           Q(i, :) = (P'*W_i{i}*P + PP + gamma*eye(K)) \ ((P'*W_i{i} + P') * (C(:,i) - X * Y(i,:)'));
       end
       toc;
       
       fout=sprintf('%s/newGeoMF.mat', filename);
       save(fout,'P','Q','X','Y');
       
       test_performance_2new(filename, 10);
   end
   
   disp('Updating X...');
   tic;
   Xt = optimize_activity(Ct, Wt, X', Yt, YtY, P', Q, lambda);
   X = Xt';
   X(X < 0) = 0;
   toc;
   
   fout=sprintf('%s/newGeoMF.mat', filename);
   save(fout,'P','Q','X','Y');

   test_performance_2new(filename, 10);
end


function Xt = optimize_activity(Rt, Wt, Xt, Yt, YtY, Ut, V, reg)
YtV = Yt * V;
[L,M] = size(Xt);
user_cell = cell(M,1);
item_cell = cell(M,1);
val_cell = cell(M,1);
for i = 1:M
    w = Wt(:,i);
    r = Rt(:,i);
    Ind = w>0; wi = w(Ind); ri = r(Ind);
    if(nnz(Ind) == 0)
        Wi = zeros(0);
    else
        Wi = spdiags(sqrt(wi), 0, length(wi), length(wi));
    end
    subYt = Yt(:, Ind);
    subV = V(Ind, :);
    YC = subYt * Wi;
    
    grad_invariant =  YC * (sqrt(wi) .* (subV * Ut(:,i))) - subYt * (wi .* ri + ri) + YtV * Ut(:,i)  + reg;
    J = 1:length(grad_invariant);
    ind = grad_invariant<=0;
    
    grad_invariant = sparse(J(ind), 1, grad_invariant(ind), length(grad_invariant), 1);
    x = line_search(YC, YtY, grad_invariant, Xt(:,i));
    
    [loc, I, val ] = find(x);
    user_cell{i} = i * I;
    item_cell{i} = loc;
    val_cell{i} = val;
end
Xt = sparse(cell2mat(item_cell), cell2mat(user_cell), cell2mat(val_cell), L, M);
end


function [x] = line_search(YC, YtY, grad_i, x)
    alpha = 1; beta = 0.1;
    for iter = 1:5
        grad = grad_i + YC * (x.' * YC).' + YtY * x;
        J = 1:length(grad);
        Ind = grad < 0| x > 0;
        grad = sparse(J(Ind), 1, grad(Ind), length(grad), 1);
        for step =1:10 % search step size
            xn = max(x - alpha * grad, 0); d = xn - x;
            dt = d.';
            gradd = dt * grad;
            dyc = dt * YC; 
            dQd = dt * (YtY * d) + dyc * dyc.';
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0;
            if step == 1
                decr_alpha = ~suff_decr; xp = x;
            end
            if decr_alpha
                if suff_decr
                    x = xn; break;
                else
                    alpha = alpha * beta;
                end
            else
                if ~suff_decr || nnz(xp~=xn)==0
                    x = xp; break;
                else
                    alpha = alpha / beta; xp = xn;
                end
            end
        end
    end
end


function [ dist ] = getEarthDist(lat1,lng1,lat2,lng2)
    radLat1=rad(lat1);
    radLat2=rad(lat2);
    a=radLat1-radLat2;
    b=rad(lng1)-rad(lng2);
    
    s=2*asin(sqrt(power(sin(a/2),2)+cos(radLat1)*cos(radLat2)*power(sin(b/2),2)));
    s=s*6371.004;
    dist=s;
end


function [r]=rad(d)
    r= d*pi/180;
end



