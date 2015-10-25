function r = recode(y, num_labels)

    %       Y is a vector of labels 
    %   
    %

    r = zeros(size(y,1), num_labels);
    
    for i = 1:size(y,1)
        
       if y(i) == 0
           r(i,10) = 1;
       else
           r(i, y(i)) = 1;
       end
    end

end