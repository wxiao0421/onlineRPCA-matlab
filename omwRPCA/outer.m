
    function [val]= outer(a, b)
        c= reshape(a', 1, []);
        d= reshape(b', 1, []);
        [grid1, grid2]=meshgrid(c, d);
        val = (grid1.*grid2)';
    end